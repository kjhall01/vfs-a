"""One interval-integrated atmosphere/surface contract, outside all packages.

Cell-area amounts, SI units. Positive turbulent transfers enter the atmosphere;
precipitation and bottom radiation enter the surface. No component recomputes a
flux or chooses its own water enthalpy. The producer owns the interval.
"""

from dataclasses import dataclass, replace
import math
import torch


@dataclass(frozen=True)
class SurfaceTiles:
    """Open water then two ice categories. All fields are CPU float64 tensors.

    Fractions [C,3]; temperature K; albedo [C,3]; velocity [C,3,3] Cartesian.
    The caller explicitly supplies ice skin temperatures (no hidden heat model).
    """

    fraction: torch.Tensor
    temperature: torch.Tensor
    velocity: torch.Tensor
    albedo: torch.Tensor

    def validate(self, mesh):
        ref = mesh.cell_area
        for name in ("fraction", "temperature", "albedo", "velocity"):
            x = getattr(self, name)
            shape = (len(ref), 3, 3) if name == "velocity" else (len(ref), 3)
            if (
                x.shape != shape
                or x.dtype != ref.dtype
                or x.device != ref.device
                or not torch.isfinite(x).all()
            ):
                raise ValueError("invalid surface tile " + name)
        eps = torch.finfo(ref.dtype).eps
        if (self.fraction < 0).any() or (
            (self.fraction.sum(-1) - 1).abs() > 32 * eps
        ).any():
            raise ValueError("surface fractions must be nonnegative and sum to one")
        if (self.temperature <= 0).any() or (
            (self.albedo < 0) | (self.albedo > 1)
        ).any():
            raise ValueError("invalid skin temperature or albedo")
        radial = (self.velocity * mesh.cell_xyz[:, None]).sum(-1)
        if (radial.abs() > 128 * eps * self.velocity.abs().amax(-1).clamp_min(1)).any():
            raise ValueError("surface velocity must be tangent")


@dataclass(frozen=True)
class Exchange:
    """Immutable record of integrated amounts; tensors must not be mutated.

    vapor/sensible [C,3] kg/m2 and J/m2; stress [C,3,3] N s/m2;
    rain/snow [C] kg/m2. Energy terms [C] J/m2 use the atmospheric mixture
    reference. Water impulse is [C,3] kg m/s/m2. Stress work is positive into
    air and is finalized by the recipient's finite midpoint mechanical work.
    """

    dt: int
    start_seconds: int
    vapor: torch.Tensor
    rain: torch.Tensor
    snow: torch.Tensor
    sensible: torch.Tensor
    stress: torch.Tensor
    vapor_velocity: torch.Tensor
    precipitation_velocity: torch.Tensor
    terms: dict
    sw_surface: torch.Tensor
    lw_surface: torch.Tensor
    toa: torch.Tensor
    surface_thermal: torch.Tensor | None = None  # [C,3], J/m2; no mechanical work

    def validate(self, mesh):
        ref = mesh.cell_area
        c = len(ref)
        if (
            isinstance(self.dt, bool)
            or not isinstance(self.dt, int)
            or self.dt <= 0
            or not isinstance(self.start_seconds, int)
            or self.start_seconds < 0
        ):
            raise ValueError("invalid exchange interval")
        for name, shape in (
            ("vapor", (c, 3)),
            ("sensible", (c, 3)),
            ("stress", (c, 3, 3)),
            ("vapor_velocity", (c, 3, 3)),
            ("precipitation_velocity", (c, 3)),
            *((k, (c,)) for k in ("rain", "snow", "sw_surface", "lw_surface", "toa")),
        ):
            x = getattr(self, name)
            if (
                not isinstance(x, torch.Tensor)
                or x.shape != shape
                or x.dtype != ref.dtype
                or x.device != ref.device
                or not torch.isfinite(x).all()
            ):
                raise ValueError("invalid exchange field " + name)
        if (self.rain < 0).any() or (self.snow < 0).any():
            raise ValueError("negative precipitation")
        keys = {
            "sensible",
            "vapor_enthalpy",
            "rain_enthalpy",
            "snow_enthalpy",
            "water_potential",
            "water_kinetic",
            "radiation",
            "stress_work",
        }
        if set(self.terms) != keys:
            raise ValueError("incomplete exchange energy ledger")
        for x in self.terms.values():
            if (
                x.shape != ref.shape
                or x.dtype != ref.dtype
                or x.device != ref.device
                or not torch.isfinite(x).all()
            ):
                raise ValueError("invalid exchange energy ledger")
        eps = torch.finfo(ref.dtype).eps
        if self.surface_thermal is not None:
            x = self.surface_thermal
            if (
                x.shape != (c, 3)
                or x.dtype != ref.dtype
                or x.device != ref.device
                or not torch.isfinite(x).all()
            ):
                raise ValueError("invalid tiled surface thermal energy")
        for x in (self.vapor_velocity, self.stress):
            radial = (x * mesh.cell_xyz[:, None]).sum(-1)
            if (radial.abs() > 128 * eps * x.abs().amax(-1).clamp_min(1)).any():
                raise ValueError("exchange vectors must be tangent")
        x = self.precipitation_velocity
        if (
            (x * mesh.cell_xyz).sum(-1).abs()
            > 128 * eps * x.abs().amax(-1).clamp_min(1)
        ).any():
            raise ValueError("precipitation velocity must be tangent")

    @property
    def water(self):
        return self.vapor.sum(-1) - self.rain - self.snow

    @property
    def water_impulse(self):
        return (self.vapor[..., None] * self.vapor_velocity).sum(1) - (
            self.rain + self.snow
        )[:, None] * self.precipitation_velocity

    @property
    def atmosphere_energy(self):
        return sum(self.terms.values())

    @property
    def surface_energy(self):
        return self.toa - self.atmosphere_energy

    def with_stress_work(self, into_air):
        if into_air.shape != self.toa.shape or not torch.isfinite(into_air).all():
            raise ValueError("finite cell stress work required")
        return replace(self, terms={**self.terms, "stress_work": into_air})

    @classmethod
    def from_am2(cls, adapter, state, dt, column, out, tiles):
        tiles.validate(adapter.model.mesh)
        if not math.isfinite(dt) or dt <= 0 or not float(dt).is_integer():
            raise ValueError("positive integer exchange interval required")
        f = out["tile_fluxes"]
        vapor, sensible = dt * f["flux_q"], dt * f["flux_t"]
        rain, snow = dt * out["lprec"], dt * out["fprec"]
        if (rain < 0).any() or (snow < 0).any():
            raise ValueError("precipitation must be nonnegative")
        m = adapter.mapping
        stress = dt * (
            f["flux_u"][..., None] * m.east[:, None]
            + f["flux_v"][..., None] * m.north[:, None]
        )
        va = m.vector(state.atmosphere.u_e)[:, -1]
        vv = torch.where((vapor >= 0)[..., None], tiles.velocity, va[:, None])
        Ta = column["T"][:, -1]
        Tv = torch.where(vapor >= 0, tiles.temperature, Ta[:, None])
        th = adapter.model.dynamics.thermodynamics
        hv = th.L_v_ref + th.cp_v * (Tv - th.T_ref)
        hl = th.cp_l * (Ta - th.T_ref)
        hi = -th.L_f_ref + th.cp_i * (Ta - th.T_ref)
        zero = torch.zeros_like(rain)
        sw, lw, toa = zero, zero, zero
        r = out["radiation"]
        if r is not None:
            # Use the SAME cached interface fluxes used by the air ledger.
            # Re-emitting sigma*T_current**4 between radiation calls breaks it.
            sw = dt * (r["dfsw"][:, -1] - r["ufsw"][:, -1])
            lw = -dt * r["flxnet"][:, -1]
            toa = dt * (r["dfsw"][:, 0] - r["ufsw"][:, 0] - r["flxnet"][:, 0])
        phi = torch.as_tensor(
            state.atmosphere.column.phi_s, dtype=rain.dtype, device=rain.device
        )
        terms = dict(
            sensible=sensible.sum(-1),
            vapor_enthalpy=(vapor * hv).sum(-1),
            rain_enthalpy=-rain * hl,
            snow_enthalpy=-snow * hi,
            water_potential=(vapor.sum(-1) - rain - snow) * phi,
            water_kinetic=(0.5 * vapor * vv.square().sum(-1)).sum(-1)
            - 0.5 * (rain + snow) * va.square().sum(-1),
            radiation=toa - sw - lw,
            stress_work=(stress * tiles.velocity).sum((1, 2)),
        )
        # Partition the SAME cached net radiation, preserving its cell sum.
        # SW follows tile absorptivity; LW uses unit emissivity and a common
        # downward flux inferred from that cached net. This is a mosaic closure,
        # not three independently calculated atmospheric radiation columns.
        absorption = tiles.fraction * (1 - tiles.albedo)
        denom = absorption.sum(-1, keepdim=True)
        weights = torch.where(
            denom > 0, absorption / denom.clamp_min(1e-300), tiles.fraction
        )
        thermal = sw[:, None] * weights + lw[:, None] * tiles.fraction
        if r is not None:
            emission = 5.670374419e-8 * tiles.temperature**4
            thermal = thermal + dt * tiles.fraction * (
                (tiles.fraction * emission).sum(-1, keepdim=True) - emission
            )
        thermal = thermal - sensible - vapor * hv
        # Rain drains through ice. Snow is intercepted by each tile.
        thermal = thermal + snow[:, None] * tiles.fraction * hi[:, None]
        thermal = thermal + torch.stack((rain * hl, zero, zero), -1)
        return cls(
            int(dt),
            state.time_seconds,
            vapor,
            rain,
            snow,
            sensible,
            stress,
            vv,
            va,
            terms,
            sw,
            lw,
            toa,
            thermal,
        )
