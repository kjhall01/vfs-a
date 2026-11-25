"""Zero-layer snow on moving category ice, with explicit frozen-mass ownership.

CategoryIce.volume stores ALL solid water in ice-equivalent m3/m2. snow_mass
labels its snow subset in kg/m2; it is not added to H a second time. Mechanics
retain the existing ice-equivalent hydrostatic geometry. Porous snow depth is
used for conduction/albedo, not for a new gravitational energy term. This is
an explicit approximation, not a resolved snow-enthalpy or freeboard model.
"""

from dataclasses import dataclass, replace
import math
import torch
from vfs.ice.category_phase import CategoryThermodynamics
from vfs.ice.ocean_ice import safe
from vfs.ice.ridging import RidgingState


@dataclass(frozen=True)
class SnowParameters:
    density: float = 330.0  # kg/m3, diagnostic porous depth
    conductivity: float = 0.30  # W/(m K)
    albedo: float = 0.85
    ice_albedo: float = 0.60
    water_albedo: float = 0.06
    patch_depth: float = 0.02  # m, smooth snow-cover transition
    air_conductance: float = 10.0  # W/(m2 K), prescribed bulk exchange

    def __post_init__(self):
        positive = (
            self.density,
            self.conductivity,
            self.patch_depth,
            self.air_conductance,
        )
        albedos = (self.albedo, self.ice_albedo, self.water_albedo)
        if any(not math.isfinite(x) or x <= 0 for x in positive) or any(
            not math.isfinite(x) or not 0 <= x <= 1 for x in albedos
        ):
            raise ValueError(
                "positive finite snow parameters and albedos in [0,1] required"
            )


@dataclass(frozen=True)
class SnowState:
    coupled: RidgingState  # volume includes the snow mass
    snow_mass: torch.Tensor  # [cell,2], kg/m2 of horizontal cell, subset of solid mass


@dataclass(frozen=True)
class SurfaceFlux:
    heat: torch.Tensor  # [cell,1], net external W/m2
    albedo: torch.Tensor  # [cell,1], grid-area mean; zero on land
    absorbed_shortwave: torch.Tensor  # [cell,1], W/m2
    reflected_shortwave: torch.Tensor  # [cell,1], W/m2
    surface_temperature: torch.Tensor  # [cell,2], deg C on each frozen category


@dataclass(frozen=True)
class SnowReport:
    external_energy: torch.Tensor  # J, heat + wind + incoming snow energy/work
    snowfall_mass: torch.Tensor  # kg, only snow intercepted by covered area
    snowfall_energy: torch.Tensor  # J, boundary enthalpy plus incorporation work
    melted_snow: torch.Tensor  # kg, transferred to liquid by phase change
    freshwater: tuple | None = None  # optional reports from the two boundary half steps


class SnowPhase(CategoryThermodynamics):
    """Same scalar energy solve, keeping footprint during snow-only melting."""

    def __init__(self, coupled, initial_snow):
        super().__init__(coupled)
        self.initial_snow = initial_snow

    @staticmethod
    def _melt_area(old_area, old_ice_mass, new_mass):
        fraction = torch.where(
            old_ice_mass > 0,
            torch.minimum(new_mass / safe(old_ice_mass), torch.ones_like(new_mass)),
            (new_mass > 0).to(new_mass.dtype),
        )
        return old_area * fraction

    def geometry(self, initial, m):
        ice, da, dv, thickness = super().geometry(initial, m)
        rho = self.H.slab.parameters.density
        old = initial.ice
        old_ice_mass = rho * old.volume - self.initial_snow
        new_mass = rho * ice.volume
        # Snow melts first. Until underlying ice itself melts, losing snow
        # reduces the column's depth, not its horizontal footprint.
        melt_area = self._melt_area(old.area, old_ice_mass, new_mass)
        melt_da = torch.where(
            new_mass < old_ice_mass, old.area * rho * dv / safe(old_ice_mass), 0.0
        )
        melting = m < self.H.mass(initial)
        area = torch.where(melting, melt_area, ice.area)
        da = torch.where(melting, melt_da, da)
        virtual = old_ice_mass / (rho * safe(old.area))
        thickness = torch.where(
            area > 0, ice.volume / safe(area), torch.where(melting, virtual, thickness)
        )
        return replace(ice, area=area), da, dv, thickness


class SnowOceanIce:
    """Compose snow-labelled frozen water with the existing coupled integrator.

    Snowfall is kg/m2/s per ice-covered area. Only its covered-area interception
    enters this model; precipitation on open water belongs to the ocean forcing
    driver. Snow arrives co-moving with ice, at the zero-layer solid enthalpy
    datum. The returned boundary ledger includes that mass, latent energy and
    the mechanical work of incorporation. No atmospheric sensible/impact heat.
    """

    def __init__(self, coupled, parameters=SnowParameters()):
        self.coupled = coupled
        self.parameters = parameters
        self.hamiltonian = coupled.hamiltonian
        self.phase = CategoryThermodynamics(coupled)

    def check(self, state):
        """Validate ocean, sea ice, and snow at an externally supplied state."""
        self.coupled.check(state.coupled)
        self._check_snow_stores(state)

    def _check_snow_stores(self, state):
        """Validate snow only when the same coupled stage is checked below."""
        m = state.snow_mass
        total = self.hamiltonian.slab.parameters.density * state.coupled.ice.volume
        if (
            m.shape != total.shape
            or m.dtype != total.dtype
            or m.device != total.device
            or not torch.isfinite(m).all()
            or (m < 0).any()
            or (m > total).any()
        ):
            raise ValueError(
                "snow_mass must be finite [cell,2], matching state, between zero and total solid mass"
            )

    def bare(self, state):
        """Wrap an existing ice state with no snow; no change to energy or mass."""
        result = SnowState(state, torch.zeros_like(state.ice.volume))
        self.check(result)
        return result

    def depths(self, state):
        """Actual ice depth and diagnostic porous snow depth, per covered area."""
        ice = state.coupled.ice
        rho = self.hamiltonian.slab.parameters.density
        hi = (ice.volume - state.snow_mass / rho) / safe(ice.area)
        hs = state.snow_mass / (self.parameters.density * safe(ice.area))
        return hi, hs

    def field(self, state, value, name, *, nonnegative=False):
        sample = state.coupled.ice.area[:, :1]
        if (
            value.shape != sample.shape
            or value.dtype != sample.dtype
            or value.device != sample.device
            or not torch.isfinite(value).all()
            or (nonnegative and (value < 0).any())
        ):
            raise ValueError(
                f"{name} must be finite [cell,1], matching state"
                + (" and nonnegative" if nonnegative else "")
            )
        return value

    def surface_flux(self, state, air_temperature, shortwave=None):
        self.check(state)
        air = self.field(state, air_temperature, "air_temperature")
        sw = (
            torch.zeros_like(air)
            if shortwave is None
            else self.field(state, shortwave, "shortwave", nonnegative=True)
        )
        s = state.coupled.ocean.ocean
        ice = state.coupled.ice
        p = self.parameters
        ip = self.hamiltonian.slab.parameters
        hi, hs = self.depths(state)
        fraction = hs / (hs + p.patch_depth)
        alpha = p.ice_albedo + fraction * (p.albedo - p.ice_albedo)
        absorbed = (1 - alpha) * sw
        resistance = hi / ip.conductivity + hs / p.conductivity
        base = ip.freezing_temperature(s.salinity()[:, :1])
        surface = (base + resistance * (p.air_conductance * air + absorbed)) / (
            1 + p.air_conductance * resistance
        )
        surface = torch.minimum(
            surface, torch.zeros_like(surface)
        )  # physical melting constraint
        covered = (ice.area * (p.air_conductance * (air - surface) + absorbed)).sum(
            -1, keepdim=True
        )
        wet = self.hamiltonian.model.energy.mass_area(s) > 0
        open_fraction = wet.to(s.h.dtype) - ice.area.sum(-1, keepdim=True)
        heat = covered + open_fraction * (
            p.air_conductance * (air - s.temperature()[:, :1])
            + (1 - p.water_albedo) * sw
        )
        albedo = (ice.area * alpha).sum(
            -1, keepdim=True
        ) + open_fraction * p.water_albedo
        reflected = albedo * sw
        absorbed_grid = wet.to(s.h.dtype) * sw - reflected
        return SurfaceFlux(heat, albedo, absorbed_grid, reflected, surface)

    def accumulate(self, state, dt, snowfall):
        """Deposit co-moving solid water; return (state, incoming kg, incoming J)."""
        self.check(state)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("positive finite dt required")
        rate = self.field(state, snowfall, "snowfall", nonnegative=True)
        old = state.coupled
        H = self.hamiltonian
        s, ice = old.ocean.ocean, old.ice
        rho = H.slab.parameters.density
        dm = dt * rate * ice.area
        dv = dm / rho
        end = replace(old, ice=replace(ice, volume=ice.volume + dv))
        result = SnowState(end, state.snow_mass + dm)
        # Exact integral of the declared incoming-mass conjugate along addition
        # at fixed category area and velocity. This is a boundary-work formula,
        # not subtraction of a computed global energy residual.
        m = H.mass(old)
        mass_added = dm.sum(-1, keepdim=True)
        V = ice.volume.sum(-1, keepdim=True)
        dV = dv.sum(-1, keepdim=True)
        g = s.column.gravity
        potential = g * mass_added * s.column.eta(s.h) + g / (2 * rho) * (
            2 * m * mass_added + mass_added.square()
        )
        variance = H.coefficient(old) * (
            ((2 * ice.volume * dv + dv.square()) / safe(ice.area)).sum(-1, keepdim=True)
            - 2 * V * dV
            - dV.square()
        )
        latent = -H.slab.parameters.latent_heat * mass_added
        mesh = s.mesh
        kinetic = (
            0.5
            * (
                (mesh.dc_edge * mesh.dv_edge)[:, None]
                * mesh.cell_to_edge(mass_added)
                * ice.velocity.square()
            ).sum()
        )
        area = mesh.cell_area[:, None]
        energy = (area * (latent + potential + variance)).sum() + kinetic
        self.check(result)
        return result, (area * mass_added).sum(), energy

    def phase_step(self, state, dt, heat_flux):
        self.check(state)
        phase = (
            SnowPhase(self.coupled, state.snow_mass)
            if (state.snow_mass > 0).any()
            else self.phase
        )
        end, report = phase.apply(state.coupled, dt, heat_flux=heat_flux)
        rho = self.hamiltonian.slab.parameters.density
        loss = (rho * (state.coupled.ice.volume - end.ice.volume)).clamp_min(0)
        melt = torch.minimum(loss, state.snow_mass)
        result = SnowState(end, state.snow_mass - melt)
        self.check(result)
        area = end.ocean.ocean.mesh.cell_area[:, None]
        return result, report.external_heat.sum(), (area * melt).sum()

    def tendency(self, state, dt, **options):
        # Coupled.tendency performs the complete ocean/ice stage check on the
        # identical unmodified state. Check the snow store here; do not repeat
        # the expensive ocean temperature/caloric validation beforehand.
        self._check_snow_stores(state)
        rhs, report = self.coupled.tendency(state.coupled, dt, **options)
        ice = state.coupled.ice
        mesh = state.coupled.ocean.ocean.mesh
        c0, c1 = mesh.cellsOnEdge.T
        ds = torch.zeros_like(state.snow_mass)
        if options.get("transport", True):
            donor = torch.where(
                ice.velocity >= 0, state.snow_mass[c0], state.snow_mass[c1]
            )
            ds = -mesh.div(donor * ice.velocity)
        if options.get("ridging", True):
            # Same donor fraction as the solid-water transfer; compaction
            # changes footprint only. Snow remains with its frozen carrier.
            compression = (-mesh.div(ice.velocity)).clamp_min(0)
            ridge = self.coupled.ridge
            chi = (
                ice.area.sum(-1, keepdim=True).pow(ridge.crowding_power)
                * compression
                / (compression + ridge.strain_rate)
            )
            transfer = chi * compression * state.snow_mass[:, :1]
            ds = ds + torch.cat((-transfer, transfer), -1)
        return SnowState(rhs, ds), report

    def add(self, state, rhs, dt):
        return SnowState(
            self.coupled.add(state.coupled, rhs.coupled, dt),
            state.snow_mass + dt * rhs.snow_mass,
        )

    def blend(self, first, second, weight):
        return SnowState(
            self.coupled.blend(first.coupled, second.coupled, weight),
            (1 - weight) * first.snow_mass + weight * second.snow_mass,
        )

    def dynamics_step(self, state, dt, *, integrator="ssprk3", **options):
        """Choose the explicit ocean–ice tableau without changing stage physics."""
        if integrator == "ssprk43":
            from vfs.ice.explicit_rk import ssprk43_dynamics
            return ssprk43_dynamics(self, state, dt, **options)
        if integrator != "ssprk3":
            raise ValueError("ocean–ice integrator must be ssprk3 or ssprk43")
        # Preserve the historical three-stage path, including all arithmetic.
        r0, p0 = self.tendency(state, dt, **options)
        first = self.add(state, r0, dt)
        r1, p1 = self.tendency(first, dt, **options)
        middle = self.blend(state, self.add(first, r1, dt), 0.25)
        r2, p2 = self.tendency(middle, dt, **options)
        end = self.blend(state, self.add(middle, r2, dt), 2 / 3)
        self.check(end)
        return end, dt * (p0.wind_power / 6 + p1.wind_power / 6 + 2 * p2.wind_power / 3)

    def mixing_step(self, state, dt, *, stage_callback=None):
        """Apply the existing conservative column closure at its own cadence.

        The source state already includes the preceding dynamics/ice operations;
        this wrapper changes no mixing equations, TKE funding, or energy checks.
        It preserves snow and thermal-ice storage on the returned state.
        """
        model = self.coupled.ocean
        if model.closure is None:
            return state
        original = state.coupled.ocean
        mixed, k, _ = model.closure.apply(
            model.layers, original.ocean, original.tke_h, dt
        )
        if not torch.equal(mixed.h, original.ocean.h):
            raise ValueError("snow coupling requires fixed-geometry mixing")
        state = replace(
            state,
            coupled=replace(
                state.coupled, ocean=original.with_layers(mixed, tke_h=k)
            ),
        )
        if stage_callback is not None:
            stage_callback("vertical_tke_mixing", state, 0.0)
        return state

    def step(
        self,
        state,
        dt,
        *,
        snowfall=None,
        air_temperature=None,
        shortwave=None,
        heat_flux=None,
        wind_velocity=None,
        freshwater=None,
        stage_callback=None,
        dynamics_integrator="ssprk3",
        apply_mixing=True,
    ):
        """Full moving ocean/ice/snow step; reports all applied boundary energy."""
        self.check(state)
        if freshwater is not None and snowfall is not None:
            raise ValueError("use freshwater.snowfall or snowfall, not both")
        if heat_flux is not None and (
            air_temperature is not None or shortwave is not None
        ):
            raise ValueError("choose net heat_flux or air_temperature/shortwave")
        if shortwave is not None and air_temperature is None:
            raise ValueError("shortwave requires air_temperature")
        zero = torch.zeros_like(state.snow_mass[:, :1])
        snowfall = zero if snowfall is None else snowfall
        prescribed = zero if heat_flux is None else heat_flux

        def flux(current):
            return (
                prescribed
                if air_temperature is None
                else self.surface_flux(current, air_temperature, shortwave).heat
            )

        reports = None
        if freshwater is None:
            state, dm0, ep0 = self.accumulate(state, 0.5 * dt, snowfall)
            boundary0 = ep0
        else:
            from vfs.ice.freshwater import FreshwaterBoundary

            boundary = FreshwaterBoundary(self)
            state, report0 = boundary.apply(state, 0.5 * dt, freshwater)
            dm0 = report0.inputs["snowfall_on_ice"].sum()
            ep0 = report0.energy_inputs["snowfall_on_ice"].sum()
            boundary0 = report0.external_energy
        if stage_callback is not None:
            stage_callback("snow_input_half_0", state, boundary0)
        state, q0, melt0 = self.phase_step(state, 0.5 * dt, flux(state))
        if stage_callback is not None:
            stage_callback("phase_half_0", state, q0)
        coupled, _ = self.coupled.drag_step(state.coupled, 0.5 * dt)
        state = replace(state, coupled=coupled)
        if stage_callback is not None:
            stage_callback("implicit_drag_half_0", state, 0.0)
        state, work = self.dynamics_step(
            state, dt, wind_velocity=wind_velocity, drag=False,
            integrator=dynamics_integrator,
        )
        if stage_callback is not None:
            stage_callback("coupled_" + dynamics_integrator, state, work)
        if apply_mixing:
            state = self.mixing_step(state, dt, stage_callback=stage_callback)
        coupled, _ = self.coupled.drag_step(state.coupled, 0.5 * dt)
        state = replace(state, coupled=coupled)
        if stage_callback is not None:
            stage_callback("implicit_drag_half_1", state, 0.0)
        state, q1, melt1 = self.phase_step(state, 0.5 * dt, flux(state))
        if stage_callback is not None:
            stage_callback("phase_half_1", state, q1)
        if freshwater is None:
            state, dm1, ep1 = self.accumulate(state, 0.5 * dt, snowfall)
            boundary1 = ep1
        else:
            state, report1 = boundary.apply(state, 0.5 * dt, freshwater)
            dm1 = report1.inputs["snowfall_on_ice"].sum()
            ep1 = report1.energy_inputs["snowfall_on_ice"].sum()
            boundary1 = report1.external_energy
            reports = (report0, report1)
        if stage_callback is not None:
            stage_callback("snow_input_half_1", state, boundary1)
        return state, SnowReport(
            work + q0 + q1 + boundary0 + boundary1,
            dm0 + dm1,
            ep0 + ep1,
            melt0 + melt1,
            reports,
        )
