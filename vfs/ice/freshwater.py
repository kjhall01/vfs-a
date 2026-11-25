"""Prescribed freshwater boundary for the common ocean/ice/snow Hamiltonian.

No bulk evaporation law or land hydrology is inferred. Rates are interval means.
Rain runs through ice immediately; snowfall is intercepted by frozen cover.
Liquid/snow arrival and vapor departure carry explicitly reported energy and
native momentum. The cell-local thermal solve is the boundary first law, not
a repair applied to the dynamical integrator.
"""

from dataclasses import dataclass, replace
import math
import torch
from vfs.ice.ocean_ice import safe
from vfs.ocean.temperature.roquet import RoquetWorkspace
from vfs.ocean.energy import ConstantDensityEnergy


@dataclass(frozen=True)
class LiquidWater:
    rate: torch.Tensor  # [cell,1], kg/m2/s of horizontal cell area
    enthalpy: torch.Tensor  # [cell,1], thermal J/kg in the model's cp*CT datum
    velocity: torch.Tensor  # [edge,1], incoming native velocity, m/s


@dataclass(frozen=True)
class FreshwaterFluxes:
    rain: LiquidWater | None = None
    snowfall: torch.Tensor | None = None  # [cell,1], kg/m2/s over the whole wet cell
    evaporation: torch.Tensor | None = None  # kg/m2/s per OPEN-WATER area
    sublimation: torch.Tensor | None = None  # kg/m2/s per ICE-COVERED area
    runoff: LiquidWater | None = None  # cell-area rate; not reduced by ice coverage


@dataclass(frozen=True)
class FreshwaterReport:
    inputs: dict  # named [cell,1] kg, snow split between solid and liquid recipients
    outputs: dict  # named [cell,1] kg
    energy_inputs: dict  # named [cell,1] J, signed into the combined system
    external_energy: torch.Tensor  # global J
    ocean_impulse: torch.Tensor  # [edge,1], generalized momentum input
    ice_impulse: torch.Tensor  # [edge,1], generalized momentum input
    water_residual: torch.Tensor  # [cell,1], kg
    momentum_residual: torch.Tensor  # [edge,1], combined generalized momentum
    thermal_adjustment: torch.Tensor  # [cell,1], J in the local boundary solve
    energy_residual: torch.Tensor  # [cell,1], J, evaluated independently after solve


class FreshwaterBoundary:
    """Finite water/energy/momentum exchange on a SnowOceanIce state.

    Ice is fresh, and all salt, passive tracers, TKE and eddy-energy contents stay
    in the liquid. Rain/runoff entry velocity and thermal enthalpy are explicit.
    Snow arrives co-moving with its recipient at the solid datum -Lf. Vapor leaves
    at donor velocity: cp*CT+Lv over water, Lv over zero-layer solid. Sublimation
    removes snow first, then underlying ice. Lv is an explicit constant approximation.
    Hydrostatic incorporation work includes surface elevation and overlying ice.
    """

    def __init__(self, snow, *, vaporization_heat=2.5e6):
        if not math.isfinite(vaporization_heat) or vaporization_heat <= 0:
            raise ValueError("positive finite vaporization heat required")
        self.snow, self.H = snow, snow.hamiltonian
        self.vaporization_heat = vaporization_heat

    def liquid(self, rate, temperature, velocity):
        """Declare liquid input using the idealized thermal reference h=cp*CT."""
        return LiquidWater(rate, self.H.model.config.cp * temperature, velocity)

    def _field(self, state, value, name, *, nonnegative=False):
        zero = torch.zeros_like(state.snow_mass[:, :1])
        if value is None:
            return zero
        if not isinstance(value, torch.Tensor):
            raise TypeError(name + " must be a tensor")
        return self.snow.field(state, value, name, nonnegative=nonnegative)

    def _liquid(self, state, value, name, wet, active):
        zero = torch.zeros_like(state.snow_mass[:, :1])
        velocity = torch.zeros_like(state.coupled.ice.velocity)
        if value is None:
            return zero, zero, velocity
        if not isinstance(value, LiquidWater):
            raise TypeError(name + " must be LiquidWater")
        if not isinstance(value.rate, torch.Tensor) or not isinstance(
            value.enthalpy, torch.Tensor
        ):
            raise TypeError(name + " requires explicit rate and enthalpy tensors")
        rate = self._field(state, value.rate, name + " rate", nonnegative=True)
        enthalpy = self._field(state, value.enthalpy, name + " enthalpy")
        v = value.velocity
        if (
            not isinstance(v, torch.Tensor)
            or v.shape != velocity.shape
            or v.dtype != zero.dtype
            or v.device != zero.device
            or not torch.isfinite(v).all()
        ):
            raise ValueError(name + " velocity must be finite [edge,1], matching state")
        if name == "runoff" and (rate[~wet] != 0).any():
            raise ValueError(
                "runoff must enter wet cells; land discharge cannot be discarded"
            )
        return torch.where(wet, rate, 0.0), enthalpy, torch.where(active, v, 0.0)

    def _solid_change(self, before, after):
        """Analytic solid latent/PE/KE increment, with liquid geometry fixed."""
        H = self.H
        s, i0, i1 = before.ocean.ocean, before.ice, after.ice
        p = H.slab.parameters
        m = H.mass(before)
        dv = i1.volume - i0.volume
        dV = dv.sum(-1, keepdim=True)
        dm = p.density * dV
        V = i0.volume.sum(-1, keepdim=True)
        hydro = s.column.gravity * (
            dm * s.column.eta(s.h) + (2 * m * dm + dm.square()) / (2 * p.density)
        )
        ridge = H.coefficient(before) * (
            (i1.volume * i1.thickness() - i0.volume * i0.thickness()).sum(
                -1, keepdim=True
            )
            - 2 * V * dV
            - dV.square()
        )
        kinetic = 0.5 * ConstantDensityEnergy.interpolation_transpose(
            s.mesh,
            H.metric(after) * i1.velocity.square()
            - H.metric(before) * i0.velocity.square(),
        )
        return (
            s.mesh.cell_area[:, None] * (-p.latent_heat * dm + hydro + ridge) + kinetic
        )

    def _sublimate(self, state, amount):
        old = state.coupled
        ice = old.ice
        rho = self.H.slab.parameters.density
        total = rho * ice.volume
        if (amount > total).any():
            raise ValueError("sublimation exhausts a category; reduce dt, no clipping")
        snow_loss = torch.minimum(amount, state.snow_mass)
        remaining = total - amount
        underlying = total - state.snow_mass
        fraction = torch.where(
            underlying > 0,
            torch.minimum(remaining / safe(underlying), torch.ones_like(total)),
            (remaining > 0).to(total.dtype),
        )
        new_ice = replace(
            ice,
            area=torch.where(amount > 0, ice.area * fraction, ice.area),
            volume=torch.where(amount > 0, remaining / rho, ice.volume),
        )
        end = replace(old, ice=new_ice)
        end = replace(
            end,
            ice=replace(
                new_ice, velocity=torch.where(self.H.metric(end) > 0, ice.velocity, 0.0)
            ),
        )
        return replace(state, coupled=end, snow_mass=state.snow_mass - snow_loss)

    def apply(self, initial, dt, fluxes):
        self.snow.check(initial)
        if not isinstance(fluxes, FreshwaterFluxes):
            raise TypeError("fluxes must be FreshwaterFluxes")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("positive finite dt required")
        H = self.H
        original = initial.coupled
        s = original.ocean.ocean
        mesh, rho = s.mesh, s.column.rho0
        area = mesh.cell_area[:, None]
        wet = H.model.energy.mass_area(s) > 0
        active = torch.ones_like(original.ice.velocity, dtype=torch.bool)
        if H.model.domain is not None:
            active = H.model.domain.active_edge[:, None]
        rain, rain_h, rain_u = self._liquid(initial, fluxes.rain, "rain", wet, active)
        runoff, runoff_h, runoff_u = self._liquid(
            initial, fluxes.runoff, "runoff", wet, active
        )
        snow = self._field(initial, fluxes.snowfall, "snowfall", nonnegative=True)
        evap = self._field(initial, fluxes.evaporation, "evaporation", nonnegative=True)
        sub = self._field(initial, fluxes.sublimation, "sublimation", nonnegative=True)
        cover = original.ice.area.sum(-1, keepdim=True)
        opened = wet.to(s.h.dtype) - cover
        dm_r, dm_q = dt * rain, dt * runoff
        dm_s, dm_e = dt * opened * snow, dt * opened * evap
        dm_sub = dt * sub * original.ice.area
        deposited, _, _ = self.snow.accumulate(initial, dt, snow)
        snowfall_energy = self._solid_change(original, deposited.coupled)
        frozen = self._sublimate(deposited, dm_sub)
        sublimation_energy = self._solid_change(deposited.coupled, frozen.coupled)
        latent_cooling = (
            -area
            * dm_sub.sum(-1, keepdim=True)
            * (self.vaporization_heat + H.slab.parameters.latent_heat)
        )

        dm = dm_r + dm_q + dm_s - dm_e
        dh = dm / rho
        h = torch.cat((s.h[:, :1] + dh, s.h[:, 1:]), -1)
        if (h[:, :1][wet] < self.snow.phase.min_liquid).any():
            raise ValueError("freshwater exchange exhausts top layer; reduce dt")
        # Freshwater changes only the surface layer, whose thickness was checked above.
        salinity = s.salt_h[:, :1] / h[:, :1]
        if ((salinity <= 0) | (salinity >= 42)).any():
            raise ValueError("freshwater exchange leaves salinity domain; reduce dt")
        thermal = torch.cat(
            (s.temp_h[:, :1] + dh * s.temperature()[:, :1], s.temp_h[:, 1:]), -1
        )
        trial = s.with_tensors(s.u_e, h, thermal, s.salt_h, s.tracer_h)
        mechanical = H.model.energy.base.mechanical
        metric0, metric1 = mechanical.metric(s)[:, :1], mechanical.metric(trial)[:, :1]
        edge_area = (mesh.dc_edge * mesh.dv_edge)[:, None]

        def edge_mass(amount):
            return torch.where(active, edge_area * mesh.cell_to_edge(amount), 0.0)

        er, eq, es, ee = [edge_mass(x) for x in (dm_r, dm_q, dm_s, dm_e)]
        u0 = s.u_e[:, :1]
        impulse = er * rain_u + eq * runoff_u + (es - ee) * u0
        u1 = torch.where(
            active, u0 + (impulse - edge_mass(dm) * u0) / safe(metric1), 0.0
        )
        trial = trial.with_tensors(
            torch.cat((u1, s.u_e[:, 1:]), -1), h, trial.temp_h, s.salt_h, s.tracer_h
        )
        candidate = replace(
            frozen,
            coupled=replace(frozen.coupled, ocean=original.ocean.with_layers(trial)),
        )
        # Boundary convention: exact hydrostatic incorporation work at fixed
        # solid mass during this liquid stage, and each stream's incoming KE.
        water_work = (
            area
            * s.column.gravity
            * (dm * (s.column.eta(s.h) + 0.5 * dh) + H.mass(frozen.coupled) * dh)
        )
        kinetic = 0.5 * ConstantDensityEnergy.interpolation_transpose(
            mesh,
            er * rain_u.square() + eq * runoff_u.square() + (es - ee) * u0.square(),
        )
        energy_inputs = dict(
            rain=area * dm_r * rain_h,
            runoff=area * dm_q * runoff_h,
            snowfall_open_water=-area * dm_s * H.slab.parameters.latent_heat,
            evaporation=-area
            * dm_e
            * (H.model.config.cp * s.temperature()[:, :1] + self.vaporization_heat),
            snowfall_on_ice=snowfall_energy,
            sublimated_solid=sublimation_energy,
            sublimation_latent=latent_cooling,
            liquid_incorporation=water_work,
            liquid_kinetic=kinetic,
        )
        external = sum(energy_inputs.values())
        before = H.terms(original)
        provisional = H.terms(candidate.coupled)
        change = sum(provisional[k] - before[k] for k in before)[:, None]
        correction = external - change
        top_mass = H.model.energy.mass_area(trial) * h[:, :1]
        specific = torch.cat(
            (correction / safe(top_mass), torch.zeros_like(h[:, 1:])), -1
        )
        physical, _, _ = RoquetWorkspace(H.model.layers, trial).heat(
            trial, trial, specific, allow_cooling=True
        )
        end = replace(
            candidate,
            coupled=replace(
                candidate.coupled,
                ocean=candidate.coupled.ocean.with_layers(physical),
            ),
        )
        self.snow.check(end)
        final = H.terms(end.coupled)
        residual = sum(final[k] - before[k] for k in before)[:, None] - external
        scale = sum(v.abs() for v in before.values())[:, None] + external.abs()
        if (
            not torch.isfinite(residual).all()
            or (
                residual.abs() > 2048 * torch.finfo(h.dtype).eps * scale.clamp_min(1)
            ).any()
        ):
            raise FloatingPointError("freshwater boundary full-H energy closure failed")
        inputs = dict(
            rain=area * dm_r,
            runoff=area * dm_q,
            snowfall_open_water=area * dm_s,
            snowfall_on_ice=area
            * (dt * snow * original.ice.area).sum(-1, keepdim=True),
        )
        outputs = dict(
            evaporation=area * dm_e, sublimation=area * dm_sub.sum(-1, keepdim=True)
        )
        expected_mass = sum(inputs.values()) - sum(outputs.values())
        water_residual = (
            area
            * (
                rho * (h - s.h).sum(-1, keepdim=True)
                + H.mass(end.coupled)
                - H.mass(original)
            )
            - expected_mass
        )
        ice_impulse = (
            H.metric(end.coupled) * end.coupled.ice.velocity
            - H.metric(original) * original.ice.velocity
        )
        momentum_residual = metric1 * u1 - metric0 * u0 - impulse
        return end, FreshwaterReport(
            inputs,
            outputs,
            energy_inputs,
            external.sum(),
            impulse,
            ice_impulse,
            water_residual,
            momentum_residual,
            correction,
            residual,
        )

    def route_runoff(self, state, discharge, temperature, receivers, weights, velocity):
        """Map river Q [river] m3/s to explicit wet receivers [river,slot].

        Each row of nonnegative weights must sum to one. River temperatures use
        the same cp*CT reference as liquid(). Receiver locations are supplied by
        the caller; this is a conservative discharge adapter, not river routing.
        """
        self.snow.check(state)
        s = state.coupled.ocean.ocean
        tensors = (discharge, temperature, weights)
        if any(
            not isinstance(x, torch.Tensor)
            or x.dtype != s.h.dtype
            or x.device != s.h.device
            or not torch.isfinite(x).all()
            for x in tensors
        ):
            raise ValueError("river fields must be finite tensors matching state")
        if (
            discharge.ndim != 1
            or temperature.shape != discharge.shape
            or receivers.ndim != 2
            or receivers.shape != weights.shape
            or receivers.shape[0] != discharge.numel()
            or receivers.dtype != torch.long
            or receivers.device != s.h.device
            or (discharge < 0).any()
            or (weights < 0).any()
            or (receivers < 0).any()
            or (receivers >= s.mesh.nCells).any()
            or not torch.allclose(
                weights.sum(-1),
                torch.ones_like(discharge),
                rtol=0,
                atol=32 * torch.finfo(s.h.dtype).eps,
            )
        ):
            raise ValueError(
                "invalid river discharge, receiver indices or normalized weights"
            )
        wet = (self.H.model.energy.mass_area(s) > 0)[:, 0]
        if (~wet[receivers] & (weights > 0)).any():
            raise ValueError("river receivers must be wet")
        mass = s.column.rho0 * discharge[:, None] * weights
        heat = mass * (self.H.model.config.cp * temperature[:, None])
        zeros = torch.zeros_like(s.h[:, 0])
        mass_cell = zeros.index_add(0, receivers.flatten(), mass.flatten())
        heat_cell = zeros.index_add(0, receivers.flatten(), heat.flatten())
        rate = (mass_cell / s.mesh.cell_area)[:, None]
        enthalpy = (heat_cell / safe(mass_cell))[:, None]
        # Validate the supplied entry momentum before returning a forcing packet.
        active = torch.ones_like(state.coupled.ice.velocity, dtype=torch.bool)
        if self.H.model.domain is not None:
            active = self.H.model.domain.active_edge[:, None]
        value = LiquidWater(rate, enthalpy, velocity)
        self._liquid(state, value, "runoff", wet[:, None], active)
        return value
