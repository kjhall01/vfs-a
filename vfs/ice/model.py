"""Thermal, snow-covered sea ice advanced jointly with the Roquet ocean."""
from dataclasses import dataclass, replace
import math

import torch

from vfs.land import LandState
from vfs.ocean.prognostics import OceanState
from .ocean_ice import OceanIceMechanics, safe
from .ridging import CategoryIce
from .snow import SnowCoveredIce, SnowParameters
from .thermal_phase import ThermalPhase


@dataclass(frozen=True)
class SeaIceState(CategoryIce):
    """Category area and total ice-equivalent volume at cells, shared velocity at edges."""
    snow_mass: torch.Tensor     # kg/m² of cell, already included in volume
    cold_content: torch.Tensor  # J/m² of cell relative to ice at 273.15 K


@dataclass(frozen=True)
class OceanIceState:
    ocean: OceanState
    ice: SeaIceState
    land: LandState | None = None  # stationary during the joint dynamics stages

    @property
    def snow_mass(self):
        """Snow store used by the common zero-layer and thermal surface operations."""
        return self.ice.snow_mass

    def with_surface(self, ocean, ice, snow_mass):
        frozen = SeaIceState(ice.area, ice.volume, ice.velocity, snow_mass, self.ice.cold_content)
        return replace(self, ocean=ocean, ice=frozen)


class OceanIce(SnowCoveredIce):
    """Advance ocean, moving category ice, snow and cold content in the same RK stages."""

    heat_capacity = 2100.0  # J/(kg K)
    melting_temperature = 273.15

    def __init__(self, ocean, *, parameters=SnowParameters(), **mechanics_options):
        super().__init__(OceanIceMechanics(ocean, **mechanics_options), parameters)

    def state(self, ocean, *, area, volume, velocity, temperature=270.0, snow_mass=None):
        """Initialize frozen water from its geometry and physical temperature in K."""
        if snow_mass is None:
            snow_mass = torch.zeros_like(area)
        T = torch.as_tensor(temperature, dtype=volume.dtype, device=volume.device)
        if not torch.isfinite(T).all() or (T <= 0).any() or (T > self.melting_temperature).any():
            raise ValueError("initial ice temperature must be in (0,273.15] K")
        energy = self.heat_capacity * self.hamiltonian.slab.parameters.density * volume * (T - self.melting_temperature)
        state = OceanIceState(ocean, SeaIceState(area, volume, velocity, snow_mass, energy))
        self.check(state)
        return state

    def capacity(self, state):
        return (
            self.heat_capacity
            * self.hamiltonian.slab.parameters.density
            * state.ice.volume
        )

    def temperature(self, state):
        return self.melting_temperature + state.ice.cold_content / safe(
            self.capacity(state)
        )

    def check(self, state):
        if not isinstance(state, OceanIceState) or not isinstance(state.ice, SeaIceState):
            raise TypeError("OceanIceState with thermal SeaIceState required")
        super().check(state)
        e, v = state.ice.cold_content, state.ice.volume
        if (
            e.shape != v.shape
            or e.dtype != v.dtype
            or e.device != v.device
            or not torch.isfinite(e).all()
            or (e > 0).any()
            or ((v == 0) & (e != 0)).any()
            or (self.temperature(state) <= 0).any()
        ):
            raise ValueError("invalid ice cold content or nonpositive temperature")

    def total_energy(self, state):
        A = state.ocean.mesh.cell_area
        return (
            self.hamiltonian.total_energy(state)
            + (A[:, None] * state.ice.cold_content).sum()
        )

    def phase_step(self, state, dt, heat_flux):
        self.check(state)
        phase = ThermalPhase(self, state)
        end, report = phase.apply(state, dt, heat_flux=heat_flux)
        rho = self.hamiltonian.slab.parameters.density
        loss = (rho * (state.ice.volume - end.ice.volume)).clamp_min(0)
        melt = torch.minimum(loss, state.ice.snow_mass)
        ice = SeaIceState(end.ice.area, end.ice.volume, end.ice.velocity,
                          state.ice.snow_mass - melt, phase.cold(end.ice))
        result = replace(state, ocean=end.ocean, ice=ice)
        self.check(result)
        A = end.ocean.mesh.cell_area[:, None]
        return result, report.external_heat.sum(), (A * melt).sum()

    def drag_step(self, state, dt):
        end, _ = self.mechanics.drag_step(state, dt)
        return end

    def tendency(self, state, dt, **options):
        self._check_snow_stores(state)
        rhs, report = self.mechanics.tendency(state, dt, **options)
        transport, ridging = options.get("transport", True), options.get("ridging", True)
        ds = self.store_tendency(state, state.ice.snow_mass, transport=transport, ridging=ridging)
        de = self.store_tendency(state, state.ice.cold_content, transport=transport, ridging=ridging)
        ice = SeaIceState(rhs.ice.area, rhs.ice.volume, rhs.ice.velocity, ds, de)
        return OceanIceState(rhs.ocean, ice), report

    def add(self, state, rhs, dt):
        end = self.mechanics.add(state, rhs, dt)
        ice = SeaIceState(end.ice.area, end.ice.volume, end.ice.velocity,
                          state.ice.snow_mass + dt * rhs.ice.snow_mass,
                          state.ice.cold_content + dt * rhs.ice.cold_content)
        return replace(state, ocean=end.ocean, ice=ice)

    def blend(self, first, second, weight):
        end = self.mechanics.blend(first, second, weight)
        ice = SeaIceState(end.ice.area, end.ice.volume, end.ice.velocity,
                          (1 - weight) * first.ice.snow_mass + weight * second.ice.snow_mass,
                          (1 - weight) * first.ice.cold_content + weight * second.ice.cold_content)
        return replace(first, ocean=end.ocean, ice=ice)

    def step(self, state, dt, **options):
        # Those standalone prescribed forcings have zero-layer semantics. The
        # thermal workflow receives all external forcing through shared Exchange.
        if any(
            value is not None
            for name, value in options.items()
            if name not in ("stage_callback", "dynamics_integrator", "apply_mixing")
        ):
            raise ValueError(
                "thermal ice uses shared atmospheric exchange for external forcing"
            )
        return super().step(state, dt, **options)

    def surface_heat(self, initial, candidate, integrated_heat, dt):
        """Backward-Euler conduction plus the physical melting complementarity.

        E1-E0 = Q_surface - dt*K*(T1-T_base) - Q_excess.
        E1<=0, Q_excess>=0, and E1*Q_excess=0. The shared first-law
        receiver transfers conduction and excess to ocean/phase exactly once.
        No temperature floor or unreported energy removal is applied.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("positive finite dt required")
        if (
            integrated_heat.shape != initial.ice.cold_content.shape
            or not torch.isfinite(integrated_heat).all()
        ):
            raise ValueError("finite category surface heat required")
        hi, hs = self.depths(candidate)
        p = self.hamiltonian.slab.parameters
        C = self.capacity(candidate)
        resistance = hi / p.conductivity + hs / self.parameters.conductivity
        K = candidate.ice.area / safe(resistance)
        base = p.freezing_temperature(candidate.ocean.salinity()[:, :1])
        provisional = (initial.ice.cold_content + integrated_heat + dt * K * base) / (
            1 + dt * K / safe(C)
        )
        e = torch.where(
            C > 0, torch.minimum(provisional, torch.zeros_like(provisional)), 0.0
        )
        result = replace(candidate, ice=replace(candidate.ice, cold_content=e))
        self.check(result)
        conduction = dt * K * (e / safe(C) - base)
        excess = initial.ice.cold_content + integrated_heat - e - conduction
        return result, conduction, excess
