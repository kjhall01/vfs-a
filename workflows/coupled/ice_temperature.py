"""One prognostic thermal node per moving ice/snow category.

E = cp_i * rho_i * volume * (T_skin - 273.15), J/m2 of horizontal cell.
The node represents the whole frozen column (snow uses the same heat capacity).
H_total = H_ocean_ice_snow + sum(A*E). It is an explicit low-order surface
model, not a resolved vertical temperature/brine profile.
"""

from dataclasses import dataclass, replace
import math
import torch
from vfs.land import LandState
from .ocean_ice import OceanIceReceiver
from .ice_phase_energy import ReducedPhaseEnergy
from snow import SnowState, SnowOceanIce, SnowPhase
from category_phase import CategoryFootprint, bisect_phase_mass
from ocean_ice import safe


@dataclass(frozen=True)
class ThermalIceState(SnowState):
    cold_content: torch.Tensor  # [cell,2], J/m2, <= 0; zero on absent categories
    land: LandState | None = None


class ThermalPhase(SnowPhase):
    """Include sensible frozen energy in the existing local phase root.

    Melting removes the same fraction of cold content as solid volume, so
    its warming cost is included in the root. Newly frozen mass enters at the
    zero-C surface-node datum; conduction subsequently cools that node.
    """

    def __init__(self, owner, initial):
        super().__init__(owner.coupled, initial.snow_mass)
        self.initial = initial

    def cold(self, ice):
        old = self.initial.coupled.ice.volume
        return self._retained_cold(self.initial.cold_content, safe(old), ice.volume)

    @staticmethod
    def _retained_cold(cold_content, safe_old_volume, volume):
        fraction = torch.minimum(volume / safe_old_volume, torch.ones_like(volume))
        return cold_content * fraction

    def _bisect_mass(self, initial, frozen, upper, target):
        # The energy bracket includes newly freezing open water. Compact once,
        # before the loop, and scatter the warm start before the full Newton steps.
        rows = frozen[:, 0].nonzero(as_tuple=True)[0]
        result = torch.zeros_like(target)
        if rows.numel() == 0:
            return result

        s = initial.ocean.ocean
        p = self.H.slab.parameters
        old_mass = self.H.mass(initial)[rows]
        area, volume = initial.ice.area[rows], initial.ice.volume[rows]
        footprint = CategoryFootprint(area, volume, old_mass, p.density, self.nucleation_thickness)
        old_ice_mass = p.density * volume - self.initial_snow[rows]
        safe_volume = safe(volume)
        cold = self.initial.cold_content[rows]
        top = s.h[rows, :1]
        salt_content = s.salt_h[rows, :1]
        eta = s.column.eta(s.h)[rows]
        coefficient = self.H.coefficient(initial)
        quadrature = self.H.model.energy.base._quadrature(top)

        def energy_at_mass(m):
            change = (old_mass - m) / s.column.rho0
            liquid = top + change
            salt = salt_content / liquid
            ice_area, ice_volume, *_ = footprint.at_mass(m)
            melt_area = self._melt_area(area, old_ice_mass, p.density * ice_volume)
            ice_area = torch.where(m < old_mass, melt_area, ice_area)
            energy = self._column_energy(
                s.column, coefficient, quadrature, liquid, -p.freezing_slope * salt,
                salt, eta + change, ice_volume, ice_volume / safe(ice_area),
            )
            return energy + self._retained_cold(cold, safe_volume, ice_volume).sum(-1, keepdim=True)

        result[rows] = bisect_phase_mass(upper[rows], target[rows], energy_at_mass)
        return result

    def _solve_mass(self, initial, wet, frozen, upper, target, melt_energy):
        if torch.is_grad_enabled() or target.dtype != torch.float64:
            return super()._solve_mass(initial, wet, frozen, upper, target, melt_energy)
        rows = frozen[:, 0].nonzero(as_tuple=True)[0]
        mass = torch.zeros_like(target)
        # Warm wet columns stay at zero mass. Their reference monotonicity
        # check is the same on all three Newton iterations.
        warm = (wet & ~frozen)[:, 0].nonzero(as_tuple=True)[0]
        if warm.numel():
            warm_curve = ReducedPhaseEnergy(self, initial, warm)
            warm_curve.decreasing_slope(mass[warm])
        if rows.numel():
            curve = ReducedPhaseEnergy(self, initial, rows)
            available = target[rows] - melt_energy[rows]
            bound = upper[rows]
            m = bisect_phase_mass(bound, available, curve.value)
            for _ in range(2):
                m = curve.refine(m, curve.value(m) - available, bound)
            mass[rows] = m
            # The last Newton residual uses the original Hamiltonian on the
            # reconstructed state, including its floating-point roundoff.
            stored_energy = self.thermal_energy(self.phase_state(initial, mass))
            mass[rows] = curve.refine(m, stored_energy[rows] - target[rows], bound)
        return mass

    def energy_values(self, initial, top, ct, salinity, eta, ice):
        return super().energy_values(initial, top, ct, salinity, eta, ice) + self.cold(
            ice
        ).sum(-1, keepdim=True)

    def derivative(self, initial, state, m):
        m0 = self.H.mass(initial)
        de = self.initial.cold_content.sum(-1, keepdim=True) / safe(m0)
        return super().derivative(initial, state, m) + torch.where(m <= m0, de, 0.0)


class ThermalSnowOceanIce(SnowOceanIce):
    heat_capacity = 2100.0  # J/(kg K), same for fresh ice and labelled snow
    melting_temperature = 273.15

    def initialize(self, state, temperature=270.0):
        SnowOceanIce.check(self, state)
        T = torch.as_tensor(
            temperature, dtype=state.snow_mass.dtype, device=state.snow_mass.device
        )
        if (
            not torch.isfinite(T).all()
            or (T <= 0).any()
            or (T > self.melting_temperature).any()
        ):
            raise ValueError("initial ice temperature must be in (0,273.15] K")
        E = self.capacity(state) * (T - self.melting_temperature)
        result = ThermalIceState(state.coupled, state.snow_mass, E)
        self.check(result)
        return result

    def capacity(self, state):
        return (
            self.heat_capacity
            * self.hamiltonian.slab.parameters.density
            * state.coupled.ice.volume
        )

    def temperature(self, state):
        return self.melting_temperature + state.cold_content / safe(
            self.capacity(state)
        )

    def check(self, state):
        super().check(state)
        if not isinstance(state, ThermalIceState):
            raise TypeError("ThermalIceState required")
        e, v = state.cold_content, state.coupled.ice.volume
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
        A = state.coupled.ocean.ocean.mesh.cell_area
        return (
            self.hamiltonian.total_energy(state.coupled)
            + (A[:, None] * state.cold_content).sum()
        )

    def accumulate(self, state, dt, snowfall):
        # The inherited zero-C snowfall datum contributes no sensible energy.
        # The atmospheric exchange uses its actual supplied snow enthalpy instead.
        bare, mass, energy = SnowOceanIce(self.coupled, self.parameters).accumulate(
            state, dt, snowfall
        )
        return (
            ThermalIceState(bare.coupled, bare.snow_mass, state.cold_content, state.land),
            mass,
            energy,
        )

    def phase_step(self, state, dt, heat_flux):
        self.check(state)
        phase = ThermalPhase(self, state)
        end, report = phase.apply(state.coupled, dt, heat_flux=heat_flux)
        rho = self.hamiltonian.slab.parameters.density
        loss = (rho * (state.coupled.ice.volume - end.ice.volume)).clamp_min(0)
        melt = torch.minimum(loss, state.snow_mass)
        result = ThermalIceState(end, state.snow_mass - melt, phase.cold(end.ice), state.land)
        self.check(result)
        A = end.ocean.ocean.mesh.cell_area[:, None]
        return result, report.external_heat.sum(), (A * melt).sum()

    def tendency(self, state, dt, **options):
        rhs, report = super().tendency(state, dt, **options)
        ice = state.coupled.ice
        mesh = state.coupled.ocean.ocean.mesh
        de = torch.zeros_like(state.cold_content)
        if options.get("transport", True):
            c0, c1 = mesh.cellsOnEdge.T
            donor = torch.where(
                ice.velocity >= 0, state.cold_content[c0], state.cold_content[c1]
            )
            de = -mesh.div(donor * ice.velocity)
        if options.get("ridging", True):
            compression = (-mesh.div(ice.velocity)).clamp_min(0)
            ridge = self.coupled.ridge
            chi = (
                ice.area.sum(-1, keepdim=True).pow(ridge.crowding_power)
                * compression
                / (compression + ridge.strain_rate)
            )
            transfer = chi * compression * state.cold_content[:, :1]
            de = de + torch.cat((-transfer, transfer), -1)
        return ThermalIceState(rhs.coupled, rhs.snow_mass, de), report

    def add(self, state, rhs, dt):
        bare = super().add(state, rhs, dt)
        return ThermalIceState(
            bare.coupled, bare.snow_mass, state.cold_content + dt * rhs.cold_content, state.land
        )

    def blend(self, first, second, weight):
        bare = super().blend(first, second, weight)
        return ThermalIceState(
            bare.coupled,
            bare.snow_mass,
            (1 - weight) * first.cold_content + weight * second.cold_content, first.land,
        )

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
            integrated_heat.shape != initial.cold_content.shape
            or not torch.isfinite(integrated_heat).all()
        ):
            raise ValueError("finite category surface heat required")
        hi, hs = self.depths(candidate)
        p = self.hamiltonian.slab.parameters
        C = self.capacity(candidate)
        resistance = hi / p.conductivity + hs / self.parameters.conductivity
        K = candidate.coupled.ice.area / safe(resistance)
        base = p.freezing_temperature(candidate.coupled.ocean.ocean.salinity()[:, :1])
        provisional = (initial.cold_content + integrated_heat + dt * K * base) / (
            1 + dt * K / safe(C)
        )
        e = torch.where(
            C > 0, torch.minimum(provisional, torch.zeros_like(provisional)), 0.0
        )
        result = replace(candidate, cold_content=e)
        self.check(result)
        conduction = dt * K * (e / safe(C) - base)
        excess = initial.cold_content + integrated_heat - e - conduction
        return result, conduction, excess


class ThermalIceReceiver(OceanIceReceiver):
    def tiles(self, state, ice_temperature=None):
        if ice_temperature is not None:
            raise ValueError("prognostic ice temperature cannot be overridden")
        return super().tiles(state, self.snow.temperature(state))

    def finish(self, initial, candidate, exchange, impulses, *, runoff=None, runoff_energy=None):
        if exchange.surface_thermal is None:
            raise ValueError("thermal ice requires tiled surface thermal energy")
        # H already includes -Lf*m. Remove that contribution from incoming heat
        # to obtain the sensible-storage forcing: sublimation pays Lv+Lf,
        # while intercepted snowfall supplies its sensible enthalpy.
        dm = exchange.snow[:, None] * initial.coupled.ice.area - exchange.vapor[:, 1:]
        q = exchange.surface_thermal[:, 1:] + self.H.slab.parameters.latent_heat * dm
        candidate, _, _ = self.snow.surface_heat(initial, candidate, q, exchange.dt)
        storage = (candidate.cold_content - initial.cold_content).sum(-1)
        return super().finish(
            initial, candidate, exchange, impulses, storage_change=storage, runoff=runoff, runoff_energy=runoff_energy
        )
