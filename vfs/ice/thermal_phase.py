"""Liquidus equilibrium including prognostic sensible energy of category ice."""
import torch
from .category_phase import CategoryFootprint, bisect_phase_mass
from .snow import SnowPhase
from .ocean_ice import safe
from .phase_energy import ReducedPhaseEnergy


class ThermalPhase(SnowPhase):
    """Include sensible frozen energy in the existing local phase root.

    Melting removes the same fraction of cold content as solid volume, so
    its warming cost is included in the root. Newly frozen mass enters at the
    zero-C surface-node datum; conduction subsequently cools that node.
    """

    def __init__(self, owner, initial):
        super().__init__(owner.mechanics, initial.ice.snow_mass)
        self.initial = initial

    def cold(self, ice):
        old = self.initial.ice.volume
        return self._retained_cold(self.initial.ice.cold_content, safe(old), ice.volume)

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
        cold = self.initial.ice.cold_content[rows]
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
        de = self.initial.ice.cold_content.sum(-1, keepdim=True) / safe(m0)
        return super().derivative(initial, state, m) + torch.where(m <= m0, de, 0.0)
