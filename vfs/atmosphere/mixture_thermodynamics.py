"""Experimental ideal-mixture thermodynamic kernel; not a moist dynamics RHS.

Species order is dry air, vapor, liquid, ice. Fractions use total mixture mass.
Ideal gases, negligible condensate volume, common temperature and constant
species heat capacities. No saturation adjustment or precipitation closure.
Reference values are explicit provisional choices, to be agreed with the SCM.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class MixtureThermodynamics:
    """One consistent set of e, h, s and alpha, with analytic inversions.

    Inputs T and p share the leading shape of q; q's final axis has length 4.
    Units: K, Pa, kg/kg, J/kg, J/(kg K), m^3/kg. Dry air must be present.
    Entropy references are a thermodynamic convention, not third-law values.
    """

    R_d: float = 287.0
    R_v: float = 461.5
    cp_d: float = 1004.0
    cp_v: float = 1850.0
    cp_l: float = 4186.0
    cp_i: float = 2106.0
    T_ref: float = 273.15
    p_d_ref: float = 100000.0
    p_v_ref: float = 611.657
    L_v_ref: float = 2500800.0
    L_f_ref: float = 333700.0

    def validate(self, T, p, q):
        """Explicit setup/diagnostic validation; not silently clipping states."""
        if q.shape[-1] != 4 or T.shape != q.shape[:-1] or p.shape != T.shape:
            raise ValueError("expected T,p shape (...) and q shape (...,4)")
        if not (torch.isfinite(q).all() and torch.isfinite(T).all() and torch.isfinite(p).all()):
            raise ValueError("nonfinite thermodynamic state")
        if (q < 0).any() or (q[..., 0] <= 0).any() or (T <= 0).any() or (p <= 0).any():
            raise ValueError("require nonnegative species, positive dry fraction, T and p")
        if not torch.allclose(q.sum(-1), torch.ones_like(T), atol=16*torch.finfo(q.dtype).eps, rtol=0):
            raise ValueError("mass fractions must sum to one")

    def gas_constant(self, q):
        return q[..., 0] * self.R_d + q[..., 1] * self.R_v

    def heat_capacity_p(self, q):
        return (q * q.new_tensor([self.cp_d, self.cp_v, self.cp_l, self.cp_i])).sum(-1)

    def enthalpy_reference(self, q):
        return q[..., 1] * self.L_v_ref - q[..., 3] * self.L_f_ref

    def enthalpy(self, T, q):
        return self.heat_capacity_p(q) * (T - self.T_ref) + self.enthalpy_reference(q)

    def internal_energy(self, T, q):
        return self.enthalpy(T, q) - self.gas_constant(q) * T

    def temperature_from_internal_energy(self, e, q):
        cp = self.heat_capacity_p(q)
        return (e - self.enthalpy_reference(q) + cp * self.T_ref) / (cp - self.gas_constant(q))

    def specific_volume(self, T, p, q):
        return self.gas_constant(q) * T / p

    def _entropy_offset(self, p, q):
        R = self.gas_constant(q)
        pd = p * q[..., 0] * self.R_d / R
        pv = p * q[..., 1] * self.R_v / R
        # Evaluate the continuous q_v log(q_v) value at q_v=0. The composition
        # derivative at that boundary is singular; do not use it as a finite
        # chemical potential for an absent species.
        pv_safe = torch.where(q[..., 1] > 0, pv, torch.ones_like(pv) * self.p_v_ref)
        return (self.enthalpy_reference(q) / self.T_ref
                - q[..., 0] * self.R_d * torch.log(pd / self.p_d_ref)
                - q[..., 1] * self.R_v * torch.log(pv_safe / self.p_v_ref))

    def entropy(self, T, p, q):
        return self.heat_capacity_p(q) * torch.log(T / self.T_ref) + self._entropy_offset(p, q)

    def temperature_from_entropy(self, s, p, q):
        return self.T_ref * torch.exp((s - self._entropy_offset(p, q)) / self.heat_capacity_p(q))

    def latent_heat_vaporization(self, T):
        return self.L_v_ref + (self.cp_v - self.cp_l) * (T - self.T_ref)


def water_mixing_ratios_to_mass_fractions(r):
    """Convert [...,3] water/dry ratios to [...,4] dry/vapor/liquid/ice q.

    Caller must provide finite nonnegative ratios. All water species appear
    in the denominator; vapor-only conversions are insufficient for clouds.
    """
    if r.shape[-1] != 3:
        raise ValueError("expected vapor/liquid/ice mixing ratios on the last axis")
    dry = 1 / (1 + r.sum(-1, keepdim=True))
    return torch.cat([dry, r * dry], dim=-1)


def mass_fractions_to_water_mixing_ratios(q):
    """Convert [...,4] mixture fractions to [...,3] water/dry ratios."""
    if q.shape[-1] != 4:
        raise ValueError("expected dry/vapor/liquid/ice fractions on the last axis")
    return q[..., 1:] / q[..., :1]
