"""Experimental CP thermodynamic reconstruction, not a moist dynamics RHS.

Entropy and active species share interface pressure-mass weights. This makes
linear changes of entropy reference commute with reconstruction.
"""
import torch
from .mixture_thermodynamics import MixtureThermodynamics


def interface_pressure_mass(dp):
    return torch.cat([.5*dp[..., :1], .5*(dp[..., :-1]+dp[..., 1:]), .5*dp[..., -1:]], -1)


def reconstruct_moist_cp(p_interface, entropy_content, water_content, thermo=None):
    """Return two Gauss points per layer from CP contents.

    p,S: (..., N+1); water: (..., N+1,3), vapor/liquid/ice.
    S=delta_p*s; water=delta_p*q_water. Contents use Pa, not kg/m².
    Inputs must be physical. No clipping, saturation adjustment or implicit solve.
    """
    thermo = MixtureThermodynamics() if thermo is None else thermo
    dp = p_interface[..., 1:] - p_interface[..., :-1]
    mass = interface_pressure_mass(dp)
    s = entropy_content / mass
    qw = water_content / mass[..., None]
    q = torch.cat([1-qw.sum(-1, keepdim=True), qw], -1)
    x = p_interface.new_tensor([.5-.5/3**.5, .5+.5/3**.5])
    pg = p_interface[..., :-1, None] + dp[..., None]*x
    sg = s[..., :-1, None]*(1-x) + s[..., 1:, None]*x
    qg = q[..., :-1, None, :]*(1-x[:, None]) + q[..., 1:, None, :]*x[:, None]
    T = thermo.temperature_from_entropy(sg, pg, qg)
    return dict(pressure=pg, entropy=sg, fractions=qg, temperature=T,
                internal_energy=thermo.internal_energy(T,qg),
                alpha=thermo.specific_volume(T,pg,qg), dp=dp)
