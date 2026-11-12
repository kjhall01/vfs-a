"""Shared finite-energy thermal inversion at fixed mass, salt and velocity."""
from dataclasses import replace
import math
import torch


def thermal_parts(ocean, physical):
    energy = ocean.energy
    node, weight = energy.base._quadrature(physical.h)
    z = energy.base.height(physical)[..., None]+physical.h[..., None]*(node-.5)
    psi = energy.base.parcel_potential(physical.temperature()[..., None], physical.salinity()[..., None], z)[0]
    ref = energy.reference.evaluate(physical.salinity(), physical.temperature())
    return (psi*weight).sum(-1), ref

def apply_heat(ocean, state, heat_specific, *, allow_cooling=False):
    """Invert the full layer thermal energy increment at fixed h, SP and u.

    Integrate its analytic temperature derivative along the increment to
    avoid subtracting large reference thermal energies for tiny heating.
    Three-point Gauss quadrature evaluates the temperature-path integral;
    this local integral approximation is separately checked against H.
    """
    if not allow_cooling and not torch.any(heat_specific != 0):
        return state, torch.zeros_like(heat_specific)
    energy = ocean.energy; base = energy.base; reference = energy.reference
    physical = state.physical_state(reference)
    theta0, salt = physical.temperature(), physical.salinity()
    node, weight = base._quadrature(state.h)
    z = base.height(physical)[..., None]+state.h[..., None]*(node-.5)
    def derivative(theta):
        ref = reference.evaluate(salt, theta)
        pt = base.parcel_potential(theta[..., None], salt[..., None], z)[1]
        return ref['h_theta']+(pt*weight).sum(-1)
    nodes = (.5-math.sqrt(15)/10, .5, .5+math.sqrt(15)/10)
    weights = (5/18, 4/9, 5/18)
    delta = heat_specific/derivative(theta0)
    for _ in range(6):
        gained = delta*sum(w*derivative(theta0+t*delta) for t, w in zip(nodes, weights))
        delta = delta+(heat_specific-gained)/derivative(theta0+delta)
    gained = delta*sum(w*derivative(theta0+t*delta) for t, w in zip(nodes, weights))
    residual = gained-heat_specific
    if not torch.isfinite(delta).all() or (not allow_cooling and (delta < 0).any()):
        raise FloatingPointError('heating inversion failed; no thermal clipping')
    theta = theta0+delta
    eta = reference.evaluate(salt, theta, entropy_derivatives=True)['entropy']
    if allow_cooling:
        # Increment baseline preserves zero forcing and its linear response.
        eta0 = reference.evaluate(salt, theta0, entropy_derivatives=True)['entropy']
        updated = replace(state, entropy_h=state.entropy_h+state.h*(eta-eta0))
    else:
        updated = replace(state, entropy_h=torch.where(heat_specific != 0, state.h*eta, state.entropy_h))
    ocean.check_state(updated)
    return updated, residual
