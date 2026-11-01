"""JMD95 dynamic-enthalpy Hamiltonian for an adiabatic layered ocean.

T means potential temperature (degC), S practical salinity. EOS pressure is
p(z)=-rho0*g*z/1e4 dbar at a FIXED sea-level datum, not depth below eta.
The reference is JMD95(S0,T0,p(z)). This defines buoyancy relative to a resting
reference column. It does not define seawater heat/entropy or diabatic physics.
See verification/ocean/JMD95_HAMILTONIAN.md for the discrete derivation.
"""
from __future__ import annotations

import numpy as np
import torch

from .constants import OCEAN_GRAVITY, OCEAN_RHO0, OCEAN_T0, OCEAN_S0
from .energy import ConstantDensityEnergy


def _polynomial(t, coefficients):
    """Horner value and explicit first derivative, coefficients in ascending order."""
    value = torch.zeros_like(t) + coefficients[-1]
    derivative = torch.zeros_like(t)
    for coefficient in reversed(coefficients[:-1]):
        derivative = derivative * t + value
        value = value * t + coefficient
    return value, derivative


def jmd95_partials(s, t, pressure_dbar):
    """Density and analytic (T,S,p_dbar) partials; no AD in forward equations.

    This is the same JMD95 polynomial as ocean.eos, regrouped to expose its
    derivatives. S must be nonnegative. pow(1.5) gives a finite first derivative
    at S=0; second salinity derivatives need not be finite there.
    """
    r0, r0t = _polynomial(t, (999.842594, 6.793952e-2, -9.095290e-3,
                             1.001685e-4, -1.120083e-6, 6.536332e-9))
    r1, r1t = _polynomial(t, (.824493, -4.0899e-3, 7.6438e-5,
                             -8.2467e-7, 5.3875e-9))
    r15, r15t = _polynomial(t, (-5.72466e-3, 1.0227e-4, -1.6546e-6))
    s15 = s.pow(1.5)
    ds15 = 1.5 * s.sqrt()
    rho_surface = r0 + s*r1 + s15*r15 + 4.8314e-4*s.square()
    rho_t = r0t + s*r1t + s15*r15t
    rho_s = r1 + ds15*r15 + 2*4.8314e-4*s

    k0, k0t = _polynomial(t, (1.965933e4, 1.444304e2, -1.706103,
                             9.648704e-3, -4.190253e-5))
    k1, k1t = _polynomial(t, (5.284855e1, -.3101089, 6.283263e-3, -5.084188e-5))
    k15, k15t = _polynomial(t, (.388664, 9.085835e-3, -4.619924e-4))
    a0, a0t = _polynomial(t, (3.186519, 2.212276e-2, -2.984642e-4, 1.956415e-6))
    a1, a1t = _polynomial(t, (6.704388e-3, -1.847318e-4, 2.059331e-7))
    b0, b0t = _polynomial(t, (2.102898e-4, -1.202016e-5, 1.394680e-7))
    b1, b1t = _polynomial(t, (-2.040237e-6, 6.128773e-8, 6.207323e-10))
    p = .1 * pressure_dbar  # polynomial pressure is bar
    a = a0 + s*a1 + 1.480266e-4*s15
    b = b0 + s*b1
    bulk = k0 + s*k1 + s15*k15 + p*a + p.square()*b
    bulk_t = k0t + s*k1t + s15*k15t + p*(a0t+s*a1t) + p.square()*(b0t+s*b1t)
    bulk_s = k1 + ds15*k15 + p*(a1+1.480266e-4*ds15) + p.square()*b1
    bulk_p = .1 * (a + 2*p*b)
    denominator = 1 - p/bulk
    density = rho_surface/denominator
    # d(1-p/K)/dx = p K_x/K², except p itself contributes -0.1/K.
    common = rho_surface/denominator.square()
    dt = rho_t/denominator - common*p*bulk_t/bulk.square()
    ds = rho_s/denominator - common*p*bulk_s/bulk.square()
    dp = common*(.1/bulk - p*bulk_p/bulk.square())
    return density, dt, ds, dp


class JMD95Energy:
    """K + free-surface energy + quadrature-defined dynamic enthalpy, in joules.

    Fixed Gauss quadrature is part of the Hamiltonian, not an approximate
    gradient of a different energy. All derivatives include the quadrature's
    moving nodes and the conversion T=Theta/h, S=Sigma/h.
    """

    def __init__(self, *, gravity=OCEAN_GRAVITY, rho0=OCEAN_RHO0,
                 T0=OCEAN_T0, S0=OCEAN_S0, quadrature_order=4, domain=None):
        if not isinstance(quadrature_order, int) or not 2 <= quadrature_order <= 16:
            raise ValueError('quadrature_order must be an integer from 2 to 16')
        self.gravity, self.rho0 = float(gravity), float(rho0)
        self.T0, self.S0 = float(T0), float(S0)
        self.domain = domain
        self.quadrature_order = quadrature_order
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        self.nodes, self.weights = (nodes+1)/2, weights/2
        self.mechanical = ConstantDensityEnergy(gravity=gravity, rho0=rho0, domain=domain)

    def _quadrature(self, like):
        return like.new_tensor(self.nodes), like.new_tensor(self.weights)

    def height(self, state):
        """Layer midpoints in metres above the fixed sea-level datum."""
        below_and_self = state.h.flip(-1).cumsum(-1).flip(-1)
        return -state.column.total_depth_like(state.h) + below_and_self - .5*state.h

    def density(self, state):
        pressure = -self.rho0*self.gravity*self.height(state)/1e4
        return jmd95_partials(state.salinity(), state.temperature(), pressure)[0]

    def parcel_potential(self, temperature, salinity, z):
        """Psi and (Psi_T,Psi_S,Psi_z), m²/s² and corresponding partials.

        Psi = integral_0^z g (rho(T,S,p(x))-rho_ref(p(x)))/rho0 dx.
        Psi_z differentiates the QUADRATURE, including its nodes; replacing it
        with endpoint buoyancy would break discrete derivative consistency.
        """
        node, weight = self._quadrature(z)
        x = z[..., None]*node
        pressure = -self.rho0*self.gravity*x/1e4
        rho, rt, rs, rp = jmd95_partials(salinity[..., None], temperature[..., None], pressure)
        ref, _, _, refp = jmd95_partials(torch.full_like(salinity[..., None], self.S0),
                                        torch.full_like(temperature[..., None], self.T0), pressure)
        f = self.gravity/self.rho0*(rho-ref)
        fz = -self.gravity**2/1e4*(rp-refp)
        psi = z*(f*weight).sum(-1)
        pt = z*(self.gravity/self.rho0*rt*weight).sum(-1)
        ps = z*(self.gravity/self.rho0*rs*weight).sum(-1)
        pz = ((f + x*fz)*weight).sum(-1)
        return psi, pt, ps, pz

    def stratification(self, state):
        """Anomaly energy and Euclidean fixed-CONTENT derivatives."""
        h, t, s = state.h, state.temperature(), state.salinity()
        node, weight = self._quadrature(h)
        z = self.height(state)[..., None] + h[..., None]*(node-.5)
        psi, pt, ps, pz = self.parcel_potential(t[..., None], s[..., None], z)
        mean = lambda value: (value*weight).sum(-1)
        qt, qs = mean(pt), mean(ps)
        # A deeper layer lifts all layers above it. Its own nodes move by xi dh.
        lift = h*mean(pz)
        above = lift.cumsum(-1)-lift
        qh = mean(psi) - t*qt - s*qs + h*mean(pz*node) + above
        area_mass = self.rho0*state.mesh.cell_area[:, None]
        mask = (lambda v: v) if self.domain is None else self.domain.apply_cell
        energy = (area_mass*mask(h*mean(psi))).sum()
        return energy, {'h':area_mass*mask(qh), 'temp_h':area_mass*mask(qt),
                        'salt_h':area_mass*mask(qs)}

    def terms(self, state):
        result = self.mechanical.terms(state)
        result['dynamic_enthalpy'] = self.stratification(state)[0]
        result['total'] = result['total'] + result['dynamic_enthalpy']
        return result

    def gradients(self, state):
        result = self.mechanical.gradients(state)
        _, extra = self.stratification(state)
        for name, value in extra.items():
            result[name] = result[name] + value
        return result
