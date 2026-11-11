"""Column-local full-Hamiltonian closure after potential-enthalpy ALE.

The scalar root is energy-derived; a uniform temperature increment within each
column is the explicit allocation policy, not a unique consequence of H. No
cross-column redistribution, salt/momentum change, clipping or forward AD.
"""
from dataclasses import dataclass, replace
import torch
from .equations import validate
from .enthalpy_remap import EnthalpyRemapper, EnthalpyRemapReport


def column_terms(energy, physical):
    """Exact local decomposition of the existing quadrature Hamiltonian.

    K_c=sum_k h_ck * (dK/dh)_ck uses the actual interpolation transpose.
    Its sum is the original edge kinetic energy, including the coast mask.
    The returned temperature derivative holds thickness, salt and velocity fixed.
    """
    base=energy.base;h=physical.h
    mass=energy.mass_area(physical)
    node,weight=base._quadrature(h)
    z=base.height(physical)[...,None]+h[...,None]*(node-.5)
    psi,pt,_,_=base.parcel_potential(physical.temperature()[...,None],physical.salinity()[...,None],z)
    ref=energy.reference.evaluate(physical.salinity(),physical.temperature())
    terms=dict(kinetic=(h*base.mechanical.kinetic_thickness_gradient(physical)).sum(-1),
        free_surface=(.5*base.gravity*mass*physical.column.eta(h).square()).sum(-1),
        dynamic_enthalpy=(mass*h*(psi*weight).sum(-1)).sum(-1),
        reference_thermal=(mass*h*ref['h0']).sum(-1))
    derivative=(mass*h*(ref['h_theta']+(pt*weight).sum(-1))).sum(-1)
    return terms,derivative


@dataclass(frozen=True)
class HamiltonianRemapReport(EnthalpyRemapReport):
    column_energy_residual: torch.Tensor
    column_entropy_change: torch.Tensor
    temperature_increment: torch.Tensor
    provisional_energy_change: dict
