"""Opt-in active T/S ocean derived from a JMD95 dynamic-enthalpy Hamiltonian.

The scalar-flux / pressure blocks are negative adjoints; rotation uses the
existing kinetic-metric skew operator. This establishes spatial energy closure,
not a discrete Poisson/Jacobi theorem or exact finite-timestep conservation.
Centered traces are independent of velocity sign. There is no donor work
converted into a discontinuous momentum force, and no unmodelled heat reservoir.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from vfs.integrators.explicit import ssprk3
from .constants import OCEAN_GRAVITY, OCEAN_RHO0, OCEAN_ROTATION_RATE, OCEAN_T0, OCEAN_S0
from .conservative import ConstantDensityDynamics
from .jmd95_energy import JMD95Energy


class JMD95Dynamics(nn.Module):
    """Adiabatic, hydrostatic material layers with active potential T and S.

    Use with ``Ocean(..., dynamics=JMD95Dynamics(), stepper=checked_ssprk3)``.
    No vertical remap/mixing, convection, wetting/drying, forcing or open boundary.
    Centered transport is not monotone or positivity preserving. The checked
    stepper rejects invalid full Euler stages without clipping or reducing dt.
    Existing rotation/kinetic spatial-consistency limitations still apply.
    """
    eos = 'jmd95_fixed_datum'
    energy_label = 'jmd95_hamiltonian'

    def __init__(self, *, domain=None, gravity=OCEAN_GRAVITY, rho0=OCEAN_RHO0,
                 omega=OCEAN_ROTATION_RATE, min_thickness=1e-3,
                 T0=OCEAN_T0, S0=OCEAN_S0, quadrature_order=4):
        super().__init__()
        if not math.isfinite(float(T0)) or not math.isfinite(float(S0)) or S0 < 0:
            raise ValueError('reference T0 must be finite and S0 finite/nonnegative')
        self.domain = domain
        self.gravity, self.rho0 = float(gravity), float(rho0)
        self.min_thickness = float(min_thickness)
        self.mechanical = ConstantDensityDynamics(domain=domain, gravity=gravity,
            rho0=rho0, omega=omega, min_thickness=min_thickness)
        self.energy = JMD95Energy(domain=domain, gravity=gravity, rho0=rho0,
            T0=T0, S0=S0, quadrature_order=quadrature_order)

    def check_state(self, state):
        self.mechanical._check_state(state)
        for name in ('u_e', 'h', 'temp_h', 'salt_h', 'tracer_h'):
            value = getattr(state, name)
            if value is not None and not torch.isfinite(value).all().item():
                raise FloatingPointError(f'JMD95 ocean {name} must be finite')
        if (state.salt_h < 0).any().item():
            raise FloatingPointError('JMD95 ocean requires nonnegative salinity; no clipping')
        rho = self.energy.density(state)
        if not torch.isfinite(rho).all().item() or (rho <= 0).any().item():
            raise FloatingPointError('JMD95 ocean density must be finite and positive')

    @staticmethod
    def centered_trace(mesh, concentration):
        c0, c1 = mesh.cellsOnEdge.unbind(1)
        return .5*(concentration[c0] + concentration[c1])

    def tendency_terms(self, state):
        self.check_state(state)
        mesh = state.mesh
        u = self.energy.mechanical.velocity(state)
        flux = mesh.cell_to_edge(state.h)*u
        gradients = self.energy.gradients(state)
        normalization = self.rho0*mesh.cell_area[:, None]
        bh = gradients['h']/normalization
        bt = gradients['temp_h']/normalization
        bs = gradients['salt_h']/normalization
        tf = self.centered_trace(mesh, state.temperature())
        sf = self.centered_trace(mesh, state.salinity())
        # With H_u=rho0*A_e*F, this is minus the scalar transport's adjoint.
        # No division by velocity and no sign-dependent thermodynamic traces.
        pressure = -mesh.grad(bh) - tf*mesh.grad(bt) - sf*mesh.grad(bs)
        cell = (lambda v: v) if self.domain is None else self.domain.apply_cell
        edge = (lambda v: v) if self.domain is None else self.domain.apply_edge
        tracers = state.tracers()
        tracer_dot = None if tracers is None else torch.stack([
            cell(-mesh.div(flux*self.centered_trace(mesh, tracers[..., k])))
            for k in range(tracers.shape[-1])], dim=-1)
        return {'u_e': {'rotation':self.mechanical.rotation_acceleration(state),
                        'hamiltonian_pressure':edge(pressure)},
                'h': {'horizontal':cell(-mesh.div(flux))},
                'temp_h': {'horizontal':cell(-mesh.div(flux*tf))},
                'salt_h': {'horizontal':cell(-mesh.div(flux*sf))},
                'tracer_h': {} if tracer_dot is None else {'horizontal':tracer_dot}}

    def tendency(self, state):
        terms = self.tendency_terms(state)
        return state.with_tensors(sum(terms['u_e'].values()), terms['h']['horizontal'],
            terms['temp_h']['horizontal'], terms['salt_h']['horizontal'],
            terms['tracer_h'].get('horizontal'))

    def total_energy(self, state):
        return self.energy.terms(state)['total']

    def density(self, state):
        return self.energy.density(state)


def checked_ssprk3(model, state, dt):
    """Shared SSPRK3, with acceptance of EVERY full Euler endpoint.

    Evaluating an RHS checks s+dt*L(s) before SSP convex combinations can hide
    a negative intermediate salinity/thickness. Rejection is not a positivity
    or wave-stability theorem; centered transport can reject at any dt near S=0.
    No model mutation, automatic retries, clipping or timestep adaptation.
    """
    if not isinstance(model.dynamics, JMD95Dynamics):
        raise TypeError('checked_ssprk3 requires JMD95Dynamics')
    if model.sources:
        raise ValueError('JMD95 checked step currently supports unforced dynamics only')
    if not math.isfinite(float(dt)) or float(dt) <= 0:
        raise ValueError('dt must be finite and positive')

    def validate(s):
        model.check_state(s)
        model.dynamics.check_state(s)

    def rhs(s):
        validate(s)
        tendency = model.tendency(s)
        validate(s + float(dt)*tendency)
        return tendency

    validate(state)
    result = ssprk3(rhs, state, dt)
    validate(result)
    return result
