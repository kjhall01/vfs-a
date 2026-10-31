"""Energy-derived constant-density ocean equations, selected explicitly.

With F = I(h) u, h_dot = -D F and B = H_h/(rho0 A_c), grad/div
adjointness gives H_h*h_dot = rho0 sum A_e F G(B). Thus u_dot=-G(B)
cancels that work under H_u = rho0 A_e I(h) u. H_h includes dK/dh;
omitting it would lose the evolving-layer kinetic-energy contribution.

Rotation is antisymmetrized in that same metric, with its vorticity held fixed
when forming the transpose. This proves zero work, not a Poisson Jacobi identity.
The reference EOS, vertical remap and mixing are NOT inherited by this class.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from .constants import OCEAN_GRAVITY, OCEAN_RHO0, OCEAN_ROTATION_RATE
from .energy import ConstantDensityEnergy


class ConstantDensityDynamics(nn.Module):
    """Experimental constant-density multilayer free surface with passive contents.

    Supported boundary for the closure argument: global sphere or static closed
    coast masks. No open boundary, wetting/drying or time-dependent bathymetry.
    Centered thickness transport is still not generally positivity preserving.
    SSPRK3 has temporal energy error even when the spatial RHS closes energy.
    """
    energy_label = 'constant_density_mechanical_energy'

    def __init__(self, *, domain=None, gravity=OCEAN_GRAVITY, rho0=OCEAN_RHO0,
                 omega=OCEAN_ROTATION_RATE, min_thickness=1e-3):
        super().__init__()
        self.domain = domain
        self.gravity = float(gravity)
        self.rho0 = float(rho0)
        self.rotation_rate = float(omega)
        self.min_thickness = float(min_thickness)
        if not all(math.isfinite(x) for x in (self.gravity,self.rho0,self.rotation_rate,self.min_thickness)):
            raise ValueError('ocean coefficients must be finite')
        if self.gravity <= 0 or self.rho0 <= 0 or self.min_thickness < 0:
            raise ValueError('gravity/rho0 must be positive and min_thickness nonnegative')
        self.energy = ConstantDensityEnergy(gravity=gravity,rho0=rho0,domain=domain)

    def _check_state(self, state):
        state.validate()
        if state.h.ndim != 2:
            raise ValueError('constant-density ocean currently supports unbatched states')
        state.column.validate_thickness(state.h,min_thickness=self.min_thickness)
        if self.domain is not None and state.mesh is not self.domain.mesh:
            raise ValueError('state and domain must share the same mesh')

    def rotation_acceleration(self, state):
        """M^-1 [M R - R^T M] u / 2, with R built from this state's vorticity.

        R uses the mesh's oriented TRiSK weights. Its transpose is a scatter
        with EXACTLY those coefficients, rather than another call that would
        recompute vorticity from a different vector. Coast input/output masks
        give the corresponding P R P restriction.
        """
        mesh = state.mesh
        u = self.energy.velocity(state)
        metric = self.energy.metric(state)
        absolute_vorticity = mesh.curl(u) + self.rotation_rate * mesh.coriolis_unit_vertex[:, None]
        v0,v1 = mesh.verticesOnEdge[:,0],mesh.verticesOnEdge[:,1]
        q = .5 * (absolute_vorticity[v0] + absolute_vorticity[v1])
        stencil = mesh.safe_rotation_stencil
        coefficients = -.5 * (q[:,None,:] + q[stencil]) * mesh.rotation_weights_masked[...,None]
        forward = (coefficients * u[stencil]).sum(dim=1)
        contributions = coefficients * (metric*u)[:,None,:]
        transpose = torch.zeros_like(u).index_add(0,stencil.reshape(-1),
                                                  contributions.reshape(-1,u.shape[-1]))
        acceleration = .5 * (metric*forward-transpose) / metric
        return acceleration if self.domain is None else self.domain.apply_edge(acceleration)

    def tendency_terms(self, state):
        self._check_state(state)
        mesh = state.mesh
        u = self.energy.velocity(state)
        flux = u * mesh.cell_to_edge(state.h)
        h_dot = -mesh.div(flux)
        bernoulli = self.energy.thickness_gradient(state) / (self.rho0*mesh.cell_area[:,None])
        pressure_and_kinetic = -mesh.grad(bernoulli)
        if self.domain is not None:
            h_dot = self.domain.apply_cell(h_dot)
            pressure_and_kinetic = self.domain.apply_edge(pressure_and_kinetic)

        def content_tendency(concentration):
            # Passive donor-cell flux uses the same carrier F, so a uniform
            # concentration follows h exactly. This is not a carrier limiter.
            c0,c1 = mesh.cellsOnEdge[:,0],mesh.cellsOnEdge[:,1]
            upwind = torch.where(u>=0,concentration[c0],concentration[c1])
            result = -mesh.div(flux*upwind)
            return result if self.domain is None else self.domain.apply_cell(result)

        tracers = state.tracers()
        tracer_dot = None if tracers is None else torch.stack(
            [content_tendency(tracers[...,k]) for k in range(tracers.shape[-1])],dim=-1)
        return {'u_e': {'rotation':self.rotation_acceleration(state),
                        'bernoulli':pressure_and_kinetic},
                'h': {'horizontal':h_dot},
                'temp_h': {'horizontal':content_tendency(state.temperature())},
                'salt_h': {'horizontal':content_tendency(state.salinity())},
                'tracer_h': {} if tracer_dot is None else {'horizontal':tracer_dot}}

    def tendency(self, state):
        terms = self.tendency_terms(state)
        return state.with_tensors(terms['u_e']['rotation']+terms['u_e']['bernoulli'],
            terms['h']['horizontal'],terms['temp_h']['horizontal'],terms['salt_h']['horizontal'],
            terms['tracer_h'].get('horizontal'))

    def total_energy(self, state):
        return self.energy.terms(state)['total']

    def density(self, state):
        """Density for this equation set; transported T/S do not change it."""
        return torch.full_like(state.h, self.rho0)
