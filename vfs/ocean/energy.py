"""Discrete mechanical energy for a constant-density layered ocean.

H = rho0/2 sum_e,k A_e I(h)_e,k u_e,k**2
    + rho0*g/2 sum_c A_c eta_c**2,       eta = sum_k h_k - H_bottom.

A_e = dc*dv is the grad/div edge metric. I is the actual mesh cell-to-edge
interpolation. Temperature, salinity and tracers are passive in THIS model.
This is not an energy functional for the archived active-EOS pressure force.
"""
from __future__ import annotations
import torch
from .constants import OCEAN_GRAVITY, OCEAN_RHO0


class ConstantDensityEnergy:
    """One energy definition and its analytic derivatives; no forward autograd."""

    def __init__(self, *, gravity=OCEAN_GRAVITY, rho0=OCEAN_RHO0, domain=None):
        self.gravity = float(gravity)
        self.rho0 = float(rho0)
        self.domain = domain

    def velocity(self, state):
        return state.u_e if self.domain is None else self.domain.apply_edge(state.u_e)

    def metric(self, state):
        """Diagonal kinetic metric M = rho0 A_e I(h), before coast masking."""
        mesh = state.mesh
        return self.rho0 * (mesh.dc_edge * mesh.dv_edge)[:, None] * mesh.cell_to_edge(state.h)

    @staticmethod
    def interpolation_transpose(mesh, edge_covector):
        """Apply I^T: scatter each edge covector using its two interpolation weights.

        This is a Euclidean transpose, not the mesh's differently weighted
        edge_to_cell average. No additional cell-area factor belongs here.
        """
        area = mesh.edge_side_area
        weights = area / area.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        out = edge_covector.new_zeros((mesh.nCells, edge_covector.shape[-1]))
        out = out.index_add(0, mesh.cellsOnEdge[:, 0], weights[:, :1] * edge_covector)
        return out.index_add(0, mesh.cellsOnEdge[:, 1], weights[:, 1:] * edge_covector)

    def kinetic_thickness_gradient(self, state):
        """dK/dh = rho0/2 I^T(A_e u**2), at fixed velocity and contents."""
        mesh = state.mesh
        return .5 * self.rho0 * self.interpolation_transpose(
            mesh, (mesh.dc_edge * mesh.dv_edge)[:, None] * self.velocity(state).square())

    def thickness_gradient(self, state):
        eta = state.column.eta(state.h)
        surface = self.rho0 * self.gravity * state.mesh.cell_area[:, None] * eta
        if self.domain is not None:
            surface = self.domain.apply_cell(surface)
        return self.kinetic_thickness_gradient(state) + surface

    def terms(self, state):
        kinetic = .5 * (self.metric(state) * self.velocity(state).square()).sum()
        surface_density = state.column.eta(state.h).square()
        if self.domain is not None:
            surface_density = self.domain.apply_cell(surface_density)
        surface = .5 * self.rho0 * self.gravity * (state.mesh.cell_area[:, None] * surface_density).sum()
        return {'kinetic': kinetic, 'free_surface': surface, 'total': kinetic + surface}

    def gradients(self, state):
        """Euclidean derivatives with respect to the stored prognostic tensors."""
        return {'u_e': self.metric(state) * self.velocity(state),
                'h': self.thickness_gradient(state),
                'temp_h': torch.zeros_like(state.temp_h),
                'salt_h': torch.zeros_like(state.salt_h),
                'tracer_h': None if state.tracer_h is None else torch.zeros_like(state.tracer_h)}
