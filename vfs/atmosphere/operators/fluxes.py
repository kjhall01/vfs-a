"""Reusable Field-aware horizontal scalar-flux packages.

The implementation is adapted from the proven MUSCL pathway in
``voronoi-flow-solver`` but follows main VFS's ``[..., cell, level]`` axis
convention.  Keeping face reconstruction separate from the dry dynamics makes
the active transport flux inspectable and reusable by future matched momentum
experiments without importing their unvalidated wind bundles.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...field import Field


def _values(value):
    return value.values if isinstance(value, Field) else value


@dataclass(frozen=True)
class HorizontalScalarFluxPackage:
    """Donor-cell or limited-MUSCL CP-interface horizontal fluxes."""

    scheme: str = "upwind"
    blend: float = 1.0
    limiter_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.scheme not in {"upwind", "muscl"}:
            raise ValueError("scheme must be 'upwind' or 'muscl'")
        if not 0.0 <= float(self.blend) <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        if float(self.limiter_epsilon) <= 0.0:
            raise ValueError("limiter_epsilon must be positive")

    @classmethod
    def baseline_upwind(cls) -> "HorizontalScalarFluxPackage":
        return cls("upwind")

    @classmethod
    def muscl_horizontal(
        cls,
        *,
        blend: float = 1.0,
        limiter_epsilon: float = 1.0e-6,
    ) -> "HorizontalScalarFluxPackage":
        return cls("muscl", blend=float(blend), limiter_epsilon=float(limiter_epsilon))

    @staticmethod
    def _neighbor_index(mesh, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        cell_index = torch.arange(int(mesh.nCells), device=device, dtype=torch.long).view(-1, 1)
        safe_edges = mesh.safe_edges_on_cell.to(device=device, dtype=torch.long)
        edge_cells = mesh.safe_cells_on_edge.to(device=device, dtype=torch.long)
        neighbor0 = edge_cells[safe_edges, 0]
        neighbor1 = edge_cells[safe_edges, 1]
        neighbor = torch.where(neighbor0 == cell_index, neighbor1, neighbor0)
        valid = mesh.valid_edges_on_cell.to(device=device)
        return torch.where(valid, neighbor, cell_index), valid

    @staticmethod
    def _basis(mesh, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        xyz = mesh.cell_xyz.to(dtype=dtype, device=device)
        z = torch.zeros_like(xyz)
        z[:, 2] = 1.0
        x = torch.zeros_like(xyz)
        x[:, 0] = 1.0
        first = torch.cross(z, xyz, dim=-1)
        fallback = torch.cross(x, xyz, dim=-1)
        first = torch.where(first.norm(dim=-1, keepdim=True) < 1.0e-8, fallback, first)
        first = first / first.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
        second = torch.cross(xyz, first, dim=-1)
        return first, second / second.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)

    def _reconstructed_sides(self, mesh, theta_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return limited values reconstructed from each endpoint of every edge."""

        dtype, device = theta_i.dtype, theta_i.device
        leading_ndim = theta_i.ndim - 2
        lead = (1,) * leading_ndim
        neighbor, valid = self._neighbor_index(mesh, device)
        basis_x, basis_y = self._basis(mesh, dtype, device)
        cell_xyz = mesh.cell_xyz.to(dtype=dtype, device=device)
        safe_edges = mesh.safe_edges_on_cell.to(device=device, dtype=torch.long)

        neighbor_xyz = cell_xyz[neighbor]
        center_xyz = cell_xyz[:, None, :]
        tangent = neighbor_xyz - (neighbor_xyz * center_xyz).sum(dim=-1, keepdim=True) * center_xyz
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
        distance = mesh.dc_edge.to(dtype=dtype, device=device)[safe_edges]
        distance = torch.where(valid, distance, torch.zeros_like(distance))
        x = distance * (tangent * basis_x[:, None, :]).sum(dim=-1)
        y = distance * (tangent * basis_y[:, None, :]).sum(dim=-1)

        neighbor_theta = theta_i[..., neighbor, :]
        center_theta = theta_i.unsqueeze(-2)
        valid_view = valid.view(*lead, int(mesh.nCells), -1, 1)
        neighbor_theta = torch.where(valid_view, neighbor_theta, center_theta)
        delta_theta = neighbor_theta - center_theta
        x_view = x.view(*lead, int(mesh.nCells), -1, 1)
        y_view = y.view(*lead, int(mesh.nCells), -1, 1)
        a11 = (x * x).sum(dim=-1).to(dtype=dtype).view(*lead, int(mesh.nCells), 1)
        a12 = (x * y).sum(dim=-1).to(dtype=dtype).view(*lead, int(mesh.nCells), 1)
        a22 = (y * y).sum(dim=-1).to(dtype=dtype).view(*lead, int(mesh.nCells), 1)
        b1 = (x_view * delta_theta).sum(dim=-2)
        b2 = (y_view * delta_theta).sum(dim=-2)
        determinant = (a11 * a22 - a12.square()).clamp_min(torch.finfo(dtype).tiny)
        grad_x = (a22 * b1 - a12 * b2) / determinant
        grad_y = (-a12 * b1 + a11 * b2) / determinant
        theta_min = torch.minimum(theta_i, neighbor_theta.min(dim=-2).values)
        theta_max = torch.maximum(theta_i, neighbor_theta.max(dim=-2).values)

        c0 = mesh.cellsOnEdge[:, 0].to(device=device, dtype=torch.long)
        c1 = mesh.cellsOnEdge[:, 1].to(device=device, dtype=torch.long)
        edge_xyz = mesh.edge_xyz.to(dtype=dtype, device=device)

        def reconstruct(cells: torch.Tensor) -> torch.Tensor:
            center = cell_xyz[cells]
            tangent_edge = edge_xyz - (edge_xyz * center).sum(dim=-1, keepdim=True) * center
            tangent_edge = tangent_edge / tangent_edge.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
            distance_edge = float(mesh.radius) * torch.acos((edge_xyz * center).sum(dim=-1).clamp(-1.0, 1.0))
            rx = distance_edge * (tangent_edge * basis_x[cells]).sum(dim=-1)
            ry = distance_edge * (tangent_edge * basis_y[cells]).sum(dim=-1)
            phi = theta_i[..., cells, :]
            delta = grad_x[..., cells, :] * rx.view(*lead, -1, 1) + grad_y[..., cells, :] * ry.view(*lead, -1, 1)
            allowed = torch.where(delta >= 0.0, theta_max[..., cells, :] - phi, phi - theta_min[..., cells, :])
            ratio = allowed.clamp_min(0.0) / delta.abs().clamp_min(torch.finfo(dtype).tiny)
            one = torch.ones((), dtype=dtype, device=device)
            epsilon = torch.as_tensor(float(self.limiter_epsilon), dtype=dtype, device=device)
            limiter = 0.5 * (ratio + one - torch.sqrt((ratio - one).square() + epsilon.square()))
            return phi + limiter.clamp(0.0, 1.0) * delta

        return reconstruct(c0), reconstruct(c1)

    def face_values(
        self,
        state,
        theta_i: Field | torch.Tensor,
        mass_flux: Field | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return upper/lower CP-interface face values at each horizontal edge."""

        theta = _values(theta_i)
        flux = _values(mass_flux)
        c0 = state.mesh.cellsOnEdge[:, 0].to(device=theta.device, dtype=torch.long)
        c1 = state.mesh.cellsOnEdge[:, 1].to(device=theta.device, dtype=torch.long)
        upper_upwind = torch.where(flux >= 0.0, theta[..., c0, :-1], theta[..., c1, :-1])
        lower_upwind = torch.where(flux >= 0.0, theta[..., c0, 1:], theta[..., c1, 1:])
        if self.scheme == "upwind":
            return upper_upwind, lower_upwind
        left, right = self._reconstructed_sides(state.mesh, theta)
        upper_high = torch.where(flux >= 0.0, left[..., :-1], right[..., :-1])
        lower_high = torch.where(flux >= 0.0, left[..., 1:], right[..., 1:])
        return (
            upper_upwind + float(self.blend) * (upper_high - upper_upwind),
            lower_upwind + float(self.blend) * (lower_high - lower_upwind),
        )

    def scalar_tendency(self, state, theta_i: Field | torch.Tensor, mass_flux: Field | torch.Tensor) -> torch.Tensor:
        """Return conservative horizontal CP-interface ``mtheta`` tendency."""

        theta = _values(theta_i)
        flux = _values(mass_flux)
        upper, lower = self.face_values(state, theta, flux)
        div_upper = state.mesh.div(0.5 * flux * upper)
        div_lower = state.mesh.div(0.5 * flux * lower)
        out = torch.zeros_like(theta)
        out[..., :-1] -= div_upper
        out[..., 1:] -= div_lower
        return out


__all__ = ["HorizontalScalarFluxPackage"]
