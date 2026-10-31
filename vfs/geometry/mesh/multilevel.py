"""Field-native geometric cell transfers for Voronoi multilevel solvers.

The main VFS mesh is an arbitrary spherical Voronoi tessellation rather than
the donor tree's refined-icosahedral grid.  These transfers therefore use a
small inverse-distance cell stencil and its normalized area adjoint.  They
are suitable for smooth coarse corrections, preserve constants, and make the
cell location explicit at the public boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from vfs.field import Field, require_field

from .grid import VoronoiMesh


def _barycentric_stencil(
    fine_xyz: np.ndarray,
    coarse_xyz: np.ndarray,
    coarse_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate fine rays in the coarse triangulation and return linear weights."""

    incident: list[list[int]] = [[] for _ in range(coarse_xyz.shape[0])]
    for face_id, face in enumerate(coarse_faces):
        for vertex in face:
            incident[int(vertex)].append(face_id)
    nearest = cKDTree(coarse_xyz).query(fine_xyz, k=1)[1]
    indices = np.empty((fine_xyz.shape[0], 3), dtype=np.int64)
    weights = np.empty((fine_xyz.shape[0], 3), dtype=np.float64)

    for row, (point, nearest_vertex) in enumerate(zip(fine_xyz, nearest, strict=True)):
        best_face: np.ndarray | None = None
        best_weights: np.ndarray | None = None
        best_minimum = -np.inf
        for face_id in incident[int(nearest_vertex)]:
            face = coarse_faces[face_id]
            a, b, c = coarse_xyz[face]
            normal = np.cross(b - a, c - a)
            ray_dot = float(np.dot(normal, point))
            if abs(ray_dot) <= np.finfo(np.float64).eps:
                continue
            intersection = point * (float(np.dot(normal, a)) / ray_dot)
            v0, v1, v2 = b - a, c - a, intersection - a
            d00, d01, d11 = float(np.dot(v0, v0)), float(np.dot(v0, v1)), float(np.dot(v1, v1))
            d20, d21 = float(np.dot(v2, v0)), float(np.dot(v2, v1))
            denominator = d00 * d11 - d01 * d01
            if abs(denominator) <= np.finfo(np.float64).eps:
                continue
            wb = (d11 * d20 - d01 * d21) / denominator
            wc = (d00 * d21 - d01 * d20) / denominator
            candidate = np.asarray((1.0 - wb - wc, wb, wc))
            minimum = float(candidate.min())
            if minimum > best_minimum:
                best_face, best_weights, best_minimum = face, candidate, minimum
        if best_face is None or best_weights is None or best_minimum < -1.0e-7:
            raise RuntimeError("could not locate a fine cell in the coarse icosahedral triangulation")
        indices[row] = best_face
        clipped = np.maximum(best_weights, 0.0)
        weights[row] = clipped / clipped.sum()
    return indices, weights


def _horizontal_axis(values: torch.Tensor, axis: int) -> int:
    """Normalize and validate a tensor horizontal axis."""

    normalized = axis if axis >= 0 else values.ndim + axis
    if normalized < 0 or normalized >= values.ndim:
        raise ValueError(f"horizontal axis {axis} is invalid for shape {tuple(values.shape)}")
    return normalized


@dataclass(frozen=True)
class CellTransfer:
    """Constant-preserving area-adjoint transfer between native cell meshes.

    ``prolong`` interpolates coarse cell values to the fine mesh.  ``restrict``
    is its normalized area adjoint, so the pair obeys the discrete identity
    ``<P coarse, fine>_fine = <coarse, R fine>_coarse`` under ``fine.cell_area``
    and :attr:`coarse_measure` respectively.  Tensor inputs follow the main
    VFS convention and default to horizontal axis ``-2`` (or ``-1`` for a
    scalar one-dimensional tensor); :class:`~vfs.Field` inputs determine the
    axis from their placement metadata.
    """

    coarse_indices: torch.Tensor
    prolongation_weights: torch.Tensor
    restriction_weights: torch.Tensor
    coarse_measure: torch.Tensor
    n_fine_cells: int
    n_coarse_cells: int

    @classmethod
    def from_meshes(
        cls,
        fine: VoronoiMesh,
        coarse: VoronoiMesh,
        *,
        stencil_size: int = 3,
    ) -> "CellTransfer":
        """Build the donor-equivalent transfer for two spherical meshes.

        A refined-icosahedral coarse mesh uses barycentric prolongation on its
        Delaunay triangles. Other mesh pairs retain the compact inverse-distance
        fallback, preserving general geometry utility outside the IMEX route.
        """

        if stencil_size < 1:
            raise ValueError("stencil_size must be positive")
        count = min(int(stencil_size), int(coarse.nCells))
        if count < 1:  # pragma: no cover - VoronoiMesh always has cells.
            raise ValueError("coarse mesh must contain at least one cell")

        dtype = fine.cell_area.dtype
        device = fine.cell_area.device
        if count == 3 and getattr(coarse, "mesh_kind", None) == "icosahedral":
            indices_np, weights_np = _barycentric_stencil(
                fine.cell_xyz.detach().to(dtype=torch.float64, device="cpu").numpy(),
                coarse.cell_xyz.detach().to(dtype=torch.float64, device="cpu").numpy(),
                coarse.cellsOnVertex.detach().to(device="cpu").numpy(),
            )
            indices = torch.as_tensor(indices_np, dtype=torch.long, device=device)
            weights = torch.as_tensor(weights_np, dtype=dtype, device=device)
        else:
            fine_xyz = fine.cell_xyz.detach().to(dtype=torch.float64)
            coarse_xyz = coarse.cell_xyz.detach().to(dtype=torch.float64, device=fine_xyz.device)
            distances, indices = torch.topk(torch.cdist(fine_xyz, coarse_xyz), count, dim=-1, largest=False)
            tiny = torch.finfo(distances.dtype).eps
            exact = distances <= 32.0 * tiny
            inverse_distance = distances.clamp_min(tiny).reciprocal().square()
            weights = inverse_distance / inverse_distance.sum(dim=-1, keepdim=True)
            if bool(exact.any()):
                exact_weights = exact.to(dtype=weights.dtype)
                exact_weights = exact_weights / exact_weights.sum(dim=-1, keepdim=True).clamp_min(tiny)
                weights = torch.where(exact.any(dim=-1, keepdim=True), exact_weights, weights)
            indices = indices.to(dtype=torch.long, device=device)
            weights = weights.to(dtype=dtype, device=device)
        weighted_area = weights * fine.cell_area.to(dtype=dtype, device=device).unsqueeze(-1)
        coarse_measure = torch.zeros(int(coarse.nCells), dtype=dtype, device=device).scatter_add(
            0,
            indices.reshape(-1),
            weighted_area.reshape(-1),
        )
        restriction = weighted_area / coarse_measure[indices].clamp_min(torch.finfo(dtype).tiny)
        return cls(
            coarse_indices=indices,
            prolongation_weights=weights,
            restriction_weights=restriction,
            coarse_measure=coarse_measure,
            n_fine_cells=int(fine.nCells),
            n_coarse_cells=int(coarse.nCells),
        )

    def _prolong_values(self, values: torch.Tensor, *, horizontal_axis: int) -> torch.Tensor:
        axis = _horizontal_axis(values, horizontal_axis)
        moved = values.movedim(axis, -1)
        if moved.shape[-1] != self.n_coarse_cells:
            raise ValueError(
                f"expected {self.n_coarse_cells} coarse cells on axis {horizontal_axis}, "
                f"got shape {tuple(values.shape)}"
            )
        indices = self.coarse_indices.to(device=moved.device)
        weights = self.prolongation_weights.to(dtype=moved.dtype, device=moved.device)
        gathered = moved[..., indices]
        weight_view = (1,) * (gathered.ndim - 2) + tuple(weights.shape)
        return (gathered * weights.reshape(weight_view)).sum(dim=-1).movedim(-1, axis)

    def _restrict_values(self, values: torch.Tensor, *, horizontal_axis: int) -> torch.Tensor:
        axis = _horizontal_axis(values, horizontal_axis)
        moved = values.movedim(axis, -1)
        if moved.shape[-1] != self.n_fine_cells:
            raise ValueError(
                f"expected {self.n_fine_cells} fine cells on axis {horizontal_axis}, "
                f"got shape {tuple(values.shape)}"
            )
        indices = self.coarse_indices.to(device=moved.device)
        weights = self.restriction_weights.to(dtype=moved.dtype, device=moved.device)
        weight_view = (1,) * (moved.ndim - 1) + tuple(weights.shape)
        contributions = (moved.unsqueeze(-1) * weights.reshape(weight_view)).flatten(start_dim=-2)
        scatter_indices = indices.reshape(-1).reshape(
            (1,) * (moved.ndim - 1) + (-1,)
        ).expand_as(contributions)
        out = torch.zeros(*moved.shape[:-1], self.n_coarse_cells, dtype=moved.dtype, device=moved.device)
        return torch.scatter_add(out, -1, scatter_indices, contributions).movedim(-1, axis)

    def prolong(self, values: Field | torch.Tensor, *, horizontal_axis: int | None = None) -> Field | torch.Tensor:
        """Interpolate a coarse cell Field/tensor to the fine mesh."""

        if isinstance(values, Field):
            field = require_field(values, location="cell", name="coarse_values")
            return field.with_values(self._prolong_values(field.values, horizontal_axis=field.horizontal_axis))
        if not isinstance(values, torch.Tensor):
            raise TypeError("values must be a vfs.Field or torch.Tensor")
        axis = (-2 if values.ndim >= 2 else -1) if horizontal_axis is None else int(horizontal_axis)
        return self._prolong_values(values, horizontal_axis=axis)

    def restrict(self, values: Field | torch.Tensor, *, horizontal_axis: int | None = None) -> Field | torch.Tensor:
        """Apply the normalized area-adjoint to a fine cell Field/tensor."""

        if isinstance(values, Field):
            field = require_field(values, location="cell", name="fine_values")
            return field.with_values(self._restrict_values(field.values, horizontal_axis=field.horizontal_axis))
        if not isinstance(values, torch.Tensor):
            raise TypeError("values must be a vfs.Field or torch.Tensor")
        axis = (-2 if values.ndim >= 2 else -1) if horizontal_axis is None else int(horizontal_axis)
        return self._restrict_values(values, horizontal_axis=axis)


@dataclass(frozen=True)
class VoronoiMeshHierarchy:
    """A compact geometric hierarchy for smooth cell-centered corrections.

    The fine VFS mesh is retained exactly. Coarser levels are the donor's
    deterministic class-I refined-icosahedral Voronoi meshes. An arbitrary
    VFS/MPAS fine mesh is therefore still the simulation mesh; only the
    preconditioner sees the static icosahedral coarse hierarchy.
    """

    meshes: tuple[VoronoiMesh, ...]
    transfers: tuple[CellTransfer, ...]
    frequencies: tuple[int, ...]

    @classmethod
    def build(
        cls,
        fine: VoronoiMesh,
        *,
        coarsening: int = 3,
        max_levels: int = 4,
        minimum_frequency: int = 3,
        stencil_size: int = 3,
    ) -> "VoronoiMeshHierarchy":
        """Build deterministic coarse Voronoi meshes and their cell transfers."""

        if coarsening < 2:
            raise ValueError("coarsening must be at least two")
        if max_levels < 2:
            raise ValueError("max_levels must be at least two")
        if minimum_frequency < 1:
            raise ValueError("minimum_frequency must be positive")

        fine_frequency = getattr(fine, "icosahedral_frequency", None)
        fine_is_icosahedral = getattr(fine, "mesh_kind", None) == "icosahedral" and fine_frequency is not None
        frequency = (
            int(fine_frequency)
            if fine_is_icosahedral
            else max(1, int(round(math.sqrt(max(int(fine.nCells) - 2, 1) / 10.0))))
        )
        frequencies = [frequency]
        if not fine_is_icosahedral:
            frequencies.append(max(1, int(round(frequency / int(coarsening)))))
        while len(frequencies) < int(max_levels):
            current = frequencies[-1]
            if current <= int(minimum_frequency):
                break
            next_frequency = max(int(minimum_frequency), int(round(current / int(coarsening))))
            if next_frequency >= current:
                break
            frequencies.append(next_frequency)
        if len(frequencies) < 2:
            raise ValueError(
                f"mesh with {fine.nCells} cells is too coarse for a multilevel hierarchy "
                f"at minimum_frequency={minimum_frequency}"
            )

        meshes = [fine]
        for frequency in frequencies[1:]:
            meshes.append(
                VoronoiMesh.refined_icosahedron(
                    int(frequency),
                    radius=fine.radius,
                    device=fine.cell_xyz.device,
                    verbose=False,
                )
            )
        transfers = [
            CellTransfer.from_meshes(fine_mesh, coarse_mesh, stencil_size=stencil_size)
            for fine_mesh, coarse_mesh in zip(meshes, meshes[1:])
        ]
        return cls(tuple(meshes), tuple(transfers), tuple(frequencies))


__all__ = ["CellTransfer", "VoronoiMeshHierarchy"]


