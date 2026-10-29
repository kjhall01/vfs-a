"""Diagnostic targets for passive shadow-tracer transport experiments."""

from __future__ import annotations

import math
from typing import Sequence

import torch


DEFAULT_SHADOW_DIAGNOSTIC_NAMES = (
    "grad_mag",
    "grad_east_orientation",
    "grad_north_orientation",
    "deformation_proxy",
    "laplacian_proxy",
    "front_strength",
    "divergence_proxy",
    "vorticity_proxy",
)


def _as_cell_level(q: torch.Tensor) -> torch.Tensor:
    if q.ndim == 2:
        return q
    if q.ndim == 3:
        return q[..., 0]
    raise ValueError("q must have shape [cell, level] or [cell, level, tracer]")


def _local_east_north(mesh, *, dtype: torch.dtype, device) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = mesh.cell_xyz.to(dtype=dtype, device=device)
    lat = torch.asin(xyz[:, 2].clamp(-1.0, 1.0))
    lon = torch.atan2(xyz[:, 1], xyz[:, 0])
    east = torch.stack(
        [-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)],
        dim=-1,
    )
    north = torch.stack(
        [
            -torch.sin(lat) * torch.cos(lon),
            -torch.sin(lat) * torch.sin(lon),
            torch.cos(lat),
        ],
        dim=-1,
    )
    return east, north


def _normalize_positive(value: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    scale = value.square().mean(dim=(0, 1), keepdim=True).sqrt().clamp_min(float(eps))
    return torch.tanh(value.abs() / scale)


def _normalize_signed(value: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    scale = value.square().mean(dim=(0, 1), keepdim=True).sqrt().clamp_min(float(eps))
    return torch.tanh(value / scale)


def _cell_average_from_vertices(mesh, vertex_field: torch.Tensor) -> torch.Tensor:
    vertices = mesh.verticesOnCell.to(device=vertex_field.device).long()
    valid = vertices >= 0
    safe = torch.where(valid, vertices, torch.zeros_like(vertices))
    gathered = vertex_field[safe]
    gathered = torch.where(valid[..., None].to(device=vertex_field.device), gathered, torch.zeros_like(gathered))
    count = valid.sum(dim=-1).to(dtype=vertex_field.dtype, device=vertex_field.device).clamp_min(1.0)
    return gathered.sum(dim=-2) / count[:, None]


def compute_shadow_diagnostics(
    mesh,
    q: torch.Tensor,
    velocity: torch.Tensor | None = None,
    *,
    names: Sequence[str] = DEFAULT_SHADOW_DIAGNOSTIC_NAMES,
    normalize: bool = True,
) -> torch.Tensor:
    """Return interpretable shadow-tracer diagnostic targets.

    Parameters
    ----------
    mesh:
        VFS Voronoi mesh.
    q:
        Tracer concentration with shape ``[cell, level]`` or
        ``[cell, level, tracer]``.  The first tracer is used when a tracer
        dimension is present.
    velocity:
        Optional edge-normal velocity with shape ``[edge, level]``.
    names:
        Ordered channel names to return.  Defaults to the eight initial
        diagnostic targets used by the shadow-tracer benchmark.
    normalize:
        If true, positive channels are squashed to roughly ``[0, 1]`` and
        signed channels to roughly ``[-1, 1]`` with RMS-based scales.
    """

    q_cl = _as_cell_level(q)
    if q_cl.shape[0] != mesh.nCells:
        raise ValueError("q leading dimension must match mesh.nCells")
    dtype = q_cl.dtype
    device = q_cl.device
    eps = torch.finfo(dtype).eps

    grad_e = mesh.grad(q_cl)
    grad_vec = mesh.cell_wind_vector(grad_e)
    east, north = _local_east_north(mesh, dtype=dtype, device=device)
    grad_x = (grad_vec * east[:, None, :]).sum(dim=-1)
    grad_y = (grad_vec * north[:, None, :]).sum(dim=-1)
    grad_square = (grad_x.square() + grad_y.square()).clamp_min(0.0)
    grad_mag = (torch.sqrt(grad_square + eps) - math.sqrt(float(eps))).clamp_min(0.0)
    orientation_x = grad_x / grad_mag.clamp_min(eps)
    orientation_y = grad_y / grad_mag.clamp_min(eps)

    laplacian = mesh.div(grad_e)
    front_strength = grad_mag * laplacian.abs()

    if velocity is None:
        velocity = torch.zeros(mesh.nEdges, q_cl.shape[-1], dtype=dtype, device=device)
    else:
        velocity = velocity.to(dtype=dtype, device=device)
    if velocity.ndim != 2:
        raise ValueError("velocity must have shape [edge, level]")

    velocity_vec = mesh.cell_wind_vector(velocity)
    speed = velocity_vec.norm(dim=-1)
    speed_grad = mesh.edge_to_cell(mesh.grad(speed).abs())
    divergence = mesh.div(velocity)
    vorticity_vertex = mesh.curl(velocity)
    vorticity = _cell_average_from_vertices(mesh, vorticity_vertex)

    raw = {
        "grad_mag": grad_mag,
        "grad_east_orientation": orientation_x,
        "grad_north_orientation": orientation_y,
        "deformation_proxy": speed_grad,
        "laplacian_proxy": laplacian,
        "front_strength": front_strength,
        "divergence_proxy": divergence,
        "vorticity_proxy": vorticity,
    }
    channels: list[torch.Tensor] = []
    for name in names:
        if name not in raw:
            raise ValueError(f"unknown shadow diagnostic channel {name!r}")
        value = raw[name]
        if normalize:
            if name in {"grad_east_orientation", "grad_north_orientation"}:
                value = value.clamp(-1.0, 1.0)
            elif name in {"laplacian_proxy", "divergence_proxy", "vorticity_proxy"}:
                value = _normalize_signed(value)
            else:
                value = _normalize_positive(value)
        channels.append(torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0))
    return torch.stack(channels, dim=-1)


__all__ = [
    "DEFAULT_SHADOW_DIAGNOSTIC_NAMES",
    "compute_shadow_diagnostics",
]
