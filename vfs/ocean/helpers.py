"""Private helper functions shared by VFS-O modules."""

from __future__ import annotations

import json
from typing import Any

import torch

from .constants import OCEAN_RHO0


def _combine_optional(left: torch.Tensor | None, right: torch.Tensor | None, op) -> torch.Tensor | None:
    if left is None and right is None:
        return None
    if left is not None and right is not None:
        return op(left, right)
    raise ValueError("cannot combine OceanState objects with inconsistent tracer_h presence")


def _require_trailing_shape(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if value.ndim < len(shape) or tuple(int(dim) for dim in value.shape[-len(shape) :]) != tuple(shape):
        raise ValueError(f"{name} must have trailing shape {shape}; got {tuple(value.shape)}")


def _require_tracer_shape(value: torch.Tensor, n_cells: int, n_layers: int) -> None:
    _normalize_tracer_tensor(value, n_cells, n_layers)


def _normalize_tracer_tensor(value: torch.Tensor, n_cells: int, n_layers: int) -> torch.Tensor:
    if value.ndim >= 2 and value.shape[-2] == n_cells and value.shape[-1] == n_layers:
        value = value.unsqueeze(-1)
    if value.ndim < 3 or value.shape[-3] != n_cells or value.shape[-2] != n_layers:
        raise ValueError(
            "ocean tracer tensor must have shape [..., nCells, nLayers] or "
            f"[..., nCells, nLayers, nTracers]; got {tuple(value.shape)}"
        )
    return value


def _cell_midlayer_spacing(state: OceanState) -> torch.Tensor:
    mid = state.column.mid_depth(state.h)
    return (mid[..., 1:] - mid[..., :-1]).clamp_min(torch.finfo(state.h.dtype).tiny)


def _edge_midlayer_spacing(h_edge: torch.Tensor) -> torch.Tensor:
    mid = torch.cumsum(h_edge, dim=-1) - 0.5 * h_edge
    return (mid[..., 1:] - mid[..., :-1]).clamp_min(torch.finfo(h_edge.dtype).tiny)


def _edge_east_unit(lon: torch.Tensor) -> torch.Tensor:
    return torch.cat([-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)], dim=-1)


def _edge_column_field(value: Any, like: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if tensor.ndim == 0:
        return torch.full_like(like, float(tensor.item()))
    if tensor.ndim == 1:
        tensor = tensor[:, None]
    if tensor.shape == like.shape:
        return tensor
    if tensor.ndim == 2 and int(tensor.shape[0]) == int(like.shape[0]) and int(tensor.shape[1]) >= 1:
        return tensor[..., :1]
    raise ValueError(f"{name} must be scalar, [nEdges], or [nEdges, nColumns]; got {tuple(tensor.shape)}")


def _ocean_vertical_momentum_stress_tendency(
    state: OceanState,
    *,
    viscosity: float = 0.0,
    surface_stress_edge: torch.Tensor | None = None,
    bottom_stress_edge: torch.Tensor | None = None,
    rho0: float = OCEAN_RHO0,
) -> torch.Tensor:
    """Return edge/layer velocity tendency from vertical momentum stress.

    Surface and bottom stresses are dynamic stresses in N m-2. Interior
    viscous stresses are kinematic stresses, then converted to the same dynamic
    stress convention with ``rho0``.
    """

    h_edge = state.mesh.cell_to_edge(state.h).clamp_min(torch.finfo(state.h.dtype).tiny)
    out = torch.zeros_like(state.u_e)
    if surface_stress_edge is not None:
        tau = _edge_column_field(surface_stress_edge, state.u_e[..., :1], name="surface_stress_edge")
        out[..., :1] = out[..., :1] + tau / (float(rho0) * h_edge[..., :1])
    if float(viscosity) != 0.0 and int(state.column.n_layers) >= 2:
        spacing = _edge_midlayer_spacing(h_edge)
        stress = float(rho0) * float(viscosity) * (state.u_e[..., 1:] - state.u_e[..., :-1]) / spacing
        out[..., :-1] += stress / (float(rho0) * h_edge[..., :-1])
        out[..., 1:] -= stress / (float(rho0) * h_edge[..., 1:])
    if bottom_stress_edge is not None:
        tau_bottom = _edge_column_field(bottom_stress_edge, state.u_e[..., -1:], name="bottom_stress_edge")
        out[..., -1:] = out[..., -1:] - tau_bottom / (float(rho0) * h_edge[..., -1:])
    return out


def _netcdf_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in dict(attrs or {}).items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().cpu().item()
        if isinstance(value, bool):
            out[str(key)] = "true" if value else "false"
        elif isinstance(value, (str, int, float)):
            out[str(key)] = value
        else:
            try:
                out[str(key)] = json.dumps(value, sort_keys=True)
            except TypeError:
                continue
    return out
