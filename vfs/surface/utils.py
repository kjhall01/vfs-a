"""Small tensor helpers shared by surface-system components."""

from __future__ import annotations

from typing import Any

import torch

from vfs.field import Field


def field_values(value: Any) -> Any:
    """Return numerical storage for a public Field at surface kernel boundaries."""

    return value.values if isinstance(value, Field) else value


def as_cell_column(
    value: Any,
    *,
    n_cells: int | None = None,
    like: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    name: str = "surface field",
) -> torch.Tensor:
    """Normalize a scalar/cell vector to VFS' canonical ``[nCells, 1]`` shape."""

    like = field_values(like)
    value = field_values(value)
    if like is not None:
        n_cells = int(like.shape[-2])
        dtype = like.dtype if dtype is None else dtype
        device = like.device if device is None else device
    if dtype is None:
        dtype = value.dtype if torch.is_tensor(value) else torch.float64
    if device is None and torch.is_tensor(value):
        device = value.device
    tensor = value.to(dtype=dtype, device=device) if torch.is_tensor(value) else torch.as_tensor(
        value,
        dtype=dtype,
        device=device,
    )
    if n_cells is None:
        if tensor.ndim == 0:
            raise ValueError("n_cells is required for scalar surface fields when like is not supplied")
        n_cells = int(tensor.shape[0])
    if tensor.ndim == 0:
        tensor = torch.ones((int(n_cells), 1), dtype=dtype, device=device) * tensor
    elif tensor.ndim == 1:
        tensor = tensor[:, None]
    if tuple(tensor.shape) != (int(n_cells), 1):
        raise ValueError(f"{name} must have shape ({int(n_cells)},) or ({int(n_cells)}, 1); got {tuple(tensor.shape)}")
    return tensor


def require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(field_values(value)).all().detach().cpu().item()):
        raise ValueError(f"{name} must be finite")


def cell_area_column(mesh: Any | None, like: torch.Tensor) -> torch.Tensor:
    """Return native cell areas as ``[nCells, 1]`` or unit areas without a mesh."""

    values = field_values(like)
    if mesh is None:
        return torch.ones_like(values)
    return mesh.cell_area.to(dtype=values.dtype, device=values.device).reshape(-1, 1)


def sum_extensive(value: torch.Tensor) -> torch.Tensor:
    """Sum a scalar or per-cell extensive tensor without detaching it."""

    values = field_values(value)
    return values.sum() if torch.is_tensor(values) else torch.as_tensor(values).sum()


__all__ = ["as_cell_column", "cell_area_column", "field_values", "require_finite", "sum_extensive"]
