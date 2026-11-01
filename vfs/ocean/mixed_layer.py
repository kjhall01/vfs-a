"""Torch-native mixed-layer helper formulas for VFS-O."""

from __future__ import annotations

import torch


def density_threshold_mixed_layer_depth(
    density: torch.Tensor,
    mid_depth: torch.Tensor,
    *,
    density_threshold: float = 0.03,
    keepdim: bool = False,
) -> torch.Tensor:
    """Return first-crossing density-threshold mixed-layer depth.

    The criterion is the first depth where density exceeds the surface density
    by ``density_threshold``.  The crossing is linearly interpolated between
    layer midpoints.  If no crossing exists, the deepest midpoint is returned.
    This avoids the old whole-layer jump and "deepest layer that still happens
    to be below threshold" behavior.
    """

    if density.shape != mid_depth.shape:
        raise ValueError(
            "density and mid_depth must have identical shapes; "
            f"got {tuple(density.shape)} and {tuple(mid_depth.shape)}"
        )
    if density.ndim < 1:
        raise ValueError("density must have at least one layer dimension")
    if int(density.shape[-1]) < 2:
        out = mid_depth[..., :1]
        return out if keepdim else out[..., 0]

    rho_excess = density - density[..., :1]
    above = rho_excess > float(density_threshold)
    found = above.any(dim=-1, keepdim=True)
    first_above = above.to(dtype=torch.int64).argmax(dim=-1, keepdim=True).clamp_min(1)
    previous = first_above - 1

    rho0 = torch.gather(rho_excess, -1, previous)
    rho1 = torch.gather(rho_excess, -1, first_above)
    depth0 = torch.gather(mid_depth, -1, previous)
    depth1 = torch.gather(mid_depth, -1, first_above)

    delta = rho1 - rho0
    safe_delta = torch.where(
        delta.abs() > torch.finfo(density.dtype).tiny,
        delta,
        torch.ones_like(delta),
    )
    weight = ((float(density_threshold) - rho0) / safe_delta).clamp(0.0, 1.0)
    crossing = depth0 + weight * (depth1 - depth0)
    deepest = mid_depth[..., -1:]
    out = torch.where(found, crossing, deepest)
    return out if keepdim else out[..., 0]


def mixed_layer_interface_weight(
    interface_depth: torch.Tensor,
    mixed_layer_depth: torch.Tensor,
    *,
    transition_thickness: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return smooth 0..1 mixed-layer weights at vertical interfaces.

    With ``transition_thickness=None`` this is the binary depth mask.  With a
    positive transition thickness, only the boundary interface is partially
    weighted; interfaces well inside the mixed layer remain one and interfaces
    well below remain zero.
    """

    if transition_thickness is None:
        return (interface_depth <= mixed_layer_depth).to(dtype=interface_depth.dtype)
    width = transition_thickness.to(dtype=interface_depth.dtype, device=interface_depth.device).clamp_min(
        torch.finfo(interface_depth.dtype).tiny
    )
    return ((mixed_layer_depth - interface_depth) / width + 0.5).clamp(0.0, 1.0)


__all__ = [
    "density_threshold_mixed_layer_depth",
    "mixed_layer_interface_weight",
]
