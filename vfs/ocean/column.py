"""Vertical ocean column geometry."""

from __future__ import annotations

import torch
import torch.nn as nn

from .constants import OCEAN_GRAVITY, OCEAN_RHO0


class OceanColumn(nn.Module):
    """Depth/layer geometry for a global multilayer ocean column."""

    def __init__(
        self,
        n_layers: int,
        *,
        total_depth: float | torch.Tensor = 4000.0,
        rest_thickness: torch.Tensor | None = None,
        gravity: float = OCEAN_GRAVITY,
        rho0: float = OCEAN_RHO0,
    ) -> None:
        super().__init__()
        n_layers = int(n_layers)
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if rest_thickness is None:
            rest = torch.full((n_layers,), float(total_depth) / float(n_layers), dtype=torch.float64)
        else:
            rest = torch.as_tensor(rest_thickness, dtype=torch.float64).reshape(-1)
            if tuple(rest.shape) != (n_layers,):
                raise ValueError(f"rest_thickness must have shape ({n_layers},), got {tuple(rest.shape)}")
        if not torch.isfinite(rest).all() or float(rest.min().item()) <= 0.0:
            raise ValueError("rest_thickness must be finite and strictly positive")
        depth = torch.as_tensor(total_depth, dtype=torch.float64)
        if depth.ndim > 2:
            raise ValueError("total_depth must be scalar, [nCells], or [nCells, 1]")
        if depth.ndim == 1:
            depth = depth[:, None]
        if not torch.isfinite(depth).all() or float(depth.min().item()) <= 0.0:
            raise ValueError("total_depth must be finite and strictly positive")
        if depth.ndim == 0:
            total = rest.sum()
            if abs(float(total.item()) - float(depth.item())) > 1.0e-8 * max(float(depth.item()), 1.0):
                raise ValueError("rest_thickness must sum to total_depth")

        self.n_layers = n_layers
        self.gravity = float(gravity)
        self.rho0 = float(rho0)
        self.register_buffer("rest_thickness", rest)
        self.register_buffer("total_depth", depth)

    @classmethod
    def top_refined(
        cls,
        n_layers: int,
        *,
        total_depth: float = 4000.0,
        stretch: float = 1.35,
        gravity: float = OCEAN_GRAVITY,
        rho0: float = OCEAN_RHO0,
    ) -> "OceanColumn":
        """Return a positive top-refined column with geometric layer growth.

        ``stretch=1`` recovers uniform layers. Larger values make the upper
        ocean thinner and the deep ocean thicker while preserving total depth.
        """

        n_layers = int(n_layers)
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        stretch = float(stretch)
        if stretch < 1.0:
            raise ValueError("top-refined ocean column stretch must be >= 1")
        depth = float(total_depth)
        if depth <= 0.0:
            raise ValueError("top-refined ocean column total_depth must be positive")
        weights = torch.pow(
            torch.as_tensor(stretch, dtype=torch.float64),
            torch.arange(n_layers, dtype=torch.float64),
        )
        rest = depth * weights / weights.sum()
        return cls(
            n_layers,
            total_depth=depth,
            rest_thickness=rest,
            gravity=gravity,
            rho0=rho0,
        )

    def rest_thickness_like(self, h: torch.Tensor) -> torch.Tensor:
        rest = self.rest_thickness.to(dtype=h.dtype, device=h.device)
        shape = (1,) * (h.ndim - 1) + (self.n_layers,)
        return rest.reshape(shape).expand_as(h)

    def total_depth_like(self, h: torch.Tensor) -> torch.Tensor:
        depth = self.total_depth.to(dtype=h.dtype, device=h.device)
        target = h.shape[:-1] + (1,)
        if depth.ndim == 0:
            return depth.reshape(()).expand(target)
        if depth.ndim == 1:
            depth = depth[:, None]
        shape = (1,) * (h.ndim - 2) + tuple(depth.shape)
        return depth.reshape(shape).expand(target)

    def eta(self, h: torch.Tensor) -> torch.Tensor:
        """Free-surface displacement relative to resting total depth."""

        return h.sum(dim=-1, keepdim=True) - self.total_depth_like(h)

    def interface_depth(self, h: torch.Tensor) -> torch.Tensor:
        """Layer interface depth below the free surface, positive downward."""

        zero = torch.zeros_like(h[..., :1])
        return torch.cat([zero, torch.cumsum(h, dim=-1)], dim=-1)

    def mid_depth(self, h: torch.Tensor) -> torch.Tensor:
        z_i = self.interface_depth(h)
        return 0.5 * (z_i[..., :-1] + z_i[..., 1:])

    def validate_thickness(self, h: torch.Tensor, *, min_thickness: float = 0.0) -> None:
        if not torch.isfinite(h).all().item():
            raise FloatingPointError("ocean layer thickness is not finite")
        h_min = float(h.min().item())
        if h_min <= float(min_thickness):
            raise FloatingPointError(f"ocean layer thickness below minimum: h_min={h_min:.3e}")
