"""Atmospheric terrain geopotential, surface pressure and hybrid columns."""
from __future__ import annotations
from typing import Any
import torch
from .column import CharneyPhillipsColumn, make_hybrid_coefficients
from vfs.data.elevation import _as_tensor

GRAVITY = 9.80665
EARTH_GRAVITY = GRAVITY
R_D = 287.05
DRY_AIR_GAS_CONSTANT = R_D
DEFAULT_SCALE_HEIGHT = 8000.0

def build_atmospheric_topographic_column(
    mesh,
    surface_geopotential: torch.Tensor | Any,
    *,
    n_levels: int,
    p_top: float = 5000.0,
    sigma_power: float = 1.0,
    phi_full_mode: str = "interface_mean",
) -> CharneyPhillipsColumn:
    """Construct an atmospheric Charney-Phillips column with surface terrain."""

    phi_s = _as_tensor(surface_geopotential, dtype=torch.float64).reshape(-1)
    if int(phi_s.numel()) != int(mesh.nCells):
        raise ValueError(f"surface_geopotential must have nCells={int(mesh.nCells)} values; got {tuple(phi_s.shape)}")
    A, B = make_hybrid_coefficients(int(n_levels), p_top=float(p_top), sigma_power=float(sigma_power))
    return CharneyPhillipsColumn(
        int(n_levels),
        A,
        B,
        phi_s=phi_s,
        phi_full_mode=phi_full_mode,
    )


def build_topographic_column(
    mesh,
    surface_geopotential: torch.Tensor | Any,
    *,
    n_levels: int,
    p_top: float = 5000.0,
    sigma_power: float = 1.0,
    phi_full_mode: str = "interface_mean",
) -> CharneyPhillipsColumn:
    """Alias for ``build_atmospheric_topographic_column``."""

    return build_atmospheric_topographic_column(
        mesh,
        surface_geopotential,
        n_levels=n_levels,
        p_top=p_top,
        sigma_power=sigma_power,
        phi_full_mode=phi_full_mode,
    )


def atmospheric_surface_pressure_from_phi_s(
    phi_s: torch.Tensor | Any,
    *,
    p0: float = 1.0e5,
    temperature: float = 288.0,
    scale_height: float | None = None,
) -> torch.Tensor:
    """Approximate hydrostatic surface pressure from terrain geopotential."""

    phi = _as_tensor(phi_s, dtype=torch.float64) if not torch.is_tensor(phi_s) else (
        phi_s if phi_s.is_floating_point() else phi_s.to(dtype=torch.float64)
    )
    if scale_height is None:
        pressure_scale = float(R_D) * float(temperature)
    else:
        pressure_scale = float(GRAVITY) * float(scale_height)
    return float(p0) * torch.exp(-phi / pressure_scale)


def earth_surface_pressure_from_phi_s(
    phi_s: torch.Tensor | Any,
    *,
    p0: float = 1.0e5,
    temperature: float = 288.0,
    scale_height: float | None = None,
) -> torch.Tensor:
    """Alias for ``atmospheric_surface_pressure_from_phi_s``."""

    return atmospheric_surface_pressure_from_phi_s(
        phi_s,
        p0=p0,
        temperature=temperature,
        scale_height=scale_height,
    )


def atmospheric_surface_height_from_elevation(
    elevation_m: torch.Tensor | Any,
    *,
    ocean_threshold_m: float = 0.0,
) -> torch.Tensor:
    """Return atmospheric surface height from elevation data.

    ETOPO-style elevation is positive over land and negative over ocean
    bathymetry. Atmospheric terrain should follow land elevation but stay at
    sea level over ocean, so negative elevations are clipped away.
    """

    elevation = _as_tensor(elevation_m, dtype=torch.float64) if not torch.is_tensor(elevation_m) else (
        elevation_m if elevation_m.is_floating_point() else elevation_m.to(dtype=torch.float64)
    )
    return (elevation - float(ocean_threshold_m)).clamp_min(0.0)


def atmospheric_surface_geopotential_from_elevation(
    elevation_m: torch.Tensor | Any,
    *,
    ocean_threshold_m: float = 0.0,
    gravity: float = GRAVITY,
) -> torch.Tensor:
    """Return ``phi_s`` for atmospheric columns from ETOPO-style elevation."""

    return float(gravity) * atmospheric_surface_height_from_elevation(
        elevation_m,
        ocean_threshold_m=float(ocean_threshold_m),
    )


