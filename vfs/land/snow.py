"""Snow-on-land helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .parameters import LandSurfaceParameters
from .state import LandSurfaceState


@dataclass(frozen=True)
class SnowBoundaryProperties:
    cover_fraction: torch.Tensor
    albedo: torch.Tensor
    roughness: torch.Tensor
    insulation_factor: torch.Tensor


def snow_melt_rate(
    state: LandSurfaceState,
    *,
    land_fraction: torch.Tensor,
    dt: torch.Tensor,
    params: LandSurfaceParameters,
) -> torch.Tensor:
    land = land_fraction.to(dtype=state.soil_water.dtype, device=state.soil_water.device)
    active = land > 0.0
    potential = (
        float(params.snow_melt_factor)
        * (state.skin_temperature - float(params.melt_temperature)).clamp_min(0.0)
    )
    available = state.snow_water_equivalent / dt.clamp_min(torch.finfo(state.soil_water.dtype).tiny)
    return torch.where(active, torch.minimum(available, potential), torch.zeros_like(available))


def update_snow_storage(
    state: LandSurfaceState,
    *,
    land_fraction: torch.Tensor,
    snowfall_rate: torch.Tensor,
    melt_rate: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    land = land_fraction.to(dtype=state.soil_water.dtype, device=state.soil_water.device)
    active = land > 0.0
    snow = snowfall_rate.to(dtype=state.soil_water.dtype, device=state.soil_water.device).clamp_min(0.0)
    snow = torch.where(active, snow, torch.zeros_like(snow))
    snow_next = (state.snow_water_equivalent + dt * (snow - melt_rate)).clamp_min(0.0)
    snow_next = torch.where(active, snow_next, state.snow_water_equivalent)
    return snow_next, snow


def snow_cover_fraction(state: LandSurfaceState, params: LandSurfaceParameters) -> torch.Tensor:
    return (state.snow_water_equivalent / float(params.snow_cover_swe)).clamp(0.0, 1.0)


def snow_insulation_factor(state: LandSurfaceState, params: LandSurfaceParameters) -> torch.Tensor:
    return (1.0 / (1.0 + state.snow_water_equivalent / float(params.snow_insulation_swe_scale))).clamp(0.0, 1.0)


def snow_boundary_properties(
    state: LandSurfaceState,
    params: LandSurfaceParameters,
    *,
    base_albedo: float | torch.Tensor,
    base_roughness: float | torch.Tensor,
) -> SnowBoundaryProperties:
    cover = snow_cover_fraction(state, params)
    base_albedo_t = torch.as_tensor(base_albedo, dtype=state.skin_temperature.dtype, device=state.skin_temperature.device)
    base_roughness_t = torch.as_tensor(base_roughness, dtype=state.skin_temperature.dtype, device=state.skin_temperature.device)
    albedo = (1.0 - cover) * base_albedo_t + cover * float(params.snow_albedo)
    roughness = (1.0 - cover) * base_roughness_t + cover * float(params.snow_roughness)
    return SnowBoundaryProperties(
        cover,
        albedo.clamp(0.0, 1.0),
        roughness.clamp_min(0.0),
        snow_insulation_factor(state, params),
    )
