"""Sea-ice boundary property helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .parameters import SeaIceParameters
from .state import SeaIceState


@dataclass(frozen=True)
class SeaIceBoundaryProperties:
    surface_temperature: torch.Tensor
    albedo: torch.Tensor
    roughness: torch.Tensor
    snow_cover_fraction: torch.Tensor
    insulation_factor: torch.Tensor


def sea_ice_snow_cover_fraction(state: SeaIceState, params: SeaIceParameters) -> torch.Tensor:
    return (state.snow_on_sea_ice / float(params.snow_cover_swe)).clamp(0.0, 1.0)


def sea_ice_snow_insulation_factor(state: SeaIceState, params: SeaIceParameters) -> torch.Tensor:
    return (1.0 / (1.0 + state.snow_on_sea_ice / float(params.snow_insulation_swe_scale))).clamp(0.0, 1.0)


def sea_ice_boundary_properties(state: SeaIceState, params: SeaIceParameters) -> SeaIceBoundaryProperties:
    cover = sea_ice_snow_cover_fraction(state, params)
    albedo = (1.0 - cover) * float(params.bare_ice_albedo) + cover * float(params.snow_albedo)
    roughness = (1.0 - cover) * float(params.ice_roughness) + cover * float(params.snow_roughness)
    return SeaIceBoundaryProperties(
        state.sea_ice_surface_temperature,
        albedo.clamp(0.0, 1.0),
        roughness.clamp_min(0.0),
        cover,
        sea_ice_snow_insulation_factor(state, params),
    )
