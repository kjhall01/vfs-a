"""Parameter containers for the VFS land model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class LandSurfaceParameters:
    """Land model parameters shared by hydrology, snow, and soil thermal code."""

    soil_layer_thickness: tuple[float, ...] = (0.10, 0.30, 0.60, 1.00)
    soil_water_capacity: float = 150.0
    critical_soil_water: float = 50.0
    field_capacity_fraction: float = 0.70
    wilting_fraction: float = 0.15
    saturated_hydraulic_conductivity: float = 1.0e-5
    latent_heat_vaporization: float = 2.5e6
    latent_heat_fusion: float = 3.34e5
    canopy_water_capacity: float = 0.5
    melt_temperature: float = 273.15
    snow_melt_factor: float = 2.0e-5
    snow_cover_swe: float = 10.0
    snow_albedo: float = 0.80
    snow_roughness: float = 1.0e-3
    snow_insulation_swe_scale: float = 20.0
    skin_relaxation_timescale: float = 86400.0
    skin_heat_capacity: float = 2.0e6
    soil_heat_capacity: float = 2.0e6
    soil_thermal_conductivity: float = 1.5
    root_fraction: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "soil_layer_thickness", _as_tuple(self.soil_layer_thickness, "soil_layer_thickness"))
        if self.root_fraction is not None:
            object.__setattr__(self, "root_fraction", _as_tuple(self.root_fraction, "root_fraction"))

    def validate(self) -> None:
        if len(self.soil_layer_thickness) == 0:
            raise ValueError("soil_layer_thickness must include at least one layer")
        if any(thickness <= 0.0 for thickness in self.soil_layer_thickness):
            raise ValueError("soil_layer_thickness values must be positive")
        if self.soil_water_capacity <= 0.0:
            raise ValueError("soil_water_capacity must be positive")
        if self.critical_soil_water <= 0.0:
            raise ValueError("critical_soil_water must be positive")
        if not 0.0 <= self.wilting_fraction < self.field_capacity_fraction <= 1.0:
            raise ValueError("require 0 <= wilting_fraction < field_capacity_fraction <= 1")
        if self.saturated_hydraulic_conductivity < 0.0:
            raise ValueError("saturated_hydraulic_conductivity must be non-negative")
        if self.latent_heat_vaporization <= 0.0:
            raise ValueError("latent_heat_vaporization must be positive")
        if self.latent_heat_fusion <= 0.0:
            raise ValueError("latent_heat_fusion must be positive")
        if self.canopy_water_capacity < 0.0:
            raise ValueError("canopy_water_capacity must be non-negative")
        if self.snow_melt_factor < 0.0:
            raise ValueError("snow_melt_factor must be non-negative")
        if self.snow_cover_swe <= 0.0:
            raise ValueError("snow_cover_swe must be positive")
        if not 0.0 <= self.snow_albedo <= 1.0:
            raise ValueError("snow_albedo must be in [0, 1]")
        if self.snow_roughness < 0.0:
            raise ValueError("snow_roughness must be non-negative")
        if self.snow_insulation_swe_scale <= 0.0:
            raise ValueError("snow_insulation_swe_scale must be positive")
        if self.skin_heat_capacity <= 0.0:
            raise ValueError("skin_heat_capacity must be positive")
        if self.soil_heat_capacity <= 0.0:
            raise ValueError("soil_heat_capacity must be positive")
        if self.soil_thermal_conductivity < 0.0:
            raise ValueError("soil_thermal_conductivity must be non-negative")
        if self.root_fraction is not None:
            if len(self.root_fraction) != len(self.soil_layer_thickness):
                raise ValueError("root_fraction must match soil_layer_thickness length")
            if any(value < 0.0 for value in self.root_fraction):
                raise ValueError("root_fraction values must be non-negative")
            if sum(self.root_fraction) <= 0.0:
                raise ValueError("root_fraction must have positive total")

    @property
    def n_soil_layers(self) -> int:
        return len(self.soil_layer_thickness)

    def soil_layer_thickness_tensor(self, ref: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.soil_layer_thickness, dtype=ref.dtype, device=ref.device)

    def root_fraction_tensor(self, ref: torch.Tensor) -> torch.Tensor:
        if self.root_fraction is None:
            thickness = self.soil_layer_thickness_tensor(ref)
            return thickness / thickness.sum().clamp_min(torch.finfo(ref.dtype).tiny)
        root = torch.as_tensor(self.root_fraction, dtype=ref.dtype, device=ref.device)
        return root / root.sum().clamp_min(torch.finfo(ref.dtype).tiny)


def _as_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of numbers")
    return tuple(float(value) for value in values)
