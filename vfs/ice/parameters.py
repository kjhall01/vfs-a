"""Parameter containers for sea-ice physics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeaIceParameters:
    ice_density: float = 917.0
    latent_heat_fusion: float = 3.34e5
    ice_formation_thickness: float = 0.5
    melt_temperature: float = 273.15
    bare_ice_albedo: float = 0.55
    snow_albedo: float = 0.80
    ice_roughness: float = 1.0e-3
    snow_roughness: float = 2.0e-3
    snow_cover_swe: float = 10.0
    snow_insulation_swe_scale: float = 20.0

    def validate(self) -> None:
        if self.ice_density <= 0.0:
            raise ValueError("ice_density must be positive")
        if self.latent_heat_fusion <= 0.0:
            raise ValueError("latent_heat_fusion must be positive")
        if self.ice_formation_thickness <= 0.0:
            raise ValueError("ice_formation_thickness must be positive")
        if self.melt_temperature <= 0.0:
            raise ValueError("melt_temperature must be positive")
        if not 0.0 <= self.bare_ice_albedo <= 1.0:
            raise ValueError("bare_ice_albedo must be in [0, 1]")
        if not 0.0 <= self.snow_albedo <= 1.0:
            raise ValueError("snow_albedo must be in [0, 1]")
        if self.ice_roughness < 0.0 or self.snow_roughness < 0.0:
            raise ValueError("ice and snow roughness values must be non-negative")
        if self.snow_cover_swe <= 0.0:
            raise ValueError("snow_cover_swe must be positive")
        if self.snow_insulation_swe_scale <= 0.0:
            raise ValueError("snow_insulation_swe_scale must be positive")
