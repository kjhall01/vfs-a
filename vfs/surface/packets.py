"""Explicit surface boundary and flux packets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vfs.field import Field

from .utils import as_cell_column, field_values, require_finite


def _normalize_packet_field(value: Any, *, like: torch.Tensor, name: str) -> torch.Tensor:
    tensor = as_cell_column(value, like=like, name=name)
    require_finite(name, tensor)
    return tensor


def _packet_field(values: Field | torch.Tensor, *, name: str) -> Field:
    return Field(field_values(values), location="cell", vertical_location="surface", name=name)


@dataclass(frozen=True)
class SurfaceBoundaryPacket:
    """Effective surface properties consumed by SCM/radiation code."""

    albedo: Field | torch.Tensor
    skin_temperature: Field | torch.Tensor
    emissivity: Field | torch.Tensor
    roughness: Field | torch.Tensor
    moisture_availability: Field | torch.Tensor

    def __post_init__(self) -> None:
        like = as_cell_column(self.skin_temperature, n_cells=int(field_values(self.skin_temperature).numel()))
        object.__setattr__(self, "skin_temperature", _packet_field(like, name="skin_temperature"))
        for name in ("albedo", "emissivity", "roughness", "moisture_availability"):
            object.__setattr__(self, name, _packet_field(_normalize_packet_field(getattr(self, name), like=like, name=name), name=name))

    def to(self, *args, **kwargs) -> "SurfaceBoundaryPacket":
        return SurfaceBoundaryPacket(
            self.albedo.to(*args, **kwargs),
            self.skin_temperature.to(*args, **kwargs),
            self.emissivity.to(*args, **kwargs),
            self.roughness.to(*args, **kwargs),
            self.moisture_availability.to(*args, **kwargs),
        )


@dataclass(frozen=True)
class AtmosSurfaceFluxPacket:
    """Atmosphere/SCM surface fluxes on cells.

    Heat fluxes are W m-2. Precipitation is kg m-2 s-1 downward. Wind stress
    components are Pa in local east/north directions.
    """

    net_shortwave: Field | torch.Tensor
    net_longwave: Field | torch.Tensor
    sensible_heat: Field | torch.Tensor
    latent_heat: Field | torch.Tensor
    precipitation_rain: Field | torch.Tensor
    precipitation_snow: Field | torch.Tensor
    wind_stress_x: Field | torch.Tensor
    wind_stress_y: Field | torch.Tensor

    @classmethod
    def zeros_like(cls, like: torch.Tensor) -> "AtmosSurfaceFluxPacket":
        z = torch.zeros_like(like)
        return cls(z, z, z, z, z, z, z, z)

    def __post_init__(self) -> None:
        like = as_cell_column(self.net_shortwave, n_cells=int(field_values(self.net_shortwave).numel()))
        object.__setattr__(self, "net_shortwave", _packet_field(like, name="net_shortwave"))
        for name in (
            "net_longwave",
            "sensible_heat",
            "latent_heat",
            "precipitation_rain",
            "precipitation_snow",
            "wind_stress_x",
            "wind_stress_y",
        ):
            object.__setattr__(self, name, _packet_field(_normalize_packet_field(getattr(self, name), like=like, name=name), name=name))


@dataclass(frozen=True)
class OceanBoundaryFluxPacket:
    """Fluxes that should be applied to ocean cells after surface routing.

    Heat is W m-2 positive into ocean. Water fluxes are kg m-2 s-1 positive
    into ocean. Salt is kg m-2 s-1 positive into ocean.
    """

    heat_flux_to_ocean: Field | torch.Tensor
    freshwater_flux_to_ocean: Field | torch.Tensor
    salt_flux_to_ocean: Field | torch.Tensor
    runoff_to_ocean: Field | torch.Tensor
    iceberg_melt_to_ocean: Field | torch.Tensor

    @classmethod
    def zeros_like(cls, like: torch.Tensor) -> "OceanBoundaryFluxPacket":
        z = torch.zeros_like(like)
        return cls(z, z, z, z, z)

    def __post_init__(self) -> None:
        like = as_cell_column(self.heat_flux_to_ocean, n_cells=int(field_values(self.heat_flux_to_ocean).numel()))
        object.__setattr__(self, "heat_flux_to_ocean", _packet_field(like, name="heat_flux_to_ocean"))
        for name in (
            "freshwater_flux_to_ocean",
            "salt_flux_to_ocean",
            "runoff_to_ocean",
            "iceberg_melt_to_ocean",
        ):
            object.__setattr__(self, name, _packet_field(_normalize_packet_field(getattr(self, name), like=like, name=name), name=name))


__all__ = ["AtmosSurfaceFluxPacket", "OceanBoundaryFluxPacket", "SurfaceBoundaryPacket"]
