"""External boundary field contracts, regridding, and NetCDF I/O."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from . import topography
from vfs.core.config import DataFieldRequirement, DataRequirement
from vfs.geometry.remap import LatLonGrid


SURFACE_GEOPOTENTIAL_ALIASES = (
    "geopotential_at_surface",
    "surface_geopotential",
    "orography_geopotential",
)
SURFACE_TEMPERATURE_ALIASES = (
    "surface_temperature",
    "2m_temperature",
    "skin_temperature",
    "sea_surface_temperature",
)


def synthetic_earth_boundary(
    grid: LatLonGrid,
    *,
    topography_height_m: float = 2500.0,
    surface_temperature_equator: float = 302.0,
    surface_temperature_pole: float = 245.0,
    lapse_rate: float = 6.5e-3,
) -> dict[str, dict[str, Any]]:
    """Return smooth synthetic topography and surface temperature on a lat/lon grid."""

    lat_deg, lon_deg = grid.meshgrid()
    lat_rad = torch.deg2rad(lat_deg)
    lon_rad = torch.deg2rad(lon_deg)

    def bump(lat0: float, lon0: float, lat_width: float, lon_width: float, amp: float) -> torch.Tensor:
        dlat = (lat_deg - float(lat0)) / float(lat_width)
        dlon = torch.atan2(
            torch.sin(lon_rad - math.radians(float(lon0))),
            torch.cos(lon_rad - math.radians(float(lon0))),
        ) / math.radians(float(lon_width))
        return float(amp) * torch.exp(-(dlat.square() + dlon.square()))

    continents = (
        bump(42.0, -105.0, 24.0, 34.0, 0.9)
        + bump(-18.0, -60.0, 28.0, 25.0, 0.55)
        + bump(15.0, 25.0, 34.0, 45.0, 0.65)
        + bump(48.0, 95.0, 22.0, 55.0, 1.0)
        + bump(-25.0, 135.0, 18.0, 24.0, 0.35)
    )
    mountains = (
        bump(31.0, 82.0, 10.0, 24.0, 1.25)
        + bump(-17.0, -70.0, 18.0, 8.0, 1.0)
        + bump(38.0, -112.0, 12.0, 12.0, 0.55)
    )
    raw_height = (0.55 * continents + 0.65 * mountains).clamp_min(0.0)
    height = float(topography_height_m) * raw_height / raw_height.max().clamp_min(1.0e-12)

    lat_gradient = float(surface_temperature_equator) - (
        float(surface_temperature_equator) - float(surface_temperature_pole)
    ) * torch.sin(lat_rad).square()
    surface_temperature = (lat_gradient - float(lapse_rate) * height).clamp(180.0, 330.0)
    surface_geopotential = float(topography.GRAVITY) * height

    return {
        "topography_height": {
            "values": height,
            "attrs": {"units": "m", "quantity": "topography_height"},
        },
        "surface_geopotential": {
            "values": surface_geopotential,
            "attrs": {"units": "m2 s-2", "quantity": "surface_geopotential"},
        },
        "surface_temperature": {
            "values": surface_temperature,
            "attrs": {"units": "K", "quantity": "surface_temperature"},
        },
    }


def earth_boundary_requirement(*, include_surface_temperature: bool = True) -> DataRequirement:
    """Data contract for topographic dry-boundary fields on a lat/lon grid."""

    return DataRequirement(
        fields=(
            DataFieldRequirement(
                "surface_geopotential",
                aliases=SURFACE_GEOPOTENTIAL_ALIASES,
                trailing_shape=("nlat", "nlon"),
                units="m2 s-2",
                quantity="surface_geopotential",
            ),
            DataFieldRequirement(
                "surface_temperature",
                aliases=SURFACE_TEMPERATURE_ALIASES,
                trailing_shape=("nlat", "nlon"),
                units="K",
                quantity="surface_temperature",
                optional=not bool(include_surface_temperature),
            ),
        )
    )


def validate_boundary_data(
    data: Mapping[str, Any],
    *,
    nlat: int,
    nlon: int,
    include_surface_temperature: bool = True,
    owner: str = "earth boundary",
) -> Mapping[str, Any]:
    """Validate external earth-boundary data using the project data contract."""

    earth_boundary_requirement(include_surface_temperature=include_surface_temperature).validate(
        data,
        owner=owner,
        context={"nlat": int(nlat), "nlon": int(nlon)},
    )
    return data


def regrid_boundary_data(data: Mapping[str, Any], grid_to_mesh) -> dict[str, dict[str, Any]]:
    """Map canonical boundary fields from a lat/lon grid to mesh cells."""

    out: dict[str, dict[str, Any]] = {}
    for canonical, aliases in (
        ("topography_height", ("topography_height",)),
        ("surface_geopotential", SURFACE_GEOPOTENTIAL_ALIASES),
        ("surface_temperature", SURFACE_TEMPERATURE_ALIASES),
    ):
        key = _resolve_key(data, aliases)
        if key is None:
            continue
        value = data[key]
        payload = _as_tensor(_payload(value), dtype=torch.float64)
        native = grid_to_mesh(payload)
        out[canonical] = {"values": native, "attrs": dict(_attrs(value))}
    if "surface_geopotential" in out and "topography_height" not in out:
        phi = out["surface_geopotential"]["values"]
        out["topography_height"] = {
            "values": phi / float(topography.GRAVITY),
            "attrs": {"units": "m", "quantity": "topography_height"},
        }
    return out


__all__ = [
    "SURFACE_GEOPOTENTIAL_ALIASES",
    "SURFACE_TEMPERATURE_ALIASES",
    "earth_boundary_requirement",
    "load_native_boundary_field",
    "regrid_boundary_data",
    "synthetic_earth_boundary",
    "validate_boundary_data",
    "write_boundary_netcdf",
]

from vfs.io import (write_boundary_netcdf, load_native_boundary_field, _resolve_key, _payload, _attrs, _as_tensor)
