"""External boundary field contracts, regridding, and NetCDF I/O."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

import vfs.data.topography as topography
from vfs.core.config import DataFieldRequirement, DataRequirement
from vfs.geometry.regrid import LatLonGrid


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


def write_boundary_netcdf(
    path: str | Path,
    grid: LatLonGrid,
    boundary_data: Mapping[str, Any],
    native_data: Mapping[str, Any] | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Write source-grid and optional mesh-native boundary fields."""

    try:
        import xarray as xr
    except Exception as exc:  # pragma: no cover - optional backend dependent.
        raise RuntimeError("Boundary NetCDF writing requires xarray and a NetCDF backend") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coords: dict[str, Any] = {
        "lat": grid.lat_deg.detach().cpu().numpy(),
        "lon": grid.lon_deg.detach().cpu().numpy(),
    }
    data_vars: dict[str, Any] = {}
    for name, value in boundary_data.items():
        payload = _as_tensor(_payload(value), dtype=torch.float64)
        if payload.ndim >= 2 and tuple(payload.shape[:2]) == (grid.nlat, grid.nlon):
            dims = ("lat", "lon", *[f"dim_{i}" for i in range(payload.ndim - 2)])
            data_vars[str(name)] = (dims, payload.detach().cpu().numpy())
    if native_data:
        first_native = None
        for value in native_data.values():
            payload = _as_tensor(_payload(value), dtype=torch.float64)
            if payload.ndim >= 1:
                first_native = int(payload.shape[0])
                break
        if first_native is not None:
            coords["cell"] = list(range(first_native))
        for name, value in native_data.items():
            payload = _as_tensor(_payload(value), dtype=torch.float64)
            if payload.ndim >= 1:
                dims = ("cell", *[f"native_dim_{i}" for i in range(payload.ndim - 1)])
                data_vars[f"native_{name}"] = (dims, payload.detach().cpu().numpy())
    attrs = {"created_by": "vfs.boundary.write_boundary_netcdf"}
    attrs.update({str(k): str(v) for k, v in dict(metadata or {}).items()})
    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    try:
        ds.to_netcdf(path, format="NETCDF4", engine="netcdf4")
    except ModuleNotFoundError as exc:
        raise ImportError("Writing boundary NetCDF requires netCDF4") from exc
    return str(path)


def load_native_boundary_field(
    path: str | Path,
    field_name: str,
    *,
    n_cells: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load a mesh-native field written by ``write_boundary_netcdf``."""

    try:
        import xarray as xr
    except Exception as exc:  # pragma: no cover - optional backend dependent.
        raise RuntimeError("Loading boundary NetCDF requires xarray") from exc

    boundary_path = Path(path)
    source_name = str(field_name)
    if not source_name.startswith("native_"):
        source_name = f"native_{source_name}"
    ds = xr.open_dataset(boundary_path)
    try:
        if source_name not in ds.variables:
            raise KeyError(f"{boundary_path} does not contain {source_name}")
        value = torch.as_tensor(ds[source_name].values, dtype=dtype)
        attrs = dict(ds.attrs)
    finally:
        ds.close()
    if value.ndim == 2 and int(value.shape[-1]) == 1:
        value = value[:, 0]
    if n_cells is not None and (value.ndim < 1 or int(value.shape[0]) != int(n_cells)):
        raise ValueError(
            f"{boundary_path} {source_name} must have leading n_cells={int(n_cells)}, got {tuple(value.shape)}"
        )
    return value, {"boundary_file": str(boundary_path), "source_name": source_name, **attrs}


def _resolve_key(data: Mapping[str, Any], aliases: tuple[str, ...]) -> str | None:
    for name in aliases:
        if name in data:
            return str(name)
    return None


def _payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in ("values", "data", "value"):
            if key in value:
                return value[key]
    for attr in ("values", "data"):
        payload = getattr(value, attr, None)
        if payload is not None and not callable(payload):
            return payload
    return value


def _attrs(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        attrs = value.get("attrs", {})
        return attrs if isinstance(attrs, Mapping) else {}
    attrs = getattr(value, "attrs", {})
    return attrs if isinstance(attrs, Mapping) else {}


def _as_tensor(value: Any, *, dtype: torch.dtype, device: torch.device | str | None = None) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(dtype=dtype, device=device if device is not None else value.device)
    return torch.as_tensor(value, dtype=dtype, device=device)


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
