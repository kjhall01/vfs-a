"""Shared coordinates, metadata, scheduling and NetCDF file mechanics.

Components supply already interpreted fields; this module knows no state class.
Xarray/netCDF are imported only at the explicit output boundary.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any, Iterable
from collections.abc import Mapping
from vfs.geometry.remap import LatLonGrid

import numpy as np
import torch


SNAPSHOT_IO_VERSION = "state_netcdf_v1"


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _clean_metadata_value(value: Any) -> str | int | float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _clean_metadata_value(value.detach().cpu().item())
        return None
    if isinstance(value, np.generic):
        return _clean_metadata_value(value.item())
    if value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


def _cell_lat_lon_deg(mesh) -> tuple[np.ndarray, np.ndarray]:
    xyz = mesh.cell_xyz.detach().cpu()
    lat = torch.asin(xyz[:, 2].clamp(-1.0, 1.0))
    lon = torch.atan2(xyz[:, 1], xyz[:, 0])
    return _as_numpy(torch.rad2deg(lat)), _as_numpy(torch.rad2deg(lon))


def _edge_lat_lon_deg(mesh) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(mesh, "edge_xyz"):
        xyz = mesh.edge_xyz.detach().cpu()
    else:
        c0 = mesh.cell_xyz[mesh.cellsOnEdge[:, 0]].detach().cpu()
        c1 = mesh.cell_xyz[mesh.cellsOnEdge[:, 1]].detach().cpu()
        xyz = c0 + c1
        xyz = xyz / xyz.norm(dim=-1, keepdim=True).clamp_min(1.0e-15)
    lat = torch.asin(xyz[:, 2].clamp(-1.0, 1.0))
    lon = torch.atan2(xyz[:, 1], xyz[:, 0])
    return _as_numpy(torch.rad2deg(lat)), _as_numpy(torch.rad2deg(lon))


def _vertex_lat_lon_deg(mesh) -> tuple[np.ndarray, np.ndarray]:
    xyz = mesh.vertex_xyz.detach().cpu()
    lat = torch.asin(xyz[:, 2].clamp(-1.0, 1.0))
    lon = torch.atan2(xyz[:, 1], xyz[:, 0])
    return _as_numpy(torch.rad2deg(lat)), _as_numpy(torch.rad2deg(lon))


def _local_east_north(lat: torch.Tensor, lon: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    east = torch.stack([-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)], dim=-1)
    north = torch.stack(
        [-torch.sin(lat) * torch.cos(lon), -torch.sin(lat) * torch.sin(lon), torch.cos(lat)],
        dim=-1,
    )
    return east, north


def _normalise_days(values: Iterable[float], days: float) -> list[float]:
    out = {0.0, float(days)}
    for value in values:
        day = min(max(0.0, float(value)), float(days))
        out.add(round(day, 10))
    return sorted(out)


def snapshot_schedule(
    days: float,
    *,
    first_phase_every: float = 30.0,
    first_phase_until: float = 300.0,
    later_every: float = 100.0,
) -> list[float]:
    """Held-Suarez-style default snapshot cadence.

    The default schedule saves day 0, every 30 days through day 300, every
    100 days after day 300, and the final requested day.
    """

    values: list[float] = []
    if first_phase_every > 0.0:
        n_first = int(math.floor(min(float(days), float(first_phase_until)) / float(first_phase_every)))
        values.extend(i * float(first_phase_every) for i in range(n_first + 1))
    start = float(first_phase_until) + float(later_every)
    if later_every > 0.0 and float(days) >= start:
        n_later = int(math.floor((float(days) - start) / float(later_every)))
        values.extend(start + i * float(later_every) for i in range(n_later + 1))
    return _normalise_days(values, float(days))


def daily_schedule(days: float) -> list[float]:
    """Daily snapshot schedule including day 0 and final day."""

    return _normalise_days(range(int(math.floor(float(days))) + 1), float(days))


def custom_schedule(
    days: float,
    *,
    snapshot_every_days: float | None = None,
    snapshot_days: Iterable[float] | None = None,
) -> list[float]:
    """Custom snapshot schedule from explicit days and/or fixed cadence."""

    values = list(snapshot_days or [])
    if snapshot_every_days is not None and float(snapshot_every_days) > 0.0:
        count = int(math.floor(float(days) / float(snapshot_every_days)))
        values.extend(i * float(snapshot_every_days) for i in range(count + 1))
    return _normalise_days(values, float(days))


def schedule_steps(days: Iterable[float], *, dt: float, n_steps: int) -> dict[int, float]:
    """Map scheduled days to nearest integration steps."""

    out: dict[int, float] = {}
    for day in days:
        step = int(round(float(day) * 86400.0 / float(dt)))
        step = min(max(0, step), int(n_steps))
        out[step] = float(day)
    return out


__all__ = [
    "SNAPSHOT_IO_VERSION",
    "custom_schedule",
    "daily_schedule",
    "schedule_steps",
    "snapshot_schedule",
]


def write_netcdf(data_vars, coords, attrs, path):
    """Write already interpreted fields; components own names, units and staggering."""
    import xarray as xr
    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    try:
        ds.to_netcdf(path, format="NETCDF4", engine="netcdf4")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Writing NetCDF snapshots requires the netCDF4 Python package. "
            "Install the project environment from environment.yml or run "
            "`pip install netCDF4`."
        ) from exc
    return str(path)


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

