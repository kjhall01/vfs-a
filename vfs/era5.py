"""WeatherBench2/ERA5 data loading and atmospheric State initialization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from vfs.boundary import (
    SURFACE_GEOPOTENTIAL_ALIASES as _SURFACE_GEOPOTENTIAL_ALIASES,
    SURFACE_TEMPERATURE_ALIASES as _SURFACE_TEMPERATURE_ALIASES,
)
from vfs.atmosphere.column import CharneyPhillipsColumn
from vfs.moisture import (
    VAPOR_QUANTITY_ATTR,
    VAPOR_QUANTITY_DRY_MIXING_RATIO,
    VAPOR_TRACER_INDEX_ATTR,
)
from vfs.geometry.regrid import LatLonGrid
from vfs.atmosphere.state import State, StateSpec, TracerSpec
from vfs.data.topography import GRAVITY, earth_surface_pressure_from_phi_s


WEATHERBENCH2_ERA5_WB13_6H_URI = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)
WEATHERBENCH2_ERA5_64X32_6H_URI = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
)
WEATHERBENCH2_ERA5_DEFAULT_URI = WEATHERBENCH2_ERA5_WB13_6H_URI

SURFACE_PRESSURE_ALIASES = ("surface_pressure", "ps")
TEMPERATURE_ALIASES = ("temperature", "t")
U_WIND_ALIASES = ("u_component_of_wind", "u")
V_WIND_ALIASES = ("v_component_of_wind", "v")
U10_WIND_ALIASES = ("10m_u_component_of_wind", "u10", "10m_u_wind")
V10_WIND_ALIASES = ("10m_v_component_of_wind", "v10", "10m_v_wind")
SPECIFIC_HUMIDITY_ALIASES = ("specific_humidity", "q")


def open_weatherbench2_era5(
    uri: str = WEATHERBENCH2_ERA5_DEFAULT_URI,
    *,
    chunks: dict[str, Any] | str | None = "auto",
    consolidated: bool = True,
    storage_options: Mapping[str, Any] | None = None,
):
    """Open a WeatherBench2 ERA5 Zarr store as an xarray Dataset."""

    try:
        import xarray as xr
    except Exception as exc:  # pragma: no cover - optional backend dependent.
        raise ImportError("WeatherBench2 loading requires xarray") from exc

    uri = str(uri)
    if uri.startswith("gs://"):
        try:
            import fsspec
        except Exception as exc:  # pragma: no cover - optional backend dependent.
            raise ImportError("WeatherBench2 gs:// loading requires fsspec, gcsfs, and zarr") from exc
        options = {"token": "anon"}
        options.update(dict(storage_options or {}))
        try:
            mapper = fsspec.get_mapper(uri, **options)
        except Exception as exc:  # pragma: no cover - optional backend dependent.
            raise ImportError("WeatherBench2 gs:// loading requires gcsfs and zarr") from exc
        return xr.open_zarr(mapper, chunks=chunks, consolidated=bool(consolidated))
    return xr.open_zarr(uri, chunks=chunks, consolidated=bool(consolidated))


def latlon_grid_from_xarray(ds_or_da) -> LatLonGrid:
    """Build a regular LatLonGrid from xarray latitude/longitude coordinates."""

    lat_name, lon_name = _lat_lon_names(ds_or_da)
    lat = torch.as_tensor(ds_or_da[lat_name].values, dtype=torch.float64)
    lon = torch.remainder(torch.as_tensor(ds_or_da[lon_name].values, dtype=torch.float64), 360.0)
    lat = torch.sort(lat).values
    lon = torch.sort(lon).values
    return LatLonGrid(lat, lon)


def weatherbench2_boundary_data(
    ds,
    *,
    time: Any | None = None,
    surface_temperature_var: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Extract canonical topographic boundary fields from a WeatherBench2 Dataset."""

    phi_name = _resolve_dataset_var(ds, _SURFACE_GEOPOTENTIAL_ALIASES)
    if phi_name is None:
        raise KeyError(f"dataset is missing a surface geopotential variable; tried {_SURFACE_GEOPOTENTIAL_ALIASES}")
    out = {
        "surface_geopotential": {
            "values": _field_to_latlon_tensor(_select_time(ds[phi_name], time)),
            "attrs": {"units": "m2 s-2", "quantity": "surface_geopotential", "source_name": str(phi_name)},
        }
    }
    temp_name = surface_temperature_var or _resolve_dataset_var(ds, _SURFACE_TEMPERATURE_ALIASES)
    if temp_name is not None:
        out["surface_temperature"] = {
            "values": _field_to_latlon_tensor(_select_time(ds[temp_name], time)),
            "attrs": {"units": "K", "quantity": "surface_temperature", "source_name": str(temp_name)},
        }
    out["topography_height"] = {
        "values": out["surface_geopotential"]["values"] / float(GRAVITY),
        "attrs": {"units": "m", "quantity": "topography_height", "source_name": str(phi_name)},
    }
    return out


def weatherbench2_dry_state_from_dataset(
    ds,
    *,
    mesh,
    column: CharneyPhillipsColumn,
    grid_to_mesh,
    time: Any | None = None,
    dyn=None,
    carry_vapor: bool = False,
    ps0: float = 1.0e5,
    surface_temperature_var: str | None = None,
    u10_var: str | None = None,
    v10_var: str | None = None,
) -> State:
    """Build a dry VFS State from one WeatherBench2/ERA5 pressure-level snapshot."""

    ps_name = _resolve_dataset_var(ds, SURFACE_PRESSURE_ALIASES)
    temp_name = _require_dataset_var(ds, TEMPERATURE_ALIASES)
    u_name = _require_dataset_var(ds, U_WIND_ALIASES)
    v_name = _require_dataset_var(ds, V_WIND_ALIASES)
    q_name = _resolve_dataset_var(ds, SPECIFIC_HUMIDITY_ALIASES)
    surface_temp_name = surface_temperature_var or _resolve_dataset_var(ds, _SURFACE_TEMPERATURE_ALIASES)
    u10_name = u10_var or _resolve_dataset_var(ds, U10_WIND_ALIASES)
    v10_name = v10_var or _resolve_dataset_var(ds, V10_WIND_ALIASES)

    if ps_name is None:
        phi_s = column.phi_s.detach()
        if phi_s.ndim == 0:
            phi_s = phi_s.expand(int(mesh.nCells))
        ps = earth_surface_pressure_from_phi_s(phi_s, p0=float(ps0)).reshape(int(mesh.nCells), 1)
    else:
        ps_grid = _field_to_latlon_tensor(_select_time(ds[ps_name], time))
        ps = grid_to_mesh(ps_grid).to(dtype=torch.float64)
        if ps.ndim == 1:
            ps = ps[:, None]
    ps = ps.clamp_min(float(column.A_interface[0].detach().cpu().item()) + 1000.0)

    source_pressure = _source_pressure_pa(ds[temp_name])
    T_native = grid_to_mesh(_field_to_latlon_tensor(_select_time(ds[temp_name], time))).to(dtype=torch.float64)
    u_native = grid_to_mesh(_field_to_latlon_tensor(_select_time(ds[u_name], time))).to(dtype=torch.float64)
    v_native = grid_to_mesh(_field_to_latlon_tensor(_select_time(ds[v_name], time))).to(dtype=torch.float64)
    surface_temperature = _optional_native_surface_field(
        ds,
        surface_temp_name,
        grid_to_mesh=grid_to_mesh,
        time=time,
        dtype=torch.float64,
        n_cells=int(mesh.nCells),
    )
    u10_native = _optional_native_surface_field(
        ds,
        u10_name,
        grid_to_mesh=grid_to_mesh,
        time=time,
        dtype=torch.float64,
        n_cells=int(mesh.nCells),
    )
    v10_native = _optional_native_surface_field(
        ds,
        v10_name,
        grid_to_mesh=grid_to_mesh,
        time=time,
        dtype=torch.float64,
        n_cells=int(mesh.nCells),
    )

    T_i, temperature_anchor_count = _interp_log_pressure_with_surface_anchor(
        T_native,
        source_pressure,
        column.interface_pressure(ps),
        surface_pressure_pa=ps,
        surface_value=surface_temperature,
    )
    u_full, u_anchor_count = _interp_log_pressure_with_surface_anchor(
        u_native,
        source_pressure,
        column.full_pressure(ps),
        surface_pressure_pa=ps,
        surface_value=u10_native,
    )
    v_full, v_anchor_count = _interp_log_pressure_with_surface_anchor(
        v_native,
        source_pressure,
        column.full_pressure(ps),
        surface_pressure_pa=ps,
        surface_value=v10_native,
    )
    u_e = _edge_normal_wind_from_cell_uv(mesh, u_full, v_full)
    theta_i = T_i / (column.exner_interface(ps) / column.c_p).clamp_min(torch.finfo(T_i.dtype).tiny)

    q_tracers = None
    spec = None
    attrs = {
        "gravity": float(getattr(dyn, "gravity", GRAVITY)),
        "initial_condition": "weatherbench2_era5_dry_snapshot",
        "weatherbench2_time": "" if time is None else str(time),
        "weatherbench2_temperature_surface_anchor_source": "" if surface_temperature is None else str(surface_temp_name),
        "weatherbench2_temperature_surface_anchor_count": int(temperature_anchor_count),
        "weatherbench2_u_wind_surface_anchor_source": "" if u10_native is None else str(u10_name),
        "weatherbench2_u_wind_surface_anchor_count": int(u_anchor_count),
        "weatherbench2_v_wind_surface_anchor_source": "" if v10_native is None else str(v10_name),
        "weatherbench2_v_wind_surface_anchor_count": int(v_anchor_count),
    }
    if bool(carry_vapor):
        if q_name is None:
            raise KeyError(f"carry_vapor=True requires specific humidity; tried {SPECIFIC_HUMIDITY_ALIASES}")
        q_native = grid_to_mesh(_field_to_latlon_tensor(_select_time(ds[q_name], time))).to(dtype=torch.float64)
        q_full = _interp_log_pressure(q_native, source_pressure, column.full_pressure(ps)).clamp(0.0, 0.99)
        r_v = q_full / (1.0 - q_full).clamp_min(1.0e-12)
        q_tracers = r_v[..., None]
        attrs[VAPOR_TRACER_INDEX_ATTR] = 0
        attrs[VAPOR_QUANTITY_ATTR] = VAPOR_QUANTITY_DRY_MIXING_RATIO
        attrs["vapor_units"] = "kg kg-1 dry_air"
        spec = StateSpec(
            tracers=(
                TracerSpec(
                    name="vapor",
                    index=0,
                    role="passive",
                    quantity=VAPOR_QUANTITY_DRY_MIXING_RATIO,
                    units="kg kg-1 dry_air",
                ),
            )
        )

    return State.from_theta(
        u_e,
        theta_i,
        ps,
        mesh=mesh,
        column=column,
        q_tracers=q_tracers,
        attrs=attrs,
        spec=spec,
    )


def _resolve_dataset_var(ds, aliases: tuple[str, ...]) -> str | None:
    for name in aliases:
        if name in ds.variables:
            return str(name)
    return None


def _require_dataset_var(ds, aliases: tuple[str, ...]) -> str:
    name = _resolve_dataset_var(ds, aliases)
    if name is None:
        raise KeyError(f"dataset is missing required variable; tried {aliases}")
    return name


def _lat_lon_names(ds_or_da) -> tuple[str, str]:
    lat_name = next((name for name in ("latitude", "lat") if name in ds_or_da.coords), None)
    lon_name = next((name for name in ("longitude", "lon") if name in ds_or_da.coords), None)
    if lat_name is None or lon_name is None:
        raise KeyError("expected latitude/longitude or lat/lon coordinates")
    return str(lat_name), str(lon_name)


def _select_time(da, time: Any | None):
    if "time" not in da.dims:
        return da
    if time is None:
        return da.isel(time=0)
    return da.sel(time=time, method="nearest")


def _field_to_latlon_tensor(da) -> torch.Tensor:
    lat_name, lon_name = _lat_lon_names(da)
    if bool((da[lat_name][0] > da[lat_name][-1]).item()):
        da = da.sortby(lat_name)
    lon_values = (da[lon_name] % 360.0).values
    da = da.assign_coords({lon_name: lon_values}).sortby(lon_name)
    trailing = [dim for dim in da.dims if dim not in (lat_name, lon_name)]
    da = da.transpose(lat_name, lon_name, *trailing)
    return torch.as_tensor(da.values, dtype=torch.float64)


def _source_pressure_pa(da) -> torch.Tensor:
    for name in ("level", "pressure", "isobaricInhPa"):
        if name in da.coords:
            values = torch.as_tensor(da[name].values, dtype=torch.float64)
            units = str(getattr(da[name], "attrs", {}).get("units", "")).lower()
            if units in ("pa", "pascal", "pascals") or values.max() > 2000.0:
                return values
            return 100.0 * values
    raise KeyError("pressure-level variables require a level/pressure coordinate")


def _optional_native_surface_field(
    ds,
    name: str | None,
    *,
    grid_to_mesh,
    time: Any | None,
    dtype: torch.dtype,
    n_cells: int,
) -> torch.Tensor | None:
    if name is None:
        return None
    if name not in ds.variables:
        raise KeyError(f"dataset is missing requested surface field {name!r}")
    value = grid_to_mesh(_field_to_latlon_tensor(_select_time(ds[name], time))).to(dtype=dtype)
    if value.ndim == 1:
        return value.reshape(int(n_cells))
    value = value.reshape(int(n_cells), -1)
    if int(value.shape[1]) != 1:
        raise ValueError(f"surface field {name!r} must map to one value per mesh cell; got {tuple(value.shape)}")
    return value[:, 0]


def _interp_log_pressure(profile: torch.Tensor, source_p_pa: torch.Tensor, target_p_pa: torch.Tensor) -> torch.Tensor:
    if profile.ndim != 2:
        raise ValueError(f"pressure profile must have shape [nCells, nSourceLevels], got {tuple(profile.shape)}")
    src = source_p_pa.to(dtype=profile.dtype, device=profile.device).reshape(-1)
    order = torch.argsort(src)
    src_log = torch.log(src[order].clamp_min(1.0))
    values = profile[:, order]
    target_log = torch.log(target_p_pa.to(dtype=profile.dtype, device=profile.device).clamp_min(1.0))
    target_log = target_log.clamp(src_log[0], src_log[-1])
    hi = torch.searchsorted(src_log.contiguous(), target_log.contiguous(), right=False)
    hi = hi.clamp(1, int(src_log.numel()) - 1)
    lo = hi - 1
    x0 = src_log[lo]
    x1 = src_log[hi]
    v0 = torch.gather(values, 1, lo)
    v1 = torch.gather(values, 1, hi)
    w = ((target_log - x0) / (x1 - x0).clamp_min(torch.finfo(profile.dtype).tiny)).clamp(0.0, 1.0)
    return v0 + w * (v1 - v0)


def _interp_log_pressure_with_surface_anchor(
    profile: torch.Tensor,
    source_p_pa: torch.Tensor,
    target_p_pa: torch.Tensor,
    *,
    surface_pressure_pa: torch.Tensor,
    surface_value: torch.Tensor | None,
) -> tuple[torch.Tensor, int]:
    """Log-pressure interpolation with an optional local near-surface anchor."""

    base = _interp_log_pressure(profile, source_p_pa, target_p_pa)
    if surface_value is None:
        return base, 0
    if profile.ndim != 2:
        raise ValueError(f"pressure profile must have shape [nCells, nSourceLevels], got {tuple(profile.shape)}")

    src = source_p_pa.to(dtype=profile.dtype, device=profile.device).reshape(-1)
    source_max_index = int(torch.argmax(src).detach().cpu().item())
    source_max_pressure = src[source_max_index].clamp_min(1.0)
    source_max_value = profile[:, source_max_index]

    target = target_p_pa.to(dtype=profile.dtype, device=profile.device).clamp_min(1.0)
    ps = surface_pressure_pa.to(dtype=profile.dtype, device=profile.device).reshape(profile.shape[0], -1)[:, :1].clamp_min(1.0)
    surf = surface_value.to(dtype=profile.dtype, device=profile.device).reshape(profile.shape[0], 1)

    finite_anchor = torch.isfinite(ps) & torch.isfinite(surf)
    has_lower_anchor = finite_anchor & (ps > source_max_pressure)
    use_anchor = has_lower_anchor & (target > source_max_pressure)
    if not bool(use_anchor.detach().cpu().any().item()):
        return base, 0

    target_for_anchor = torch.minimum(target, ps)
    x0 = torch.log(source_max_pressure)
    x1 = torch.log(ps)
    xt = torch.log(target_for_anchor)
    w = ((xt - x0) / (x1 - x0).clamp_min(torch.finfo(profile.dtype).tiny)).clamp(0.0, 1.0)
    anchored = source_max_value[:, None] + w * (surf - source_max_value[:, None])
    count = int(use_anchor.detach().cpu().sum().item())
    return torch.where(use_anchor, anchored, base), count


def _edge_normal_wind_from_cell_uv(mesh, u_east: torch.Tensor, v_north: torch.Tensor) -> torch.Tensor:
    xyz = mesh.cell_xyz.to(dtype=u_east.dtype, device=u_east.device)
    lat = torch.asin(xyz[:, 2].clamp(-1.0, 1.0))
    lon = torch.atan2(xyz[:, 1], xyz[:, 0])
    east = torch.stack([-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)], dim=-1)
    north = torch.stack(
        [-torch.sin(lat) * torch.cos(lon), -torch.sin(lat) * torch.sin(lon), torch.cos(lat)],
        dim=-1,
    )
    cell_vec = u_east[..., None] * east[:, None, :] + v_north[..., None] * north[:, None, :]
    cells = mesh.cellsOnEdge.to(device=u_east.device, dtype=torch.long)
    edge_vec = 0.5 * (cell_vec[cells[:, 0]] + cell_vec[cells[:, 1]])
    normal = mesh.edge_normal.to(dtype=u_east.dtype, device=u_east.device)
    return (edge_vec * normal[:, None, :]).sum(dim=-1)


__all__ = [
    "SPECIFIC_HUMIDITY_ALIASES",
    "SURFACE_PRESSURE_ALIASES",
    "TEMPERATURE_ALIASES",
    "U10_WIND_ALIASES",
    "U_WIND_ALIASES",
    "V10_WIND_ALIASES",
    "V_WIND_ALIASES",
    "WEATHERBENCH2_ERA5_64X32_6H_URI",
    "WEATHERBENCH2_ERA5_DEFAULT_URI",
    "WEATHERBENCH2_ERA5_WB13_6H_URI",
    "latlon_grid_from_xarray",
    "open_weatherbench2_era5",
    "weatherbench2_boundary_data",
    "weatherbench2_dry_state_from_dataset",
]
