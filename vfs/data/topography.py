"""Shared terrain and bathymetry helpers for VFS workflows.

This module is the common home for atmospheric topography and ocean
bathymetry/ETOPO ingest utilities. It owns terrain/bathymetry setup, while
model-specific earth/ocean modules consume these helpers.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlretrieve

import torch

from vfs.geometry.column import CharneyPhillipsColumn, make_hybrid_coefficients
from vfs.geometry.regrid import LatLonGrid, LatLonToMeshBilinear


GRAVITY = 9.80665
EARTH_GRAVITY = GRAVITY
R_D = 287.05
DRY_AIR_GAS_CONSTANT = R_D
DEFAULT_SCALE_HEIGHT = 8000.0
ETOPO_2022_60S_BEDROCK_OPENDAP_URL = (
    "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/60s/"
    "60s_bed_elev_netcdf/ETOPO_2022_v1_60s_N90W180_bed.nc"
)
ETOPO_2022_60S_BEDROCK_DOWNLOAD_URL = (
    "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/"
    "60s_bed_elev_netcdf/ETOPO_2022_v1_60s_N90W180_bed.nc"
)
ETOPO_2022_60S_BEDROCK_NETCDF_URL = ETOPO_2022_60S_BEDROCK_DOWNLOAD_URL
ETOPO_2022_METADATA_URL = "https://www.ncei.noaa.gov/products/etopo-global-relief-model"
ETOPO_ELEVATION_ALIASES = ("z", "elevation", "elev", "Band1", "topography", "relief")

OCEAN_BATHYMETRY_DEFAULT_GRAVITY = GRAVITY
OCEAN_BATHYMETRY_DEFAULT_RHO0 = 1025.0


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


def download_etopo2022(
    path_or_url: str = ETOPO_2022_60S_BEDROCK_DOWNLOAD_URL,
    *,
    cache_dir: str | Path = "data/etopo",
    filename: str | None = None,
    force: bool = False,
) -> Path:
    """Download an ETOPO 2022 NetCDF file and return the local path."""

    source = str(path_or_url)
    if not _is_remote_url(source):
        return Path(source)
    url = _etopo_direct_download_url(source)
    parsed = urlparse(url)
    target_name = filename or Path(parsed.path).name
    if not target_name:
        raise ValueError(f"could not infer ETOPO filename from URL {url!r}")
    target = Path(cache_dir).expanduser() / target_name
    if target.exists() and not bool(force):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    try:
        urlretrieve(url, partial)
        partial.replace(target)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return target


def open_etopo2022(
    path_or_url: str = ETOPO_2022_60S_BEDROCK_NETCDF_URL,
    *,
    cache_dir: str | Path | None = None,
    allow_download: bool = False,
    **kwargs,
):
    """Open an ETOPO 2022 NetCDF/OPeNDAP dataset with xarray.

    Direct NOAA fileServer URLs are downloaded to ``cache_dir`` when
    ``allow_download=True``. This avoids netCDF-C interpreting HTTPS files as
    OPeNDAP datasets and reporting authorization failures.
    """

    try:
        import xarray as xr
    except Exception as exc:  # pragma: no cover - optional backend dependent.
        raise RuntimeError("ETOPO loading requires xarray and a NetCDF/OPeNDAP backend") from exc
    source = str(path_or_url)
    if _is_remote_url(source) and bool(allow_download):
        local_path = download_etopo2022(source, cache_dir="data/etopo" if cache_dir is None else cache_dir)
        return xr.open_dataset(local_path, **kwargs)
    try:
        return xr.open_dataset(source, **kwargs)
    except Exception as exc:
        if _is_remote_url(source):
            download_url = _etopo_direct_download_url(source)
            raise RuntimeError(
                "Remote ETOPO open failed. netCDF backends often treat HTTPS "
                "paths as OPeNDAP and NOAA's DODS endpoint may return "
                "authorization failures. Download the NetCDF first and pass the "
                f"local path, or call open_etopo2022(..., allow_download=True). "
                f"Direct download URL: {download_url}"
            ) from exc
        raise


def etopo_grid_from_xarray(ds_or_da) -> LatLonGrid:
    """Build a regular ``LatLonGrid`` from an ETOPO-like xarray object."""

    lat_name, lon_name = _etopo_lat_lon_names(ds_or_da)
    lat = torch.as_tensor(ds_or_da[lat_name].values, dtype=torch.float64)
    lon = torch.remainder(torch.as_tensor(ds_or_da[lon_name].values, dtype=torch.float64), 360.0)
    return LatLonGrid(torch.sort(lat).values, torch.sort(lon).values)


def etopo_elevation_from_xarray(
    ds_or_da,
    *,
    elevation_var: str | None = None,
    coarsen: int = 1,
    coarsen_cache_dir: str | Path | None = None,
) -> tuple[LatLonGrid, torch.Tensor, dict[str, Any]]:
    """Return sorted ETOPO elevation in meters on a ``LatLonGrid``."""

    da = _etopo_elevation_dataarray(ds_or_da, elevation_var=elevation_var)
    coarsen = int(coarsen)
    if coarsen < 1:
        raise ValueError("coarsen must be >= 1")
    cache_path = None
    if coarsen_cache_dir is not None and coarsen > 1:
        cache_path = _etopo_coarsen_cache_path(ds_or_da, da, cache_dir=coarsen_cache_dir, coarsen=coarsen)
        if cache_path.exists():
            return _read_etopo_coarsen_cache(cache_path)

    lat_name, lon_name = _etopo_lat_lon_names(da)
    if bool((da[lat_name][0] > da[lat_name][-1]).item()):
        da = da.sortby(lat_name)
    lon_values = (da[lon_name] % 360.0).values
    da = da.assign_coords({lon_name: lon_values}).sortby(lon_name)
    da = da.transpose(lat_name, lon_name)
    if coarsen > 1:
        da = da.coarsen({lat_name: coarsen, lon_name: coarsen}, boundary="trim").mean()
    grid = LatLonGrid(
        torch.as_tensor(da[lat_name].values.copy(), dtype=torch.float64),
        torch.remainder(torch.as_tensor(da[lon_name].values.copy(), dtype=torch.float64), 360.0),
    )
    elevation = torch.as_tensor(da.values, dtype=torch.float64)
    attrs = {
        "source_name": str(getattr(da, "name", "") or elevation_var or ""),
        "units": str(getattr(da, "attrs", {}).get("units", "m")),
        "coarsen": int(coarsen),
        "nlat": int(grid.nlat),
        "nlon": int(grid.nlon),
    }
    if cache_path is not None:
        _write_etopo_coarsen_cache(cache_path, grid, elevation, attrs=attrs)
        attrs["coarsen_cache_path"] = str(cache_path)
        attrs["coarsen_cache_hit"] = False
    return grid, elevation, attrs


def etopo_native_elevation_from_xarray(
    ds_or_da,
    mesh,
    *,
    grid_to_mesh: LatLonToMeshBilinear | None = None,
    elevation_var: str | None = None,
    coarsen: int = 1,
    coarsen_cache_dir: str | Path | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Map ETOPO-like elevation data to native mesh cells."""

    grid, elevation_grid, attrs = etopo_elevation_from_xarray(
        ds_or_da,
        elevation_var=elevation_var,
        coarsen=int(coarsen),
        coarsen_cache_dir=coarsen_cache_dir,
    )
    if grid_to_mesh is None:
        grid_to_mesh = LatLonToMeshBilinear.from_mesh(grid, mesh)
    elevation_native = grid_to_mesh(elevation_grid).to(dtype=torch.float64).reshape(int(mesh.nCells), -1)
    if int(elevation_native.shape[1]) != 1:
        raise ValueError(f"ETOPO elevation must map to one value per mesh cell, got {tuple(elevation_native.shape)}")
    return elevation_native[:, 0], attrs


def build_ocean_domain_from_native_elevation(
    mesh,
    elevation_native: torch.Tensor | Any,
    *,
    n_layers: int,
    ocean_threshold_m: float = 0.0,
    min_depth: float = 50.0,
    max_depth: float | None = 6000.0,
    land_depth: float | None = None,
    depth_smoothing_passes: int = 0,
    depth_smoothing_alpha: float = 0.5,
    preserve_wet_mean_depth: bool = True,
    min_wet_component_cells: int = 0,
    keep_largest_wet_component: bool = False,
    rest_thickness: torch.Tensor | None = None,
    gravity: float = OCEAN_BATHYMETRY_DEFAULT_GRAVITY,
    rho0: float = OCEAN_BATHYMETRY_DEFAULT_RHO0,
    metadata: dict[str, Any] | None = None,
):
    """Build ``OceanDomain`` and ``OceanColumn`` from native elevation."""

    from vfs.ocean import OceanColumn, OceanDomain

    elevation_native = torch.as_tensor(elevation_native, dtype=torch.float64).reshape(int(mesh.nCells))
    finite = torch.isfinite(elevation_native)
    wet_cell = finite & (elevation_native < float(ocean_threshold_m))
    if not bool(wet_cell.any().item()):
        raise ValueError("native elevation produced no wet ocean cells")
    component_metadata = filter_wet_components_on_mesh(
        mesh,
        wet_cell,
        min_component_cells=int(min_wet_component_cells),
        keep_largest=bool(keep_largest_wet_component),
    )
    wet_cell = component_metadata["wet_cell"]
    if not bool(wet_cell.any().item()):
        raise ValueError("ETOPO wet-component filtering removed all wet ocean cells")
    elevation_native = torch.where(finite, elevation_native, torch.zeros_like(elevation_native))
    depth = (float(ocean_threshold_m) - elevation_native).clamp_min(float(min_depth))
    if max_depth is not None:
        depth = depth.clamp_max(float(max_depth))
    depth = smooth_ocean_depth_on_mesh(
        mesh,
        depth,
        wet_cell,
        passes=int(depth_smoothing_passes),
        alpha=float(depth_smoothing_alpha),
        min_depth=float(min_depth),
        max_depth=max_depth,
        preserve_wet_mean=bool(preserve_wet_mean_depth),
    )
    dummy_depth = float(min_depth if land_depth is None else land_depth)
    if dummy_depth <= 0.0:
        raise ValueError("land_depth must be positive")
    total_depth = torch.where(wet_cell, depth, torch.full_like(depth, dummy_depth)).reshape(int(mesh.nCells), 1)
    if rest_thickness is None:
        rest_thickness = torch.full(
            (int(n_layers),),
            float(depth[wet_cell].mean().item()) / float(n_layers),
            dtype=torch.float64,
        )
    column = OceanColumn(
        int(n_layers),
        total_depth=total_depth.detach().cpu(),
        rest_thickness=rest_thickness.detach().cpu(),
        gravity=float(gravity),
        rho0=float(rho0),
    )
    domain = OceanDomain(mesh, wet_cell.detach().cpu(), depth=total_depth.detach().cpu())
    wet_depth = depth[wet_cell]
    out_metadata = dict(metadata or {})
    out_metadata.update(
        {
        "ocean_threshold_m": float(ocean_threshold_m),
        "min_depth": float(min_depth),
        "max_depth": None if max_depth is None else float(max_depth),
        "land_depth": float(dummy_depth),
        "depth_smoothing_passes": int(depth_smoothing_passes),
        "depth_smoothing_alpha": float(depth_smoothing_alpha),
        "preserve_wet_mean_depth": bool(preserve_wet_mean_depth),
        "min_wet_component_cells": int(min_wet_component_cells),
        "keep_largest_wet_component": bool(keep_largest_wet_component),
        "wet_component_count_raw": int(component_metadata["component_count_raw"]),
        "wet_component_count_kept": int(component_metadata["component_count_kept"]),
        "wet_cells_removed_by_component_filter": int(component_metadata["wet_cells_removed"]),
        "largest_wet_component_cells": int(component_metadata["largest_component_cells"]),
        "wet_cell_count": int(wet_cell.sum().item()),
        "land_cell_count": int((~wet_cell).sum().item()),
        "wet_fraction": float(wet_cell.to(dtype=torch.float64).mean().item()),
        "depth_min": float(wet_depth.min().item()),
        "depth_max": float(wet_depth.max().item()),
        "depth_mean": float(wet_depth.mean().item()),
        "elevation_min": float(elevation_native.min().item()),
        "elevation_max": float(elevation_native.max().item()),
        }
    )
    return domain, column, out_metadata


def build_ocean_domain_from_etopo(
    ds_or_da,
    mesh,
    *,
    n_layers: int,
    grid_to_mesh: LatLonToMeshBilinear | None = None,
    elevation_var: str | None = None,
    coarsen: int = 1,
    coarsen_cache_dir: str | Path | None = None,
    ocean_threshold_m: float = 0.0,
    min_depth: float = 50.0,
    max_depth: float | None = 6000.0,
    land_depth: float | None = None,
    depth_smoothing_passes: int = 0,
    depth_smoothing_alpha: float = 0.5,
    preserve_wet_mean_depth: bool = True,
    min_wet_component_cells: int = 0,
    keep_largest_wet_component: bool = False,
    rest_thickness: torch.Tensor | None = None,
    gravity: float = OCEAN_BATHYMETRY_DEFAULT_GRAVITY,
    rho0: float = OCEAN_BATHYMETRY_DEFAULT_RHO0,
):
    """Build ``OceanDomain`` and ``OceanColumn`` from ETOPO elevation."""

    elevation_native, attrs = etopo_native_elevation_from_xarray(
        ds_or_da,
        mesh,
        grid_to_mesh=grid_to_mesh,
        elevation_var=elevation_var,
        coarsen=int(coarsen),
        coarsen_cache_dir=coarsen_cache_dir,
    )
    metadata = {
        "source": "ETOPO_2022",
        "source_url": ETOPO_2022_METADATA_URL,
        "elevation_var": attrs.get("source_name"),
        "coarsen": int(coarsen),
        "coarsen_cache_path": attrs.get("coarsen_cache_path"),
        "coarsen_cache_hit": attrs.get("coarsen_cache_hit"),
    }
    return build_ocean_domain_from_native_elevation(
        mesh,
        elevation_native,
        n_layers=n_layers,
        ocean_threshold_m=ocean_threshold_m,
        min_depth=min_depth,
        max_depth=max_depth,
        land_depth=land_depth,
        depth_smoothing_passes=depth_smoothing_passes,
        depth_smoothing_alpha=depth_smoothing_alpha,
        preserve_wet_mean_depth=preserve_wet_mean_depth,
        min_wet_component_cells=min_wet_component_cells,
        keep_largest_wet_component=keep_largest_wet_component,
        rest_thickness=rest_thickness,
        gravity=gravity,
        rho0=rho0,
        metadata=metadata,
    )


def smooth_ocean_depth_on_mesh(
    mesh,
    depth: torch.Tensor,
    wet_cell: torch.Tensor,
    *,
    passes: int = 0,
    alpha: float = 0.5,
    min_depth: float | None = None,
    max_depth: float | None = None,
    preserve_wet_mean: bool = True,
) -> torch.Tensor:
    """Smooth wet-cell ocean depth over wet-wet mesh edges."""

    depth_in = torch.as_tensor(depth, dtype=torch.float64)
    original_shape = tuple(depth_in.shape)
    depth_flat = depth_in.reshape(int(mesh.nCells), -1)
    if int(depth_flat.shape[1]) != 1:
        raise ValueError(f"depth must map to one value per mesh cell, got {original_shape}")
    out = depth_flat[:, 0].clone()
    wet = torch.as_tensor(wet_cell, dtype=torch.bool, device=out.device).reshape(int(mesh.nCells))
    if not bool(wet.any().item()):
        raise ValueError("cannot smooth ocean depth with no wet cells")
    n_passes = int(passes)
    if n_passes < 0:
        raise ValueError("passes must be >= 0")
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    if n_passes == 0 or alpha == 0.0:
        return out.reshape(original_shape)

    cells = mesh.cellsOnEdge.to(device=out.device)
    c0 = cells[:, 0]
    c1 = cells[:, 1]
    valid = (c0 >= 0) & (c1 >= 0)
    c0 = c0[valid].to(dtype=torch.long)
    c1 = c1[valid].to(dtype=torch.long)
    active = wet[c0] & wet[c1]
    c0 = c0[active]
    c1 = c1[active]
    if int(c0.numel()) == 0:
        return out.reshape(original_shape)

    for _ in range(n_passes):
        old_mean = out[wet].mean()
        accum = torch.zeros_like(out)
        count = torch.zeros_like(out)
        one = torch.ones_like(c0, dtype=out.dtype)
        accum.index_add_(0, c0, out[c1])
        accum.index_add_(0, c1, out[c0])
        count.index_add_(0, c0, one)
        count.index_add_(0, c1, one)
        has_neighbor = count > 0.0
        neighbor_mean = torch.where(has_neighbor, accum / count.clamp_min(1.0), out)
        proposal = out + alpha * (neighbor_mean - out)
        if min_depth is not None:
            proposal = proposal.clamp_min(float(min_depth))
        if max_depth is not None:
            proposal = proposal.clamp_max(float(max_depth))
        if preserve_wet_mean:
            new_mean = proposal[wet].mean()
            proposal = torch.where(wet, proposal + (old_mean - new_mean), proposal)
            if min_depth is not None:
                proposal = proposal.clamp_min(float(min_depth))
            if max_depth is not None:
                proposal = proposal.clamp_max(float(max_depth))
        out = torch.where(wet, proposal, out)
    return out.reshape(original_shape)


def filter_wet_components_on_mesh(
    mesh,
    wet_cell: torch.Tensor,
    *,
    min_component_cells: int = 0,
    keep_largest: bool = False,
) -> dict[str, Any]:
    """Return a wet mask with optional small-component filtering metadata."""

    wet = torch.as_tensor(wet_cell, dtype=torch.bool).detach().cpu().reshape(int(mesh.nCells))
    cells = mesh.cellsOnEdge.detach().cpu().to(dtype=torch.long)
    adjacency: list[list[int]] = [[] for _ in range(int(mesh.nCells))]
    for c0_t, c1_t in cells:
        c0 = int(c0_t.item())
        c1 = int(c1_t.item())
        if c0 < 0 or c1 < 0 or not bool(wet[c0].item()) or not bool(wet[c1].item()):
            continue
        adjacency[c0].append(c1)
        adjacency[c1].append(c0)

    visited = torch.zeros_like(wet)
    components: list[list[int]] = []
    for start in torch.nonzero(wet, as_tuple=False).reshape(-1).tolist():
        start = int(start)
        if bool(visited[start].item()):
            continue
        stack = [start]
        visited[start] = True
        component: list[int] = []
        while stack:
            cell = stack.pop()
            component.append(cell)
            for neighbor in adjacency[cell]:
                if not bool(visited[neighbor].item()):
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(component)

    keep = torch.zeros_like(wet)
    if components:
        largest = max(components, key=len)
        min_size = max(0, int(min_component_cells))
        for component in components:
            if bool(keep_largest):
                should_keep = component is largest
            else:
                should_keep = min_size <= 1 or len(component) >= min_size
            if should_keep:
                keep[torch.as_tensor(component, dtype=torch.long)] = True
        if not bool(keep.any().item()):
            keep[torch.as_tensor(largest, dtype=torch.long)] = True

    filtered = bool(keep_largest) or int(min_component_cells) > 1
    result_wet = keep if filtered else wet
    return {
        "wet_cell": result_wet,
        "component_count_raw": len(components),
        "component_count_kept": int(sum(1 for component in components if bool(result_wet[component[0]].item()))) if components else 0,
        "wet_cells_removed": int(wet.sum().item() - result_wet.sum().item()) if filtered else 0,
        "largest_component_cells": max((len(component) for component in components), default=0),
    }


def _as_tensor(value: Any, *, dtype: torch.dtype, device: torch.device | str | None = None) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(dtype=dtype, device=device if device is not None else value.device)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _etopo_lat_lon_names(ds_or_da) -> tuple[str, str]:
    names = set(getattr(ds_or_da, "coords", {})) | set(getattr(ds_or_da, "dims", {}))
    lat_name = next((name for name in ("lat", "latitude", "y") if name in names), None)
    lon_name = next((name for name in ("lon", "longitude", "x") if name in names), None)
    if lat_name is None or lon_name is None:
        raise KeyError("expected ETOPO latitude/longitude coordinates named lat/lon, latitude/longitude, or y/x")
    return str(lat_name), str(lon_name)


def _etopo_elevation_dataarray(ds_or_da, *, elevation_var: str | None = None):
    if hasattr(ds_or_da, "data_vars"):
        if elevation_var is not None:
            if elevation_var not in ds_or_da:
                raise KeyError(f"ETOPO dataset is missing requested elevation variable {elevation_var!r}")
            da = ds_or_da[elevation_var]
        else:
            da = None
            for name in ETOPO_ELEVATION_ALIASES:
                if name in ds_or_da:
                    da = ds_or_da[name]
                    break
            if da is None:
                for _name, candidate in ds_or_da.data_vars.items():
                    try:
                        lat_name, lon_name = _etopo_lat_lon_names(candidate)
                    except KeyError:
                        continue
                    if lat_name in candidate.dims and lon_name in candidate.dims:
                        da = candidate
                        break
            if da is None:
                raise KeyError(f"ETOPO dataset is missing an elevation variable; tried {ETOPO_ELEVATION_ALIASES}")
    else:
        da = ds_or_da
        if elevation_var is not None and getattr(da, "name", None) not in (None, elevation_var):
            raise KeyError(f"ETOPO DataArray name {getattr(da, 'name', None)!r} does not match {elevation_var!r}")
    lat_name, lon_name = _etopo_lat_lon_names(da)
    extra_dims = [dim for dim in da.dims if dim not in (lat_name, lon_name)]
    if extra_dims:
        raise ValueError(f"ETOPO elevation variable must be 2D lat/lon; extra dims={extra_dims}")
    return da


def _etopo_coarsen_cache_path(ds_or_da, da, *, cache_dir: str | Path, coarsen: int) -> Path:
    source = _etopo_xarray_source_id(ds_or_da, da)
    parsed = urlparse(source)
    source_name = Path(parsed.path).stem if parsed.path else Path(source).stem
    if not source_name:
        source_name = "etopo"
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    var_name = str(getattr(da, "name", "") or "elevation")
    safe_source = _etopo_cache_token(source_name)
    safe_var = _etopo_cache_token(var_name)
    return Path(cache_dir).expanduser() / "coarsened" / f"{safe_source}.{digest}.{safe_var}.coarsen{int(coarsen)}.nc"


def _etopo_xarray_source_id(ds_or_da, da) -> str:
    for obj in (ds_or_da, da):
        encoding = getattr(obj, "encoding", {}) or {}
        source = encoding.get("source")
        if source:
            return str(source)
    attrs = getattr(ds_or_da, "attrs", {}) or {}
    for key in ("source", "source_url", "title"):
        if attrs.get(key):
            return str(attrs[key])
    return "etopo"


def _etopo_cache_token(value: str) -> str:
    token = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in str(value))
    return token[:96] or "unknown"


def _read_etopo_coarsen_cache(cache_path: Path) -> tuple[LatLonGrid, torch.Tensor, dict[str, Any]]:
    try:
        import xarray as xr
    except Exception as exc:  # pragma: no cover - optional backend dependent.
        raise RuntimeError("ETOPO coarsen-cache loading requires xarray") from exc
    with xr.open_dataset(cache_path) as ds:
        grid, elevation, attrs = etopo_elevation_from_xarray(ds, elevation_var="z", coarsen=1)
        attrs["coarsen"] = int(ds.attrs.get("source_coarsen", attrs.get("coarsen", 1)))
        attrs["source_name"] = str(ds.attrs.get("source_name", attrs.get("source_name", "")))
        attrs["units"] = str(ds["z"].attrs.get("units", attrs.get("units", "m")))
    attrs["coarsen_cache_path"] = str(cache_path)
    attrs["coarsen_cache_hit"] = True
    return grid, elevation, attrs


def _write_etopo_coarsen_cache(cache_path: Path, grid: LatLonGrid, elevation: torch.Tensor, *, attrs: dict[str, Any]) -> None:
    try:
        import xarray as xr
    except Exception as exc:  # pragma: no cover - optional backend dependent.
        raise RuntimeError("ETOPO coarsen-cache writing requires xarray") from exc
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(cache_path.name + ".part")
    if tmp.exists():
        tmp.unlink()
    ds = xr.Dataset(
        data_vars={
            "z": (
                ("lat", "lon"),
                elevation.detach().cpu().numpy(),
                {
                    "units": str(attrs.get("units", "m")),
                    "source_name": str(attrs.get("source_name", "")),
                },
            )
        },
        coords={
            "lat": grid.lat_deg.detach().cpu().numpy(),
            "lon": grid.lon_deg.detach().cpu().numpy(),
        },
        attrs={
            "vfs_etopo_coarsen_cache_version": 1,
            "source_coarsen": int(attrs.get("coarsen", 1)),
            "source_name": str(attrs.get("source_name", "")),
        },
    )
    try:
        ds.to_netcdf(tmp)
        tmp.replace(cache_path)
    finally:
        ds.close()
        if tmp.exists():
            tmp.unlink()


def _is_remote_url(value: str) -> bool:
    return urlparse(str(value)).scheme in {"http", "https"}


def _etopo_direct_download_url(path_or_url: str) -> str:
    return str(path_or_url).replace("/thredds/dodsC/", "/thredds/fileServer/")


__all__ = [
    "ETOPO_2022_60S_BEDROCK_DOWNLOAD_URL",
    "ETOPO_2022_60S_BEDROCK_NETCDF_URL",
    "ETOPO_2022_60S_BEDROCK_OPENDAP_URL",
    "ETOPO_2022_METADATA_URL",
    "ETOPO_ELEVATION_ALIASES",
    "DEFAULT_SCALE_HEIGHT",
    "DRY_AIR_GAS_CONSTANT",
    "EARTH_GRAVITY",
    "GRAVITY",
    "R_D",
    "atmospheric_surface_pressure_from_phi_s",
    "atmospheric_surface_geopotential_from_elevation",
    "atmospheric_surface_height_from_elevation",
    "build_atmospheric_topographic_column",
    "build_ocean_domain_from_etopo",
    "build_ocean_domain_from_native_elevation",
    "build_topographic_column",
    "download_etopo2022",
    "earth_surface_pressure_from_phi_s",
    "etopo_native_elevation_from_xarray",
    "etopo_elevation_from_xarray",
    "etopo_grid_from_xarray",
    "filter_wet_components_on_mesh",
    "open_etopo2022",
    "smooth_ocean_depth_on_mesh",
]
