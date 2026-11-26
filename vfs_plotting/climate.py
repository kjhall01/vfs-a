"""Read coupled VFS states and calculate climate summaries on the native mesh."""
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import xarray as xr

DAY = 86400
G = 9.8  # Coupled atmosphere uses pyam21.constants.GRAV.


@contextmanager
def open_state(path):
    """Yield one state on its native grid with snapshot names and physical units."""
    with xr.open_dataset(path, decode_times=False) as raw:
        kind = raw.attrs.get("vfs_format")
        if kind == "coupled_snapshot_v1":
            if raw.sizes.get("time") != 1:
                raise ValueError(f"{path} must contain one snapshot time")
            yield raw.isel(time=0)
            return
        if kind != "coupled_restart_v1":
            raise ValueError(f"{path} is not a coupled snapshot or restart")

        import torch
        from vfs.atmosphere.mixture_thermodynamics import MixtureThermodynamics
        from vfs.atmosphere.moist_cp import interface_pressure_mass

        ps = raw.air_atmosphere_ps.squeeze("singleton", drop=True)
        p = (raw.air_hybrid_a + raw.air_hybrid_b * ps).transpose("cell", "air_interface")
        pressure = torch.from_numpy(p.values)
        mass = interface_pressure_mass(pressure.diff())
        content = raw.air_atmosphere_water_content.sel(water_species=["vapor", "liquid", "ice"])
        water = torch.from_numpy(content.values) / mass[..., None]
        q = torch.cat([1 - water.sum(-1, keepdim=True), water], -1)
        entropy = torch.from_numpy(raw.air_atmosphere_entropy_content.values) / mass
        temp = MixtureThermodynamics().temperature_from_entropy(entropy, pressure, q)
        yield raw.assign(
            air_pressure=p.assign_attrs(units="Pa"),
            air_temperature=(p.dims, temp.numpy(), {"units": "K"}),
            air_vapor=(p.dims, water[..., 0].numpy(), {"units": "kg kg-1"}),
            air_normal_wind=raw.air_atmosphere_u_e,
            ocean_temperature=(raw.ocean_temp_h / raw.ocean_h.where(raw.ocean_h > 0)).assign_attrs(units="degC"),
        )


def _wind_weights(grid):
    """VFS cell_wind_vector least-squares fit, in geographic east/north axes.

    Read the saved cell pair at each edge, preserving its direction and index.
    Building these two sparse maps once avoids rebuilding a dynamics mesh.
    """
    from scipy.sparse import csr_matrix

    xyz = grid.cell_xyz.values
    pairs = grid.cells_on_edge.values.astype(int)
    normals = xyz[pairs[:, 1]] - xyz[pairs[:, 0]]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    lon = np.deg2rad(grid.longitude.values)
    east = np.column_stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)))
    north = np.cross(xyz, east)
    cells = pairs.ravel()
    edges = np.repeat(np.arange(len(pairs)), 2)
    basis = np.stack((east[cells], north[cells]), axis=1)
    directions = np.einsum("nij,nj->ni", basis, normals[edges])
    gram = np.broadcast_to(1e-10 * np.eye(2), (len(xyz), 2, 2)).copy()
    np.add.at(gram, cells, directions[:, :, None] * directions[:, None, :])
    weights = np.linalg.solve(gram[cells], directions[..., None])[..., 0]
    return tuple(csr_matrix((w, (cells, edges)), shape=(len(xyz), len(pairs))) for w in weights.T)


def _at_pressure(field, pressure, target):
    """Interpolate in log pressure within each column's sampled range."""
    log_target = np.log(target)
    return np.stack([np.interp(log_target, np.log(p), values, left=np.nan, right=np.nan)
                     for values, p in zip(field, pressure)])


def _zonal(values, bands, area, n):
    result = np.full((n, *values.shape[1:]), np.nan)
    for i in np.unique(bands):
        field = values[bands == i]
        weights = area[bands == i].reshape((-1,) + (1,) * (values.ndim - 1))
        denominator = np.sum(weights * np.isfinite(field), axis=0)
        result[i] = np.nansum(weights * field, axis=0) / np.where(denominator > 0, denominator, np.nan)
    return result


def _sample_fields(state, wind_weights, latitude, bands, pressure):
    p = state.air_pressure.values
    dp = np.diff(p, axis=1)
    if not np.all(dp > 0):
        raise ValueError("interface pressures must increase from lid to surface")
    area = state.cell_area.values
    u, v = (w @ state.air_normal_wind.values for w in wind_weights)
    temperature = state.air_temperature.values
    q = state.air_vapor.values
    zonal_args = (bands, area, len(latitude))
    T = _zonal(_at_pressure(temperature, p, pressure), *zonal_args)
    U = _zonal(_at_pressure(u, (p[:, 1:] + p[:, :-1]) / 2, pressure), *zonal_args)

    # Integrate northward v dp from the lid. Terrain contributes zero transport
    # below ground. Keep the full band area in the transport denominator.
    transport = np.concatenate([np.zeros_like(v[:, :1]), np.cumsum(v * dp, axis=1)], axis=1)
    integral = np.stack([np.interp(pressure, levels, row, left=0)
                         for levels, row in zip(p, transport)])
    psi = _zonal(integral, *zonal_args)
    psi[_zonal((p[:, -1:] >= pressure).astype(float), *zonal_args) == 0] = np.nan
    circumference = 2 * np.pi * state.attrs["sphere_radius_m"] * np.cos(np.deg2rad(latitude))
    psi *= circumference[:, None] / (G * 1e9)

    maps = {
        "surface_air": temperature[:, -1] - 273.15,
        "sst": state.ocean_temperature.isel(ocean_level=0).where(state.land_mask == 0).values,
        "column_vapor": np.sum((q[:, 1:] + q[:, :-1]) * 0.5 * dp, axis=1) / G,
    }
    # Optional CF precipitation flux, positive downward, liquid-water equivalent.
    for var in state.data_vars.values():
        if var.attrs.get("standard_name") == "precipitation_flux" or var.name == "precipitation_flux":
            if var.dims != ("cell",) or var.attrs.get("units") not in {"kg m-2 s-1", "kg m^-2 s^-1", "kg m**-2 s**-1"}:
                raise ValueError("precipitation_flux must be cellwise in kg m-2 s-1")
            maps["precipitation"] = var.values * DAY
            break
    result = {name: (("cell",), field) for name, field in maps.items()}
    for name, field in maps.items():
        result[name + "_zonal"] = (("lat",), _zonal(field, *zonal_args))
    for name, section in (("temperature", T), ("zonal_wind", U), ("streamfunction", psi)):
        result[name] = (("lat", "pressure_hpa"), section)
    return xr.Dataset(result, coords={"lat": latitude, "pressure_hpa": pressure / 100})


def summarize(paths, days=None, lat_bins=36):
    """Return an xarray Dataset of latitude–pressure sections, cell maps and profiles.

    Pass a path for an instantaneous state or an iterable of files for a mean.
    Optional days=(start, end) selects sample times inclusively. Consecutive
    samples receive trapezoidal time weights, including across any gaps.
    Latitude bands use cell area for weighting. Pressure is in hPa.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    if lat_bins < 2:
        raise ValueError("use at least two latitude bands")
    files = []
    for path in paths:
        with xr.open_dataset(path, decode_times=False) as ds:
            seconds = float(ds.time.item())
            if days is None or days[0] <= seconds / DAY <= days[1]:
                files.append((seconds, Path(path)))
    if not files:
        raise ValueError("no states in the selected time window")
    files.sort()
    times = np.array([t for t, _ in files])
    if np.any(np.diff(times) <= 0):
        raise ValueError("duplicate model time; select one snapshot or restart per time")
    weights = np.ones(len(times))
    if len(times) > 1:
        half_interval = np.diff(times) / 2
        weights = np.r_[0, half_interval] + np.r_[half_interval, 0]
    bounds = np.linspace(-90, 90, lat_bins + 1)
    latitude = (bounds[1:] + bounds[:-1]) / 2
    for i, ((_, path), weight) in enumerate(zip(files, weights)):
        with open_state(path) as state:
            if i == 0:
                # Keep the native polygons for map drawing; no horizontal regridding.
                names = ["cell_xyz", "longitude", "latitude", "cell_area", "cells_on_edge",
                         "vertex_longitude", "vertex_latitude", "vertices_on_edge",
                         "vertices_on_cell", "land_mask"]
                grid = state[names].load()
                wind_weights = _wind_weights(state)
                bands = np.clip(np.searchsorted(bounds, state.latitude.values, side="right") - 1, 0, lat_bins-1)
                lid = float(state.air_pressure.min())
                pressure = np.geomspace(lid, 100000., 48)
                clock = state.time.attrs
            elif (not np.array_equal(state.cell_xyz, grid.cell_xyz)
                  or not np.array_equal(state.cells_on_edge, grid.cells_on_edge)
                  or not np.array_equal(state.land_mask, grid.land_mask)
                  or clock != state.time.attrs):
                raise ValueError(f"incompatible mesh, land mask, or clock in {path}")
            sample = _sample_fields(state, wind_weights, latitude, bands, pressure)
            if i == 0:
                total = xr.zeros_like(sample)
                valid = xr.zeros_like(sample)
            elif set(sample.data_vars) != set(total.data_vars):
                raise ValueError("field availability changed within the selected files")
            total += sample.fillna(0) * weight
            valid += sample.notnull() * weight
    result = (total / valid.where(valid > 0)).assign_coords({n: grid[n].variable for n in names})
    units = {"temperature": "K", "zonal_wind": "m s-1", "streamfunction": "1e9 kg s-1",
             "surface_air": "degC", "sst": "degC", "column_vapor": "kg m-2", "precipitation": "mm day-1"}
    for name in result.data_vars:
        result[name].attrs["units"] = units[name.removesuffix("_zonal")]
    result.lat.attrs["units"] = "degrees_north"
    result.pressure_hpa.attrs["units"] = "hPa"
    result.attrs["period"] = (f"Day {times[0]/DAY:g}" if len(files) == 1 else
                              f"Days {times[0]/DAY:g}–{times[-1]/DAY:g} · time mean of {len(files)} states")
    return result
