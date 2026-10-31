"""Atmospheric physical fields and native snapshot conversion."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
from .state import State
from .moisture import CLOUD_LIQUID_TRACER_INDEX_ATTR, VAPOR_TRACER_INDEX_ATTR
from vfs.io import (SNAPSHOT_IO_VERSION, _as_numpy, _git_commit, _clean_metadata_value,
                    _cell_lat_lon_deg, _edge_lat_lon_deg, _vertex_lat_lon_deg,
                    _local_east_north, write_netcdf)

def _vapor_tracer_index(attrs: dict[str, Any], q_tracer: torch.Tensor | None) -> int | None:
    return _named_tracer_index(attrs, q_tracer, VAPOR_TRACER_INDEX_ATTR)


def _cloud_liquid_tracer_index(attrs: dict[str, Any], q_tracer: torch.Tensor | None) -> int | None:
    return _named_tracer_index(attrs, q_tracer, CLOUD_LIQUID_TRACER_INDEX_ATTR)


def _named_tracer_index(attrs: dict[str, Any], q_tracer: torch.Tensor | None, attr_name: str) -> int | None:
    if q_tracer is None:
        return None
    value = attrs.get(attr_name)
    if value is None:
        return None
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    if index < 0 or index >= int(q_tracer.shape[-1]):
        return None
    return index


def _theta_full(column, theta_i: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
    T_full = column.temperature_full(theta_i, ps)
    if column.phi_full_mode in ("linear_pfull_adjoint", "direct_u_linear_adjoint", "direct_u_gl2_adjoint"):
        pi_full = column.exner_full_pmid(ps)
    else:
        pi_full = column.exner_full(ps)
    return T_full / (pi_full / column.c_p)


def _save_native_state_netcdf(
    path: str | Path,
    state: State,
    mesh=None,
    column=None,
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Write an unbatched VFS state snapshot as NetCDF.

    The file stores native prognostic variables and common recovered fields so
    plotting and offline analysis can proceed without rerunning a simulation.
    """

    try:
        import xarray as xr
    except Exception as exc:  # pragma: no cover - optional backend dependent.
        raise RuntimeError("NetCDF snapshot writing requires xarray and a NetCDF backend") from exc

    mesh = mesh or state.mesh
    column = column or state.column
    if mesh is None or column is None:
        raise ValueError("State.to_netcdf requires State.mesh and State.column or explicit mesh/column")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if state.u_e.ndim != 2 or state.mtheta_i.ndim != 2 or state.ps.ndim != 2:
        raise ValueError("State.to_netcdf currently supports unbatched states only")

    theta_i = state.theta()
    theta_full = _theta_full(column, theta_i, state.ps)
    temperature_full = column.temperature_full(theta_i, state.ps)
    pressure_interface = column.interface_pressure(state.ps)
    pressure_full = column.full_pressure(state.ps)
    dp = state.layer_dp()
    lat_cell, lon_cell = _cell_lat_lon_deg(mesh)
    lat_edge, lon_edge = _edge_lat_lon_deg(mesh)
    lat_vertex, lon_vertex = _vertex_lat_lon_deg(mesh)

    wind_vec = mesh.cell_wind_vector(state.u_e)
    lat = torch.asin(mesh.cell_xyz[:, 2].clamp(-1.0, 1.0))
    lon = torch.atan2(mesh.cell_xyz[:, 1], mesh.cell_xyz[:, 0])
    east, north = _local_east_north(lat, lon)
    u_east = (wind_vec * east[:, None, :]).sum(dim=-1)
    v_north = (wind_vec * north[:, None, :]).sum(dim=-1)
    wind_speed = (u_east.square() + v_north.square()).sqrt()
    relative_vorticity = state.curl(state.u_e)
    specific_volume_full = state.specific_volume()
    geopotential_full = state.geopotential()
    surface_geopotential = column.phi_s.to(dtype=state.ps.dtype, device=state.ps.device)
    if surface_geopotential.ndim == 0:
        surface_geopotential = surface_geopotential.expand(int(mesh.nCells))
    surface_geopotential = surface_geopotential.reshape(int(mesh.nCells))

    reference_ps = torch.full_like(state.ps, float(column.p0))
    coords: dict[str, Any] = {
        "cell": np.arange(int(mesh.nCells)),
        "edge": np.arange(int(mesh.nEdges)),
        "vertex": np.arange(int(mesh.nVertices)),
        "level": np.arange(int(column.n_levels)),
        "interface": np.arange(int(column.n_levels) + 1),
    }
    data_vars: dict[str, Any] = {
        "lat_cell_deg": (("cell",), lat_cell),
        "lon_cell_deg": (("cell",), lon_cell),
        "cell_area": (("cell",), _as_numpy(mesh.cell_area)),
        "lat_edge_deg": (("edge",), lat_edge),
        "lon_edge_deg": (("edge",), lon_edge),
        "dc_edge": (("edge",), _as_numpy(mesh.dc_edge)),
        "dv_edge": (("edge",), _as_numpy(mesh.dv_edge)),
        "lat_vertex_deg": (("vertex",), lat_vertex),
        "lon_vertex_deg": (("vertex",), lon_vertex),
        "dual_area": (("vertex",), _as_numpy(mesh.dual_area)),
        "A_interface": (("interface",), _as_numpy(column.A_interface)),
        "B_interface": (("interface",), _as_numpy(column.B_interface)),
        "sigma_interface_reference": (("interface",), _as_numpy(column.interface_pressure(reference_ps)[0] / column.p0)),
        "sigma_full_reference": (("level",), _as_numpy(column.full_pressure(reference_ps)[0] / column.p0)),
        "surface_geopotential": (("cell",), _as_numpy(surface_geopotential)),
        "surface_height": (("cell",), _as_numpy(surface_geopotential / float(column.g))),
        "ps": (("cell",), _as_numpy(state.ps[:, 0])),
        "u_edge": (("edge", "level"), _as_numpy(state.u_e)),
        "mtheta_interface": (("cell", "interface"), _as_numpy(state.mtheta_i)),
        "theta_interface": (("cell", "interface"), _as_numpy(theta_i)),
        "theta_full": (("cell", "level"), _as_numpy(theta_full)),
        "temperature_full": (("cell", "level"), _as_numpy(temperature_full)),
        "pressure_full": (("cell", "level"), _as_numpy(pressure_full)),
        "pressure_interface": (("cell", "interface"), _as_numpy(pressure_interface)),
        "dp": (("cell", "level"), _as_numpy(dp)),
        "cell_wind_u_east": (("cell", "level"), _as_numpy(u_east)),
        "cell_wind_v_north": (("cell", "level"), _as_numpy(v_north)),
        "cell_wind_speed": (("cell", "level"), _as_numpy(wind_speed)),
        "relative_vorticity": (("vertex", "level"), _as_numpy(relative_vorticity)),
        "specific_volume_full": (("cell", "level"), _as_numpy(specific_volume_full)),
        "geopotential_full": (("cell", "level"), _as_numpy(geopotential_full)),
    }

    q_tracer = state.tracers()
    vapor_index = _vapor_tracer_index(state.attrs or {}, q_tracer)
    cloud_liquid_index = _cloud_liquid_tracer_index(state.attrs or {}, q_tracer)
    if state.tracer_m is not None and q_tracer is not None:
        if state.tracer_m.ndim != 3:
            raise ValueError("NetCDF tracer snapshots currently support unbatched tracer_m only")
        coords["tracer"] = np.arange(int(state.tracer_m.shape[-1]))
        data_vars["tracer_m"] = (("cell", "level", "tracer"), _as_numpy(state.tracer_m))
        data_vars["q_tracer"] = (("cell", "level", "tracer"), _as_numpy(q_tracer))
        if vapor_index is not None:
            r_v = q_tracer[..., int(vapor_index)]
            q_v = r_v / (1.0 + r_v).clamp_min(torch.finfo(r_v.dtype).tiny)
            data_vars["vapor_mixing_ratio"] = (("cell", "level"), _as_numpy(r_v))
            data_vars["vapor_specific_humidity"] = (("cell", "level"), _as_numpy(q_v))
        if cloud_liquid_index is not None:
            r_l = q_tracer[..., int(cloud_liquid_index)]
            data_vars["cloud_liquid_mixing_ratio"] = (("cell", "level"), _as_numpy(r_l))

    attrs = {
        "model": "VFS",
        "snapshot_io_version": SNAPSHOT_IO_VERSION,
        "phi_full_mode": column.phi_full_mode,
        "n_cells": int(mesh.nCells),
        "n_edges": int(mesh.nEdges),
        "n_levels": int(column.n_levels),
        "mesh_operator_version": str(getattr(mesh, "mesh_operator_version", "")),
        "thuburn_corner_convention": str(getattr(mesh, "thuburn_corner_convention", "")),
        "p_top": float(column.A_interface[0].item()),
        "p0": float(column.p0),
        "created_by": "vfs.core.state.State.to_netcdf",
        "git_commit": _git_commit() or "",
    }
    for key, value in (state.attrs or {}).items():
        clean_value = _clean_metadata_value(value)
        if isinstance(clean_value, (str, int, float)) or clean_value is None:
            attrs[str(key)] = "" if clean_value is None else clean_value
    if vapor_index is not None:
        attrs["vapor_tracer_index"] = int(vapor_index)
        attrs["vapor_quantity"] = str((state.attrs or {}).get("vapor_quantity", "dry_air_mixing_ratio"))
        attrs["vapor_units"] = str((state.attrs or {}).get("vapor_units", "kg kg-1 dry_air"))
    if cloud_liquid_index is not None:
        attrs["cloud_liquid_tracer_index"] = int(cloud_liquid_index)
        attrs["cloud_liquid_quantity"] = str((state.attrs or {}).get("cloud_liquid_quantity", "dry_air_mixing_ratio"))
        attrs["cloud_liquid_units"] = str((state.attrs or {}).get("cloud_liquid_units", "kg kg-1 dry_air"))
    for key, value in (metadata or {}).items():
        clean_value = _clean_metadata_value(value)
        if isinstance(clean_value, (str, int, float)) or clean_value is None:
            attrs[str(key)] = "" if clean_value is None else clean_value

    return write_netcdf(data_vars, coords, attrs, path)


