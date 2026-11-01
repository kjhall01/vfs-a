"""Explicit ocean snapshot boundary; the archived format checks sizes only.

The caller supplies the original mesh/column; this is not a self-contained
restart format or an identity check for bathymetry/topology.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from .state import OceanState
from .helpers import _netcdf_attrs, _normalize_tracer_tensor

def save_snapshot(state: OceanState, path, *, metadata: dict[str, Any] | None = None) -> str:
    try:
        import xarray as xr
    except Exception as exc:  # pragma: no cover - optional backend dependent.
        raise RuntimeError("OceanState NetCDF writing requires xarray and a NetCDF backend") from exc

    state.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = state._require_mesh()
    column = state._require_column()
    attrs = _netcdf_attrs(state.attrs)
    attrs.update(_netcdf_attrs(metadata or {}))
    attrs.update(
        {
            "created_by": "vfs.ocean.OceanState.to_netcdf",
            "state_kind": "ocean",
            "n_cells": int(mesh.nCells),
            "n_edges": int(mesh.nEdges),
            "n_layers": int(column.n_layers),
            "gravity": float(column.gravity),
            "rho0": float(column.rho0),
        }
    )
    data_vars = {
        "u_edge": (("edge", "layer"), state.u_e.detach().cpu().numpy()),
        "h": (("cell", "layer"), state.h.detach().cpu().numpy()),
        "temperature": (("cell", "layer"), state.temperature().detach().cpu().numpy()),
        "salinity": (("cell", "layer"), state.salinity().detach().cpu().numpy()),
        "temp_h": (("cell", "layer"), state.temp_h.detach().cpu().numpy()),
        "salt_h": (("cell", "layer"), state.salt_h.detach().cpu().numpy()),
        "eta": (("cell",), state.eta().detach().cpu().reshape(-1).numpy()),
        "rest_thickness": (("layer",), column.rest_thickness.detach().cpu().numpy()),
    }
    if state.tracer_h is not None:
        tracer_h = _normalize_tracer_tensor(state.tracer_h, mesh.nCells, column.n_layers)
        data_vars["tracer_h"] = (("cell", "layer", "tracer"), tracer_h.detach().cpu().numpy())
    lat = torch.rad2deg(torch.asin(mesh.cell_xyz[:, 2].detach().cpu().clamp(-1.0, 1.0)))
    lon = torch.remainder(torch.rad2deg(torch.atan2(mesh.cell_xyz[:, 1].detach().cpu(), mesh.cell_xyz[:, 0].detach().cpu())), 360.0)
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "cell": list(range(int(mesh.nCells))),
            "edge": list(range(int(mesh.nEdges))),
            "layer": list(range(int(column.n_layers))),
            "lat_cell_deg": (("cell",), lat.numpy()),
            "lon_cell_deg": (("cell",), lon.numpy()),
        },
        attrs=attrs,
    )
    ds.to_netcdf(path, format="NETCDF4", engine="netcdf4")
    ds.close()
    return str(path)

def load_snapshot(path, *, mesh, column, dtype=None, device=None) -> "OceanState":
    if mesh is None or column is None:
        raise ValueError("load_snapshot requires mesh=... and column=...")
    try:
        import xarray as xr
    except Exception as exc:  # pragma: no cover - optional backend dependent.
        raise RuntimeError("OceanState NetCDF loading requires xarray") from exc

    ds = xr.open_dataset(path)
    try:
        required = ("u_edge", "h", "temp_h", "salt_h")
        missing = [name for name in required if name not in ds.variables]
        if missing:
            raise ValueError(f"NetCDF snapshot is missing required OceanState variables: {missing}")
        u_e = torch.as_tensor(ds["u_edge"].values)
        h = torch.as_tensor(ds["h"].values)
        temp_h = torch.as_tensor(ds["temp_h"].values)
        salt_h = torch.as_tensor(ds["salt_h"].values)
        tracer_h = torch.as_tensor(ds["tracer_h"].values) if "tracer_h" in ds.variables else None
        attrs = dict(ds.attrs)
    finally:
        ds.close()
    if dtype is not None or device is not None:
        target_dtype = dtype if dtype is not None else u_e.dtype
        target_device = device if device is not None else u_e.device
        u_e = u_e.to(dtype=target_dtype, device=target_device)
        h = h.to(dtype=target_dtype, device=target_device)
        temp_h = temp_h.to(dtype=target_dtype, device=target_device)
        salt_h = salt_h.to(dtype=target_dtype, device=target_device)
        if tracer_h is not None:
            tracer_h = tracer_h.to(dtype=target_dtype, device=target_device)
    out = OceanState(u_e, h, temp_h, salt_h, tracer_h, mesh=mesh, column=column, attrs=attrs)
    _validate_restart_metadata(out)
    out.validate()
    return out

def _validate_restart_metadata(state) -> None:
    mesh = state._require_mesh()
    column = state._require_column()
    attrs = dict(state.attrs or {})
    expected = {"n_cells": int(mesh.nCells), "n_edges": int(mesh.nEdges), "n_layers": int(column.n_layers)}
    for key, value in expected.items():
        if key in attrs and str(attrs[key]) not in ("", str(value)):
            raise ValueError(f"OceanState restart metadata {key}={attrs[key]!r} does not match expected {value}")
