"""Idealized VFS-O initial conditions."""

from __future__ import annotations

import torch

from .column import OceanColumn
from .state import OceanState


def ocean_planet_initial_state(
    mesh,
    column: OceanColumn,
    *,
    eta_amplitude: float = 0.1,
    surface_temperature: float = 20.0,
    bottom_temperature: float = 4.0,
    meridional_temperature_amplitude: float = 2.0,
    salinity: float = 35.0,
    meridional_salinity_amplitude: float = 0.2,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> OceanState:
    """Return a smooth global ocean-planet initial state."""

    device = torch.device(device) if device is not None else mesh.cell_xyz.device
    xyz = mesh.cell_xyz.to(dtype=dtype, device=device)
    lat = torch.asin(xyz[:, 2:3].clamp(-1.0, 1.0))
    lon = torch.atan2(xyz[:, 1:2], xyz[:, 0:1])
    area = mesh.cell_area.to(dtype=dtype, device=device)[:, None]
    eta = float(eta_amplitude) * torch.cos(lat) * torch.cos(lon)
    eta = eta - (eta * area).sum() / area.sum().clamp_min(torch.finfo(dtype).tiny)

    rest = column.rest_thickness.to(dtype=dtype, device=device).reshape(1, column.n_layers)
    total_depth = rest.sum()
    h = rest * ((total_depth + eta) / total_depth.clamp_min(torch.finfo(dtype).tiny))
    mid_fraction = (torch.cumsum(rest, dim=-1) - 0.5 * rest) / total_depth.clamp_min(torch.finfo(dtype).tiny)
    vertical_temp = float(surface_temperature) + (float(bottom_temperature) - float(surface_temperature)) * mid_fraction
    temp = vertical_temp + float(meridional_temperature_amplitude) * torch.cos(lat)
    salt = float(salinity) + float(meridional_salinity_amplitude) * torch.sin(lat)
    salt = salt.expand_as(h)
    u_e = torch.zeros(int(mesh.nEdges), int(column.n_layers), dtype=dtype, device=device)
    attrs = {
        "initial_condition": "ocean_planet_smooth",
        "state_kind": "ocean",
        "eta_amplitude": float(eta_amplitude),
    }
    return OceanState.from_temperature(u_e, h, temp.expand_as(h), salt, mesh=mesh, column=column, attrs=attrs)
