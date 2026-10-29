"""Earth-shaped dry initial-state and prescribed-forcing helpers."""

from __future__ import annotations

from typing import Any

import torch

import vfs.data.topography as topography
from vfs.geometry.column import CharneyPhillipsColumn
from vfs.physics import HeldSuarezPhysics
from vfs.core.state import State


def prescribed_earth_temperature_interfaces(
    mesh,
    column: CharneyPhillipsColumn,
    ps: torch.Tensor,
    *,
    surface_temperature: torch.Tensor | Any | None = None,
    lapse_rate: float = 6.5e-3,
    stratosphere_temperature: float = 200.0,
    scale_height: float = topography.DEFAULT_SCALE_HEIGHT,
) -> torch.Tensor:
    """Simple lat/topography-shaped equilibrium temperature at CP interfaces."""

    ref = ps
    if surface_temperature is None:
        lat = torch.asin(mesh.cell_xyz[:, 2:3].to(dtype=ref.dtype, device=ref.device).clamp(-1.0, 1.0))
        surface_temperature_t = 302.0 - 57.0 * torch.sin(lat).square()
    else:
        surface_temperature_t = _as_tensor(surface_temperature, dtype=ref.dtype, device=ref.device)
        if surface_temperature_t.ndim == 1:
            surface_temperature_t = surface_temperature_t[:, None]
    p_i = column.interface_pressure(ps)
    height_above_surface = float(scale_height) * torch.log(ps.clamp_min(1.0) / p_i.clamp_min(1.0))
    troposphere = surface_temperature_t - float(lapse_rate) * height_above_surface
    floor = torch.as_tensor(float(stratosphere_temperature), dtype=troposphere.dtype, device=troposphere.device)
    return torch.maximum(troposphere, floor)


def earth_like_dry_initial_state(
    mesh,
    column: CharneyPhillipsColumn,
    dyn=None,
    *,
    surface_temperature: torch.Tensor | Any | None = None,
    wind_noise: float = 0.0,
    seed: int = 0,
    ps0: float = 1.0e5,
    min_ps_margin: float = 1000.0,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> State:
    """Construct a topography-aware dry rest state with optional tiny wind noise."""

    dtype = dtype or mesh.cell_xyz.dtype
    device = torch.device(device) if device is not None else mesh.cell_xyz.device
    phi_s = column.phi_s.to(dtype=dtype, device=device)
    if phi_s.ndim == 0:
        phi_s = phi_s.expand(int(mesh.nCells))
    p_top = float(column.A_interface[0].detach().cpu().item())
    ps = topography.earth_surface_pressure_from_phi_s(phi_s, p0=float(ps0)).to(dtype=dtype, device=device)
    ps = ps.clamp_min(p_top + float(min_ps_margin)).reshape(int(mesh.nCells), 1)
    T_i = prescribed_earth_temperature_interfaces(
        mesh,
        column,
        ps,
        surface_temperature=surface_temperature,
    )
    theta_i = T_i / (column.exner_interface(ps) / column.c_p).clamp_min(torch.finfo(T_i.dtype).tiny)
    u_e = torch.zeros(int(mesh.nEdges), int(column.n_levels), dtype=dtype, device=device)
    if float(wind_noise) != 0.0:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            u_e = u_e + float(wind_noise) * torch.randn_like(u_e)
    attrs = {
        "gravity": float(getattr(dyn, "gravity", topography.GRAVITY)),
        "initial_condition": "earth_like_dry_topographic",
        "surface_pressure_formula": "p0*exp(-phi_s/(R_d*T_ref))",
    }
    return State.from_theta(u_e, theta_i, ps, mesh=mesh, column=column, attrs=attrs)


def build_earth_like_dry_physics(
    mesh,
    column: CharneyPhillipsColumn,
    *,
    surface_temperature: torch.Tensor | Any | None = None,
    thermal_relaxation_days: float = 20.0,
    lower_relaxation_days: float = 5.0,
    drag_days: float = 1.0,
    sigma_b: float = 0.7,
    p0: float = 1.0e5,
) -> HeldSuarezPhysics:
    """Build a prescribed dry thermal-relaxation/Rayleigh-drag source provider."""

    phi_s = column.phi_s.detach().to(dtype=torch.float64)
    if phi_s.ndim == 0:
        phi_s = phi_s.expand(int(mesh.nCells))
    ps = topography.earth_surface_pressure_from_phi_s(phi_s, p0=float(p0)).reshape(int(mesh.nCells), 1)
    ps = ps.clamp_min(float(column.A_interface[0].detach().cpu().item()) + 1000.0)
    Teq = prescribed_earth_temperature_interfaces(
        mesh,
        column,
        ps,
        surface_temperature=surface_temperature,
    ).detach()
    p_i = column.interface_pressure(ps)
    sigma_i = (p_i / ps.clamp_min(1.0)).clamp_min(1.0e-12)
    ramp_i = ((sigma_i - float(sigma_b)) / (1.0 - float(sigma_b))).clamp(0.0, 1.0)
    k_upper = 1.0 / (float(thermal_relaxation_days) * 86400.0)
    k_lower = 1.0 / (float(lower_relaxation_days) * 86400.0)
    kT = k_upper + (k_lower - k_upper) * ramp_i

    p_full = column.full_pressure(ps)
    sigma_full = (p_full / ps.clamp_min(1.0)).clamp_min(1.0e-12)
    ramp_full = ((sigma_full - float(sigma_b)) / (1.0 - float(sigma_b))).clamp(0.0, 1.0)
    kv = (1.0 / (float(drag_days) * 86400.0)) * ramp_full
    return HeldSuarezPhysics(Teq=Teq, kT=kT.detach(), kv=kv.detach())


def _as_tensor(value: Any, *, dtype: torch.dtype, device: torch.device | str | None = None) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(dtype=dtype, device=device if device is not None else value.device)
    return torch.as_tensor(value, dtype=dtype, device=device)


__all__ = [
    "build_earth_like_dry_physics",
    "earth_like_dry_initial_state",
    "prescribed_earth_temperature_interfaces",
]
