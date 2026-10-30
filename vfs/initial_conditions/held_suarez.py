"""Held-Suarez initial-condition builders."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from vfs.atmosphere import DryHydrostaticConservativeDynamics
from .equilibrium import HeldSuarezConstants, held_suarez_equilibrium_temperature
from vfs.core.state import State


@dataclass(frozen=True)
class HeldSuarezInitialConditionInfo:
    """Metadata for a Held-Suarez initial state."""

    initial_condition: str
    equilibrium_wind: str
    initial_wind_noise: float
    held_suarez_constants: dict[str, Any]
    initial_theta_min: float
    initial_theta_max: float
    initial_T_min: float
    initial_T_max: float
    initial_ps_min: float
    initial_ps_max: float
    initial_u_max: float
    initial_equator_to_pole_T_contrast: float | None


def normalize_held_suarez_initial_condition(value: Any) -> str:
    aliases = {
        "isothermal": "isothermal_rest",
        "rest": "isothermal_rest",
        "flat_isothermal": "isothermal_rest",
        "isothermal_rest": "isothermal_rest",
        "equilibrium": "hs_equilibrium",
        "hs_eq": "hs_equilibrium",
        "hs_equilibrium": "hs_equilibrium",
    }
    try:
        return aliases[str(value)]
    except KeyError as exc:
        raise ValueError("initial_condition must be isothermal_rest or hs_equilibrium") from exc


def _cell_lat(mesh) -> torch.Tensor:
    return torch.asin(torch.clamp(mesh.cell_xyz[:, 2:3], -1.0, 1.0))


def _wind_for_mode(
    mesh,
    column,
    *,
    wind_mode: str,
    initial_wind_noise: float,
    seed: int,
) -> torch.Tensor:
    wind_mode = str(wind_mode)
    if wind_mode == "none":
        return torch.zeros((mesh.nEdges, column.n_levels), dtype=torch.float64)
    if wind_mode == "small_noise":
        gen = torch.Generator().manual_seed(int(seed))
        return float(initial_wind_noise) * torch.randn((mesh.nEdges, column.n_levels), generator=gen, dtype=torch.float64)
    if wind_mode == "thermal_wind":
        raise NotImplementedError("equilibrium_wind='thermal_wind' is not implemented; choose none or small_noise")
    raise ValueError("equilibrium_wind must be none, small_noise, or thermal_wind")


def _metadata(
    mesh,
    *,
    initial_condition: str,
    equilibrium_wind: str,
    initial_wind_noise: float,
    constants: HeldSuarezConstants,
    theta_i: torch.Tensor,
    T_i: torch.Tensor,
    ps: torch.Tensor,
    u_e: torch.Tensor,
) -> HeldSuarezInitialConditionInfo:
    lat = _cell_lat(mesh)
    T_surface = T_i[:, -1]
    trop = lat[:, 0].abs() <= math.radians(15.0)
    pole = lat[:, 0].abs() >= math.radians(60.0)
    contrast = None
    if bool(trop.any()) and bool(pole.any()):
        contrast = float(T_surface[trop].mean().item() - T_surface[pole].mean().item())
    return HeldSuarezInitialConditionInfo(
        initial_condition=initial_condition,
        equilibrium_wind=equilibrium_wind,
        initial_wind_noise=float(initial_wind_noise),
        held_suarez_constants=asdict(constants),
        initial_theta_min=float(theta_i.min().item()),
        initial_theta_max=float(theta_i.max().item()),
        initial_T_min=float(T_i.min().item()),
        initial_T_max=float(T_i.max().item()),
        initial_ps_min=float(ps.min().item()),
        initial_ps_max=float(ps.max().item()),
        initial_u_max=float(u_e.abs().max().item()),
        initial_equator_to_pole_T_contrast=contrast,
    )


def isothermal_rest_state(
    mesh,
    column,
    *,
    dyn: DryHydrostaticConservativeDynamics | None = None,
    constants: HeldSuarezConstants | None = None,
    temperature: float = 288.0,
    wind_mode: str = "none",
    initial_wind_noise: float = 0.05,
    seed: int = 0,
    return_info: bool = False,
) -> State | tuple[State, HeldSuarezInitialConditionInfo]:
    constants = constants or HeldSuarezConstants()
    ps = torch.full((mesh.nCells, 1), float(constants.p0), dtype=torch.float64)
    T_i = torch.full((mesh.nCells, column.n_levels + 1), float(temperature), dtype=torch.float64)
    theta_i = T_i / (column.exner_interface(ps) / column.c_p).clamp_min(1.0e-12)
    u_e = _wind_for_mode(mesh, column, wind_mode=wind_mode, initial_wind_noise=initial_wind_noise, seed=seed)
    attrs = {}
    if dyn is not None:
        attrs["gravity"] = dyn.gravity
    state = State.from_theta(u_e, theta_i, ps, mesh=mesh, column=column, attrs=attrs)
    info = _metadata(
        mesh,
        initial_condition="isothermal_rest",
        equilibrium_wind=wind_mode,
        initial_wind_noise=initial_wind_noise,
        constants=constants,
        theta_i=theta_i,
        T_i=T_i,
        ps=ps,
        u_e=u_e,
    )
    return (state, info) if return_info else state


def held_suarez_equilibrium_state(
    mesh,
    column,
    *,
    dyn: DryHydrostaticConservativeDynamics | None = None,
    constants: HeldSuarezConstants | None = None,
    equilibrium_wind: str = "small_noise",
    initial_wind_noise: float = 0.05,
    seed: int = 0,
    return_info: bool = False,
) -> State | tuple[State, HeldSuarezInitialConditionInfo]:
    constants = constants or HeldSuarezConstants()
    ps = torch.full((mesh.nCells, 1), float(constants.p0), dtype=torch.float64)
    p_i = column.interface_pressure(ps)
    T_i = held_suarez_equilibrium_temperature(column, constants, _cell_lat(mesh), p_i)
    theta_i = T_i / (column.exner_interface(ps) / column.c_p).clamp_min(1.0e-12)
    u_e = _wind_for_mode(
        mesh,
        column,
        wind_mode=str(equilibrium_wind),
        initial_wind_noise=initial_wind_noise,
        seed=seed,
    )
    attrs = {}
    if dyn is not None:
        attrs["gravity"] = dyn.gravity
    state = State.from_theta(u_e, theta_i, ps, mesh=mesh, column=column, attrs=attrs)
    info = _metadata(
        mesh,
        initial_condition="hs_equilibrium",
        equilibrium_wind=str(equilibrium_wind),
        initial_wind_noise=initial_wind_noise,
        constants=constants,
        theta_i=theta_i,
        T_i=T_i,
        ps=ps,
        u_e=u_e,
    )
    return (state, info) if return_info else state


__all__ = [
    "HeldSuarezInitialConditionInfo",
    "held_suarez_equilibrium_state",
    "isothermal_rest_state",
    "normalize_held_suarez_initial_condition",
]
