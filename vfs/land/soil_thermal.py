"""Layered soil thermal diffusion for the land model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .parameters import LandSurfaceParameters
from .state import LandSurfaceState


@dataclass(frozen=True)
class SoilThermalResult:
    soil_temperature: torch.Tensor
    skin_temperature: torch.Tensor
    conductive_flux_down: torch.Tensor
    soil_energy_tendency: torch.Tensor


def relax_skin_temperature(
    state: LandSurfaceState,
    *,
    dt: torch.Tensor,
    params: LandSurfaceParameters,
    soil_temperature: torch.Tensor | None = None,
) -> torch.Tensor:
    if params.skin_relaxation_timescale <= 0.0:
        return state.skin_temperature
    relax = (dt / float(params.skin_relaxation_timescale)).clamp(0.0, 1.0)
    top_soil = state.top_soil_temperature if soil_temperature is None else soil_temperature[:, :1]
    return state.skin_temperature + relax * (top_soil - state.skin_temperature)


def diffuse_soil_temperature(
    state: LandSurfaceState,
    *,
    land_fraction: torch.Tensor,
    dt: torch.Tensor,
    params: LandSurfaceParameters,
    top_conductance_factor: torch.Tensor | None = None,
) -> SoilThermalResult:
    """Diffuse heat through soil layers with a skin-temperature upper boundary."""

    ref = state.soil_temperature
    land = land_fraction.to(dtype=ref.dtype, device=ref.device)
    dt = dt.to(dtype=ref.dtype, device=ref.device).clamp_min(torch.finfo(ref.dtype).tiny)
    if params.soil_thermal_conductivity <= 0.0:
        skin = relax_skin_temperature(state, dt=dt, params=params)
        return SoilThermalResult(
            ref,
            skin,
            torch.zeros(ref.shape[0], ref.shape[1] + 1, dtype=ref.dtype, device=ref.device),
            torch.zeros_like(land),
        )

    substeps = _thermal_substeps(float(dt.detach().cpu().item()), params)
    sub_dt = dt / float(substeps)
    temperature = ref
    capacity = soil_heat_capacity_per_area(ref, params)
    active = land > 0.0
    active_layers = active.expand_as(ref)
    last_flux = torch.zeros(ref.shape[0], ref.shape[1] + 1, dtype=ref.dtype, device=ref.device)
    for _ in range(substeps):
        flux = conductive_flux_down(
            temperature,
            state.skin_temperature,
            params,
            top_conductance_factor=top_conductance_factor,
        )
        flux = torch.where(active, flux, torch.zeros_like(flux))
        tendency = (flux[:, :-1] - flux[:, 1:]) / capacity.clamp_min(torch.finfo(ref.dtype).tiny)
        updated = temperature + sub_dt * tendency
        temperature = torch.where(active_layers, updated, ref)
        last_flux = flux

    skin = relax_skin_temperature(state, dt=dt, params=params, soil_temperature=temperature)
    energy_tendency = ((temperature - ref) * capacity).sum(dim=1, keepdim=True) / dt
    energy_tendency = torch.where(active, energy_tendency, torch.zeros_like(energy_tendency))
    return SoilThermalResult(temperature, skin, last_flux, energy_tendency)


def soil_heat_capacity_per_area(ref: torch.Tensor, params: LandSurfaceParameters) -> torch.Tensor:
    thickness = params.soil_layer_thickness_tensor(ref).reshape(1, -1)
    return thickness * float(params.soil_heat_capacity)


def conductive_flux_down(
    soil_temperature: torch.Tensor,
    skin_temperature: torch.Tensor,
    params: LandSurfaceParameters,
    *,
    top_conductance_factor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return interface conductive heat flux, W m-2 positive downward."""

    dz = params.soil_layer_thickness_tensor(soil_temperature)
    k = float(params.soil_thermal_conductivity)
    n_layers = int(soil_temperature.shape[1])
    flux = torch.zeros(
        soil_temperature.shape[0],
        n_layers + 1,
        dtype=soil_temperature.dtype,
        device=soil_temperature.device,
    )
    top_distance = (0.5 * dz[0]).clamp_min(torch.finfo(soil_temperature.dtype).tiny)
    flux[:, :1] = k * (skin_temperature - soil_temperature[:, :1]) / top_distance
    if top_conductance_factor is not None:
        factor = top_conductance_factor.to(dtype=soil_temperature.dtype, device=soil_temperature.device)
        flux[:, :1] = flux[:, :1] * factor.clamp(0.0, 1.0)
    if n_layers > 1:
        distance = (0.5 * (dz[:-1] + dz[1:])).reshape(1, -1)
        flux[:, 1:-1] = k * (soil_temperature[:, :-1] - soil_temperature[:, 1:]) / distance
    return flux


def _thermal_substeps(dt: float, params: LandSurfaceParameters) -> int:
    if params.soil_thermal_conductivity <= 0.0:
        return 1
    dz_min = min(params.soil_layer_thickness)
    max_dt = 0.45 * float(params.soil_heat_capacity) * dz_min * dz_min / float(params.soil_thermal_conductivity)
    if max_dt <= 0.0:
        return 1
    return max(1, int(math.ceil(float(dt) / max_dt)))
