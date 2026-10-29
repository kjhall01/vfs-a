"""Layered hydrology helpers for land-surface stepping."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .parameters import LandSurfaceParameters
from .state import LandSurfaceState


@dataclass(frozen=True)
class HydrologyResult:
    soil_liquid_water: torch.Tensor
    runoff_mass: torch.Tensor
    surface_runoff_mass: torch.Tensor
    drainage_mass: torch.Tensor
    rain_rate: torch.Tensor
    evaporation_rate: torch.Tensor
    infiltration_rate: torch.Tensor
    layer_evaporation_rate: torch.Tensor
    layer_drainage_rate: torch.Tensor


def soil_moisture_availability(
    state: LandSurfaceState,
    params: LandSurfaceParameters,
    *,
    root_fraction: torch.Tensor | None = None,
) -> torch.Tensor:
    available = _plant_available_water(state, params)
    root = _root_fraction_like(state.soil_liquid_water, params, root_fraction)
    return (available * root).sum(dim=1, keepdim=True).clamp(0.0, 1.0)


def bucket_water_budget(
    state: LandSurfaceState,
    *,
    land_fraction: torch.Tensor,
    rain_rate: torch.Tensor,
    melt_rate: torch.Tensor,
    latent_heat_flux: torch.Tensor,
    dt: torch.Tensor,
    cell_area: torch.Tensor,
    params: LandSurfaceParameters,
    evaporation_factor: torch.Tensor | None = None,
    root_fraction: torch.Tensor | None = None,
    potential_evaporation_mass: torch.Tensor | None = None,
) -> HydrologyResult:
    ref = state.soil_liquid_water
    land = land_fraction.to(dtype=ref.dtype, device=ref.device)
    active_land = land > 0.0
    dt = dt.to(dtype=ref.dtype, device=ref.device).clamp_min(torch.finfo(ref.dtype).tiny)
    area = cell_area.to(dtype=ref.dtype, device=ref.device)
    rain = rain_rate.to(dtype=ref.dtype, device=ref.device).clamp_min(0.0)
    melt = melt_rate.to(dtype=ref.dtype, device=ref.device).clamp_min(0.0)

    incoming_mass = (rain + melt) * dt
    infiltration_capacity = _hydraulic_conductivity_water_rate(params, ref) * dt
    infiltrated_mass = torch.minimum(incoming_mass, infiltration_capacity)
    surface_runoff_mass = (incoming_mass - infiltrated_mass).clamp_min(0.0) * area * land

    water = _add_to_top_layer(ref, infiltrated_mass)
    water, saturation_excess = _cascade_saturation_excess(water, params)
    surface_runoff_mass = surface_runoff_mass + saturation_excess * area * land

    water, layer_drainage_mass, bottom_drainage_depth = _gravitational_drainage(water, dt, params)
    water, bottom_excess = _cascade_saturation_excess(water, params)
    drainage_mass = ((bottom_drainage_depth + bottom_excess) * area * land).clamp_min(0.0)

    if potential_evaporation_mass is None:
        potential_evaporation_mass = (
            latent_heat_flux.to(dtype=ref.dtype, device=ref.device).clamp_min(0.0)
            / float(params.latent_heat_vaporization)
            * dt
        )
        if evaporation_factor is not None:
            factor = evaporation_factor.to(dtype=ref.dtype, device=ref.device).clamp_min(0.0)
            potential_evaporation_mass = potential_evaporation_mass * factor
    else:
        potential_evaporation_mass = potential_evaporation_mass.to(dtype=ref.dtype, device=ref.device).clamp_min(0.0)
    layer_evaporation_mass = _root_extraction(water, potential_evaporation_mass, params, root_fraction=root_fraction)
    water = (water - layer_evaporation_mass).clamp_min(0.0)

    active_layers = active_land.expand_as(water)
    water = torch.where(active_layers, water, ref)
    surface_runoff_mass = torch.where(active_land, surface_runoff_mass, torch.zeros_like(surface_runoff_mass))
    drainage_mass = torch.where(active_land, drainage_mass, torch.zeros_like(drainage_mass))
    layer_drainage_mass = torch.where(active_layers, layer_drainage_mass, torch.zeros_like(layer_drainage_mass))
    layer_evaporation_mass = torch.where(active_layers, layer_evaporation_mass, torch.zeros_like(layer_evaporation_mass))
    infiltrated_mass = torch.where(active_land, infiltrated_mass, torch.zeros_like(infiltrated_mass))
    rain = torch.where(active_land, rain, torch.zeros_like(rain))
    melt = torch.where(active_land, melt, torch.zeros_like(melt))

    evaporation_rate = layer_evaporation_mass.sum(dim=1, keepdim=True) / dt
    runoff_mass = surface_runoff_mass + drainage_mass
    return HydrologyResult(
        water,
        runoff_mass,
        surface_runoff_mass,
        drainage_mass,
        rain,
        evaporation_rate,
        infiltrated_mass / dt,
        layer_evaporation_mass / dt,
        layer_drainage_mass / dt,
    )


def layer_water_capacity(state: LandSurfaceState, params: LandSurfaceParameters) -> torch.Tensor:
    return _layer_water_capacity_like(state.soil_liquid_water, params)


def layer_field_capacity(state: LandSurfaceState, params: LandSurfaceParameters) -> torch.Tensor:
    return layer_water_capacity(state, params) * float(params.field_capacity_fraction)


def layer_wilting_point(state: LandSurfaceState, params: LandSurfaceParameters) -> torch.Tensor:
    return layer_water_capacity(state, params) * float(params.wilting_fraction)


def _plant_available_water(state: LandSurfaceState, params: LandSurfaceParameters) -> torch.Tensor:
    wilting = layer_wilting_point(state, params)
    field = layer_field_capacity(state, params)
    return ((state.soil_liquid_water - wilting) / (field - wilting).clamp_min(1.0e-12)).clamp(0.0, 1.0)


def _hydraulic_conductivity_water_rate(params: LandSurfaceParameters, ref: torch.Tensor) -> torch.Tensor:
    rho_water = 1000.0
    return torch.as_tensor(float(params.saturated_hydraulic_conductivity) * rho_water, dtype=ref.dtype, device=ref.device)


def _add_to_top_layer(water: torch.Tensor, input_mass: torch.Tensor) -> torch.Tensor:
    layers = torch.zeros_like(water)
    layers[:, :1] = input_mass
    return water + layers


def _cascade_saturation_excess(water: torch.Tensor, params: LandSurfaceParameters) -> tuple[torch.Tensor, torch.Tensor]:
    capacity = _layer_water_capacity_like(water, params)
    next_water = water
    n_layers = int(water.shape[1])
    bottom_excess = water[:, :1] * 0.0
    for k in range(n_layers):
        cap = capacity[:, k : k + 1]
        excess = (next_water[:, k : k + 1] - cap).clamp_min(0.0)
        next_water = next_water.clone()
        next_water[:, k : k + 1] = next_water[:, k : k + 1] - excess
        if k + 1 < n_layers:
            next_water[:, k + 1 : k + 2] = next_water[:, k + 1 : k + 2] + excess
        else:
            bottom_excess = bottom_excess + excess
    return next_water, bottom_excess


def _gravitational_drainage(
    water: torch.Tensor,
    dt: torch.Tensor,
    params: LandSurfaceParameters,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    capacity = _layer_water_capacity_like(water, params)
    field = capacity * float(params.field_capacity_fraction)
    conductivity = _hydraulic_conductivity_water_rate(params, water)
    next_water = water
    layer_flux = torch.zeros_like(water)
    bottom_drainage = water[:, :1] * 0.0
    n_layers = int(water.shape[1])
    for k in range(n_layers):
        layer = next_water[:, k : k + 1]
        rel_sat = (layer / capacity[:, k : k + 1].clamp_min(torch.finfo(water.dtype).tiny)).clamp(0.0, 1.0)
        available = (layer - field[:, k : k + 1]).clamp_min(0.0)
        drainage = torch.minimum(available, conductivity * dt * rel_sat.square())
        next_water = next_water.clone()
        next_water[:, k : k + 1] = next_water[:, k : k + 1] - drainage
        layer_flux = layer_flux.clone()
        layer_flux[:, k : k + 1] = drainage
        if k + 1 < n_layers:
            next_water[:, k + 1 : k + 2] = next_water[:, k + 1 : k + 2] + drainage
        else:
            bottom_drainage = drainage
    return next_water, layer_flux, bottom_drainage


def _root_extraction(
    water: torch.Tensor,
    potential_evaporation_mass: torch.Tensor,
    params: LandSurfaceParameters,
    *,
    root_fraction: torch.Tensor | None = None,
) -> torch.Tensor:
    wilting = _layer_water_capacity_like(water, params) * float(params.wilting_fraction)
    available = (water - wilting).clamp_min(0.0)
    root = _root_fraction_like(water, params, root_fraction)
    demand = potential_evaporation_mass * root
    first_pass = torch.minimum(available, demand)
    unmet = (potential_evaporation_mass - first_pass.sum(dim=1, keepdim=True)).clamp_min(0.0)
    remaining = (available - first_pass).clamp_min(0.0)
    remaining_total = remaining.sum(dim=1, keepdim=True)
    extra_total = torch.minimum(unmet, remaining_total)
    extra = torch.where(
        remaining_total > torch.finfo(water.dtype).tiny,
        remaining / remaining_total.clamp_min(torch.finfo(water.dtype).tiny) * extra_total,
        torch.zeros_like(remaining),
    )
    return first_pass + extra


def _layer_water_capacity_like(ref: torch.Tensor, params: LandSurfaceParameters) -> torch.Tensor:
    thickness = params.soil_layer_thickness_tensor(ref).reshape(1, -1)
    weights = thickness / thickness.sum().clamp_min(torch.finfo(ref.dtype).tiny)
    return float(params.soil_water_capacity) * weights


def _root_fraction_like(
    ref: torch.Tensor,
    params: LandSurfaceParameters,
    root_fraction: torch.Tensor | None,
) -> torch.Tensor:
    if root_fraction is None:
        root = params.root_fraction_tensor(ref).reshape(1, -1)
    else:
        root = root_fraction.to(dtype=ref.dtype, device=ref.device)
        if root.ndim == 1:
            root = root.reshape(1, -1)
        if root.ndim != 2:
            raise ValueError("root_fraction must have shape [n_soil_layers] or [n_cells, n_soil_layers]")
        if int(root.shape[1]) != int(ref.shape[1]):
            raise ValueError(f"root_fraction has {int(root.shape[1])} layers, expected {int(ref.shape[1])}")
        if int(root.shape[0]) == 1:
            root = root.expand_as(ref)
        elif int(root.shape[0]) != int(ref.shape[0]):
            raise ValueError(f"root_fraction has {int(root.shape[0])} cells, expected {int(ref.shape[0])}")
    root = root.clamp_min(0.0)
    return root / root.sum(dim=1, keepdim=True).clamp_min(torch.finfo(ref.dtype).tiny)
