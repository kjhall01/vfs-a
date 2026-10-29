"""Surface energy balance for land skin temperature."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..surface.packets import AtmosSurfaceFluxPacket
from .parameters import LandSurfaceParameters
from .state import LandSurfaceState


@dataclass(frozen=True)
class SurfaceEnergyBalanceResult:
    skin_temperature: torch.Tensor
    net_radiation: torch.Tensor
    turbulent_heat_flux_up: torch.Tensor
    ground_heat_flux_down: torch.Tensor
    snow_melt_energy_flux: torch.Tensor
    surface_energy_tendency: torch.Tensor
    skin_temperature_tendency: torch.Tensor
    energy_residual: torch.Tensor

@dataclass(frozen=True)
class EnergyLimitedSnowMeltResult:
    melt_rate: torch.Tensor
    potential_melt_rate: torch.Tensor
    energy_limited_melt_rate: torch.Tensor
    available_energy_flux: torch.Tensor


def energy_limited_snow_melt_rate(
    *,
    state: LandSurfaceState,
    land_fraction: torch.Tensor,
    fluxes: AtmosSurfaceFluxPacket,
    params: LandSurfaceParameters,
    ground_heat_flux_down: torch.Tensor,
    potential_melt_rate: torch.Tensor,
) -> EnergyLimitedSnowMeltResult:
    """Limit potential snow melt to energy available at the skin surface."""

    ref = state.skin_temperature
    land = land_fraction.to(dtype=ref.dtype, device=ref.device)
    active = land > 0.0
    net_radiation = (
        fluxes.net_shortwave.to(dtype=ref.dtype, device=ref.device)
        + fluxes.net_longwave.to(dtype=ref.dtype, device=ref.device)
    )
    turbulent = (
        fluxes.sensible_heat.to(dtype=ref.dtype, device=ref.device)
        + fluxes.latent_heat.to(dtype=ref.dtype, device=ref.device)
    )
    ground = ground_heat_flux_down.to(dtype=ref.dtype, device=ref.device)
    available = net_radiation - turbulent - ground
    energy_limited = available.clamp_min(0.0) / float(params.latent_heat_fusion)
    potential = potential_melt_rate.to(dtype=ref.dtype, device=ref.device).clamp_min(0.0)
    melt = torch.minimum(potential, energy_limited)

    zero = torch.zeros_like(ref)
    return EnergyLimitedSnowMeltResult(
        torch.where(active, melt, zero),
        torch.where(active, potential, zero),
        torch.where(active, energy_limited, zero),
        torch.where(active, available, zero),
    )


def solve_surface_energy_balance(
    state: LandSurfaceState,
    *,
    land_fraction: torch.Tensor,
    fluxes: AtmosSurfaceFluxPacket,
    dt: torch.Tensor,
    params: LandSurfaceParameters,
    ground_heat_flux_down: torch.Tensor,
    snow_melt_rate: torch.Tensor,
) -> SurfaceEnergyBalanceResult:
    """Advance land skin temperature with a local explicit energy balance.

    Sign convention: radiative fluxes are positive downward into the surface;
    sensible, latent, ground, and snow-melt energy fluxes are positive upward
    or away from the skin reservoir.
    """

    ref = state.skin_temperature
    land = land_fraction.to(dtype=ref.dtype, device=ref.device)
    active = land > 0.0
    dt = dt.to(dtype=ref.dtype, device=ref.device).clamp_min(torch.finfo(ref.dtype).tiny)
    net_shortwave = fluxes.net_shortwave.to(dtype=ref.dtype, device=ref.device)
    net_longwave = fluxes.net_longwave.to(dtype=ref.dtype, device=ref.device)
    sensible = fluxes.sensible_heat.to(dtype=ref.dtype, device=ref.device)
    latent = fluxes.latent_heat.to(dtype=ref.dtype, device=ref.device)
    ground = ground_heat_flux_down.to(dtype=ref.dtype, device=ref.device)
    melt = snow_melt_rate.to(dtype=ref.dtype, device=ref.device).clamp_min(0.0)

    net_radiation = net_shortwave + net_longwave
    turbulent = sensible + latent
    melt_energy = melt * float(params.latent_heat_fusion)
    surface_energy = net_radiation - turbulent - ground - melt_energy
    temperature_tendency = surface_energy / float(params.skin_heat_capacity)
    skin_next = ref + dt * temperature_tendency
    residual = net_radiation - turbulent - ground - melt_energy - surface_energy

    zero = torch.zeros_like(ref)
    return SurfaceEnergyBalanceResult(
        torch.where(active, skin_next, ref),
        torch.where(active, net_radiation, zero),
        torch.where(active, turbulent, zero),
        torch.where(active, ground, zero),
        torch.where(active, melt_energy, zero),
        torch.where(active, surface_energy, zero),
        torch.where(active, temperature_tendency, zero),
        torch.where(active, residual, zero),
    )
