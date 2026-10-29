"""Thermodynamic sea-ice model with conservative freeze/melt transfers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..core.registry import register_component
from ..surface.conservation import ConservativeTransfer
from ..surface.fractions import SurfaceFractions
from ..surface.utils import as_cell_column, cell_area_column
from .albedo import SeaIceBoundaryProperties, sea_ice_boundary_properties, sea_ice_snow_insulation_factor
from .parameters import SeaIceParameters
from .state import SeaIceState, SeaIceStepResult


class SeaIceThermo(nn.Module):
    """Freeze/melt-only sea-ice thermodynamics.

    No advection, rheology, ridging, or thickness distribution is represented.
    The only job here is to make water/salt/energy ownership explicit.
    """

    recipe_name = "sea_ice_thermo"

    def __init__(
        self,
        *,
        ice_density: float = 917.0,
        latent_heat_fusion: float = 3.34e5,
        ice_formation_thickness: float = 0.5,
        evolve_fraction: bool = True,
        melt_temperature: float = 273.15,
        bare_ice_albedo: float = 0.55,
        snow_albedo: float = 0.80,
        ice_roughness: float = 1.0e-3,
        snow_roughness: float = 2.0e-3,
        snow_cover_swe: float = 10.0,
        snow_insulation_swe_scale: float = 20.0,
    ) -> None:
        super().__init__()
        self.parameters = SeaIceParameters(
            ice_density=float(ice_density),
            latent_heat_fusion=float(latent_heat_fusion),
            ice_formation_thickness=float(ice_formation_thickness),
            melt_temperature=float(melt_temperature),
            bare_ice_albedo=float(bare_ice_albedo),
            snow_albedo=float(snow_albedo),
            ice_roughness=float(ice_roughness),
            snow_roughness=float(snow_roughness),
            snow_cover_swe=float(snow_cover_swe),
            snow_insulation_swe_scale=float(snow_insulation_swe_scale),
        )
        self.parameters.validate()
        self.evolve_fraction = bool(evolve_fraction)

    @property
    def ice_density(self) -> float:
        return self.parameters.ice_density

    @property
    def latent_heat_fusion(self) -> float:
        return self.parameters.latent_heat_fusion

    @property
    def ice_formation_thickness(self) -> float:
        return self.parameters.ice_formation_thickness

    @property
    def melt_temperature(self) -> float:
        return self.parameters.melt_temperature

    @staticmethod
    def freezing_temperature(salinity: torch.Tensor) -> torch.Tensor:
        return -0.054 * salinity

    def boundary_properties(self, state: SeaIceState) -> SeaIceBoundaryProperties:
        return sea_ice_boundary_properties(state, self.parameters)

    def step(
        self,
        state: SeaIceState,
        fractions: SurfaceFractions,
        *,
        sst_celsius: torch.Tensor,
        ocean_salinity: torch.Tensor,
        heat_flux_to_ocean: torch.Tensor,
        dt: float,
        snowfall_rate: torch.Tensor | None = None,
        mesh: Any | None = None,
    ) -> SeaIceStepResult:
        like = state.sea_ice_thickness
        area = cell_area_column(mesh, like)
        ocean_fraction = fractions.ocean_fraction.to(dtype=like.dtype, device=like.device)
        ice_fraction = state.sea_ice_fraction.to(dtype=like.dtype, device=like.device).clamp(0.0, 1.0)
        sst = as_cell_column(sst_celsius, like=like, name="sst_celsius")
        salinity = as_cell_column(ocean_salinity, like=like, name="ocean_salinity").clamp_min(0.0)
        heat = as_cell_column(heat_flux_to_ocean, like=like, name="heat_flux_to_ocean")
        snowfall = (
            torch.zeros_like(like)
            if snowfall_rate is None
            else as_cell_column(snowfall_rate, like=like, name="snowfall_rate").clamp_min(0.0)
        )
        dt_t = torch.as_tensor(float(dt), dtype=like.dtype, device=like.device)
        tiny = torch.finfo(like.dtype).tiny
        latent_heat = float(self.latent_heat_fusion)

        snow_mass = state.snow_on_sea_ice.clamp_min(0.0) * ice_fraction * area
        snowfall_mass = snowfall * dt_t * area * ice_fraction
        melt_energy_available = heat.clamp_min(0.0) * dt_t * area * ice_fraction
        snow_available = snow_mass + snowfall_mass
        snow_melt_mass = torch.minimum(melt_energy_available / latent_heat, snow_available)
        remaining_melt_energy = (melt_energy_available - snow_melt_mass * latent_heat).clamp_min(0.0)

        insulation = sea_ice_snow_insulation_factor(state, self.parameters)
        cooling_factor = torch.where(ice_fraction > 0.0, insulation, torch.ones_like(insulation))
        freezing = (sst < self.freezing_temperature(salinity)) & (heat < 0.0)
        freeze_energy = (
            (-heat).clamp_min(0.0)
            * freezing.to(dtype=like.dtype)
            * cooling_factor
            * dt_t
            * area
            * ocean_fraction
        )
        freeze_mass = freeze_energy / latent_heat
        salinity_fraction = salinity / 1000.0
        freeze_salt = freeze_mass * salinity_fraction

        ice_mass = state.sea_ice_thickness.clamp_min(0.0) * ice_fraction * float(self.ice_density) * area
        ice_melt_mass = torch.minimum(remaining_melt_energy / latent_heat, ice_mass + freeze_mass)
        ice_mass_next = (ice_mass + freeze_mass - ice_melt_mass).clamp_min(0.0)

        fraction_next = ice_fraction
        if self.evolve_fraction:
            target_fraction = ice_mass_next / (
                float(self.ice_density) * area * float(self.ice_formation_thickness)
            )
            fraction_next = torch.where(
                ice_mass_next > 0.0,
                torch.maximum(ice_fraction, target_fraction),
                torch.zeros_like(ice_fraction),
            )
            fraction_next = torch.minimum(fraction_next, ocean_fraction).clamp(0.0, 1.0)
        thickness_next = ice_mass_next / (float(self.ice_density) * area * fraction_next.clamp_min(tiny))
        thickness_next = torch.where(ice_mass_next > 0.0, thickness_next, torch.zeros_like(thickness_next))
        snow_after_melt = (snow_available - snow_melt_mass).clamp_min(0.0)
        retained_snow_mass = torch.where(ice_mass_next > 0.0, snow_after_melt, torch.zeros_like(snow_after_melt))
        snow_area_loss_mass = (snow_after_melt - retained_snow_mass).clamp_min(0.0)
        snow_next = retained_snow_mass / (area * fraction_next.clamp_min(tiny))
        snow_next = torch.where(ice_mass_next > 0.0, snow_next, torch.zeros_like(snow_next))
        fractions_next = fractions.with_overlays(sea_ice_fraction=fraction_next)
        surface_temperature_next = torch.where(
            ice_mass_next > 0.0,
            state.sea_ice_surface_temperature.clamp_max(float(self.melt_temperature)),
            state.sea_ice_surface_temperature,
        )
        next_state = SeaIceState(
            fraction_next,
            thickness_next,
            snow_next,
            surface_temperature_next,
            state.sea_ice_u_e,
            state.sea_ice_stress_xx,
            state.sea_ice_stress_xy,
            state.sea_ice_stress_yy,
        )
        snow_to_ocean_mass = snow_melt_mass + snow_area_loss_mass
        ocean_water_mass = snow_to_ocean_mass + ice_melt_mass - freeze_mass
        ocean_energy = snow_melt_mass * latent_heat + ice_melt_mass * latent_heat - freeze_energy
        flux_denom = (dt_t * area).clamp_min(tiny)
        ocean_heat_flux = ocean_energy / flux_denom
        ocean_freshwater_flux = ocean_water_mass / flux_denom
        ocean_salt_flux = torch.zeros_like(ocean_heat_flux)
        transfers = (
            ConservativeTransfer(
                "snow_to_sea_ice",
                "atmosphere",
                "sea_ice",
                water_mass=snowfall_mass,
            ),
            ConservativeTransfer(
                "sea_ice_freeze_seawater",
                "ocean",
                "sea_ice",
                water_mass=freeze_mass,
                salt_mass=freeze_salt,
                energy=freeze_energy,
            ),
            ConservativeTransfer(
                "sea_ice_brine_rejection",
                "sea_ice",
                "ocean",
                salt_mass=freeze_salt,
            ),
            ConservativeTransfer(
                "sea_ice_snow_melt",
                "sea_ice",
                "ocean",
                water_mass=snow_to_ocean_mass,
                energy=snow_melt_mass * latent_heat,
            ),
            ConservativeTransfer(
                "sea_ice_melt",
                "sea_ice",
                "ocean",
                water_mass=ice_melt_mass,
                energy=ice_melt_mass * latent_heat,
            ),
        )
        diagnostics = {
            "sea_ice_freeze_mass": freeze_mass.sum(),
            "sea_ice_melt_mass": ice_melt_mass.sum(),
            "sea_ice_snowfall_mass": snowfall_mass.sum(),
            "sea_ice_snow_melt_mass": snow_to_ocean_mass.sum(),
            "sea_ice_snow_storage": (next_state.snow_on_sea_ice * fraction_next * area).sum(),
            "sea_ice_ice_storage": (next_state.sea_ice_thickness * fraction_next * float(self.ice_density) * area).sum(),
            "sea_ice_water_storage": next_state.total_water(mesh=mesh, ice_density=self.ice_density),
            "sea_ice_fraction_mean": fraction_next.mean(),
            "sea_ice_ocean_heat_flux": ocean_heat_flux,
            "sea_ice_ocean_freshwater_flux": ocean_freshwater_flux,
            "sea_ice_ocean_salt_flux": ocean_salt_flux,
            "sea_ice_brine_rejection_salt_flux": freeze_salt / flux_denom,
            "sea_ice_snow_insulation_factor": insulation,
        }
        return SeaIceStepResult(
            next_state,
            fractions_next,
            ocean_heat_flux,
            ocean_freshwater_flux,
            ocean_salt_flux,
            transfers,
            diagnostics,
        )


register_component("sea_ice_thermo", SeaIceThermo)


__all__ = ["SeaIceBoundaryProperties", "SeaIceState", "SeaIceStepResult", "SeaIceThermo"]
