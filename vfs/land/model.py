"""Land-surface model component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from ..core.registry import register_component
from ..surface.conservation import ConservativeTransfer
from ..surface.fractions import SurfaceFractions
from ..surface.packets import AtmosSurfaceFluxPacket
from ..surface.utils import cell_area_column
from .energy_balance import energy_limited_snow_melt_rate, solve_surface_energy_balance
from .hydrology import bucket_water_budget, soil_moisture_availability
from .parameters import LandSurfaceParameters
from .snow import (
    snow_boundary_properties,
    snow_cover_fraction,
    snow_insulation_factor,
    snow_melt_rate,
    update_snow_storage,
)
from .soil_thermal import diffuse_soil_temperature
from .state import LandSurfaceState, LandSurfaceStepResult
from .vegetation import (
    LandUseFields,
    LandUseParameterTable,
    canopy_interception,
    land_use_root_fraction,
)


@dataclass(frozen=True)
class LandBoundaryProperties:
    skin_temperature: torch.Tensor
    albedo: torch.Tensor
    roughness: torch.Tensor
    moisture_availability: torch.Tensor
    snow_cover_fraction: torch.Tensor
    snow_insulation_factor: torch.Tensor
    land_use_type: torch.Tensor
    surface_emissions_factor: torch.Tensor


class LandSurface(nn.Module):
    """Layered land hydrology, soil thermal diffusion, and skin energy balance."""

    recipe_name = "land_surface"

    def __init__(
        self,
        *,
        soil_layer_thickness: tuple[float, ...] = (0.10, 0.30, 0.60, 1.00),
        soil_water_capacity: float = 150.0,
        critical_soil_water: float = 50.0,
        field_capacity_fraction: float = 0.70,
        wilting_fraction: float = 0.15,
        saturated_hydraulic_conductivity: float = 1.0e-5,
        latent_heat_vaporization: float = 2.5e6,
        latent_heat_fusion: float = 3.34e5,
        canopy_water_capacity: float = 0.5,
        melt_temperature: float = 273.15,
        snow_melt_factor: float = 2.0e-5,
        snow_cover_swe: float = 10.0,
        snow_albedo: float = 0.80,
        snow_roughness: float = 1.0e-3,
        snow_insulation_swe_scale: float = 20.0,
        skin_relaxation_timescale: float = 86400.0,
        skin_heat_capacity: float = 2.0e6,
        soil_heat_capacity: float = 2.0e6,
        soil_thermal_conductivity: float = 1.5,
        root_fraction: tuple[float, ...] | None = None,
        land_use_table: LandUseParameterTable | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(land_use_table, Mapping):
            self.land_use_table = LandUseParameterTable(**land_use_table)
        else:
            self.land_use_table = land_use_table or LandUseParameterTable()
        self.parameters = LandSurfaceParameters(
            soil_layer_thickness=soil_layer_thickness,
            soil_water_capacity=float(soil_water_capacity),
            critical_soil_water=float(critical_soil_water),
            field_capacity_fraction=float(field_capacity_fraction),
            wilting_fraction=float(wilting_fraction),
            saturated_hydraulic_conductivity=float(saturated_hydraulic_conductivity),
            latent_heat_vaporization=float(latent_heat_vaporization),
            latent_heat_fusion=float(latent_heat_fusion),
            canopy_water_capacity=float(canopy_water_capacity),
            melt_temperature=float(melt_temperature),
            snow_melt_factor=float(snow_melt_factor),
            snow_cover_swe=float(snow_cover_swe),
            snow_albedo=float(snow_albedo),
            snow_roughness=float(snow_roughness),
            snow_insulation_swe_scale=float(snow_insulation_swe_scale),
            skin_relaxation_timescale=float(skin_relaxation_timescale),
            skin_heat_capacity=float(skin_heat_capacity),
            soil_heat_capacity=float(soil_heat_capacity),
            soil_thermal_conductivity=float(soil_thermal_conductivity),
            root_fraction=root_fraction,
        )
        self.parameters.validate()

    @property
    def soil_water_capacity(self) -> float:
        return self.parameters.soil_water_capacity

    @property
    def n_soil_layers(self) -> int:
        return self.parameters.n_soil_layers

    @property
    def critical_soil_water(self) -> float:
        return self.parameters.critical_soil_water

    @property
    def latent_heat_vaporization(self) -> float:
        return self.parameters.latent_heat_vaporization

    @property
    def melt_temperature(self) -> float:
        return self.parameters.melt_temperature

    @property
    def snow_melt_factor(self) -> float:
        return self.parameters.snow_melt_factor

    @property
    def skin_relaxation_timescale(self) -> float:
        return self.parameters.skin_relaxation_timescale

    def initial_state(
        self,
        n_cells: int,
        *,
        soil_water: float = 50.0,
        soil_temperature: float = 280.0,
        snow_water_equivalent: float = 0.0,
        skin_temperature: float = 280.0,
        canopy_water: float = 0.0,
        land_use_type: Any = 0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> LandSurfaceState:
        encoded_land_use = self.land_use_table.encode(land_use_type, n_cells=n_cells, device=device)
        return LandSurfaceState.uniform(
            n_cells,
            soil_water=soil_water,
            soil_temperature=soil_temperature,
            snow_water_equivalent=snow_water_equivalent,
            skin_temperature=skin_temperature,
            canopy_water=canopy_water,
            land_use_type=encoded_land_use,
            soil_layer_thickness=self.parameters.soil_layer_thickness,
            dtype=dtype,
            device=device,
        )

    def land_use_fields(self, state: LandSurfaceState) -> LandUseFields:
        return self.land_use_table.fields(state.land_use_type, like=state.soil_water)

    def root_fraction(self, state: LandSurfaceState) -> torch.Tensor:
        return land_use_root_fraction(self.land_use_fields(state), self.parameters, state.soil_liquid_water)

    def moisture_availability(
        self,
        state: LandSurfaceState,
        *,
        land_use: LandUseFields | None = None,
        root_fraction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        land_use = self.land_use_fields(state) if land_use is None else land_use
        root_fraction = (
            land_use_root_fraction(land_use, self.parameters, state.soil_liquid_water)
            if root_fraction is None
            else root_fraction
        )
        return (
            soil_moisture_availability(state, self.parameters, root_fraction=root_fraction)
            * land_use.evaporation_factor
        ).clamp(0.0, 1.0)

    def beta(self, state: LandSurfaceState) -> torch.Tensor:
        return self.moisture_availability(state)

    def boundary_properties(
        self,
        state: LandSurfaceState,
        *,
        base_albedo: float | torch.Tensor | None = None,
        base_roughness: float | torch.Tensor | None = None,
    ) -> LandBoundaryProperties:
        land_use = self.land_use_fields(state)
        snow = snow_boundary_properties(
            state,
            self.parameters,
            base_albedo=land_use.albedo if base_albedo is None else base_albedo,
            base_roughness=land_use.roughness_length if base_roughness is None else base_roughness,
        )
        return LandBoundaryProperties(
            state.skin_temperature,
            snow.albedo,
            snow.roughness,
            self.moisture_availability(state, land_use=land_use),
            snow.cover_fraction,
            snow.insulation_factor,
            state.land_use_type,
            land_use.emissions_factor,
        )

    def step(
        self,
        state: LandSurfaceState,
        fractions: SurfaceFractions,
        fluxes: AtmosSurfaceFluxPacket,
        dt: float,
        *,
        mesh: Any | None = None,
    ) -> LandSurfaceStepResult:
        like = state.soil_water
        land = fractions.land_fraction.to(dtype=like.dtype, device=like.device)
        area = cell_area_column(mesh, like)
        dt_t = torch.as_tensor(float(dt), dtype=like.dtype, device=like.device)
        land_use = self.land_use_fields(state)
        root_fraction = land_use_root_fraction(land_use, self.parameters, state.soil_liquid_water)
        beta = self.moisture_availability(state, land_use=land_use, root_fraction=root_fraction)

        canopy = canopy_interception(
            state,
            land_fraction=land,
            rain_rate=fluxes.precipitation_rain,
            snow_rate=fluxes.precipitation_snow,
            latent_heat_flux=fluxes.latent_heat,
            dt=dt_t,
            params=self.parameters,
            land_use_fields=land_use,
        )
        potential_melt = snow_melt_rate(state, land_fraction=land, dt=dt_t, params=self.parameters)
        thermal = diffuse_soil_temperature(
            state,
            land_fraction=land,
            dt=dt_t,
            params=self.parameters,
            top_conductance_factor=snow_insulation_factor(state, self.parameters),
        )
        melt_limit = energy_limited_snow_melt_rate(
            state=state,
            land_fraction=land,
            fluxes=fluxes,
            params=self.parameters,
            ground_heat_flux_down=thermal.conductive_flux_down[:, :1],
            potential_melt_rate=potential_melt,
        )
        melt = melt_limit.melt_rate
        energy = solve_surface_energy_balance(
            state,
            land_fraction=land,
            fluxes=fluxes,
            dt=dt_t,
            params=self.parameters,
            ground_heat_flux_down=thermal.conductive_flux_down[:, :1],
            snow_melt_rate=melt,
        )
        snow_next, snow = update_snow_storage(
            state,
            land_fraction=land,
            snowfall_rate=canopy.throughfall_snow_rate,
            melt_rate=melt,
            dt=dt_t,
        )
        hydrology = bucket_water_budget(
            state,
            land_fraction=land,
            rain_rate=canopy.throughfall_rain_rate,
            melt_rate=melt,
            latent_heat_flux=fluxes.latent_heat,
            dt=dt_t,
            cell_area=area,
            params=self.parameters,
            root_fraction=root_fraction,
            potential_evaporation_mass=canopy.remaining_evaporation_mass,
        )
        next_state = LandSurfaceState(
            hydrology.soil_liquid_water,
            thermal.soil_temperature,
            snow_next,
            energy.skin_temperature,
            state.soil_ice,
            canopy.canopy_water,
            state.land_use_type,
        )

        rain = fluxes.precipitation_rain.to(dtype=like.dtype, device=like.device).clamp_min(0.0)
        snow_input = fluxes.precipitation_snow.to(dtype=like.dtype, device=like.device).clamp_min(0.0)
        evaporation_rate = canopy.canopy_evaporation_rate + hydrology.evaporation_rate
        evaporation_mass = evaporation_rate * dt_t * area * land
        transfers = (
            ConservativeTransfer(
                "rain_to_land",
                "atmosphere",
                "land",
                water_mass=rain * dt_t * area * land,
            ),
            ConservativeTransfer(
                "snow_to_land",
                "atmosphere",
                "land",
                water_mass=snow_input * dt_t * area * land,
            ),
            ConservativeTransfer(
                "land_evaporation",
                "land",
                "atmosphere",
                water_mass=evaporation_mass,
                energy=evaporation_mass * float(self.parameters.latent_heat_vaporization),
            ),
            ConservativeTransfer(
                "land_runoff",
                "land",
                "runoff",
                water_mass=hydrology.runoff_mass,
            ),
        )
        diagnostics = {
            "land_beta_mean": (beta * land).sum() / land.sum().clamp_min(torch.finfo(like.dtype).tiny),
            "land_vegetation_evaporation_factor": land_use.evaporation_factor,
            "land_vegetation_root_fraction": root_fraction,
            "land_surface_emissions_factor": land_use.emissions_factor,
            "land_runoff_mass": hydrology.runoff_mass.sum(),
            "land_surface_runoff_mass": hydrology.surface_runoff_mass.sum(),
            "land_drainage_mass": hydrology.drainage_mass.sum(),
            "land_infiltration_mass": (hydrology.infiltration_rate * dt_t * area * land).sum(),
            "land_evaporation_mass": evaporation_mass.sum(),
            "land_canopy_evaporation_mass": (canopy.canopy_evaporation_rate * dt_t * area * land).sum(),
            "land_soil_evaporation_mass": (hydrology.evaporation_rate * dt_t * area * land).sum(),
            "land_canopy_storage": (next_state.canopy_water * land * area).sum(),
            "land_rain_interception_mass": (canopy.rain_interception_rate * dt_t * area * land).sum(),
            "land_snow_interception_mass": (canopy.snow_interception_rate * dt_t * area * land).sum(),
            "land_throughfall_rain_mass": (canopy.throughfall_rain_rate * dt_t * area * land).sum(),
            "land_throughfall_snow_mass": (canopy.throughfall_snow_rate * dt_t * area * land).sum(),
            "land_snow_melt_mass": (melt * dt_t * area * land).sum(),
            "land_snow_storage": (next_state.snow_water_equivalent * land * area).sum(),
            "land_snow_cover_fraction": snow_cover_fraction(next_state, self.parameters),
            "land_soil_energy_tendency": thermal.soil_energy_tendency,
            "land_soil_conductive_flux_down": thermal.conductive_flux_down,
            "land_surface_net_radiation": energy.net_radiation,
            "land_surface_turbulent_heat_flux_up": energy.turbulent_heat_flux_up,
            "land_surface_ground_heat_flux_down": energy.ground_heat_flux_down,
            "land_surface_snow_melt_energy_flux": energy.snow_melt_energy_flux,
            "land_surface_available_melt_energy_flux": melt_limit.available_energy_flux,
            "land_potential_snow_melt_mass": (melt_limit.potential_melt_rate * dt_t * area * land).sum(),
            "land_energy_limited_snow_melt_mass": (
                melt_limit.energy_limited_melt_rate * dt_t * area * land
            ).sum(),
            "land_surface_energy_tendency": energy.surface_energy_tendency,
            "land_skin_temperature_tendency": energy.skin_temperature_tendency,
            "land_surface_energy_residual": (energy.energy_residual * area * land).sum(),
            "land_water_storage": next_state.total_water(fractions, mesh=mesh),
            "land_root_zone_water": (
                next_state.soil_liquid_water * root_fraction
            ).sum(dim=1, keepdim=True),
        }
        return LandSurfaceStepResult(next_state, hydrology.runoff_mass, transfers, diagnostics)


register_component("land_surface", LandSurface)
