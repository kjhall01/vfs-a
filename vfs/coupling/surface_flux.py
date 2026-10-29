"""Thermodynamic air-sea exchange operators."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..core.registry import register_component
from ..core.state import DEFAULT_GRAVITY, State
from ..moisture import saturation_mixing_ratio, vapor_mixing_ratio, vapor_tracer_index
from .momentum import AirSeaMomentumCoupler


class AirSeaSurfaceFluxCoupler(nn.Module):
    """Bulk thermodynamic air-sea exchange on VFS native states.

    Sign conventions are explicit:

    - sensible/latent heat fluxes are positive upward into atmosphere-side
      reservoirs;
    - ocean heat flux is positive downward into the ocean;
    - freshwater flux is positive downward into the ocean and is applied as a
      virtual salt flux by the ocean tendency.

    The default latent-heat mode keeps latent energy in a diagnosed vapor
    reservoir instead of immediately warming dry atmospheric temperature.
    """

    recipe_name = "air_sea_surface_flux_coupler"

    def __init__(
        self,
        *,
        air_density: float = 1.225,
        heat_exchange_coefficient: float = 1.1e-3,
        moisture_exchange_coefficient: float = 1.1e-3,
        atmosphere_level: int = -1,
        wind_speed_min: float = 0.0,
        c_p_air: float = 1004.0,
        latent_heat_vaporization: float = 2.5e6,
        ocean_density: float = 1025.0,
        ocean_heat_capacity: float = 3990.0,
        water_density: float = 1000.0,
        reference_salinity: float = 35.0,
        default_air_mixing_ratio: float = 0.0,
        latent_heat_mode: str = "vapor_reservoir",
        domain: Any | None = None,
    ) -> None:
        super().__init__()
        self.air_density = float(air_density)
        self.heat_exchange_coefficient = float(heat_exchange_coefficient)
        self.moisture_exchange_coefficient = float(moisture_exchange_coefficient)
        self.atmosphere_level = int(atmosphere_level)
        self.wind_speed_min = float(wind_speed_min)
        self.c_p_air = float(c_p_air)
        self.latent_heat_vaporization = float(latent_heat_vaporization)
        self.ocean_density = float(ocean_density)
        self.ocean_heat_capacity = float(ocean_heat_capacity)
        self.water_density = float(water_density)
        self.reference_salinity = float(reference_salinity)
        self.default_air_mixing_ratio = float(default_air_mixing_ratio)
        self.latent_heat_mode = str(latent_heat_mode)
        self.domain = domain
        if self.latent_heat_mode not in ("vapor_reservoir", "temperature"):
            raise ValueError("latent_heat_mode must be 'vapor_reservoir' or 'temperature'")
        for name in (
            "air_density",
            "heat_exchange_coefficient",
            "moisture_exchange_coefficient",
            "wind_speed_min",
            "c_p_air",
            "latent_heat_vaporization",
            "ocean_density",
            "ocean_heat_capacity",
            "water_density",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.c_p_air <= 0.0 or self.ocean_density <= 0.0 or self.ocean_heat_capacity <= 0.0:
            raise ValueError("heat capacities and ocean density must be positive")
        if self.water_density <= 0.0:
            raise ValueError("water_density must be positive")

    def _level_index(self, n_levels: int) -> int:
        level = self.atmosphere_level
        if level < 0:
            level = int(n_levels) + level
        if level < 0 or level >= int(n_levels):
            raise ValueError(
                f"atmosphere_level {self.atmosphere_level} is out of range for {int(n_levels)} levels"
            )
        return int(level)

    @staticmethod
    def _tensor_like(value: Any, like: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(dtype=like.dtype, device=like.device)
        return torch.as_tensor(value, dtype=like.dtype, device=like.device)

    @classmethod
    def _cell_column_field(cls, value: Any, like: torch.Tensor, *, name: str) -> torch.Tensor:
        tensor = cls._tensor_like(value, like)
        if tensor.ndim == 0:
            return torch.ones_like(like) * tensor
        if tensor.ndim == 1:
            tensor = tensor[:, None]
        if tensor.shape == like.shape:
            return tensor
        if (
            tensor.ndim == 2
            and int(tensor.shape[0]) == int(like.shape[-2])
            and int(tensor.shape[1]) == 1
        ):
            return tensor
        raise ValueError(f"{name} must be scalar, [nCells], or [nCells, 1]; got {tuple(tensor.shape)}")

    def _apply_cell_domain(self, value: torch.Tensor) -> torch.Tensor:
        if self.domain is None:
            return value
        return self.domain.apply_cell(value)

    def relative_wind_speed_cell(
        self,
        atmosphere_state: State,
        ocean_state: Any,
        *,
        relative_wind_speed_cell: Any | None = None,
    ) -> torch.Tensor:
        """Return cell-centered air-minus-ocean wind speed in m s-1."""

        like = ocean_state.h[..., :1]
        if relative_wind_speed_cell is not None:
            speed = self._cell_column_field(relative_wind_speed_cell, like, name="relative_wind_speed_cell")
            return self._apply_cell_domain(speed.clamp_min(0.0))
        AirSeaMomentumCoupler._same_native_mesh(atmosphere_state, ocean_state)
        level = self._level_index(int(atmosphere_state.u_e.shape[-1]))
        air_vec = atmosphere_state.mesh.cell_wind_vector(atmosphere_state.u_e[..., level : level + 1])
        ocean_vec = ocean_state.mesh.cell_wind_vector(ocean_state.u_e[..., :1])
        speed2 = (air_vec - ocean_vec).square().sum(dim=-1)
        if self.wind_speed_min > 0.0:
            speed2 = speed2 + float(self.wind_speed_min) ** 2
        return self._apply_cell_domain(speed2.clamp_min(0.0).sqrt())

    def fluxes(
        self,
        atmosphere_state: State,
        ocean_state: Any,
        *,
        relative_wind_speed_cell: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return surface-exchange flux tensors with documented signs."""

        level = self._level_index(int(atmosphere_state.u_e.shape[-1]))
        air_temperature = atmosphere_state.temperature()[..., level : level + 1]
        air_pressure = atmosphere_state.full_pressure()[..., level : level + 1]
        sst_kelvin = ocean_state.temperature()[..., :1] + 273.15
        speed = self.relative_wind_speed_cell(
            atmosphere_state,
            ocean_state,
            relative_wind_speed_cell=relative_wind_speed_cell,
        )
        r_air = vapor_mixing_ratio(atmosphere_state)
        if r_air is None:
            r_air_level = torch.full_like(air_temperature, float(self.default_air_mixing_ratio))
        else:
            r_air_level = r_air[..., level : level + 1].to(dtype=air_temperature.dtype, device=air_temperature.device)
        r_air_level = r_air_level.clamp_min(0.0)
        r_sat_surface = saturation_mixing_ratio(sst_kelvin, air_pressure)

        sensible_up = (
            float(self.air_density)
            * float(self.c_p_air)
            * float(self.heat_exchange_coefficient)
            * speed
            * (sst_kelvin - air_temperature)
        )
        evaporation_up = (
            float(self.air_density)
            * float(self.moisture_exchange_coefficient)
            * speed
            * (r_sat_surface - r_air_level)
        )
        latent_up = float(self.latent_heat_vaporization) * evaporation_up
        temperature_heat_up = sensible_up
        latent_reservoir_up = latent_up
        if self.latent_heat_mode == "temperature":
            temperature_heat_up = sensible_up + latent_up
            latent_reservoir_up = torch.zeros_like(latent_up)
        ocean_heat_down = -(sensible_up + latent_up)
        freshwater_down = -evaporation_up / float(self.water_density)
        return {
            "relative_wind_speed_cell": speed,
            "air_temperature": air_temperature,
            "sea_surface_temperature": sst_kelvin,
            "air_pressure": air_pressure,
            "air_vapor_mixing_ratio": r_air_level,
            "surface_saturation_mixing_ratio": r_sat_surface,
            "sensible_heat_flux_up": self._apply_cell_domain(sensible_up),
            "latent_heat_flux_up": self._apply_cell_domain(latent_up),
            "evaporation_mass_flux_up": self._apply_cell_domain(evaporation_up),
            "atmosphere_temperature_heat_flux_up": self._apply_cell_domain(temperature_heat_up),
            "latent_reservoir_heat_flux_up": self._apply_cell_domain(latent_reservoir_up),
            "ocean_surface_heat_flux_down": self._apply_cell_domain(ocean_heat_down),
            "freshwater_flux_down": self._apply_cell_domain(freshwater_down),
        }

    def atmospheric_tendency(
        self,
        atmosphere_state: State,
        ocean_state: Any | None = None,
        *,
        fluxes: dict[str, torch.Tensor] | None = None,
        relative_wind_speed_cell: Any | None = None,
    ) -> State:
        """Return atmosphere-side bottom-layer heat and vapor tendencies."""

        if fluxes is None:
            if ocean_state is None:
                raise ValueError("ocean_state is required when fluxes are not supplied")
            fluxes = self.fluxes(
                atmosphere_state,
                ocean_state,
                relative_wind_speed_cell=relative_wind_speed_cell,
            )
        level = self._level_index(int(atmosphere_state.u_e.shape[-1]))
        dp = atmosphere_state.layer_dp()[..., level : level + 1]
        pi_full = (atmosphere_state.exner_full()[..., level : level + 1] / float(atmosphere_state.column.c_p)).clamp_min(
            torch.finfo(dp.dtype).tiny
        )
        gravity = float((atmosphere_state.attrs or {}).get("gravity", DEFAULT_GRAVITY))
        heat_up = fluxes["atmosphere_temperature_heat_flux_up"].to(dtype=dp.dtype, device=dp.device)
        dtheta_full = gravity * heat_up / (
            float(self.c_p_air) * dp.clamp_min(torch.finfo(dp.dtype).tiny) * pi_full
        )
        layer_dmtheta = dp * dtheta_full
        dmtheta = torch.zeros_like(atmosphere_state.mtheta_i)
        dmtheta[..., level : level + 1] += 0.5 * layer_dmtheta
        dmtheta[..., level + 1 : level + 2] += 0.5 * layer_dmtheta

        tracer_t = None
        if atmosphere_state.tracer_m is not None:
            tracer_t = torch.zeros_like(atmosphere_state.tracer_m)
            index = vapor_tracer_index(atmosphere_state)
            if index is not None:
                evap = fluxes["evaporation_mass_flux_up"].to(
                    dtype=atmosphere_state.tracer_m.dtype,
                    device=atmosphere_state.tracer_m.device,
                )
                tracer_t[..., level : level + 1, int(index)] = gravity * evap
        return State(
            torch.zeros_like(atmosphere_state.u_e),
            dmtheta,
            torch.zeros_like(atmosphere_state.ps),
            tracer_t,
            mesh=atmosphere_state.mesh,
            column=atmosphere_state.column,
            attrs=atmosphere_state._attrs_copy(),
        )

    def ocean_tendency(
        self,
        ocean_state: Any,
        *,
        fluxes: dict[str, torch.Tensor],
    ) -> Any:
        """Return ocean-side heat and virtual-freshwater tendencies."""

        heat_down = fluxes["ocean_surface_heat_flux_down"].to(dtype=ocean_state.h.dtype, device=ocean_state.h.device)
        freshwater_down = fluxes["freshwater_flux_down"].to(dtype=ocean_state.h.dtype, device=ocean_state.h.device)
        u_t = torch.zeros_like(ocean_state.u_e)
        h_t = torch.zeros_like(ocean_state.h)
        temp_h_t = torch.zeros_like(ocean_state.temp_h)
        salt_h_t = torch.zeros_like(ocean_state.salt_h)
        temp_h_t[..., :1] = heat_down / (float(self.ocean_density) * float(self.ocean_heat_capacity))
        salt_h_t[..., :1] = -float(self.reference_salinity) * freshwater_down
        if self.domain is not None:
            u_t = self.domain.apply_edge(u_t)
            h_t = self.domain.apply_cell(h_t)
            temp_h_t = self.domain.apply_cell(temp_h_t)
            salt_h_t = self.domain.apply_cell(salt_h_t)
        tracer_h_t = None if ocean_state.tracer_h is None else torch.zeros_like(ocean_state.tracer_h)
        return ocean_state.with_tensors(u_t, h_t, temp_h_t, salt_h_t, tracer_h_t)

    def coupled_thermodynamic_tendencies(
        self,
        atmosphere_state: State,
        ocean_state: Any,
        *,
        relative_wind_speed_cell: Any | None = None,
        include_audit: bool = True,
    ) -> dict[str, Any]:
        """Return atmosphere/ocean thermodynamic exchange tendencies."""

        fluxes = self.fluxes(
            atmosphere_state,
            ocean_state,
            relative_wind_speed_cell=relative_wind_speed_cell,
        )
        atmosphere_rhs = self.atmospheric_tendency(atmosphere_state, fluxes=fluxes)
        ocean_rhs = self.ocean_tendency(ocean_state, fluxes=fluxes)
        out: dict[str, Any] = {
            "atmosphere_rhs": atmosphere_rhs,
            "ocean_rhs": ocean_rhs,
            "fluxes": fluxes,
        }
        if include_audit:
            out["flux_summary"] = self.flux_summary(atmosphere_state, ocean_state, fluxes=fluxes)
        return out

    def flux_summary(
        self,
        atmosphere_state: State,
        ocean_state: Any,
        *,
        fluxes: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """Return detached scalar diagnostics for the surface exchange."""

        if fluxes is None:
            fluxes = self.fluxes(atmosphere_state, ocean_state)

        def mean(value: torch.Tensor) -> float:
            return float(value.detach().to(dtype=torch.float32).mean().cpu().item())

        def rms(value: torch.Tensor) -> float:
            return float(value.detach().to(dtype=torch.float32).square().mean().sqrt().cpu().item())

        def max_abs(value: torch.Tensor) -> float:
            return float(value.detach().to(dtype=torch.float32).abs().max().cpu().item())

        energy_residual = (
            fluxes["ocean_surface_heat_flux_down"]
            + fluxes["atmosphere_temperature_heat_flux_up"]
            + fluxes["latent_reservoir_heat_flux_up"]
        )
        water_residual = (
            float(self.water_density) * fluxes["freshwater_flux_down"]
            + fluxes["evaporation_mass_flux_up"]
        )
        heat_scale = rms(fluxes["sensible_heat_flux_up"]) + rms(fluxes["latent_heat_flux_up"])
        water_scale = rms(fluxes["evaporation_mass_flux_up"])
        return {
            "surface_relative_wind_speed_mean": mean(fluxes["relative_wind_speed_cell"]),
            "surface_sensible_heat_flux_up_mean": mean(fluxes["sensible_heat_flux_up"]),
            "surface_latent_heat_flux_up_mean": mean(fluxes["latent_heat_flux_up"]),
            "surface_evaporation_mass_flux_up_mean": mean(fluxes["evaporation_mass_flux_up"]),
            "surface_ocean_heat_flux_down_mean": mean(fluxes["ocean_surface_heat_flux_down"]),
            "surface_freshwater_flux_down_mean": mean(fluxes["freshwater_flux_down"]),
            "surface_energy_residual_rms": rms(energy_residual),
            "surface_energy_residual_max_abs": max_abs(energy_residual),
            "surface_energy_relative_residual": max_abs(energy_residual) / max(heat_scale, 1.0e-300),
            "surface_water_residual_rms": rms(water_residual),
            "surface_water_residual_max_abs": max_abs(water_residual),
            "surface_water_relative_residual": max_abs(water_residual) / max(water_scale, 1.0e-300),
        }


register_component("air_sea_surface_flux_coupler", AirSeaSurfaceFluxCoupler)


__all__ = ["AirSeaSurfaceFluxCoupler"]
