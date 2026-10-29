"""VFS-O forcing, flux, and Ekman surface physics."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vfs.core.registry import register_component

from .constants import OCEAN_CP, OCEAN_EOS_MODES, OCEAN_RHO0, OCEAN_S0
from .eos import jmd95_seawater_density, linear_seawater_density
from .domain import OceanDomain
from .helpers import (
    _cell_midlayer_spacing,
    _edge_column_field,
    _edge_east_unit,
    _edge_midlayer_spacing,
    _normalize_tracer_tensor,
    _ocean_vertical_momentum_stress_tendency,
)
from .mixed_layer import density_threshold_mixed_layer_depth, mixed_layer_interface_weight
from .state import OceanState, OceanStateRequirement


class OceanSurfaceForcing(nn.Module):
    """Idealized wind, drag, and surface scalar forcing for VFS-O smokes."""

    recipe_name = "ocean_surface_forcing"

    def __init__(
        self,
        *,
        domain: OceanDomain | None = None,
        wind_stress_amplitude: float = 0.05,
        wind_stress_pattern: str = "zonal_sin2lat",
        temperature_relaxation_timescale: float = 30.0 * 86400.0,
        surface_temperature_equator: float = 26.0,
        surface_temperature_pole: float = 4.0,
        salinity_relaxation_timescale: float = 90.0 * 86400.0,
        surface_salinity_mean: float = 35.0,
        surface_salinity_meridional_amplitude: float = 0.0,
        linear_drag_timescale: float = 0.0,
        bottom_drag_timescale: float = 20.0 * 86400.0,
        rho0: float = OCEAN_RHO0,
    ) -> None:
        super().__init__()
        self.domain = domain
        self.wind_stress_amplitude = float(wind_stress_amplitude)
        self.wind_stress_pattern = str(wind_stress_pattern)
        self.temperature_relaxation_timescale = float(temperature_relaxation_timescale)
        self.surface_temperature_equator = float(surface_temperature_equator)
        self.surface_temperature_pole = float(surface_temperature_pole)
        self.salinity_relaxation_timescale = float(salinity_relaxation_timescale)
        self.surface_salinity_mean = float(surface_salinity_mean)
        self.surface_salinity_meridional_amplitude = float(surface_salinity_meridional_amplitude)
        self.linear_drag_timescale = float(linear_drag_timescale)
        self.bottom_drag_timescale = float(bottom_drag_timescale)
        self.rho0 = float(rho0)
        if self.wind_stress_pattern not in ("none", "zonal_sin2lat"):
            raise ValueError("wind_stress_pattern must be 'none' or 'zonal_sin2lat'")
        for name in (
            "temperature_relaxation_timescale",
            "salinity_relaxation_timescale",
            "linear_drag_timescale",
            "bottom_drag_timescale",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")

    def _cell_lat(self, state: OceanState) -> torch.Tensor:
        xyz = state.mesh.cell_xyz.to(dtype=state.h.dtype, device=state.h.device)
        return torch.asin(xyz[:, 2:3].clamp(-1.0, 1.0))

    def _edge_lat_lon(self, state: OceanState) -> tuple[torch.Tensor, torch.Tensor]:
        xyz = state.mesh.edge_xyz.to(dtype=state.u_e.dtype, device=state.u_e.device)
        lat = torch.asin(xyz[:, 2:3].clamp(-1.0, 1.0))
        lon = torch.atan2(xyz[:, 1:2], xyz[:, 0:1])
        return lat, lon

    def wind_stress_east(self, state: OceanState) -> torch.Tensor:
        """Return prescribed eastward wind stress at edges in N m-2."""

        lat, _lon = self._edge_lat_lon(state)
        if self.wind_stress_pattern == "none" or self.wind_stress_amplitude == 0.0:
            return torch.zeros_like(lat)
        return float(self.wind_stress_amplitude) * torch.sin(2.0 * lat)

    def wind_stress_edge(self, state: OceanState) -> torch.Tensor:
        """Return edge-normal wind stress in N m-2."""

        tau_east = self.wind_stress_east(state)
        _lat, lon = self._edge_lat_lon(state)
        east = torch.cat([-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)], dim=-1)
        normal = state.mesh.edge_normal.to(dtype=state.u_e.dtype, device=state.u_e.device)
        tau = tau_east * (east * normal).sum(dim=-1, keepdim=True)
        return tau if self.domain is None else self.domain.apply_edge(tau)

    def surface_temperature_target(self, state: OceanState) -> torch.Tensor:
        """Return top-layer restoring temperature target at cells."""

        lat = self._cell_lat(state)
        weight = torch.cos(lat).square()
        return float(self.surface_temperature_pole) + (
            float(self.surface_temperature_equator) - float(self.surface_temperature_pole)
        ) * weight

    def surface_salinity_target(self, state: OceanState) -> torch.Tensor:
        """Return top-layer restoring salinity target at cells."""

        lat = self._cell_lat(state)
        return float(self.surface_salinity_mean) + float(self.surface_salinity_meridional_amplitude) * torch.sin(lat)

    def required_state_requirement(self) -> OceanStateRequirement:
        return OceanStateRequirement()

    def tendency(self, state: OceanState) -> OceanState:
        """Return forcing tendencies as an ``OceanState``-shaped RHS."""

        if self.domain is not None:
            self.domain.check_compatible(state)
        h = state.h
        u_t = torch.zeros_like(state.u_e)
        h_t = torch.zeros_like(h)
        temp_h_t = torch.zeros_like(state.temp_h)
        salt_h_t = torch.zeros_like(state.salt_h)

        tau_e = self.wind_stress_edge(state)
        if self.wind_stress_pattern != "none" and self.wind_stress_amplitude != 0.0:
            h_top_edge = state.mesh.cell_to_edge(h[..., :1]).clamp_min(torch.finfo(h.dtype).tiny)
            u_t[..., :1] = u_t[..., :1] + tau_e / (float(self.rho0) * h_top_edge)

        if self.linear_drag_timescale > 0.0:
            u_t = u_t - state.u_e / float(self.linear_drag_timescale)
        if self.bottom_drag_timescale > 0.0:
            u_t[..., -1:] = u_t[..., -1:] - state.u_e[..., -1:] / float(self.bottom_drag_timescale)

        if self.temperature_relaxation_timescale > 0.0:
            target = self.surface_temperature_target(state)
            tendency = (target - state.temperature()[..., :1]) / float(self.temperature_relaxation_timescale)
            temp_h_t[..., :1] = temp_h_t[..., :1] + h[..., :1] * tendency

        if self.salinity_relaxation_timescale > 0.0:
            target = self.surface_salinity_target(state)
            tendency = (target - state.salinity()[..., :1]) / float(self.salinity_relaxation_timescale)
            salt_h_t[..., :1] = salt_h_t[..., :1] + h[..., :1] * tendency

        tracer_h_t = None
        if state.tracer_h is not None:
            tracer_h_t = torch.zeros_like(_normalize_tracer_tensor(state.tracer_h, state.mesh.nCells, state.column.n_layers))
        if self.domain is not None:
            u_t = self.domain.apply_edge(u_t)
            h_t = self.domain.apply_cell(h_t)
            temp_h_t = self.domain.apply_cell(temp_h_t)
            salt_h_t = self.domain.apply_cell(salt_h_t)
            if tracer_h_t is not None:
                tracer_h_t = self.domain.apply_cell(tracer_h_t)
        return state.with_tensors(u_t, h_t, temp_h_t, salt_h_t, tracer_h_t)


class OceanSurfaceFluxPhysics(nn.Module):
    """Flux-form surface physics for idealized VFS-O ocean smokes.

    Heat flux is positive downward into the ocean. Freshwater flux is positive
    downward and is applied as a virtual salt flux, so it changes salinity
    content but not layer thickness.
    """

    recipe_name = "ocean_surface_flux_physics"

    def __init__(
        self,
        *,
        domain: OceanDomain | None = None,
        wind_stress_amplitude: float = 0.05,
        wind_stress_pattern: str = "zonal_sin2lat",
        surface_heat_flux_mean: float = 0.0,
        surface_heat_flux_amplitude: float = 50.0,
        surface_heat_flux_pattern: str = "cos2_lat_anomaly",
        freshwater_flux_mean: float = 0.0,
        freshwater_flux_amplitude: float = 0.0,
        freshwater_flux_pattern: str = "cos2_lat_anomaly",
        reference_salinity: float = OCEAN_S0,
        linear_drag_timescale: float = 0.0,
        bottom_drag_timescale: float = 20.0 * 86400.0,
        mixed_layer_depth: float = 0.0,
        mixed_layer_scalar_diffusivity: float = 0.0,
        mixed_layer_momentum_viscosity: float = 0.0,
        rho0: float = OCEAN_RHO0,
        heat_capacity: float = OCEAN_CP,
    ) -> None:
        super().__init__()
        self.domain = domain
        self.wind_stress_amplitude = float(wind_stress_amplitude)
        self.wind_stress_pattern = str(wind_stress_pattern)
        self.surface_heat_flux_mean = float(surface_heat_flux_mean)
        self.surface_heat_flux_amplitude = float(surface_heat_flux_amplitude)
        self.surface_heat_flux_pattern = str(surface_heat_flux_pattern)
        self.freshwater_flux_mean = float(freshwater_flux_mean)
        self.freshwater_flux_amplitude = float(freshwater_flux_amplitude)
        self.freshwater_flux_pattern = str(freshwater_flux_pattern)
        self.reference_salinity = float(reference_salinity)
        self.linear_drag_timescale = float(linear_drag_timescale)
        self.bottom_drag_timescale = float(bottom_drag_timescale)
        self.mixed_layer_depth = float(mixed_layer_depth)
        self.mixed_layer_scalar_diffusivity = float(mixed_layer_scalar_diffusivity)
        self.mixed_layer_momentum_viscosity = float(mixed_layer_momentum_viscosity)
        self.rho0 = float(rho0)
        self.heat_capacity = float(heat_capacity)
        if self.wind_stress_pattern not in ("none", "zonal_sin2lat"):
            raise ValueError("wind_stress_pattern must be 'none' or 'zonal_sin2lat'")
        for name in ("surface_heat_flux_pattern", "freshwater_flux_pattern"):
            if getattr(self, name) not in ("none", "cos2_lat_anomaly", "sin_lat"):
                raise ValueError(f"{name} must be 'none', 'cos2_lat_anomaly', or 'sin_lat'")
        for name in (
            "linear_drag_timescale",
            "bottom_drag_timescale",
            "mixed_layer_depth",
            "mixed_layer_scalar_diffusivity",
            "mixed_layer_momentum_viscosity",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.rho0 <= 0.0:
            raise ValueError("rho0 must be positive")
        if self.heat_capacity <= 0.0:
            raise ValueError("heat_capacity must be positive")

    def _cell_lat(self, state: OceanState) -> torch.Tensor:
        xyz = state.mesh.cell_xyz.to(dtype=state.h.dtype, device=state.h.device)
        return torch.asin(xyz[:, 2:3].clamp(-1.0, 1.0))

    def _edge_lat_lon(self, state: OceanState) -> tuple[torch.Tensor, torch.Tensor]:
        xyz = state.mesh.edge_xyz.to(dtype=state.u_e.dtype, device=state.u_e.device)
        lat = torch.asin(xyz[:, 2:3].clamp(-1.0, 1.0))
        lon = torch.atan2(xyz[:, 1:2], xyz[:, 0:1])
        return lat, lon

    def _area_mean_zero(self, state: OceanState, field: torch.Tensor) -> torch.Tensor:
        area = state.mesh.cell_area.to(dtype=field.dtype, device=field.device)[:, None]
        if self.domain is None:
            return field - (field * area).sum() / area.sum().clamp_min(torch.finfo(field.dtype).tiny)
        wet = self.domain.cell_mask(field)
        wet_area = area * wet
        mean = (field * wet_area).sum() / wet_area.sum().clamp_min(torch.finfo(field.dtype).tiny)
        return (field - mean) * wet

    def _cell_pattern(self, state: OceanState, pattern: str) -> torch.Tensor:
        lat = self._cell_lat(state)
        if pattern == "none":
            return torch.zeros_like(lat)
        if pattern == "cos2_lat_anomaly":
            return self._area_mean_zero(state, torch.cos(lat).square())
        if pattern == "sin_lat":
            return self._area_mean_zero(state, torch.sin(lat))
        raise AssertionError(f"unsupported surface flux pattern {pattern!r}")

    def wind_stress_east(self, state: OceanState) -> torch.Tensor:
        """Return prescribed eastward wind stress at edges in N m-2."""

        lat, _lon = self._edge_lat_lon(state)
        if self.wind_stress_pattern == "none" or self.wind_stress_amplitude == 0.0:
            return torch.zeros_like(lat)
        return float(self.wind_stress_amplitude) * torch.sin(2.0 * lat)

    def wind_stress_edge(self, state: OceanState) -> torch.Tensor:
        """Return edge-normal wind stress in N m-2."""

        tau_east = self.wind_stress_east(state)
        _lat, lon = self._edge_lat_lon(state)
        east = torch.cat([-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)], dim=-1)
        normal = state.mesh.edge_normal.to(dtype=state.u_e.dtype, device=state.u_e.device)
        tau = tau_east * (east * normal).sum(dim=-1, keepdim=True)
        return tau if self.domain is None else self.domain.apply_edge(tau)

    def surface_heat_flux(self, state: OceanState) -> torch.Tensor:
        """Return downward surface heat flux in W m-2."""

        pattern = self._cell_pattern(state, self.surface_heat_flux_pattern)
        flux = float(self.surface_heat_flux_mean) + float(self.surface_heat_flux_amplitude) * pattern
        return flux if self.domain is None else self.domain.apply_cell(flux)

    def freshwater_flux(self, state: OceanState) -> torch.Tensor:
        """Return downward freshwater flux in m s-1."""

        pattern = self._cell_pattern(state, self.freshwater_flux_pattern)
        flux = float(self.freshwater_flux_mean) + float(self.freshwater_flux_amplitude) * pattern
        return flux if self.domain is None else self.domain.apply_cell(flux)

    def surface_temperature_content_tendency(self, state: OceanState) -> torch.Tensor:
        """Return top-layer ``h*T`` tendency from surface heat flux."""

        return self.surface_heat_flux(state) / (float(self.rho0) * float(self.heat_capacity))

    def surface_salinity_content_tendency(self, state: OceanState) -> torch.Tensor:
        """Return top-layer virtual-salt ``h*S`` tendency from freshwater flux."""

        return -float(self.reference_salinity) * self.freshwater_flux(state)

    def required_state_requirement(self) -> OceanStateRequirement:
        return OceanStateRequirement()

    def _mixed_layer_cell_mask(self, state: OceanState) -> torch.Tensor:
        if self.mixed_layer_depth <= 0.0 or int(state.column.n_layers) < 2:
            return torch.zeros(state.h.shape[:-1] + (int(state.column.n_layers) - 1,), dtype=state.h.dtype, device=state.h.device)
        interface_depth = state.column.interface_depth(state.h)[..., 1:-1]
        mask = (interface_depth <= float(self.mixed_layer_depth)).to(dtype=state.h.dtype)
        return mask if self.domain is None else self.domain.apply_cell(mask)

    def _mixed_layer_edge_mask(self, state: OceanState) -> torch.Tensor:
        if self.mixed_layer_depth <= 0.0 or int(state.column.n_layers) < 2:
            return torch.zeros(state.u_e.shape[:-1] + (int(state.column.n_layers) - 1,), dtype=state.u_e.dtype, device=state.u_e.device)
        h_edge = state.mesh.cell_to_edge(state.h)
        interface_depth = torch.cumsum(h_edge, dim=-1)[..., :-1]
        mask = (interface_depth <= float(self.mixed_layer_depth)).to(dtype=state.u_e.dtype)
        return mask if self.domain is None else self.domain.apply_edge(mask)

    def mixed_layer_scalar_diffusion_tendency(self, state: OceanState, content: torch.Tensor) -> torch.Tensor:
        """Return conservative mixed-layer scalar diffusion tendency."""

        kappa = float(self.mixed_layer_scalar_diffusivity)
        if kappa == 0.0 or int(state.column.n_layers) < 2:
            return torch.zeros_like(content)
        h = state.h
        spacing = _cell_midlayer_spacing(state)
        mask = self._mixed_layer_cell_mask(state)
        if content.ndim == h.ndim:
            q = content / h.clamp_min(torch.finfo(h.dtype).tiny)
            flux = kappa * mask * (q[..., 1:] - q[..., :-1]) / spacing
            out = torch.zeros_like(content)
            out[..., :-1] += flux
            out[..., 1:] -= flux
            return out
        if content.ndim == h.ndim + 1:
            h_b = h[..., None].clamp_min(torch.finfo(h.dtype).tiny)
            q = content / h_b
            flux = kappa * mask[..., None] * (q[..., 1:, :] - q[..., :-1, :]) / spacing[..., None]
            out = torch.zeros_like(content)
            out[..., :-1, :] += flux
            out[..., 1:, :] -= flux
            return out
        raise ValueError(
            "mixed-layer scalar diffusion expects content with shape "
            "[..., nCells, nLayers] or [..., nCells, nLayers, nTracers]"
        )

    def mixed_layer_velocity_diffusion_tendency(self, state: OceanState) -> torch.Tensor:
        """Return mixed-layer vertical viscosity tendency for edge velocity."""

        nu = float(self.mixed_layer_momentum_viscosity)
        if nu == 0.0 or int(state.column.n_layers) < 2:
            return torch.zeros_like(state.u_e)
        h_edge = state.mesh.cell_to_edge(state.h).clamp_min(torch.finfo(state.h.dtype).tiny)
        spacing = _edge_midlayer_spacing(h_edge)
        mask = self._mixed_layer_edge_mask(state)
        stress = nu * mask * (state.u_e[..., 1:] - state.u_e[..., :-1]) / spacing
        out = torch.zeros_like(state.u_e)
        out[..., :-1] += stress / h_edge[..., :-1]
        out[..., 1:] -= stress / h_edge[..., 1:]
        return out

    def flux_diagnostics(self, state: OceanState) -> dict[str, torch.Tensor]:
        """Return tensor diagnostics for the applied surface fluxes."""

        heat_flux = self.surface_heat_flux(state)
        freshwater = self.freshwater_flux(state)
        return {
            "wind_stress_edge": self.wind_stress_edge(state),
            "surface_heat_flux": heat_flux,
            "freshwater_flux": freshwater,
            "temperature_content_tendency": heat_flux / (float(self.rho0) * float(self.heat_capacity)),
            "salinity_content_tendency": -float(self.reference_salinity) * freshwater,
        }

    def tendency(self, state: OceanState) -> OceanState:
        """Return surface-flux physics tendencies as an ``OceanState`` RHS."""

        if self.domain is not None:
            self.domain.check_compatible(state)
        h = state.h
        u_t = torch.zeros_like(state.u_e)
        h_t = torch.zeros_like(h)
        temp_h_t = torch.zeros_like(state.temp_h)
        salt_h_t = torch.zeros_like(state.salt_h)

        tau_e = self.wind_stress_edge(state)
        if self.wind_stress_pattern != "none" and self.wind_stress_amplitude != 0.0:
            h_top_edge = state.mesh.cell_to_edge(h[..., :1]).clamp_min(torch.finfo(h.dtype).tiny)
            u_t[..., :1] = u_t[..., :1] + tau_e / (float(self.rho0) * h_top_edge)

        if self.linear_drag_timescale > 0.0:
            u_t = u_t - state.u_e / float(self.linear_drag_timescale)
        if self.bottom_drag_timescale > 0.0:
            u_t[..., -1:] = u_t[..., -1:] - state.u_e[..., -1:] / float(self.bottom_drag_timescale)

        temp_h_t[..., :1] = temp_h_t[..., :1] + self.surface_temperature_content_tendency(state)
        salt_h_t[..., :1] = salt_h_t[..., :1] + self.surface_salinity_content_tendency(state)

        temp_h_t = temp_h_t + self.mixed_layer_scalar_diffusion_tendency(state, state.temp_h)
        salt_h_t = salt_h_t + self.mixed_layer_scalar_diffusion_tendency(state, state.salt_h)
        u_t = u_t + self.mixed_layer_velocity_diffusion_tendency(state)

        tracer_h_t = None
        if state.tracer_h is not None:
            tracer_h = _normalize_tracer_tensor(state.tracer_h, state.mesh.nCells, state.column.n_layers)
            tracer_h_t = self.mixed_layer_scalar_diffusion_tendency(state, tracer_h)
        if self.domain is not None:
            u_t = self.domain.apply_edge(u_t)
            h_t = self.domain.apply_cell(h_t)
            temp_h_t = self.domain.apply_cell(temp_h_t)
            salt_h_t = self.domain.apply_cell(salt_h_t)
            if tracer_h_t is not None:
                tracer_h_t = self.domain.apply_cell(tracer_h_t)
        return state.with_tensors(u_t, h_t, temp_h_t, salt_h_t, tracer_h_t)


class OceanBulkMixedLayerPhysics(nn.Module):
    """Simple diagnostic bulk mixed-layer vertical mixing for VFS-O.

    This is a column-local parameterization, not a full KPP or turbulence
    closure. It diagnoses a mixed-layer depth from near-surface density,
    enhances vertical scalar/momentum mixing over that depth, and always mixes
    statically unstable interfaces. All tendencies are conservative for layer
    contents and column-integrated momentum.
    """

    recipe_name = "ocean_bulk_mixed_layer_physics"

    def __init__(
        self,
        *,
        domain: OceanDomain | None = None,
        surface_flux_component: nn.Module | None = None,
        density_threshold: float = 0.03,
        min_mixed_layer_depth: float = 10.0,
        max_mixed_layer_depth: float = 250.0,
        background_scalar_diffusivity: float = 0.0,
        background_momentum_viscosity: float = 0.0,
        mixed_layer_scalar_diffusivity: float = 1.0e-2,
        mixed_layer_momentum_viscosity: float = 1.0e-2,
        convective_scalar_diffusivity: float = 1.0e-2,
        convective_momentum_viscosity: float = 1.0e-2,
        wind_mixing_coefficient: float = 0.2,
        max_wind_scalar_diffusivity: float = 1.0e-2,
        max_wind_momentum_viscosity: float = 1.0e-2,
        density_mode: str = "linear",
        convective_adjustment_timescale: float = 6.0 * 3600.0,
        convective_adjustment_passes: int | None = None,
        redistribute_surface_stress: bool = True,
        rho0: float = OCEAN_RHO0,
    ) -> None:
        super().__init__()
        self.domain = domain
        self.surface_flux_component = surface_flux_component
        self.density_threshold = float(density_threshold)
        self.min_mixed_layer_depth = float(min_mixed_layer_depth)
        self.max_mixed_layer_depth = float(max_mixed_layer_depth)
        self.background_scalar_diffusivity = float(background_scalar_diffusivity)
        self.background_momentum_viscosity = float(background_momentum_viscosity)
        self.mixed_layer_scalar_diffusivity = float(mixed_layer_scalar_diffusivity)
        self.mixed_layer_momentum_viscosity = float(mixed_layer_momentum_viscosity)
        self.convective_scalar_diffusivity = float(convective_scalar_diffusivity)
        self.convective_momentum_viscosity = float(convective_momentum_viscosity)
        self.wind_mixing_coefficient = float(wind_mixing_coefficient)
        self.max_wind_scalar_diffusivity = float(max_wind_scalar_diffusivity)
        self.max_wind_momentum_viscosity = float(max_wind_momentum_viscosity)
        self.density_mode = str(density_mode)
        self.convective_adjustment_timescale = float(convective_adjustment_timescale)
        self.convective_adjustment_passes = None if convective_adjustment_passes is None else int(convective_adjustment_passes)
        self.redistribute_surface_stress = bool(redistribute_surface_stress)
        self.rho0 = float(rho0)
        if self.density_mode not in OCEAN_EOS_MODES:
            raise ValueError(f"density_mode must be one of {OCEAN_EOS_MODES}, got {self.density_mode!r}")
        if self.convective_adjustment_passes is not None and self.convective_adjustment_passes < 1:
            raise ValueError("convective_adjustment_passes must be >= 1 when provided")
        for name in (
            "density_threshold",
            "min_mixed_layer_depth",
            "max_mixed_layer_depth",
            "background_scalar_diffusivity",
            "background_momentum_viscosity",
            "mixed_layer_scalar_diffusivity",
            "mixed_layer_momentum_viscosity",
            "convective_scalar_diffusivity",
            "convective_momentum_viscosity",
            "wind_mixing_coefficient",
            "max_wind_scalar_diffusivity",
            "max_wind_momentum_viscosity",
            "convective_adjustment_timescale",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.max_mixed_layer_depth <= 0.0:
            raise ValueError("max_mixed_layer_depth must be positive")
        if self.min_mixed_layer_depth > self.max_mixed_layer_depth:
            raise ValueError("min_mixed_layer_depth must be <= max_mixed_layer_depth")
        if self.rho0 <= 0.0:
            raise ValueError("rho0 must be positive")

    def required_state_requirement(self) -> OceanStateRequirement:
        return OceanStateRequirement()

    def density(self, state: OceanState) -> torch.Tensor:
        """Return the density used by the bulk mixed-layer diagnosis."""

        return self._density_from_contents(state, state.temp_h, state.salt_h)

    def _density_from_contents(
        self,
        state: OceanState,
        temp_h: torch.Tensor,
        salt_h: torch.Tensor,
    ) -> torch.Tensor:
        h_safe = state.h.clamp_min(torch.finfo(state.h.dtype).tiny)
        temperature = temp_h / h_safe
        salinity = salt_h / h_safe
        if self.density_mode == "linear":
            return linear_seawater_density(temperature, salinity, rho0=self.rho0)
        if self.density_mode == "jmd95":
            mid = state.column.mid_depth(state.h)
            pressure_dbar = float(self.rho0) * float(state.column.gravity) * mid / 1.0e4
            return jmd95_seawater_density(salinity, temperature, pressure_dbar)
        raise AssertionError(f"unsupported bulk mixed-layer density mode {self.density_mode!r}")

    def diagnostic_mixed_layer_depth(self, state: OceanState) -> torch.Tensor:
        """Return first-crossing density-threshold mixed-layer depth in meters."""

        rho = self.density(state)
        mid = state.column.mid_depth(state.h)
        depth = density_threshold_mixed_layer_depth(
            rho,
            mid,
            density_threshold=float(self.density_threshold),
            keepdim=True,
        )
        column_depth = state.h.sum(dim=-1, keepdim=True)
        min_depth = torch.minimum(torch.full_like(depth, float(self.min_mixed_layer_depth)), column_depth)
        max_depth = torch.minimum(torch.full_like(depth, float(self.max_mixed_layer_depth)), column_depth)
        depth = torch.minimum(torch.maximum(depth, min_depth), max_depth)
        return depth if self.domain is None else self.domain.apply_cell(depth)

    def mixed_layer_interface_mask(self, state: OceanState) -> torch.Tensor:
        if int(state.column.n_layers) < 2:
            return torch.zeros(state.h.shape[:-1] + (0,), dtype=state.h.dtype, device=state.h.device)
        interface_depth = state.column.interface_depth(state.h)[..., 1:-1]
        mask = mixed_layer_interface_weight(
            interface_depth,
            self.diagnostic_mixed_layer_depth(state),
            transition_thickness=_cell_midlayer_spacing(state),
        )
        return mask if self.domain is None else self.domain.apply_cell(mask)

    def convective_interface_mask(self, state: OceanState) -> torch.Tensor:
        if int(state.column.n_layers) < 2:
            return torch.zeros(state.h.shape[:-1] + (0,), dtype=state.h.dtype, device=state.h.device)
        rho = self.density(state)
        # Stable ocean stratification has density increasing downward. Mix
        # interfaces where denser water overlies lighter water.
        mask = (rho[..., :-1] > rho[..., 1:]).to(dtype=state.h.dtype)
        return mask if self.domain is None else self.domain.apply_cell(mask)

    def convective_adjustment_depth(self, state: OceanState) -> torch.Tensor:
        if int(state.column.n_layers) < 2:
            return torch.zeros_like(state.h[..., :1])
        interface_depth = state.column.interface_depth(state.h)[..., 1:-1]
        unstable = self.convective_interface_mask(state) > 0.0
        depth = torch.where(unstable, interface_depth, torch.zeros_like(interface_depth)).max(dim=-1, keepdim=True).values
        return depth if self.domain is None else self.domain.apply_cell(depth)

    def unstable_column_mask(self, state: OceanState) -> torch.Tensor:
        if int(state.column.n_layers) < 2:
            return torch.zeros_like(state.h[..., :1])
        mask = self.convective_interface_mask(state).amax(dim=-1, keepdim=True)
        return mask if self.domain is None else self.domain.apply_cell(mask)

    def convectively_adjusted_contents(
        self,
        state: OceanState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return conservatively adjusted scalar contents for unstable columns."""

        n_layers = int(state.column.n_layers)
        if n_layers < 2:
            return state.temp_h, state.salt_h, state.tracer_h
        temp_h = state.temp_h
        salt_h = state.salt_h
        tracer_h = None if state.tracer_h is None else _normalize_tracer_tensor(
            state.tracer_h,
            state.mesh.nCells,
            state.column.n_layers,
        )
        passes = n_layers if self.convective_adjustment_passes is None else int(self.convective_adjustment_passes)
        for _ in range(passes):
            temp_h, salt_h, tracer_h = self._convective_adjustment_pair_sweep(
                state,
                temp_h,
                salt_h,
                tracer_h,
                parity=0,
            )
            temp_h, salt_h, tracer_h = self._convective_adjustment_pair_sweep(
                state,
                temp_h,
                salt_h,
                tracer_h,
                parity=1,
            )
        return temp_h, salt_h, tracer_h

    def _convective_adjustment_pair_sweep(
        self,
        state: OceanState,
        temp_h: torch.Tensor,
        salt_h: torch.Tensor,
        tracer_h: torch.Tensor | None,
        *,
        parity: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Mix one set of non-overlapping unstable layer pairs.

        Even/odd pair sweeps keep the adjustment conservative pair-by-pair
        while avoiding Python loops and repeated full-column concatenations.
        """

        n_layers = int(state.column.n_layers)
        lower = torch.arange(int(parity), n_layers - 1, 2, dtype=torch.long, device=state.h.device)
        if int(lower.numel()) == 0:
            return temp_h, salt_h, tracer_h
        upper = lower + 1
        h = state.h
        rho = self._density_from_contents(state, temp_h, salt_h)
        unstable = rho.index_select(-1, lower) > rho.index_select(-1, upper)
        if self.domain is not None:
            unstable = self.domain.apply_cell(unstable.to(dtype=h.dtype)) > 0.0

        h0 = h.index_select(-1, lower)
        h1 = h.index_select(-1, upper)
        pair_h = (h0 + h1).clamp_min(torch.finfo(h.dtype).tiny)

        temp0 = temp_h.index_select(-1, lower)
        temp1 = temp_h.index_select(-1, upper)
        salt0 = salt_h.index_select(-1, lower)
        salt1 = salt_h.index_select(-1, upper)
        temp_mean = (temp0 + temp1) / pair_h
        salt_mean = (salt0 + salt1) / pair_h

        adjusted_temp_h = temp_h.clone()
        adjusted_salt_h = salt_h.clone()
        adjusted_temp_h[..., lower] = torch.where(unstable, h0 * temp_mean, temp0)
        adjusted_temp_h[..., upper] = torch.where(unstable, h1 * temp_mean, temp1)
        adjusted_salt_h[..., lower] = torch.where(unstable, h0 * salt_mean, salt0)
        adjusted_salt_h[..., upper] = torch.where(unstable, h1 * salt_mean, salt1)

        if tracer_h is None:
            return adjusted_temp_h, adjusted_salt_h, None

        tracer0 = tracer_h.index_select(-2, lower)
        tracer1 = tracer_h.index_select(-2, upper)
        tracer_mean = (tracer0 + tracer1) / pair_h[..., None]
        unstable_tracer = unstable[..., None]
        adjusted_tracer_h = tracer_h.clone()
        adjusted_tracer_h[..., lower, :] = torch.where(unstable_tracer, h0[..., None] * tracer_mean, tracer0)
        adjusted_tracer_h[..., upper, :] = torch.where(unstable_tracer, h1[..., None] * tracer_mean, tracer1)
        return adjusted_temp_h, adjusted_salt_h, adjusted_tracer_h

    def convective_adjustment_tendency(
        self,
        state: OceanState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.convective_adjustment_timescale <= 0.0 or int(state.column.n_layers) < 2:
            tracer_h_t = None if state.tracer_h is None else torch.zeros_like(
                _normalize_tracer_tensor(state.tracer_h, state.mesh.nCells, state.column.n_layers)
            )
            return torch.zeros_like(state.temp_h), torch.zeros_like(state.salt_h), tracer_h_t
        adjusted_temp_h, adjusted_salt_h, adjusted_tracer_h = self.convectively_adjusted_contents(state)
        scale = 1.0 / float(self.convective_adjustment_timescale)
        tracer_h_t = None
        if state.tracer_h is not None and adjusted_tracer_h is not None:
            tracer_h = _normalize_tracer_tensor(state.tracer_h, state.mesh.nCells, state.column.n_layers)
            tracer_h_t = (adjusted_tracer_h - tracer_h) * scale
        return (adjusted_temp_h - state.temp_h) * scale, (adjusted_salt_h - state.salt_h) * scale, tracer_h_t

    def _wind_mixing_diffusivity(self, state: OceanState, *, edge: bool = False) -> torch.Tensor:
        component = self.surface_flux_component
        if component is None or not hasattr(component, "wind_stress_edge") or self.wind_mixing_coefficient == 0.0:
            shape = state.u_e.shape[:-1] + (1,) if edge else state.h.shape[:-1] + (1,)
            like = state.u_e if edge else state.h
            return torch.zeros(shape, dtype=like.dtype, device=like.device)
        tau_edge = component.wind_stress_edge(state).abs()
        if edge:
            tau = tau_edge.to(dtype=state.u_e.dtype, device=state.u_e.device)
            depth = state.mesh.cell_to_edge(self.diagnostic_mixed_layer_depth(state)).to(
                dtype=state.u_e.dtype,
                device=state.u_e.device,
            )
        else:
            tau = state.mesh.edge_to_cell(tau_edge).to(dtype=state.h.dtype, device=state.h.device)
            depth = self.diagnostic_mixed_layer_depth(state)
        ustar = (tau / float(self.rho0)).clamp_min(0.0).sqrt()
        diffusivity = float(self.wind_mixing_coefficient) * ustar * depth
        cap = float(self.max_wind_momentum_viscosity if edge else self.max_wind_scalar_diffusivity)
        diffusivity = diffusivity.clamp_max(cap)
        return diffusivity

    def _edge_mixed_layer_overlap(self, state: OceanState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_edge = state.mesh.cell_to_edge(state.h).clamp_min(torch.finfo(state.h.dtype).tiny)
        mld_edge = state.mesh.cell_to_edge(self.diagnostic_mixed_layer_depth(state)).to(
            dtype=state.u_e.dtype,
            device=state.u_e.device,
        )
        layer_bottom = torch.cumsum(h_edge, dim=-1)
        layer_top = layer_bottom - h_edge
        overlap = (torch.minimum(layer_bottom, mld_edge) - layer_top).clamp_min(0.0)
        overlap = torch.minimum(overlap, h_edge)
        effective_depth = overlap.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(h_edge.dtype).tiny)
        layer_weight = overlap / h_edge
        return h_edge, layer_weight, effective_depth

    def surface_stress_redistribution_tendency(self, state: OceanState) -> torch.Tensor:
        """Return correction from top-layer stress to mixed-layer-mass stress."""

        component = self.surface_flux_component
        if not self.redistribute_surface_stress or component is None or not hasattr(component, "wind_stress_edge"):
            return torch.zeros_like(state.u_e)
        tau = component.wind_stress_edge(state).to(dtype=state.u_e.dtype, device=state.u_e.device)
        h_edge, layer_weight, effective_depth = self._edge_mixed_layer_overlap(state)
        distributed = layer_weight * tau / (float(self.rho0) * effective_depth)
        top_only = torch.zeros_like(state.u_e)
        top_only[..., :1] = tau / (float(self.rho0) * h_edge[..., :1])
        correction = distributed - top_only
        return correction if self.domain is None else self.domain.apply_edge(correction)

    def scalar_interface_diffusivity(self, state: OceanState) -> torch.Tensor:
        if int(state.column.n_layers) < 2:
            return torch.zeros(state.h.shape[:-1] + (0,), dtype=state.h.dtype, device=state.h.device)
        ml_mask = self.mixed_layer_interface_mask(state)
        conv_mask = self.convective_interface_mask(state)
        wind = self._wind_mixing_diffusivity(state, edge=False)
        kappa = (
            torch.full_like(ml_mask, float(self.background_scalar_diffusivity))
            + ml_mask * (float(self.mixed_layer_scalar_diffusivity) + wind)
            + conv_mask * float(self.convective_scalar_diffusivity)
        )
        return kappa if self.domain is None else self.domain.apply_cell(kappa)

    def momentum_interface_viscosity(self, state: OceanState) -> torch.Tensor:
        if int(state.column.n_layers) < 2:
            return torch.zeros(state.u_e.shape[:-1] + (0,), dtype=state.u_e.dtype, device=state.u_e.device)
        h_edge = state.mesh.cell_to_edge(state.h)
        interface_depth = torch.cumsum(h_edge, dim=-1)[..., :-1]
        mld_edge = state.mesh.cell_to_edge(self.diagnostic_mixed_layer_depth(state)).to(
            dtype=state.u_e.dtype,
            device=state.u_e.device,
        )
        ml_mask = mixed_layer_interface_weight(
            interface_depth,
            mld_edge,
            transition_thickness=_edge_midlayer_spacing(h_edge),
        )
        conv_mask = state.mesh.cell_to_edge(self.convective_interface_mask(state)).to(
            dtype=state.u_e.dtype,
            device=state.u_e.device,
        )
        wind = self._wind_mixing_diffusivity(state, edge=True)
        viscosity = (
            torch.full_like(ml_mask, float(self.background_momentum_viscosity))
            + ml_mask * (float(self.mixed_layer_momentum_viscosity) + wind)
            + conv_mask * float(self.convective_momentum_viscosity)
        )
        return viscosity if self.domain is None else self.domain.apply_edge(viscosity)

    def scalar_diffusion_tendency(self, state: OceanState, content: torch.Tensor) -> torch.Tensor:
        """Return conservative scalar-content tendency from mixed-layer mixing."""

        if int(state.column.n_layers) < 2:
            return torch.zeros_like(content)
        h = state.h
        spacing = _cell_midlayer_spacing(state)
        kappa = self.scalar_interface_diffusivity(state)
        if content.ndim == h.ndim:
            q = content / h.clamp_min(torch.finfo(h.dtype).tiny)
            flux = kappa * (q[..., 1:] - q[..., :-1]) / spacing
            out = torch.zeros_like(content)
            out[..., :-1] += flux
            out[..., 1:] -= flux
            return out
        if content.ndim == h.ndim + 1:
            q = content / h[..., None].clamp_min(torch.finfo(h.dtype).tiny)
            flux = kappa[..., None] * (q[..., 1:, :] - q[..., :-1, :]) / spacing[..., None]
            out = torch.zeros_like(content)
            out[..., :-1, :] += flux
            out[..., 1:, :] -= flux
            return out
        raise ValueError(
            "mixed-layer scalar diffusion expects content with shape "
            "[..., nCells, nLayers] or [..., nCells, nLayers, nTracers]"
        )

    def momentum_diffusion_tendency(self, state: OceanState) -> torch.Tensor:
        """Return conservative column-momentum vertical viscosity tendency."""

        if int(state.column.n_layers) < 2:
            return torch.zeros_like(state.u_e)
        h_edge = state.mesh.cell_to_edge(state.h).clamp_min(torch.finfo(state.h.dtype).tiny)
        spacing = _edge_midlayer_spacing(h_edge)
        viscosity = self.momentum_interface_viscosity(state)
        stress = viscosity * (state.u_e[..., 1:] - state.u_e[..., :-1]) / spacing
        out = torch.zeros_like(state.u_e)
        out[..., :-1] += stress / h_edge[..., :-1]
        out[..., 1:] -= stress / h_edge[..., 1:]
        return out

    def mixed_layer_diagnostics(self, state: OceanState) -> dict[str, torch.Tensor]:
        stress_correction = self.surface_stress_redistribution_tendency(state)
        _h_edge, _layer_weight, stress_effective_depth = self._edge_mixed_layer_overlap(state)
        return {
            "mixed_layer_depth": self.diagnostic_mixed_layer_depth(state),
            "scalar_interface_diffusivity": self.scalar_interface_diffusivity(state),
            "momentum_interface_viscosity": self.momentum_interface_viscosity(state),
            "convective_interface_mask": self.convective_interface_mask(state),
            "unstable_column_mask": self.unstable_column_mask(state),
            "convective_adjustment_depth": self.convective_adjustment_depth(state),
            "surface_stress_effective_depth": stress_effective_depth,
            "surface_stress_redistribution_tendency": stress_correction,
        }

    def tendency(self, state: OceanState) -> OceanState:
        if self.domain is not None:
            self.domain.check_compatible(state)
        u_t = self.momentum_diffusion_tendency(state)
        h_t = torch.zeros_like(state.h)
        temp_h_t = self.scalar_diffusion_tendency(state, state.temp_h)
        salt_h_t = self.scalar_diffusion_tendency(state, state.salt_h)
        adjustment_temp_h_t, adjustment_salt_h_t, adjustment_tracer_h_t = self.convective_adjustment_tendency(state)
        temp_h_t = temp_h_t + adjustment_temp_h_t
        salt_h_t = salt_h_t + adjustment_salt_h_t
        u_t = u_t + self.surface_stress_redistribution_tendency(state)
        tracer_h_t = None
        if state.tracer_h is not None:
            tracer_h_t = self.scalar_diffusion_tendency(
                state,
                _normalize_tracer_tensor(state.tracer_h, state.mesh.nCells, state.column.n_layers),
            )
            if adjustment_tracer_h_t is not None:
                tracer_h_t = tracer_h_t + adjustment_tracer_h_t
        if self.domain is not None:
            u_t = self.domain.apply_edge(u_t)
            h_t = self.domain.apply_cell(h_t)
            temp_h_t = self.domain.apply_cell(temp_h_t)
            salt_h_t = self.domain.apply_cell(salt_h_t)
            if tracer_h_t is not None:
                tracer_h_t = self.domain.apply_cell(tracer_h_t)
        return state.with_tensors(u_t, h_t, temp_h_t, salt_h_t, tracer_h_t)


class CompositeOceanPhysics(nn.Module):
    """Additive composition of OceanState-shaped physics tendencies."""

    recipe_name = "composite_ocean_physics"

    def __init__(self, *components: nn.Module) -> None:
        super().__init__()
        if not components:
            raise ValueError("CompositeOceanPhysics requires at least one component")
        self.components = nn.ModuleList(list(components))

    def required_state_requirement(self) -> OceanStateRequirement:
        return OceanStateRequirement()

    def tendency(self, state: OceanState) -> OceanState:
        total = state.zeros_like()
        for component in self.components:
            rhs = component.tendency(state)
            if state.tracer_h is not None and rhs.tracer_h is None:
                rhs = rhs.with_tensors(rhs.u_e, rhs.h, rhs.temp_h, rhs.salt_h, torch.zeros_like(state.tracer_h))
            total = total + rhs
        return total

    def _first(self, method_name: str, state: OceanState):
        for component in self.components:
            method = getattr(component, method_name, None)
            if method is not None:
                return method(state)
        return None

    def surface_temperature_target(self, state: OceanState):
        return self._first("surface_temperature_target", state)

    def surface_salinity_target(self, state: OceanState):
        return self._first("surface_salinity_target", state)

    def wind_stress_edge(self, state: OceanState):
        return self._first("wind_stress_edge", state)

    def surface_heat_flux(self, state: OceanState):
        return self._first("surface_heat_flux", state)

    def freshwater_flux(self, state: OceanState):
        return self._first("freshwater_flux", state)

    def mixed_layer_diagnostics(self, state: OceanState) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for component in self.components:
            method = getattr(component, "mixed_layer_diagnostics", None)
            if method is not None:
                out.update(method(state))
        return out


class OceanEkmanSurfacePhysics(nn.Module):
    """Ocean-side Ekman/surface-stress physics for VFS-O research recipes.

    This component is intentionally narrow: it computes an edge-normal surface
    stress and applies it as the top boundary flux of vertical momentum stress.
    It can also add vertical and horizontal momentum viscosity plus simple
    linear/bottom damping. Heat, freshwater, and scalar physics remain owned by
    the existing flux/restoring components.
    """

    recipe_name = "ocean_ekman_surface_physics"

    def __init__(
        self,
        *,
        domain: OceanDomain | None = None,
        stress_mode: str = "analytic",
        wind_stress_amplitude: float = 0.05,
        wind_stress_pattern: str = "zonal_sin2lat",
        prescribed_stress_edge: Any | None = None,
        air_density: float = 1.225,
        drag_coefficient: float = 1.3e-3,
        vertical_momentum_viscosity: float = 0.0,
        horizontal_momentum_viscosity: float = 0.0,
        linear_drag_timescale: float = 0.0,
        bottom_drag_timescale: float = 20.0 * 86400.0,
        rho0: float = OCEAN_RHO0,
    ) -> None:
        super().__init__()
        self.domain = domain
        self.stress_mode = str(stress_mode)
        self.wind_stress_amplitude = float(wind_stress_amplitude)
        self.wind_stress_pattern = str(wind_stress_pattern)
        self.prescribed_stress_edge = prescribed_stress_edge
        self.air_density = float(air_density)
        self.drag_coefficient = float(drag_coefficient)
        self.vertical_momentum_viscosity = float(vertical_momentum_viscosity)
        self.horizontal_momentum_viscosity = float(horizontal_momentum_viscosity)
        self.linear_drag_timescale = float(linear_drag_timescale)
        self.bottom_drag_timescale = float(bottom_drag_timescale)
        self.rho0 = float(rho0)
        if self.stress_mode not in ("analytic", "prescribed", "relative"):
            raise ValueError("stress_mode must be 'analytic', 'prescribed', or 'relative'")
        if self.wind_stress_pattern not in ("none", "zonal_sin2lat"):
            raise ValueError("wind_stress_pattern must be 'none' or 'zonal_sin2lat'")
        for name in (
            "air_density",
            "drag_coefficient",
            "vertical_momentum_viscosity",
            "horizontal_momentum_viscosity",
            "linear_drag_timescale",
            "bottom_drag_timescale",
            "rho0",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.rho0 <= 0.0:
            raise ValueError("rho0 must be positive")

    def _edge_lat_lon(self, state: OceanState) -> tuple[torch.Tensor, torch.Tensor]:
        xyz = state.mesh.edge_xyz.to(dtype=state.u_e.dtype, device=state.u_e.device)
        lat = torch.asin(xyz[:, 2:3].clamp(-1.0, 1.0))
        lon = torch.atan2(xyz[:, 1:2], xyz[:, 0:1])
        return lat, lon

    def analytic_wind_stress_edge(self, state: OceanState) -> torch.Tensor:
        """Return prescribed analytic edge-normal stress in N m-2."""

        lat, lon = self._edge_lat_lon(state)
        if self.wind_stress_pattern == "none" or self.wind_stress_amplitude == 0.0:
            return torch.zeros((state.mesh.nEdges, 1), dtype=state.u_e.dtype, device=state.u_e.device)
        tau_east = float(self.wind_stress_amplitude) * torch.sin(2.0 * lat)
        east = _edge_east_unit(lon)
        normal = state.mesh.edge_normal.to(dtype=state.u_e.dtype, device=state.u_e.device)
        tau = tau_east * (east * normal).sum(dim=-1, keepdim=True)
        return tau if self.domain is None else self.domain.apply_edge(tau)

    def prescribed_wind_stress_edge(self, state: OceanState) -> torch.Tensor:
        """Return externally supplied edge-normal stress in N m-2."""

        value = self.prescribed_stress_edge
        if value is None:
            value = dict(state.attrs or {}).get("surface_stress_edge")
        if value is None:
            raise ValueError("prescribed ocean stress requires prescribed_stress_edge or state.attrs['surface_stress_edge']")
        tau = _edge_column_field(value, state.u_e[..., :1], name="surface_stress_edge")
        return tau if self.domain is None else self.domain.apply_edge(tau)

    def relative_wind_edge(self, state: OceanState) -> torch.Tensor:
        """Return edge-normal air-minus-ocean relative wind in m s-1."""

        attrs = dict(state.attrs or {})
        if "atmospheric_wind_edge" in attrs:
            wind_edge = _edge_column_field(attrs["atmospheric_wind_edge"], state.u_e[..., :1], name="atmospheric_wind_edge")
        elif "atmospheric_wind_xyz_edge" in attrs:
            wind_xyz = torch.as_tensor(attrs["atmospheric_wind_xyz_edge"], dtype=state.u_e.dtype, device=state.u_e.device)
            if wind_xyz.shape != (state.mesh.nEdges, 3):
                raise ValueError(
                    "state.attrs['atmospheric_wind_xyz_edge'] must have shape "
                    f"({state.mesh.nEdges}, 3); got {tuple(wind_xyz.shape)}"
                )
            normal = state.mesh.edge_normal.to(dtype=state.u_e.dtype, device=state.u_e.device)
            wind_edge = (wind_xyz * normal).sum(dim=-1, keepdim=True)
        else:
            raise ValueError(
                "relative wind stress requires state.attrs['atmospheric_wind_edge'] "
                "or state.attrs['atmospheric_wind_xyz_edge']"
            )
        return wind_edge - state.u_e[..., :1]

    def relative_wind_stress_edge(self, state: OceanState) -> torch.Tensor:
        """Return bulk edge-normal stress from air-ocean relative wind."""

        rel = self.relative_wind_edge(state)
        tau = float(self.air_density) * float(self.drag_coefficient) * rel.abs() * rel
        return tau if self.domain is None else self.domain.apply_edge(tau)

    def wind_stress_edge(self, state: OceanState) -> torch.Tensor:
        """Return active edge-normal surface stress in N m-2."""

        if self.stress_mode == "analytic":
            return self.analytic_wind_stress_edge(state)
        if self.stress_mode == "prescribed":
            return self.prescribed_wind_stress_edge(state)
        if self.stress_mode == "relative":
            return self.relative_wind_stress_edge(state)
        raise AssertionError(f"unsupported stress_mode {self.stress_mode!r}")

    def required_state_requirement(self) -> OceanStateRequirement:
        return OceanStateRequirement()

    def tendency(self, state: OceanState) -> OceanState:
        """Return Ekman/surface-stress tendencies as an ``OceanState`` RHS."""

        return self.tendency_with_surface_stress(state, self.wind_stress_edge(state))

    def tendency_with_surface_stress(
        self,
        state: OceanState,
        surface_stress_edge: Any,
    ) -> OceanState:
        """Return the ocean tendency for an externally averaged surface stress."""

        if self.domain is not None:
            self.domain.check_compatible(state)
        h = state.h
        u_t = torch.zeros_like(state.u_e)
        h_t = torch.zeros_like(h)
        temp_h_t = torch.zeros_like(state.temp_h)
        salt_h_t = torch.zeros_like(state.salt_h)
        tracer_h_t = None if state.tracer_h is None else torch.zeros_like(state.tracer_h)

        tau_surface = _edge_column_field(
            surface_stress_edge,
            state.u_e[..., :1],
            name="surface_stress_edge",
        )
        if self.domain is not None:
            tau_surface = self.domain.apply_edge(tau_surface)
        bottom_stress = None
        if self.bottom_drag_timescale > 0.0:
            h_edge = state.mesh.cell_to_edge(h).clamp_min(torch.finfo(h.dtype).tiny)
            bottom_stress = float(self.rho0) * h_edge[..., -1:] * state.u_e[..., -1:] / float(self.bottom_drag_timescale)
        u_t = u_t + _ocean_vertical_momentum_stress_tendency(
            state,
            viscosity=self.vertical_momentum_viscosity,
            surface_stress_edge=tau_surface,
            bottom_stress_edge=bottom_stress,
            rho0=self.rho0,
        )
        if self.horizontal_momentum_viscosity > 0.0:
            u_t = u_t + float(self.horizontal_momentum_viscosity) * state.mesh.vector_laplacian_edge(state.u_e)
        if self.linear_drag_timescale > 0.0:
            u_t = u_t - state.u_e / float(self.linear_drag_timescale)
        if self.domain is not None:
            u_t = self.domain.apply_edge(u_t)
            h_t = self.domain.apply_cell(h_t)
            temp_h_t = self.domain.apply_cell(temp_h_t)
            salt_h_t = self.domain.apply_cell(salt_h_t)
            if tracer_h_t is not None:
                tracer_h_t = self.domain.apply_cell(tracer_h_t)
        return state.with_tensors(u_t, h_t, temp_h_t, salt_h_t, tracer_h_t)


register_component("ocean_ekman_surface_physics", OceanEkmanSurfacePhysics)
register_component("ocean_bulk_mixed_layer_physics", OceanBulkMixedLayerPhysics)
register_component("composite_ocean_physics", CompositeOceanPhysics)
register_component("ocean_surface_forcing", OceanSurfaceForcing)
register_component("ocean_surface_flux_physics", OceanSurfaceFluxPhysics)
