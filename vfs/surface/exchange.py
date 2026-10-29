"""Surface boundary semantics for coupled VFS atmosphere/ocean experiments."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from ..core.registry import register_component


def _as_cell_column(value: Any, *, n_cells: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype)
    if tensor.ndim == 0:
        return torch.full((int(n_cells), 1), float(tensor.item()), dtype=dtype)
    if tensor.ndim == 1:
        tensor = tensor[:, None]
    if tuple(tensor.shape) != (int(n_cells), 1):
        raise ValueError(f"surface field must have shape ({int(n_cells)},) or ({int(n_cells)}, 1); got {tuple(tensor.shape)}")
    return tensor


def _cell_east_north_components(mesh: Any, u_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    wind = mesh.cell_wind_vector(u_e)
    xyz = mesh.cell_xyz.to(dtype=u_e.dtype, device=u_e.device)
    lat = torch.asin(xyz[:, 2].clamp(-1.0, 1.0))
    lon = torch.atan2(xyz[:, 1], xyz[:, 0])
    east = torch.stack([-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)], dim=-1)
    north = torch.stack(
        [-torch.sin(lat) * torch.cos(lon), -torch.sin(lat) * torch.sin(lon), torch.cos(lat)],
        dim=-1,
    )
    return (wind * east[:, None, :]).sum(dim=-1), (wind * north[:, None, :]).sum(dim=-1)


class SurfaceBoundary(nn.Module):
    """Static surface-type and boundary-reservoir metadata on the native mesh.

    ``ocean_cell`` is the routing mask. Ocean cells send recognized SCM
    ocean-side fluxes into ``OceanState``. Land cells use a prescribed skin
    temperature, zero ocean current, and diagnosed fluxes in this first pass.
    """

    recipe_name = "surface_boundary"

    def __init__(
        self,
        ocean_cell: Any,
        *,
        mesh: Any | None = None,
        land_skin_temperature: Any = 288.0,
        surface_elevation_m: Any | None = None,
        bathymetry_m: Any | None = None,
        land_evaporation_factor: Any = 0.05,
    ) -> None:
        super().__init__()
        ocean = torch.as_tensor(ocean_cell, dtype=torch.bool)
        if ocean.ndim == 2 and int(ocean.shape[-1]) == 1:
            ocean = ocean[:, 0]
        ocean = ocean.reshape(-1)
        n_cells = int(ocean.numel())
        if n_cells <= 0:
            raise ValueError("SurfaceBoundary requires at least one cell")
        self.mesh = mesh
        self.n_cells = n_cells
        self.register_buffer("ocean_cell", ocean[:, None])
        self.register_buffer("land_cell", (~ocean)[:, None])
        self.register_buffer("surface_type", ocean.to(dtype=torch.float64)[:, None])

        skin = _as_cell_column(land_skin_temperature, n_cells=n_cells)
        if not torch.isfinite(skin).all().item():
            raise ValueError("land_skin_temperature must be finite")
        self.register_buffer("land_skin_temperature", skin)

        elevation = torch.zeros((n_cells, 1), dtype=torch.float64) if surface_elevation_m is None else _as_cell_column(
            surface_elevation_m,
            n_cells=n_cells,
        )
        if not torch.isfinite(elevation).all().item():
            raise ValueError("surface_elevation_m must be finite")
        bathy = torch.zeros((n_cells, 1), dtype=torch.float64) if bathymetry_m is None else _as_cell_column(
            bathymetry_m,
            n_cells=n_cells,
        )
        if not torch.isfinite(bathy).all().item():
            raise ValueError("bathymetry_m must be finite")
        bathy = torch.where(ocean[:, None], bathy.clamp_min(0.0), torch.zeros_like(bathy))
        self.register_buffer("surface_elevation_m", elevation)
        self.register_buffer("topography_m", elevation.clamp_min(0.0))
        self.register_buffer("bathymetry_m", bathy)

        evap = _as_cell_column(land_evaporation_factor, n_cells=n_cells)
        if not torch.isfinite(evap).all().item():
            raise ValueError("land_evaporation_factor must be finite")
        self.register_buffer("land_evaporation_factor", evap.clamp_min(0.0))

        if mesh is not None:
            cells = mesh.cellsOnEdge.detach().cpu().to(dtype=torch.long)
            c0 = cells[:, 0]
            c1 = cells[:, 1]
            valid = (c0 >= 0) & (c1 >= 0)
            edge = torch.zeros((int(mesh.nEdges), 1), dtype=torch.bool)
            edge[valid, 0] = ocean[c0[valid]] & ocean[c1[valid]]
            self.register_buffer("ocean_edge", edge)
        else:
            self.ocean_edge = None

    @classmethod
    def global_ocean(
        cls,
        mesh: Any,
        *,
        depth: Any | None = None,
        land_skin_temperature: Any = 288.0,
    ) -> "SurfaceBoundary":
        n_cells = int(mesh.nCells)
        bathy = torch.zeros((n_cells, 1), dtype=torch.float64) if depth is None else _as_cell_column(depth, n_cells=n_cells)
        return cls(
            torch.ones(n_cells, dtype=torch.bool),
            mesh=mesh,
            land_skin_temperature=land_skin_temperature,
            bathymetry_m=bathy,
        )

    @classmethod
    def from_ocean_domain(
        cls,
        domain: Any,
        *,
        ocean_state: Any | None = None,
        land_skin_temperature: Any = 288.0,
        surface_elevation_m: Any | None = None,
        land_evaporation_factor: Any = 0.05,
    ) -> "SurfaceBoundary":
        wet = domain.wet_cell.detach().cpu().reshape(-1)
        n_cells = int(wet.numel())
        if getattr(domain, "depth", None) is not None:
            depth = domain.depth.detach().cpu().to(dtype=torch.float64).reshape(n_cells, 1)
        elif ocean_state is not None:
            depth = ocean_state.column.total_depth_like(ocean_state.h).detach().cpu().to(dtype=torch.float64)
        else:
            depth = torch.zeros((n_cells, 1), dtype=torch.float64)
        mesh = getattr(domain, "mesh", None)
        if mesh is None and ocean_state is not None:
            mesh = ocean_state.mesh
        return cls(
            wet,
            mesh=mesh,
            land_skin_temperature=land_skin_temperature,
            surface_elevation_m=surface_elevation_m,
            bathymetry_m=depth,
            land_evaporation_factor=land_evaporation_factor,
        )

    def _mask_like(self, value: torch.Tensor, mask: torch.Tensor, *, name: str) -> torch.Tensor:
        axis_size = int(mask.shape[0])
        axis = None
        if value.ndim >= 2 and int(value.shape[-2]) == axis_size:
            axis = value.ndim - 2
        elif value.ndim >= 1 and int(value.shape[-1]) == axis_size:
            axis = value.ndim - 1
        else:
            for idx in range(value.ndim):
                if int(value.shape[idx]) == axis_size:
                    axis = idx
                    break
        if axis is None:
            raise ValueError(f"cannot apply {name} surface mask to tensor with shape {tuple(value.shape)}")
        shape = [1] * value.ndim
        shape[axis] = axis_size
        if value.ndim >= 2 and axis == value.ndim - 2:
            shape[-1] = 1
        return mask.to(device=value.device).reshape(shape)

    def ocean_cell_mask(self, value: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        mask = self._mask_like(value, self.ocean_cell, name="ocean cell")
        if dtype is None:
            dtype = value.dtype if value.is_floating_point() else torch.bool
        return mask.to(dtype=dtype, device=value.device)

    def land_cell_mask(self, value: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        mask = self._mask_like(value, self.land_cell, name="land cell")
        if dtype is None:
            dtype = value.dtype if value.is_floating_point() else torch.bool
        return mask.to(dtype=dtype, device=value.device)

    def ocean_edge_mask(self, value: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        edge = self.ocean_edge
        if edge is None:
            mask = torch.ones((int(value.shape[-2]), 1), dtype=torch.bool, device=value.device)
        else:
            mask = self._mask_like(value, edge, name="ocean edge")
        if dtype is None:
            dtype = value.dtype if value.is_floating_point() else torch.bool
        return mask.to(dtype=dtype, device=value.device)

    def apply_ocean_cell(self, value: torch.Tensor) -> torch.Tensor:
        mask = self.ocean_cell_mask(value)
        if value.dtype == torch.bool:
            return value & mask
        return value * mask

    def apply_ocean_edge(self, value: torch.Tensor) -> torch.Tensor:
        mask = self.ocean_edge_mask(value)
        if value.dtype == torch.bool:
            return value & mask
        return value * mask

    def field_like(self, name: str, like: torch.Tensor) -> torch.Tensor:
        value = getattr(self, name)
        return value.to(dtype=like.dtype, device=like.device)

    def surface_temperature(self, ocean_sst_kelvin: torch.Tensor) -> torch.Tensor:
        ocean = self.ocean_cell_mask(ocean_sst_kelvin)
        skin = self.land_skin_temperature.to(dtype=ocean_sst_kelvin.dtype, device=ocean_sst_kelvin.device)
        return ocean * ocean_sst_kelvin + (1.0 - ocean) * skin


class SurfaceExchange(nn.Module):
    """SCM-facing surface context plus ocean/land output routing."""

    recipe_name = "surface_exchange"

    def __init__(
        self,
        boundary: SurfaceBoundary | None = None,
        *,
        surface_system: Any | None = None,
        surface_state: Any | None = None,
    ) -> None:
        super().__init__()
        self.boundary = boundary
        self.surface_system = surface_system
        self.surface_state = surface_state

    @staticmethod
    def _same_mesh(atmosphere_state: Any, ocean_state: Any) -> None:
        if atmosphere_state.mesh is None or ocean_state.mesh is None:
            raise ValueError("SurfaceExchange requires mesh-bound atmosphere and ocean states")
        if atmosphere_state.mesh is not ocean_state.mesh:
            raise ValueError("SurfaceExchange currently requires atmosphere and ocean states on the same native mesh")

    def surface_params(
        self,
        state: Any,
        *,
        surface_context: Any,
        surface_state: Any | None = None,
        ocean_temperature_offset: float = 273.15,
    ) -> dict[str, torch.Tensor]:
        atmosphere = state.atmosphere
        ocean = state.ocean
        self._same_mesh(atmosphere, ocean)
        level = surface_context._level_index(int(atmosphere.u_e.shape[-1]))
        sst_c = ocean.temperature()[..., :1]
        sst_k = sst_c + float(ocean_temperature_offset)
        air_u, air_v = _cell_east_north_components(atmosphere.mesh, atmosphere.u_e[..., level : level + 1])
        ocean_u, ocean_v = _cell_east_north_components(ocean.mesh, ocean.u_e[..., :1])

        boundary = self.boundary
        if boundary is None:
            ocean_mask = torch.ones_like(sst_k)
            land_mask = torch.zeros_like(sst_k)
            surface_temperature = sst_k
            topography = torch.zeros_like(sst_k)
            bathymetry = ocean.column.total_depth_like(ocean.h).to(dtype=sst_k.dtype, device=sst_k.device)
            surface_type = ocean_mask
            moisture_factor = ocean_mask
        else:
            ocean_mask = boundary.ocean_cell_mask(sst_k)
            land_mask = 1.0 - ocean_mask
            surface_temperature = boundary.surface_temperature(sst_k)
            ocean_u = ocean_u * ocean_mask
            ocean_v = ocean_v * ocean_mask
            topography = boundary.topography_m.to(dtype=sst_k.dtype, device=sst_k.device)
            bathymetry = boundary.bathymetry_m.to(dtype=sst_k.dtype, device=sst_k.device)
            surface_type = boundary.surface_type.to(dtype=sst_k.dtype, device=sst_k.device)
            land_evap = boundary.land_evaporation_factor.to(dtype=sst_k.dtype, device=sst_k.device)
            moisture_factor = ocean_mask + land_mask * land_evap

        surface_albedo = torch.full_like(sst_k, 0.07)
        surface_emissivity = torch.full_like(sst_k, 0.97)
        surface_roughness = torch.full_like(sst_k, 2.0e-4)
        sea_ice_fraction = torch.zeros_like(sst_k)
        glacier_fraction = torch.zeros_like(sst_k)
        active_surface_state = self.surface_state if surface_state is None else surface_state
        if self.surface_system is not None and active_surface_state is not None:
            packet = self.surface_system.effective_boundary_packet(active_surface_state, ocean_state=ocean).to(
                dtype=sst_k.dtype,
                device=sst_k.device,
            )
            fractions = active_surface_state.fractions.to(dtype=sst_k.dtype, device=sst_k.device)
            ocean_mask = fractions.ocean_fraction
            land_mask = fractions.land_fraction
            sea_ice_fraction = fractions.sea_ice_fraction
            glacier_fraction = fractions.glacier_fraction
            surface_temperature = packet.skin_temperature
            surface_type = ocean_mask
            moisture_factor = packet.moisture_availability
            ocean_u = ocean_u * ocean_mask
            ocean_v = ocean_v * ocean_mask
            topography = torch.zeros_like(sst_k) if boundary is None else topography
            bathymetry = ocean.column.total_depth_like(ocean.h).to(dtype=sst_k.dtype, device=sst_k.device) if boundary is None else bathymetry
            surface_albedo = packet.albedo
            surface_emissivity = packet.emissivity
            surface_roughness = packet.roughness

        relative_u = air_u - ocean_u
        relative_v = air_v - ocean_v
        relative_speed = (relative_u.square() + relative_v.square()).clamp_min(0.0).sqrt()
        heat_coeff = torch.ones_like(sst_k) * float(surface_context.heat_exchange_coefficient)
        moisture_coeff = torch.ones_like(sst_k) * float(surface_context.moisture_exchange_coefficient) * moisture_factor
        return {
            "surface_temperature": surface_temperature,
            "sea_surface_temperature": sst_k,
            "sea_surface_temperature_celsius": sst_c,
            "skin_or_sea_surface_temperature": surface_temperature,
            "relative_wind_speed_cell": relative_speed,
            "surface_wind_speed": relative_speed,
            "atmosphere_surface_wind_u_east": air_u,
            "atmosphere_surface_wind_v_north": air_v,
            "ocean_surface_current_u_east": ocean_u,
            "ocean_surface_current_v_north": ocean_v,
            "relative_surface_wind_u_east": relative_u,
            "relative_surface_wind_v_north": relative_v,
            "surface_type": surface_type,
            "ocean_cell_mask": ocean_mask,
            "land_cell_mask": land_mask,
            "ocean_fraction": ocean_mask,
            "land_fraction": land_mask,
            "sea_ice_fraction": sea_ice_fraction,
            "glacier_fraction": glacier_fraction,
            "topography_m": topography,
            "bathymetry_m": bathymetry,
            "ocean_top_layer_thickness": ocean.h[..., :1],
            "surface_exchange_coefficient_heat": heat_coeff,
            "surface_exchange_coefficient_moisture": moisture_coeff,
            "surface_albedo": surface_albedo,
            "surface_emissivity": surface_emissivity,
            "surface_roughness": surface_roughness,
            "surface_moisture_availability": moisture_factor,
        }

    def route_ocean_tendency(self, ocean_rhs: Any) -> Any:
        boundary = self.boundary
        if boundary is None:
            return ocean_rhs
        tracer = None if ocean_rhs.tracer_h is None else boundary.apply_ocean_cell(ocean_rhs.tracer_h)
        return ocean_rhs.with_tensors(
            boundary.apply_ocean_edge(ocean_rhs.u_e),
            boundary.apply_ocean_cell(ocean_rhs.h),
            boundary.apply_ocean_cell(ocean_rhs.temp_h),
            boundary.apply_ocean_cell(ocean_rhs.salt_h),
            tracer,
        )

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weighted = value * mask
        denom = mask.sum().clamp_min(torch.finfo(value.dtype).tiny)
        return weighted.sum() / denom

    def output_diagnostics(
        self,
        outputs: Mapping[str, Any],
        surface_params: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        diagnostics: dict[str, torch.Tensor] = {}
        ocean = surface_params["ocean_cell_mask"]
        land = surface_params["land_cell_mask"]
        diagnostics["surface_ocean_fraction"] = ocean.mean()
        diagnostics["surface_land_fraction"] = land.mean()
        for key in (
            "ocean_surface_heat_flux_down",
            "freshwater_flux_down",
            "evaporation_mass_flux_up",
            "surface_precipitation_rate",
        ):
            value = outputs.get(key)
            if value is None or not torch.is_tensor(value):
                continue
            field = value.to(dtype=ocean.dtype, device=ocean.device)
            if field.ndim == 1:
                field = field[:, None]
            if field.shape[-1] != 1 and field.shape[-2:] != ocean.shape:
                field = field.mean(dim=-1, keepdim=True)
            diagnostics[f"{key}_ocean_mean"] = self._masked_mean(field, ocean)
            diagnostics[f"{key}_land_mean"] = self._masked_mean(field, land) if bool((land > 0).any().detach().cpu().item()) else torch.zeros((), dtype=field.dtype, device=field.device)
        return diagnostics


register_component("surface_boundary", SurfaceBoundary)
register_component("surface_exchange", SurfaceExchange)


__all__ = ["SurfaceBoundary", "SurfaceExchange"]
