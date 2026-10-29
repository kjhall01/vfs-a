"""Surface-system composition and conservative routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..core.registry import register_component
from .conservation import ConservativeTransfer, TransferLedger
from .fractions import SurfaceFractions
from .glacier_land_ice import GlacierLandIce, GlacierState
from .icebergs import IcebergState, IcebergTracer
from ..land import LandSurface, LandSurfaceState
from .packets import AtmosSurfaceFluxPacket, OceanBoundaryFluxPacket, SurfaceBoundaryPacket
from .runoff_router import RunoffRouter, RunoffState
from .sea_ice import SeaIceState, SeaIceThermo, SeaIceTransport
from .state import SurfaceState
from .utils import as_cell_column, cell_area_column


@dataclass(frozen=True)
class SurfaceStepResult:
    state: SurfaceState
    ocean_fluxes: OceanBoundaryFluxPacket
    transfers: tuple[ConservativeTransfer, ...]
    diagnostics: dict[str, torch.Tensor]


class SurfaceSystem(nn.Module):
    """Top-level surface/cryosphere component.

    The system owns surface states and conservative routing. It deliberately
    does not own atmosphere or ocean prognostic states; those reservoirs are
    coupled through explicit packets and transfers.
    """

    recipe_name = "surface_system"

    def __init__(
        self,
        *,
        land: LandSurface | None = None,
        sea_ice: SeaIceThermo | None = None,
        sea_ice_transport: SeaIceTransport | None = None,
        glacier: GlacierLandIce | None = None,
        icebergs: IcebergTracer | None = None,
        runoff_router: RunoffRouter | None = None,
        ocean_temperature_offset: float = 273.15,
        default_land_albedo: float = 0.25,
        default_ocean_albedo: float = 0.07,
        default_sea_ice_albedo: float = 0.55,
        default_glacier_albedo: float = 0.70,
        default_snow_albedo: float = 0.80,
    ) -> None:
        super().__init__()
        self.land = land
        self.sea_ice = sea_ice
        self.sea_ice_transport = sea_ice_transport
        self.glacier = glacier
        self.icebergs = icebergs
        self.runoff_router = runoff_router
        self.ocean_temperature_offset = float(ocean_temperature_offset)
        self.default_land_albedo = float(default_land_albedo)
        self.default_ocean_albedo = float(default_ocean_albedo)
        self.default_sea_ice_albedo = float(default_sea_ice_albedo)
        self.default_glacier_albedo = float(default_glacier_albedo)
        self.default_snow_albedo = float(default_snow_albedo)

    def initial_state(
        self,
        n_cells: int,
        *,
        fractions: SurfaceFractions | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> SurfaceState:
        fractions = fractions or SurfaceFractions.global_ocean(n_cells, dtype=dtype, device=device)
        state = SurfaceState(
            fractions,
            land=None if self.land is None else self.land.initial_state(n_cells, dtype=dtype, device=device),
            sea_ice=None if self.sea_ice is None else SeaIceState.uniform(n_cells, dtype=dtype, device=device),
            glacier=None if self.glacier is None else GlacierState.uniform(n_cells, dtype=dtype, device=device),
            icebergs=None if self.icebergs is None else IcebergState.uniform(n_cells, dtype=dtype, device=device),
            runoff=None if self.runoff_router is None else RunoffState.zeros(n_cells, dtype=dtype, device=device),
            attrs={"surface_system": "v1_skeleton"},
        )
        state.validate()
        return state

    def _ocean_skin_temperature(self, surface_state: SurfaceState, ocean_state: Any | None) -> torch.Tensor:
        like = surface_state.fractions.land_fraction
        if ocean_state is None:
            return torch.full_like(like, 271.35)
        return ocean_state.temperature()[..., :1].to(dtype=like.dtype, device=like.device) + float(
            self.ocean_temperature_offset
        )

    def effective_boundary_packet(self, surface_state: SurfaceState, *, ocean_state: Any | None = None) -> SurfaceBoundaryPacket:
        """Blend diagnostic surface properties by fractional ownership."""

        fractions = surface_state.fractions
        like = fractions.land_fraction
        ocean = fractions.ocean_fraction.to(dtype=like.dtype, device=like.device)
        sea_ice = fractions.sea_ice_fraction.to(dtype=like.dtype, device=like.device)
        land = fractions.land_fraction.to(dtype=like.dtype, device=like.device)
        glacier = fractions.glacier_fraction.to(dtype=like.dtype, device=like.device)
        snow = fractions.snow_fraction.to(dtype=like.dtype, device=like.device)
        open_ocean = (ocean - sea_ice).clamp_min(0.0)
        bare_land = (land - glacier).clamp_min(0.0)

        ocean_temp = self._ocean_skin_temperature(surface_state, ocean_state)
        land_temp = torch.full_like(like, 288.0)
        land_albedo = torch.full_like(like, float(self.default_land_albedo))
        land_roughness = torch.full_like(like, 0.05)
        land_moisture = torch.zeros_like(like)
        if surface_state.land is not None:
            if self.land is not None:
                land_boundary = self.land.boundary_properties(surface_state.land)
                land_temp = land_boundary.skin_temperature
                land_albedo = land_boundary.albedo
                land_roughness = land_boundary.roughness
                land_moisture = land_boundary.moisture_availability
            else:
                land_temp = surface_state.land.skin_temperature
                land_moisture = (surface_state.land.soil_water / 50.0).clamp(0.0, 1.0)
        ice_temp = torch.full_like(like, 268.0)
        ice_albedo = torch.full_like(like, float(self.default_sea_ice_albedo))
        ice_roughness = torch.full_like(like, 1.0e-3)
        if surface_state.sea_ice is not None:
            if self.sea_ice is not None:
                ice_boundary = self.sea_ice.boundary_properties(surface_state.sea_ice)
                ice_temp = ice_boundary.surface_temperature
                ice_albedo = ice_boundary.albedo
                ice_roughness = ice_boundary.roughness
            else:
                ice_temp = surface_state.sea_ice.sea_ice_surface_temperature
        glacier_temp = (
            surface_state.glacier.glacier_surface_temperature
            if surface_state.glacier is not None
            else torch.full_like(like, 263.0)
        )
        denom = (open_ocean + bare_land + sea_ice + glacier).clamp_min(torch.finfo(like.dtype).tiny)
        skin = (open_ocean * ocean_temp + bare_land * land_temp + sea_ice * ice_temp + glacier * glacier_temp) / denom

        moisture = open_ocean + bare_land * land_moisture.clamp(0.0, 1.0) + sea_ice * 0.05
        albedo = (
            open_ocean * float(self.default_ocean_albedo)
            + bare_land * land_albedo
            + sea_ice * ice_albedo
            + glacier * float(self.default_glacier_albedo)
        ) / denom
        snow_overlay = snow.clamp(0.0, 1.0)
        albedo = (1.0 - snow_overlay) * albedo + snow_overlay * float(self.default_snow_albedo)
        emissivity = torch.ones_like(like) * 0.97
        roughness = (
            open_ocean * 2.0e-4
            + bare_land * land_roughness
            + sea_ice * ice_roughness
            + glacier * 5.0e-3
        )
        roughness = roughness / denom
        return SurfaceBoundaryPacket(albedo, skin, emissivity, roughness, moisture.clamp(0.0, 1.0))

    def step(
        self,
        state: SurfaceState,
        dt: float,
        *,
        fluxes: AtmosSurfaceFluxPacket,
        ocean_state: Any | None = None,
        sst_celsius: torch.Tensor | None = None,
        ocean_salinity: torch.Tensor | None = None,
        mesh: Any | None = None,
    ) -> SurfaceStepResult:
        state.validate()
        transfers: list[ConservativeTransfer] = []
        diagnostics: dict[str, torch.Tensor] = {}
        fractions = state.fractions
        land_state = state.land
        runoff_state = state.runoff
        sea_ice_state = state.sea_ice
        glacier_state = state.glacier
        iceberg_state = state.icebergs
        like = fractions.land_fraction
        routed_runoff = torch.zeros_like(like)
        glacier_calving = torch.zeros_like(like)
        sea_ice_heat_to_ocean = torch.zeros_like(like)
        sea_ice_freshwater_to_ocean = torch.zeros_like(like)
        sea_ice_salt_to_ocean = torch.zeros_like(like)
        # Open-ocean fluxes are applied directly to the ocean. Land, glacier,
        # and sea-ice portions are owned by their respective reservoirs below.
        # Packet convention is radiative heat positive downward and turbulent
        # sensible/latent heat positive upward from the surface.
        open_ocean = (fractions.ocean_fraction - fractions.sea_ice_fraction).clamp_min(0.0).to(
            dtype=like.dtype,
            device=like.device,
        )
        direct_ocean_heat = open_ocean * (
            fluxes.net_shortwave.to(dtype=like.dtype, device=like.device)
            + fluxes.net_longwave.to(dtype=like.dtype, device=like.device)
            - fluxes.sensible_heat.to(dtype=like.dtype, device=like.device)
            - fluxes.latent_heat.to(dtype=like.dtype, device=like.device)
        )
        direct_ocean_freshwater = open_ocean * (
            fluxes.precipitation_rain.to(dtype=like.dtype, device=like.device)
            + fluxes.precipitation_snow.to(dtype=like.dtype, device=like.device)
            - fluxes.latent_heat.to(dtype=like.dtype, device=like.device) / 2.5e6
        )

        if self.land is not None and land_state is not None:
            land_result = self.land.step(land_state, fractions, fluxes, dt, mesh=mesh)
            land_state = land_result.state
            routed_runoff = routed_runoff + land_result.runoff_mass
            transfers.extend(land_result.transfers)
            diagnostics.update(land_result.diagnostics)
        if self.glacier is not None and glacier_state is not None:
            snowfall = fluxes.precipitation_snow.to(dtype=like.dtype, device=like.device)
            glacier_result = self.glacier.step(glacier_state, fractions, dt, snowfall=snowfall, mesh=mesh)
            glacier_state = glacier_result.state
            fractions = glacier_result.fractions
            routed_runoff = routed_runoff + glacier_result.runoff_mass
            glacier_calving = glacier_calving + glacier_result.iceberg_source_mass
            transfers.extend(glacier_result.transfers)
            diagnostics.update(glacier_result.diagnostics)
        if self.runoff_router is not None and runoff_state is not None:
            runoff_result = self.runoff_router.step(runoff_state, routed_runoff, dt)
            runoff_state = runoff_result.state
            runoff_to_ocean = runoff_result.ocean_input_mass
            transfers.extend(runoff_result.transfers)
            diagnostics.update(runoff_result.diagnostics)
        else:
            runoff_to_ocean = routed_runoff
            if bool((routed_runoff != 0.0).any().detach().cpu().item()):
                transfers.append(ConservativeTransfer("runoff_to_ocean_direct", "runoff", "ocean", water_mass=routed_runoff))
        if self.icebergs is not None and iceberg_state is not None:
            iceberg_result = self.icebergs.step(iceberg_state, dt, source_mass=glacier_calving, source_component=None)
            iceberg_state = iceberg_result.state
            iceberg_melt = iceberg_result.melt_mass
            transfers.extend(iceberg_result.transfers)
            diagnostics.update(iceberg_result.diagnostics)
        else:
            iceberg_melt = torch.zeros_like(like)
        if self.sea_ice is not None and sea_ice_state is not None:
            if ocean_state is not None:
                sst_celsius = ocean_state.temperature()[..., :1]
                ocean_salinity = ocean_state.salinity()[..., :1]
            sst = torch.zeros_like(like) if sst_celsius is None else as_cell_column(sst_celsius, like=like, name="sst_celsius")
            salinity = torch.full_like(like, 35.0) if ocean_salinity is None else as_cell_column(
                ocean_salinity,
                like=like,
                name="ocean_salinity",
            )
            sea_ice_result = self.sea_ice.step(
                sea_ice_state,
                fractions,
                sst_celsius=sst,
                ocean_salinity=salinity,
                heat_flux_to_ocean=(
                    fluxes.net_shortwave
                    + fluxes.net_longwave
                    - fluxes.sensible_heat
                    - fluxes.latent_heat
                ),
                dt=dt,
                snowfall_rate=fluxes.precipitation_snow,
                mesh=mesh,
            )
            sea_ice_state = sea_ice_result.state
            fractions = sea_ice_result.fractions
            sea_ice_heat_to_ocean = sea_ice_result.ocean_heat_flux
            sea_ice_freshwater_to_ocean = sea_ice_result.ocean_freshwater_flux
            sea_ice_salt_to_ocean = sea_ice_result.ocean_salt_flux
            transfers.extend(sea_ice_result.transfers)
            diagnostics.update(sea_ice_result.diagnostics)
        if self.sea_ice_transport is not None and sea_ice_state is not None:
            if mesh is None:
                raise ValueError("sea_ice_transport requires the native mesh")
            sea_ice_transport_result = self.sea_ice_transport.step(
                sea_ice_state,
                fractions,
                mesh=mesh,
                dt=dt,
            )
            sea_ice_state = sea_ice_transport_result.state
            fractions = sea_ice_transport_result.fractions
            diagnostics.update(sea_ice_transport_result.diagnostics)

        next_state = SurfaceState(
            fractions,
            land_state,
            sea_ice_state,
            glacier_state,
            iceberg_state,
            runoff_state,
            state._attrs_copy(),
        )
        dt_t = torch.as_tensor(float(dt), dtype=like.dtype, device=like.device)
        area = cell_area_column(mesh, like)
        ocean_flux = OceanBoundaryFluxPacket(
            direct_ocean_heat + sea_ice_heat_to_ocean,
            direct_ocean_freshwater
            + sea_ice_freshwater_to_ocean
            + (runoff_to_ocean + iceberg_melt) / (dt_t * area).clamp_min(torch.finfo(like.dtype).tiny),
            sea_ice_salt_to_ocean,
            runoff_to_ocean / (dt_t * area).clamp_min(torch.finfo(like.dtype).tiny),
            iceberg_melt / (dt_t * area).clamp_min(torch.finfo(like.dtype).tiny),
        )
        diagnostics.update(TransferLedger(transfers).summary())
        diagnostics["surface_total_water"] = next_state.total_water(mesh=mesh)
        return SurfaceStepResult(next_state, ocean_flux, tuple(transfers), diagnostics)


register_component("surface_system", SurfaceSystem)


__all__ = ["SurfaceStepResult", "SurfaceSystem"]
