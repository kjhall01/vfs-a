"""Explicit tiled calls to public PyAM21 surface routines; no upstream edits."""

from dataclasses import dataclass, replace

import torch
from pyam21.sat_vapor_pres import get_sat_vapor_pres
from pyam21.physics.am2 import ALBEDO_BANDS
from workflows.exchange import SurfaceTiles


@dataclass(frozen=True)
class ColumnTiles(SurfaceTiles):
    """One fixed land or slab-ocean surface per atmospheric column."""
    land: torch.Tensor
    roughness: torch.Tensor
    availability: torch.Tensor
    supply: torch.Tensor  # evaporatable water, kg/m² of land at interval start

    tile_count = 1

    def validate(self, mesh):
        super().validate(mesh)
        ref = mesh.cell_area
        if (self.land.shape != ref.shape
                or self.land.dtype != torch.bool or self.land.device != ref.device):
            raise ValueError("invalid column land mask")
        for value in (self.roughness, self.availability, self.supply):
            if (value.shape != ref.shape or value.dtype != ref.dtype or value.device != ref.device
                    or not torch.isfinite(value).all()):
                raise ValueError("invalid land boundary properties")
        if ((self.supply < 0).any() or (self.roughness <= 0).any()
                or ((self.availability < 0) | (self.availability > 1)).any()
                or (self.velocity != 0).any()):
            raise ValueError("invalid fixed-column surface")


def prepare_surface(state, tiles):
    tiles.validate(state.atmosphere.mesh)
    # One radiation column: area-mean albedo and emission-equivalent skin T.
    surf = dict(state.surface)
    if isinstance(tiles, ColumnTiles):
        surf["land"] = tiles.land
        for key in ("rough_mom", "rough_heat", "rough_moist", "rough_scale"):
            surf[key] = torch.where(tiles.land, tiles.roughness, surf[key])
    surf["t_surf"] = (tiles.fraction * tiles.temperature**4).sum(-1) ** 0.25
    surf["albedo"] = (tiles.fraction * tiles.albedo).sum(-1)
    surf["albedo_bands"] = {k: surf["albedo"] for k in ALBEDO_BANDS}
    return replace(state, surface=surf)


def surface_layer(adapter, columns, geometry, surf, mem, tiles):
    p, m = adapter.physics, adapter.mapping
    T, u, v, R, ps = (columns[k] for k in ("T", "u", "v", "tracers", "ps"))
    land = surf["land"]
    if isinstance(tiles, ColumnTiles):
        q_air = R[:, -1, 0]
        q_sat = get_sat_vapor_pres().compute_qs(surf["t_surf"], ps)
        gradient = q_sat - q_air
        # Dry soil reduces evaporation. A cold skin can still collect dew.
        q_skin = q_air + torch.where(gradient > 0, tiles.availability * gradient, gradient)
        sfc = p.surface_layer(T, u, v, R, ps, geometry["zfull"][:, -1] - geometry["zhalf"][:, -1],
                              geometry["pfull"][:, -1], surf["t_surf"], surf, mem,
                              q_surf_in=q_skin)
        return sfc, [sfc]
    raw = []
    for i in range(3):
        # Upstream saturation lookup includes its ice/water temperature rule.
        # No seawater salinity reduction on frozen tiles.
        sea = torch.full_like(land, i == 0)
        ui = (tiles.velocity[:, i] * m.east).sum(-1)
        vi = (tiles.velocity[:, i] * m.north).sum(-1)
        raw.append(
            p.sflux(
                T[:, -1],
                R[:, -1, 0],
                u[:, -1],
                v[:, -1],
                geometry["pfull"][:, -1],
                geometry["zfull"][:, -1] - geometry["zhalf"][:, -1],
                ps,
                tiles.temperature[:, i],
                tiles.temperature[:, i],
                R[:, -1, 0],
                ui,
                vi,
                surf["rough_mom"],
                surf["rough_heat"],
                surf["rough_moist"],
                surf["rough_scale"],
                mem["gust"],
                land,
                sea,
            )
        )
    # Bulk turbulence gets area-mean surface diagnostics; the implicit bottom
    # solve uses weighted fluxes/derivatives. This is an explicit mosaic closure.
    bulk = {k: sum(tiles.fraction[:, i] * raw[i][k] for i in range(3)) for k in raw[0]}
    return bulk, raw


def tile_fluxes(raw, tiles, vd, coupled):
    # The COMMON implicit air increment belongs to every tile. Independently
    # solving each tile would imply three different bottom air states.
    delta = dict(
        flux_t=("dhdt_atm", coupled["delta_t"]),
        flux_q=("dedq_atm", coupled["delta_q"]),
        flux_u=("dtaudu_atm", vd["delta_u"]),
        flux_v=("dtaudv_atm", vd["delta_v"]),
    )
    return {
        k: torch.stack(
            [tiles.fraction[:, i] * (raw[i][k] + raw[i][deriv] * d) for i in range(len(raw))],
            -1,
        )
        for k, (deriv, d) in delta.items()
    }


def fund_column_evaporation(physics, vd, surf, sfc, coupled, tiles, dt):
    """Re-solve the implicit air humidity increment at a binding water supply."""
    limit = tiles.supply / dt
    limited = tiles.land & (coupled["flux_q"] > limit)
    if limited.any():
        sfc = dict(sfc)
        sfc["flux_q"] = torch.where(limited, limit, sfc["flux_q"])
        for key in ("dedq_atm", "dedq_surf", "dedt_surf"):
            sfc[key] = torch.where(limited, 0., sfc[key])
        coupled = physics.couple(vd, sfc, surf, update_ocean=False)
    return coupled, [sfc]
