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
        fixed_velocity = self.velocity if self.tile_count == 1 else self.velocity[self.land]
        if ((self.supply < 0).any() or (self.roughness <= 0).any()
                or ((self.availability < 0) | (self.availability > 1)).any()
                or (fixed_velocity != 0).any()):
            raise ValueError("invalid fixed-column surface")


class CoastalTiles(ColumnTiles):
    """Tile zero is land on dry cells and open water on wet cells."""
    tile_count = 3


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
    q_air = R[:, -1, 0]
    q_surface = q_air
    if isinstance(tiles, ColumnTiles):
        gradient = get_sat_vapor_pres().compute_qs(tiles.temperature[:, 0], ps) - q_air
        # Land availability limits evaporation; dew uses the full gradient.
        q_land = q_air + torch.where(gradient > 0, tiles.availability * gradient, gradient)
        q_surface = torch.where(land, q_land, q_air)
    raw = []
    for i in range(tiles.tile_count):
        sea = ~land if i == 0 else torch.zeros_like(land)
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
                q_surface if i == 0 else q_air,
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
    if tiles.tile_count == 1:
        return raw[0], raw
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


def fund_column_evaporation(physics, vd, surf, sfc, coupled, tiles, dt, raw):
    """Re-solve the implicit air humidity increment at a binding water supply."""
    limit = tiles.supply / dt
    limited = tiles.land & (coupled["flux_q"] > limit)
    if limited.any():
        sfc = dict(sfc)
        sfc["flux_q"] = torch.where(limited, limit, sfc["flux_q"])
        for key in ("dedq_atm", "dedq_surf", "dedt_surf"):
            sfc[key] = torch.where(limited, 0., sfc[key])
        coupled = physics.couple(vd, sfc, surf, update_ocean=False)
        raw[0] = dict(raw[0])
        for key in ("flux_q", "dedq_atm", "dedq_surf", "dedt_surf"):
            raw[0][key] = torch.where(limited, sfc[key], raw[0][key])
    return coupled, raw


def fund_ice_sublimation(physics, vd, surf, sfc, coupled, tiles, dt, raw):
    """Constrain each ice tile's integrated vapor loss in the shared humidity solve.

    The supply is frozen mass per cell area at interval start. Snow diagnosed
    later by moist physics first becomes available during the next interval.
    Capping one tile lowers the common air humidity increment and can cause the
    other tile to bind. At most two solves are needed for two frozen categories.
    """
    fraction = tiles.fraction[:, 1:]
    stock = tiles.frozen_mass
    q0 = torch.stack([tile["flux_q"] for tile in raw[1:]], -1)
    dq = torch.stack([tile["dedq_atm"] for tile in raw[1:]], -1)
    def demand():
        return dt * (fraction * (q0 + dq * coupled["delta_q"][:, None]))

    binding = demand() > stock
    if not binding.any():
        return coupled, raw, {}
    limited = binding
    rate = stock / torch.where(fraction > 0, fraction, 1.) / dt
    keys = ("flux_q", "dedq_atm", "dedq_surf", "dedt_surf")
    for _ in range(2):
        for i, tile in enumerate(raw[1:]):
            raw[i + 1] = {**tile, **{
                k: torch.where(binding[:, i], rate[:, i] if k == "flux_q" else 0., tile[k])
                for k in keys
            }}
        # raw[0] also carries any previously prescribed land evaporation flux.
        sfc = {**sfc, **{k: sum(tiles.fraction[:, i] * tile[k] for i, tile in enumerate(raw))
                         for k in keys}}
        coupled = physics.couple(vd, sfc, surf, update_ocean=False)
        requested = demand()
        binding = (requested > stock) & ~limited
        if not binding.any():
            break
        limited = limited | binding
    return coupled, raw, dict(ice_sublimation_limited=limited,
                              ice_sublimation_withheld=torch.where(limited, requested - stock, 0.))
