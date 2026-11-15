"""Explicit tiled calls to public PyAM21 surface routines; no upstream edits."""

import torch
from pyam21.physics.am2 import ALBEDO_BANDS


def prepare_surface(state, tiles):
    from dataclasses import replace

    tiles.validate(state.atmosphere.mesh)
    # One radiation column: area-mean albedo and emission-equivalent skin T.
    surf = dict(state.surface)
    surf["t_surf"] = (tiles.fraction * tiles.temperature**4).sum(-1) ** 0.25
    surf["albedo"] = (tiles.fraction * tiles.albedo).sum(-1)
    surf["albedo_bands"] = {k: surf["albedo"] for k in ALBEDO_BANDS}
    return replace(state, surface=surf)


def surface_layer(adapter, columns, geometry, surf, mem, tiles):
    p, m = adapter.physics, adapter.mapping
    T, u, v, R, ps = (columns[k] for k in ("T", "u", "v", "tracers", "ps"))
    land = surf["land"]
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
            [tiles.fraction[:, i] * (raw[i][k] + raw[i][deriv] * d) for i in range(3)],
            -1,
        )
        for k, (deriv, d) in delta.items()
    }
