"""Replay the Falklands and closing bottom face from the resting campaign ocean."""

import argparse
import json
from pathlib import Path

import torch
from .operators import use_openings
from .ocean_replay import energy_components, step
from .replay import load_restart
from workflows.coupled.ice_temperature import ThermalSnowOceanIce
from vfs.ice.ridging import CategoryIce, RidgingState
from vfs.ice.snow import SnowState


def fresh(coupling):
    """Rebuild the campaign resting ocean, including its initial ice phase adjustment."""
    model = coupling.ocean
    mesh = model.ocean.mesh
    column = model.ocean.column
    domain = model.ocean.domain
    h = column.layer_reference.clone()
    z = h.cumsum(1) - h / 2
    sst = (28 - 37 * mesh.cell_xyz[:, 2].square()).clamp_min(-0.054 * 34.0)
    deep = sst.clamp_max(2.0)
    ct = deep[:, None] + (sst - deep)[:, None] * torch.exp(-(z - z[:, :1]) / 250.0)
    ocean = model.state(
        u_e=h.new_zeros(mesh.nEdges, h.shape[1]),
        h=h,
        conservative_temperature=ct,
        absolute_salinity=torch.full_like(h, 34.0),
        tracer=torch.ones_like(h)[..., None],
        tke=1e-4 * domain.wet_cell[:, None],
        eddy_energy=0.1,
    )
    cover = domain.apply_cell(((mesh.cell_xyz[:, 2].abs() - 0.90) / 0.08).clamp(0, 0.7))
    a = cover[:, None] * h.new_tensor([0.8, 0.2])
    ice = CategoryIce(a, a * a.new_tensor([0.3, 1.0]), h.new_zeros(mesh.nEdges, 1))
    snow = ThermalSnowOceanIce(coupling)
    w = snow.initialize(SnowState(RidgingState(ocean, ice), torch.zeros_like(a)), 265.0)
    w, _, _ = snow.phase_step(w, 1, torch.zeros_like(a[:, :1]))
    return w.coupled.ocean.ocean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "restart", help="Earth-shelf checkpoint supplying mesh and bathymetry"
    )
    parser.add_argument("--days", type=int, default=15)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--checkpoints", type=Path, help="optional ocean-only test states"
    )
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    owner, _ = load_restart(args.restart)
    model = owner.ocean
    mechanical = use_openings(model)
    s = fresh(owner)
    s = s.with_tensors(s.u_e, s.h, s.temp_h, s.salt_h, None)
    mesh = s.mesh
    if mesh.nCells != 4096 or s.h.shape[1] != 40:
        parser.error(
            "the edge and level indices require the 4096-cell, 40-level Falklands case"
        )
    core = mesh.edgesOnVertex[mesh.verticesOnEdge[6304]].unique()
    edge, layer = 11588, 35
    area = mesh.dc_edge * mesh.dv_edge
    energy0 = energy_components(model, s)
    for interval in range(args.days * 72 + 1):
        if interval % 18 == 0:
            opening = mechanical.edge_thickness(s, s.h)
            wet = opening > 0
            velocity = torch.where(wet, s.u_e, 0.0)
            weights = area[core] * opening[core, 22]
            energy = energy_components(model, s)
            peak = velocity.abs().flatten().argmax().item()
            day = interval / 72
            print(
                json.dumps(
                    dict(
                        day=day,
                        opening_m=opening[edge, layer].item(),
                        edge_velocity=s.u_e[edge, layer].item(),
                        flux_per_width=(
                            opening[edge, layer] * s.u_e[edge, layer]
                        ).item(),
                        max_wet=velocity.abs().max().item(),
                        max_wet_slot=divmod(peak, s.h.shape[1]),
                        max_closed=torch.where(wet, 0.0, s.u_e.abs()).max().item(),
                        falklands_rms=(
                            (weights * s.u_e[core, 22].square()).sum() / weights.sum()
                        )
                        .sqrt()
                        .item(),
                        energy_change_J_m2=sum(energy[q] - energy0[q] for q in energy)
                        / mesh.cell_area.sum().item(),
                    )
                ),
                flush=True,
            )
            if args.checkpoints and day in (7.5, 7.75, 8.0, 8.25, 15.0):
                args.checkpoints.mkdir(exist_ok=True, parents=True)
                torch.save(
                    (s.u_e, s.h, s.temp_h, s.salt_h),
                    args.checkpoints / f"day-{day:g}.pt",
                )
        if interval < args.days * 72:
            s = step(model, s, 1200.0)


if __name__ == "__main__":
    main()
