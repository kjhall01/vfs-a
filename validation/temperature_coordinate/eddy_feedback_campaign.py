"""Reproducible feedback/heat budgets and short closed zonal-band composition.

The channel is a spherical wet band with closed irregular coasts, not a planar
Eady benchmark. Initial geostrophic velocity is a continuum approximation;
resolved adjustment/RK drift is measured separately from closure work.
"""

import argparse
import json
import math
import sys
from pathlib import Path
import torch

sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "ocean_column"),
]
from eddy import GMRedi
from eddy_energy import EddyMixingLength, EddyDissipation
from eddy_campaign import make
from roquet import RoquetOcean
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn, OceanDomain


def channel(cells=128, layers=8):
    mesh = VoronoiMesh(
        n_cells=cells,
        radius=1e6,
        lloyd_iters=2,
        rotation_scheme="compatible",
        verbose=False,
    )
    latitude = torch.asin(mesh.cell_xyz[:, 2:3])
    domain = OceanDomain(
        mesh, (latitude[:, 0] > math.radians(15)) & (latitude[:, 0] < math.radians(65))
    )
    column = OceanColumn(layers, total_depth=1000.0)
    model = RoquetOcean(mesh, column, domain=domain)
    h = torch.full((cells, layers), 1000.0 / layers)
    z = -(torch.arange(layers) + 0.5) * 1000.0 / layers
    # Stable stratification and weak meridional density gradient, with a small
    # zonal perturbation. Surface pressure balance is approximate by design.
    y = mesh.radius * (latitude - math.radians(40))
    ct = torch.full_like(h, 10.0)
    sa = 35.0 - 0.001 * z[None, :] + 1e-8 * y + 1e-4 * mesh.cell_xyz[:, :1]
    by = -column.gravity / column.rho0 * model.config.s * 1e-8
    edge_lat = torch.asin(mesh.edge_xyz[:, 2])
    f = 2 * 7.292115e-5 * torch.sin(edge_lat)
    safe_f = torch.where(domain.active_edge, f, torch.ones_like(f))
    u_z = -by / safe_f
    east = torch.stack(
        (-mesh.edge_xyz[:, 1], mesh.edge_xyz[:, 0], torch.zeros_like(f)), -1
    )
    east = east / torch.linalg.vector_norm(east, dim=-1, keepdim=True)
    projection = (east * mesh.edge_normal).sum(-1)
    u = projection[:, None] * u_z[:, None] * (z[None, :] + 1000.0)
    u = domain.apply_edge(u)
    state = model.state(
        u_e=u,
        h=h,
        conservative_temperature=ct,
        absolute_salinity=sa,
        tke=0.0,
        eddy_energy=0.1,
    )
    return model, state


def run(model, initial, dt, seconds, *, dynamics=False, feedback=True, decay=True):
    law = EddyMixingLength(10000.0, 0.1, 0.5)
    eddy = GMRedi(
        kappa_gm=0.0 if feedback else 1000.0 * math.sqrt(0.1),
        kappa_redi=100.0,
        energy_closure=law if feedback else None,
    )
    sink = EddyDissipation(30 * 86400.0)
    state = initial
    mass = model.hamiltonian.mass_area(initial.ocean) * initial.ocean.h
    wet = mass > 0
    total_mass = mass.sum()
    initial_H = model.total_energy(initial)
    gm = redi = heat = closure = resolved = 0.0
    maxlocal = 0.0
    kappas = []
    for _ in range(round(seconds / dt)):
        if dynamics:
            before = model.total_energy(state)
            state, _ = model.step_with_report(state, dt)
            resolved += float((model.total_energy(state) - before) / total_mass)
        state, r = eddy.apply(model, state, dt)
        gm += float(r.gm_reservoir_exchange / total_mass)
        redi += float(r.redi_reservoir_exchange / total_mass)
        closure += float(r.energy_change / total_mass)
        if r.gm_diffusivity is not None:
            kappas.append(
                [float(r.gm_diffusivity.min()), float(r.gm_diffusivity.max())]
            )
        for local in (r.gm_local_energy_residual, r.redi_local_energy_residual):
            maxlocal = max(maxlocal, float((local[wet].abs() / mass[wet]).max()))
        if decay:
            state, r = sink.apply(model, state, dt)
            heat += float(r.dissipated_energy / total_mass)
            closure += float(r.local_energy_residual.sum() / total_mass)
            maxlocal = max(
                maxlocal, float((r.local_energy_residual[wet].abs() / mass[wet]).max())
            )
    e = state.specific_eddy_energy()
    return dict(
        dt_s=dt,
        duration_s=seconds,
        dynamics=dynamics,
        feedback=feedback,
        decay=decay,
        gm_reservoir_exchange_J_kg=gm,
        redi_reservoir_exchange_J_kg=redi,
        dissipated_J_kg=heat,
        cumulative_closure_residual_J_kg=closure,
        maximum_local_closure_residual_J_kg=maxlocal,
        resolved_step_energy_change_J_kg=resolved,
        direct_total_energy_change_J_kg=float(
            (model.total_energy(state) - initial_H) / total_mass
        ),
        minimum_eddy_energy_J_kg=float(e[wet].min()),
        maximum_eddy_energy_J_kg=float(e[wet].max()),
        initial_kappa_range=None if not kappas else kappas[0],
        final_kappa_range=None if not kappas else kappas[-1],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    trials = []
    for front in ("density", "cold", "warm"):
        model, initial = make(front=front)
        for dt in (120.0, 60.0, 30.0):
            trials.append(dict(case=front, **run(model, initial, dt, 3600.0)))
    model, initial = channel()
    for feedback, decay in ((False, False), (True, False), (True, True)):
        for dt in (60.0, 30.0):
            trials.append(
                dict(
                    case="zonal_band",
                    **run(
                        model,
                        initial,
                        dt,
                        600.0,
                        dynamics=True,
                        feedback=feedback,
                        decay=decay,
                    ),
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trials, indent=2) + "\n")
    print(json.dumps(trials, indent=2))


if __name__ == "__main__":
    main()
