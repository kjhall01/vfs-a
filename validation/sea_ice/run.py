"""Prescribed-air seasonal ice columns; no ocean dynamics or lateral transport."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "validation" / p)
    for p in ("sea_ice", "temperature_coordinate", "ocean_column")
]
from ice import SeaIceState, IceOceanState, SeaIceThermodynamics, ConductiveIceBoundary
from roquet import RoquetOcean
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=32)
    parser.add_argument("--dt", type=float, default=86400.0)
    parser.add_argument("--days", type=float, default=365.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cells < 16 or args.dt <= 0 or args.days <= 0:
        parser.error("require cells>=16 and positive duration/timestep")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    mesh = VoronoiMesh(n_cells=args.cells, lloyd_iters=2, verbose=False)
    model = RoquetOcean(mesh, OceanColumn(3, total_depth=200.0))
    h = torch.tensor([20.0, 60.0, 120.0]).expand(args.cells, -1).clone()
    salinity = 34.0 + 0.5 * mesh.cell_xyz[:, :1] + torch.zeros_like(h)
    ocean = model.state(
        u_e=torch.zeros(mesh.nEdges, 3),
        h=h,
        conservative_temperature=-0.054 * salinity,
        absolute_salinity=salinity,
        tracer=torch.ones_like(h)[..., None],
        tke=1e-4,
        eddy_energy=0.01,
    )
    state = IceOceanState(ocean, SeaIceState(torch.zeros(args.cells, 1)))
    thermo = SeaIceThermodynamics(model)
    boundary = ConductiveIceBoundary()
    H = thermo.hamiltonian
    area = mesh.cell_area
    start_energy = H.energy_per_area(state).clone()
    start_water = ocean.ocean.column.rho0 * h.sum(-1, keepdim=True)
    cumulative = torch.zeros_like(state.ice.mass)
    rows, timings = [], []
    peak_step_residual = 0.0
    t = 0.0
    args.output.mkdir(parents=True, exist_ok=True)
    while t < args.days * 86400.0 - 1e-6:
        dt = min(args.dt, args.days * 86400.0 - t)
        # Winter first; spatially varying idealized air, not reanalysis forcing.
        air = (
            -5.0
            - 15.0 * math.cos(2 * math.pi * (t + 0.5 * dt) / (365.0 * 86400.0))
            + 2.0 * mesh.cell_xyz[:, :1]
        )
        flux = boundary.flux(state, air)
        tic = time.perf_counter()
        state, report = thermo.apply(state, dt, heat_flux=flux)
        timings.append(time.perf_counter() - tic)
        cumulative += dt * flux
        residual = H.energy_per_area(state) - start_energy - cumulative
        water = (
            ocean.ocean.column.rho0 * state.ocean.ocean.h.sum(-1, keepdim=True)
            + state.ice.mass
            - start_water
        )
        peak_step_residual = max(
            peak_step_residual, float((report.energy_residual / area).abs().max())
        )
        t += dt
        rows.append(
            dict(
                day=t / 86400.0,
                ice_min_m=float(state.ice.thickness(H.parameters).min()),
                ice_max_m=float(state.ice.thickness(H.parameters).max()),
                ct_min_C=float(state.ocean.ocean.temperature()[:, :1].min()),
                ct_max_C=float(state.ocean.ocean.temperature()[:, :1].max()),
                max_energy_residual_J_m2=float(residual.abs().max()),
                max_water_residual_kg_m2=float(water.abs().max()),
            )
        )
    with (args.output / "series.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        cells=args.cells,
        layers=3,
        days=args.days,
        dt=args.dt,
        steps=len(rows),
        threads=torch.get_num_threads(),
        torch=torch.__version__,
        dtype="float64",
        device="cpu",
        median_step_seconds=float(torch.tensor(timings).median()),
        peak_ice_m=max(r["ice_max_m"] for r in rows),
        peak_step_residual_J_m2=peak_step_residual,
        peak_cumulative_residual_J_m2=max(r["max_energy_residual_J_m2"] for r in rows),
        peak_water_residual_kg_m2=max(r["max_water_residual_kg_m2"] for r in rows),
        peak_relative_energy_residual=max(r["max_energy_residual_J_m2"] for r in rows)
        / float(start_energy.abs().max()),
        final=rows[-1],
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(__file__).with_name("ice.py")]
        },
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
