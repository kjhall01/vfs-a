"""Idealized moving ocean/ice freshwater forcing with complete boundary ledgers."""

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
from run_coupled import initial_case
from snow import SnowOceanIce
from freshwater import FreshwaterBoundary, FreshwaterFluxes


def run(args):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    coupled, initial = initial_case(args.cells, args.coast, args.coast)
    snow = SnowOceanIce(coupled)
    boundary = FreshwaterBoundary(snow)
    state = snow.bare(initial)
    H = snow.hamiltonian
    mesh = initial.ocean.ocean.mesh
    A = mesh.cell_area[:, None]
    wet = H.model.energy.mass_area(initial.ocean.ocean) > 0
    wet_area = A[wet].sum()
    zero = torch.zeros_like(A)
    u = torch.zeros_like(initial.ice.velocity)
    rain = boundary.liquid(zero + 2e-5, zero + 5.0, u)
    receivers = torch.where(wet[:, 0])[0][:2]
    if args.coast:
        c0, c1 = mesh.cellsOnEdge.T
        crosses = wet[c0, 0] != wet[c1, 0]
        coast = torch.unique(
            torch.cat((c0[crosses & wet[c0, 0]], c1[crosses & wet[c1, 0]]))
        )
        receivers = coast[:2]
    river = boundary.route_runoff(
        state,
        torch.tensor([2e5, 1e5]),
        torch.tensor([5.0, 10.0]),
        receivers[:, None],
        torch.ones(2, 1),
        u,
    )
    data = dict(
        rain=rain,
        snowfall=zero + 1e-5,
        evaporation=zero + 2e-5,
        sublimation=zero + 2e-6,
        runoff=river,
    )
    if args.case != "mixed":
        data = {args.case: data[args.case]}
    flux = FreshwaterFluxes(**data)
    heat = torch.where(wet, zero - 10.0, 0.0)
    wind = 2.0 * torch.cross(
        torch.tensor([0.0, 0.0, 1.0]).expand_as(mesh.cell_xyz), mesh.cell_xyz, dim=-1
    )

    def water(x):
        s = x.coupled.ocean.ocean
        return (
            A * wet * (s.column.rho0 * s.h.sum(-1, keepdim=True) + H.mass(x.coupled))
        ).sum()

    def salt(x):
        return (A * wet * x.coupled.ocean.ocean.salt_h).sum()

    E0, M0, S0 = H.total_energy(initial), water(state), salt(state)
    external, incoming = torch.zeros_like(E0), torch.zeros_like(M0)
    rows = []
    clock = time.perf_counter()
    duration, t = args.hours * 3600.0, 0.0
    with torch.no_grad():
        while t < duration - 1e-7:
            dt = min(args.dt, duration - t)
            state, report = snow.step(
                state, dt, freshwater=flux, heat_flux=heat, wind_velocity=wind
            )
            external += report.external_energy
            boundary_error = 0.0
            for r in report.freshwater:
                incoming += (sum(r.inputs.values()) - sum(r.outputs.values())).sum()
                boundary_error = max(
                    boundary_error, float((r.energy_residual / A).abs().max())
                )
            t += dt
            s = state.coupled.ocean.ocean
            rows.append(
                dict(
                    seconds=t,
                    energy_error_J_m2=float(
                        (H.total_energy(state.coupled) - E0 - external) / wet_area
                    ),
                    relative_water_error=float((water(state) - M0 - incoming) / M0),
                    relative_salt_error=float((salt(state) - S0) / S0),
                    boundary_energy_error_J_m2=boundary_error,
                    boundary_water_kg=float(incoming),
                    snow_kg=float((A * state.snow_mass).sum()),
                    solid_kg=float((A * H.mass(state.coupled)).sum()),
                    min_salinity=float(s.salinity()[wet[:, 0]].min()),
                    max_ice_speed_m_s=float(state.coupled.ice.velocity.abs().max()),
                )
            )
            if args.verbose and (len(rows) % 30 == 0 or t == duration):
                print(
                    f"{t/3600:.2f} h: energy {rows[-1]['energy_error_J_m2']:.4g} J/m2",
                    flush=True,
                )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "series.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        configuration={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        steps=len(rows),
        wall_seconds=time.perf_counter() - clock,
        peak_energy_error_J_m2=max(abs(r["energy_error_J_m2"]) for r in rows),
        peak_relative_water_error=max(abs(r["relative_water_error"]) for r in rows),
        peak_relative_salt_error=max(abs(r["relative_salt_error"]) for r in rows),
        peak_boundary_energy_error_J_m2=max(
            r["boundary_energy_error_J_m2"] for r in rows
        ),
        final=rows[-1],
        torch=torch.__version__,
        threads=1,
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in summary.items() if k != "source_sha256"}, indent=2)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells", type=int, default=32)
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--dt", type=float, default=120.0)
    p.add_argument("--coast", action="store_true")
    p.add_argument(
        "--case",
        choices=("mixed", "evaporation", "sublimation", "runoff"),
        default="mixed",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.cells < 16 or not all(math.isfinite(x) and x > 0 for x in (a.hours, a.dt)):
        p.error("cells>=16 and positive finite hours/dt required")
    run(a)
