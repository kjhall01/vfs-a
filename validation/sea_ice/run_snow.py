"""Moving ocean/ice snow accumulation then melt, with boundary mass/energy ledgers."""

from vfs.ice import category_phase as category_phase_source
from vfs.ice import ice as ice_source
from vfs.ice import ice_drag as ice_drag_source
from vfs.ice import ocean_ice as ocean_ice_source
from vfs.ice import ridging as ridging_source
from vfs.ice import snow as snow_source
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
from vfs.ice.snow import SnowOceanIce, SnowParameters


def run(args):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    coupled, initial = initial_case(args.cells, args.coast, args.patch)
    params = SnowParameters()
    if args.no_snow_feedback:
        # Keep the same mass and latent-energy physics, but give the labelled
        # snow the optical/thermal properties of the compact ice carrier.
        ip = coupled.hamiltonian.slab.parameters
        params = SnowParameters(
            density=ip.density, conductivity=ip.conductivity, albedo=params.ice_albedo
        )
    model = SnowOceanIce(coupled, params)
    state = model.bare(initial)
    H = model.hamiltonian
    mesh = initial.ocean.ocean.mesh
    A = mesh.cell_area[:, None]
    wet = H.model.energy.mass_area(initial.ocean.ocean) > 0
    wet_area = A[wet].sum()

    def water(s):
        return (
            A
            * wet
            * (
                s.coupled.ocean.ocean.column.rho0
                * s.coupled.ocean.ocean.h.sum(-1, keepdim=True)
                + H.mass(s.coupled)
            )
        ).sum()

    start = H.total_energy(initial)
    water0 = water(state)
    salt0 = (A * initial.ocean.ocean.salt_h).sum()
    wind = args.wind * torch.cross(
        torch.tensor([0.0, 0.0, 1.0]).expand_as(mesh.cell_xyz), mesh.cell_xyz, dim=-1
    )
    work = torch.zeros_like(start)
    incoming = torch.zeros_like(start)
    melted = torch.zeros_like(start)
    t = 0.0
    rows = []
    clock = time.perf_counter()
    duration = args.hours * 3600.0
    transition = duration / 2
    while t < duration - 1e-7:
        cold = t < transition - 1e-7
        dt = min(args.dt, duration - t, transition - t if cold else duration - t)
        air = torch.full_like(A, args.cold_air if cold else args.warm_air)
        sw = torch.full_like(A, args.cold_sun if cold else args.warm_sun)
        snow = torch.full_like(A, args.snowfall if cold else 0.0)
        with torch.no_grad():
            state, report = model.step(
                state,
                dt,
                snowfall=snow,
                air_temperature=air,
                shortwave=sw,
                wind_velocity=wind if args.wind else None,
            )
            work = work + report.external_energy
            incoming = incoming + report.snowfall_mass
            melted = melted + report.melted_snow
            flux = model.surface_flux(state, air, sw)
            hi, hs = model.depths(state)
        t += dt
        rows.append(
            dict(
                seconds=t,
                energy_error_J_m2=float(
                    (H.total_energy(state.coupled) - start - work) / wet_area
                ),
                relative_water_error=float((water(state) - water0 - incoming) / water0),
                relative_salt_error=float(
                    ((A * state.coupled.ocean.ocean.salt_h).sum() - salt0) / salt0
                ),
                snow_kg=float((A * state.snow_mass).sum()),
                incoming_snow_kg=float(incoming),
                melted_snow_kg=float(melted),
                solid_kg=float((A * H.mass(state.coupled)).sum()),
                max_snow_depth_m=float(hs.max()),
                mean_albedo=float((A * flux.albedo).sum() / wet_area),
                absorbed_shortwave_W_m2=float(
                    (A * flux.absorbed_shortwave).sum() / wet_area
                ),
                mean_surface_heat_W_m2=float((A * flux.heat).sum() / wet_area),
                max_ice_speed_m_s=float(state.coupled.ice.velocity.abs().max()),
            )
        )
        if args.verbose and (len(rows) % 30 == 0 or t == duration):
            print(
                f'{t/3600:.2f} h | snow {rows[-1]["max_snow_depth_m"]:.3g} m | energy {rows[-1]["energy_error_J_m2"]:.3g} J/m2',
                flush=True,
            )
    wall = time.perf_counter() - clock
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
        wall_seconds=wall,
        seconds_per_step=wall / len(rows),
        peak_energy_error_J_m2=max(abs(r["energy_error_J_m2"]) for r in rows),
        peak_relative_water_error=max(abs(r["relative_water_error"]) for r in rows),
        peak_relative_salt_error=max(abs(r["relative_salt_error"]) for r in rows),
        peak_snow_kg=max(r["snow_kg"] for r in rows),
        final=rows[-1],
        torch=torch.__version__,
        threads=1,
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(snow_source.__file__),
                Path(__file__).with_name('run_coupled.py'),
                Path(ocean_ice_source.__file__),
                Path(category_phase_source.__file__),
                Path(ice_drag_source.__file__),
                Path(ridging_source.__file__),
                Path(ice_source.__file__),

            ]
        },
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells", type=int, default=32)
    p.add_argument("--hours", type=float, default=12.0)
    p.add_argument("--dt", type=float, default=120.0)
    p.add_argument(
        "--snowfall",
        type=float,
        default=2e-4,
        help="kg/m2/s per ice-covered area during first half",
    )
    p.add_argument("--cold-air", type=float, default=-15.0)
    p.add_argument("--warm-air", type=float, default=5.0)
    p.add_argument("--cold-sun", type=float, default=0.0)
    p.add_argument("--warm-sun", type=float, default=400.0)
    p.add_argument("--wind", type=float, default=4.0)
    p.add_argument("--coast", action="store_true")
    p.add_argument("--patch", action="store_true")
    p.add_argument("--no-snow-feedback", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    values = (
        args.hours,
        args.dt,
        args.snowfall,
        args.cold_air,
        args.warm_air,
        args.cold_sun,
        args.warm_sun,
        args.wind,
    )
    if (
        args.cells < 16
        or not all(math.isfinite(x) for x in values)
        or min(args.hours, args.dt) <= 0
        or min(args.snowfall, args.cold_sun, args.warm_sun) < 0
    ):
        p.error(
            "finite forcing, positive duration/dt, nonnegative snowfall/sun, cells>=16 required"
        )
    run(args)


if __name__ == "__main__":
    main()
