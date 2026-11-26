"""Short coupled native atmosphere/ocean/ice budget experiment, CPU float64."""

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from workflows.pyam21.setup import example
from workflows.pyam21.dynamics import advance_dynamics, pressure_velocity
from workflows.coupled.ocean_ice import OceanIceReceiver
from workflows.coupled.driver import exchange_step
from vfs.ice.ocean_ice import OceanIce
from vfs.ice.ridging import RidgingState, CategoryIce
from vfs.ice.snow import SnowOceanIce
from vfs.ocean.model import Ocean
from vfs.ocean import OceanColumn


def initial_surface(adapter, air, cover):
    mesh = adapter.model.mesh
    model = Ocean(mesh, OceanColumn(3, total_depth=200.0, gravity=9.8))
    h = torch.tensor([20.0, 60.0, 120.0]).expand(mesh.nCells, -1).clone()
    frozen = mesh.cell_xyz[:, 2].abs() > 0.6
    ice_area = (
        frozen[:, None] * torch.tensor([0.4, 0.1])[None]
        if cover
        else torch.zeros(mesh.nCells, 2)
    )
    sst = air.surface["t_surf"] - 273.15
    if cover:
        sst = torch.where(frozen, torch.full_like(sst, -1.836), sst)
    ocean = model.state(
        u_e=torch.zeros(mesh.nEdges, 3),
        h=h,
        conservative_temperature=sst[:, None].expand_as(h),
        absolute_salinity=torch.full_like(h, 34.0),
        tracer=torch.ones_like(h)[..., None],
        tke=1e-4,
        eddy_energy=0.01,
    )
    ice = CategoryIce(
        ice_area, ice_area * torch.tensor([0.3, 1.0]), torch.zeros(mesh.nEdges, 1)
    )
    snow = SnowOceanIce(OceanIce(model))
    return OceanIceReceiver(snow), snow.bare(RidgingState(ocean, ice))


def run(args):
    adapter, air = example(
        args.source,
        cells=args.cells,
        levels=args.levels,
        aquaplanet=True,
        radiation_data=args.radiation_data,
        radiation_step=args.radiation_step,
    )
    receiver, water = initial_surface(adapter, air, args.ice)
    A = adapter.model.mesh.cell_area

    def energy():
        return adapter.model.diagnostics(air.atmosphere)[
            "energy"
        ] + receiver.H.total_energy(water.coupled)

    def mass():
        s = water.ocean.ocean
        return (
            adapter.model.diagnostics(air.atmosphere)["species"].sum()
            + (
                A * (s.column.rho0 * s.h.sum(-1) + receiver.H.mass(water)[:, 0])
            ).sum()
        )

    E0, W0 = energy(), mass()
    S0 = (A[:, None] * water.ocean.ocean.salt_h).sum()
    external, upstream = 0.0, 0.0
    rows = []
    start = time.perf_counter()
    with torch.no_grad():
        while air.time_seconds < args.hours * 3600:
            dt = min(args.dt, int(args.hours * 3600 - air.time_seconds))
            if args.dynamics:
                elapsed = 0.0
                while elapsed < dt:
                    sub = min(args.dynamics_dt, dt - elapsed)
                    air = advance_dynamics(adapter, air, sub)
                    elapsed += sub
                water, r = receiver.ocean_ice.step(water, dt)
                external += float(r.external_energy / A.sum())
            omega = pressure_velocity(adapter, air) if args.dynamics else None
            # Deliberate boundary assumption for this integration experiment.
            # The shared API accepts a future solved ice skin temperature.
            iceT = torch.full_like(water.snow_mass, args.ice_skin_temperature)
            air, water, x, ar, sr = exchange_step(
                adapter, air, receiver, water, dt, ice_temperature=iceT, omega=omega
            )
            external += float((A * x.toa).sum() / A.sum())
            upstream += float(
                (A * ar.energy_changes["am2_energy_residual"]).sum() / A.sum()
            )
            s = water.ocean.ocean
            residual = float((energy() - E0) / A.sum()) - external
            row = dict(
                seconds=air.time_seconds,
                combined_energy_minus_toa_J_m2=residual,
                cumulative_upstream_residual_J_m2=upstream,
                numerical_energy_residual_J_m2=residual - upstream,
                combined_water_residual_kg_m2=float((mass() - W0) / A.sum()),
                salt_relative=float(((A[:, None] * s.salt_h).sum() - S0) / S0),
                surface_energy_residual_max_J_m2=float(sr.energy_residual.abs().max()),
                atmosphere_transfer_max_J_m2=float(
                    ar.energy_transfer["layer_residual"].abs().max()
                ),
                radiation_flux_residual_W_m2=float(
                    ar.energy_changes["radiation_flux_residual"].abs().max() / dt
                ),
                SST_min_K=float(s.temperature()[:, 0].min() + 273.15),
                SST_max_K=float(s.temperature()[:, 0].max() + 273.15),
                ice_kg=float((A[:, None] * receiver.H.mass(water)).sum()),
                snow_kg=float((A[:, None] * water.snow_mass).sum()),
                max_ocean_speed_m_s=float(s.u_e.abs().max()),
                max_ice_speed_m_s=float(water.coupled.ice.velocity.abs().max()),
            )
            rows.append(row)
            if args.verbose and (
                len(rows) % 30 == 0 or air.time_seconds == args.hours * 3600
            ):
                print(
                    f'{air.time_seconds/3600:.2f} h: numerical energy {row["numerical_energy_residual_J_m2"]:.4g} J/m2; water {row["combined_water_residual_kg_m2"]:.3g} kg/m2',
                    flush=True,
                )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "series.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    summary = dict(
        configuration={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        radiation=adapter.radiation_manifest,
        steps=len(rows),
        wall_seconds=time.perf_counter() - start,
        final=rows[-1],
        peaks={
            k: max(abs(r[k]) for r in rows)
            for k in (
                "numerical_energy_residual_J_m2",
                "combined_water_residual_kg_m2",
                "salt_relative",
                "surface_energy_residual_max_J_m2",
                "atmosphere_transfer_max_J_m2",
            )
        },
        restart_fingerprint=adapter.fingerprint,
        torch=torch.__version__,
        workflow_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "workflows").rglob("*.py")
        },
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ("workflow_sha256", "radiation")
            },
            indent=2,
        )
    )


ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cells", type=int, default=32)
    p.add_argument("--levels", type=int, default=24)
    p.add_argument("--dt", type=int, default=60)
    p.add_argument("--hours", type=float, default=1)
    p.add_argument("--dynamics", action="store_true")
    p.add_argument("--dynamics-dt", type=float, default=15)
    p.add_argument("--ice", action="store_true")
    p.add_argument("--ice-skin-temperature", type=float, default=270.0)
    p.add_argument("--radiation-data", type=Path)
    p.add_argument("--radiation-step", type=int, default=3600)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    if (
        a.cells < 16
        or a.levels < 4
        or a.dt <= 0
        or not math.isfinite(a.hours)
        or a.hours <= 0
        or not float(a.hours * 3600).is_integer()
        or not math.isfinite(a.dynamics_dt)
        or a.dynamics_dt <= 0
        or not math.isfinite(a.ice_skin_temperature)
        or a.ice_skin_temperature <= 0
        or a.radiation_step < a.dt
        or a.radiation_step % a.dt
    ):
        p.error("invalid mesh, interval, radiation cadence or skin temperature")
    run(a)
