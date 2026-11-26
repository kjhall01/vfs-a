"""Paired ocean–ice SSPRK3/SSPRK(4,3) dynamics benchmark.

Does not change the outer coupling, mixing, GM–Redi, or radiation schedule.
Both candidates advance the SAME initial state for the SAME model time:
2 x SSPRK3(dt) versus 1 x SSPRK43(2*dt). This isolates dynamics economics.
Results cannot be extrapolated to full coupled SYPD without a campaign A/B.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)] + [str(ROOT / "validation" / p) for p in (
    "sea_ice", "temperature_coordinate", "ocean_column"
)]
from vfs.ice import OceanIce
from vfs.ocean.model import Ocean as RoquetOcean
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn


def make_case(cells, layers):
    mesh = VoronoiMesh(
        n_cells=cells, lloyd_iters=0, rotation_scheme="compatible", verbose=False
    )
    column = OceanColumn(layers, total_depth=5000.)
    h = torch.full((cells, layers), 5000. / layers)
    salinity = torch.full_like(h, 34.0)
    ocean_model = RoquetOcean(mesh, column)
    ocean = ocean_model.state(
        u_e=torch.zeros(mesh.nEdges, layers),
        h=h,
        conservative_temperature=-0.054 * salinity,
        absolute_salinity=salinity,
        tracer=torch.ones_like(h)[..., None],
        tke=1e-4,
        eddy_energy=0.1,
    )
    ice_fraction = 0.5 * (
        (mesh.cell_xyz[:, 2].abs() - 0.85) / 0.15
    ).clamp(0, 1)
    cover = ice_fraction[:, None] * torch.tensor([0.8, 0.2])
    snow = OceanIce(ocean_model)
    state = snow.state(ocean, area=cover, volume=cover * torch.tensor([0.3, 1.0]),
                       velocity=torch.zeros(mesh.nEdges, 1), temperature=265.)
    snow.check(state)
    return snow, state


def trial(model, initial, dt, method):
    if method == "ssprk3":
        state, _ = model.dynamics_step(initial, dt, integrator="ssprk3")
        return model.dynamics_step(state, dt, integrator="ssprk3")
    return model.dynamics_step(initial, 2 * dt, integrator="ssprk43")


def time_method(model, initial, dt, method, warmups, repeats):
    try:
        with torch.no_grad():
            for _ in range(warmups):
                trial(model, initial, dt, method)
            timings = []
            final = None
            for _ in range(repeats):
                start = time.perf_counter()
                final, _ = trial(model, initial, dt, method)
                timings.append(time.perf_counter() - start)
        return dict(
            ok=True, times_s=timings, median_s=statistics.median(timings),
            final=final,
        )
    except (ValueError, FloatingPointError, RuntimeError) as exc:
        return dict(ok=False, error=f"{type(exc).__name__}: {exc}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells", type=int, default=256)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if (args.cells < 16 or args.layers < 2 or not math.isfinite(args.dt)
            or args.dt <= 0 or args.threads < 1 or args.repeats < 1
            or args.warmups < 0):
        p.error("invalid benchmark dimensions/timing")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    model, state = make_case(args.cells, args.layers)
    a = time_method(model, state, args.dt, "ssprk3", args.warmups, args.repeats)
    b = time_method(model, state, args.dt, "ssprk43", args.warmups, args.repeats)
    report = dict(
        cells=args.cells,
        layers=args.layers,
        dt_ssprk3=args.dt,
        dt_ssprk43=2*args.dt,
        matched_model_seconds=2*args.dt,
        threads=args.threads,
        torch=str(torch.__version__),
        baseline={k:v for k,v in a.items() if k!="final"},
        candidate={k:v for k,v in b.items() if k!="final"},
    )
    if a["ok"] and b["ok"]:
        report["speedup_dynamics_only"] = a["median_s"] / b["median_s"]
        aa, bb = a["final"], b["final"]
        report["max_state_difference"] = {
            "ocean_height_m": float(
                (aa.ocean.resolved.h - bb.ocean.resolved.h).abs().max()
            ),
            "ocean_velocity_m_s": float(
                (aa.ocean.resolved.u_e - bb.ocean.resolved.u_e).abs().max()
            ),
            "ice_volume_m": float(
                (aa.ice.volume - bb.ice.volume).abs().max()
            ),
        }
    message = json.dumps(report, indent=2)
    print(message, flush=True)
    if args.output:
        args.output.write_text(message + "\n")
    return 0 if a["ok"] and b["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
