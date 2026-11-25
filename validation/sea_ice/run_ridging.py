"""Isolated compression stress/redistribution on a spherical mesh, not a coupled run."""

from vfs.ice import ridging as ridging_source
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "validation" / p)
    for p in ("sea_ice", "temperature_coordinate", "ocean_column")
]
from vfs.ice.ridging import CategoryIce, RidgingState, CompressionRidging
from vfs.ocean.temperature.roquet import RoquetOcean, RoquetWorkspace
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn, OceanDomain


def initial_case(cells, coast):
    mesh = VoronoiMesh(n_cells=cells, radius=1e5, lloyd_iters=2, verbose=False)
    domain = OceanDomain(mesh, mesh.cell_xyz[:, 2] > -0.4) if coast else None
    model = RoquetOcean(mesh, OceanColumn(3, total_depth=200.0), domain=domain)
    wet = (
        torch.ones(cells, 1)
        if domain is None
        else domain.wet_cell[:, None].to(torch.float64)
    )
    h = torch.tensor([20.0, 60.0, 120.0]).expand(cells, -1).clone()
    # Zero CT datum improves sensitivity of tiny mechanical-work measurements.
    # Phase equilibrium is NOT being advanced in this isolated experiment.
    ocean = model.state(
        u_e=torch.zeros(mesh.nEdges, 3),
        h=h,
        conservative_temperature=torch.zeros_like(h),
        absolute_salinity=torch.full_like(h, 34.0),
        tracer=torch.ones_like(h)[..., None],
        tke=1e-4 * wet,
        eddy_energy=0.01 * wet,
    )
    area = torch.tensor([0.7, 0.15]).expand(cells, -1).clone() * wet
    volume = area * torch.tensor([0.8, 4.0])
    velocity = mesh.grad(mesh.cell_xyz[:, :1]) * 1e4
    if domain:
        velocity = domain.apply_edge(velocity)
    return CompressionRidging(model), RidgingState(
        ocean, CategoryIce(area, volume, velocity)
    )


def stable_change(H, initial, end):
    """Independent algebraic full-H difference at fixed ocean h, salt and velocity.

    Retain actual ice mass roundoff changes. Only diagnostic subtraction is
    factored; this is not fed back to any prognostic variable.
    """
    s, t = initial.ocean.ocean, end.ocean.ocean
    for name in ("h", "salt_h", "u_e"):
        if not torch.equal(getattr(s, name), getattr(t, name)):
            raise ValueError("diagnostic requires fixed ocean geometry/salt/velocity")
    workspace = RoquetWorkspace(H.model.ocean, s)
    dcontent = t.temp_h - s.temp_h
    capacity = (
        workspace.reference.cp_theta + 2 * workspace.a * s.temperature() + workspace.b
    )
    heat = (
        H.model.energy.mass_area(s)
        * (capacity + workspace.a * dcontent / s.h)
        * dcontent
    ).sum()
    m0, m1 = H.mass(initial), H.mass(end)
    dm = m1 - m0
    ice = (
        s.mesh.cell_area[:, None]
        * dm
        * (
            -H.slab.parameters.latent_heat
            + s.column.gravity
            * (s.column.eta(s.h) + (m0 + m1) / (2 * H.slab.parameters.density))
        )
    ).sum()
    k0, k1 = H.metric(initial), H.metric(end)
    u0, u1 = initial.ice.velocity, end.ice.velocity
    kinetic = (
        0.25 * (k0 + k1) * (u0 + u1) * (u1 - u0)
        + 0.25 * (k1 - k0) * (u1.square() + u0.square())
    ).sum()
    ridge = (
        H.terms(end)["ridge_potential"] - H.terms(initial)["ridge_potential"]
    ).sum()
    return heat, ice, kinetic, ridge


def run(args, dt):
    op, state = initial_case(args.cells, args.coast)
    initial = state
    H = op.hamiltonian
    area = state.ocean.ocean.mesh.cell_area
    wet = H.model.energy.mass_area(state.ocean.ocean)[:, 0] > 0
    total_area = area[wet].sum()
    rows = []
    clock = time.perf_counter()
    t = 0.0
    peak_rhs = 0.0
    while t < args.seconds - 1e-9:
        _, report = op.tendency(state)
        scale = (
            report.kinetic_power.abs() + report.potential_power + report.heating_power
        ).sum()
        if scale > 0:
            peak_rhs = max(
                peak_rhs, float(report.energy_redistribution.sum().abs() / scale)
            )
        step = min(dt, args.seconds - t)
        state = op.step(state, step)
        t += step
        heat, ice, kinetic, ridge = stable_change(H, initial, state)
        rows.append(
            dict(
                seconds=t,
                ice_ke_change_J_m2=float(kinetic / total_area),
                ridge_pe_change_J_m2=float(ridge / total_area),
                heating_J_m2=float(heat / total_area),
                stable_energy_error_J_m2=float(
                    (heat + ice + kinetic + ridge) / total_area
                ),
                maximum_coverage=float(state.ice.area.sum(-1).max()),
                ridged_volume_m3=float((area * state.ice.volume[:, 1]).sum()),
            )
        )
    wall = time.perf_counter() - clock
    destination = args.output / f"dt{dt:g}"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "series.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        cells=args.cells,
        coast=args.coast,
        seconds=args.seconds,
        dt=dt,
        steps=len(rows),
        wall_seconds=wall,
        seconds_per_step=wall / len(rows),
        peak_relative_rhs_residual=peak_rhs,
        direct_full_H_error_J_m2=float(
            (H.total_energy(state) - H.total_energy(initial)) / total_area
        ),
        initial_full_H_J_m2=float(H.total_energy(initial) / total_area),
        max_cell_volume_change_m=float(
            (state.ice.volume.sum(-1) - initial.ice.volume.sum(-1)).abs().max()
        ),
        final=rows[-1],
        torch=torch.__version__,
        threads=torch.get_num_threads(),
        dtype="float64",
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(ridging_source.__file__)]
        },
    )
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=1000.0)
    parser.add_argument("--dt", type=float, nargs="+", default=[20.0, 10.0, 5.0])
    parser.add_argument("--coast", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cells < 16 or args.seconds <= 0 or min(args.dt) <= 0:
        parser.error("positive duration/timesteps, cells>=16 required")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    for dt in args.dt:
        run(args, dt)


if __name__ == "__main__":
    main()
