"""Small wind/thermal-forced ocean-ice integration with actual ocean dynamics."""

from vfs.ice import category_phase as category_phase_source
from vfs.ice import ice as ice_source
from vfs.ice import ice_drag as ice_drag_source
from vfs.ice import ocean_ice as ocean_ice_source
from vfs.ice import ridging as ridging_source
import argparse, csv, hashlib, json, math, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "validation" / p)
    for p in ("sea_ice", "temperature_coordinate", "ocean_column")
]
from vfs.ice.ocean_ice import OceanIce
from vfs.ice.ridging import CategoryIce, RidgingState
from vfs.ocean.temperature.roquet import RoquetOcean
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn, OceanDomain


def initial_case(cells, coast=False, patch=False):
    mesh = VoronoiMesh(
        n_cells=cells, lloyd_iters=2, rotation_scheme="compatible", verbose=False
    )
    domain = OceanDomain(mesh, mesh.cell_xyz[:, 2] > -0.5) if coast else None
    model = RoquetOcean(mesh, OceanColumn(3, total_depth=200.0), domain=domain)
    wet = (
        torch.ones(cells, 1)
        if domain is None
        else domain.wet_cell[:, None].to(torch.float64)
    )
    h = torch.tensor([20.0, 60.0, 120.0]).expand(cells, -1).clone()
    salinity = torch.full_like(h, 34.0)
    ocean = model.state(
        u_e=torch.zeros(mesh.nEdges, 3),
        h=h,
        conservative_temperature=-0.054 * salinity,
        absolute_salinity=salinity,
        tracer=torch.ones_like(h)[..., None],
        tke=1e-4 * wet,
        eddy_energy=0.01 * wet,
    )
    area = wet * torch.tensor([0.5, 0.1]) * (1 + 0.1 * mesh.cell_xyz[:, :1])
    if patch:
        area = torch.where(mesh.cell_xyz[:, 2:3] > 0, area, 0.0)
    ice = CategoryIce(
        area, area * torch.tensor([0.3, 1.0]), torch.zeros(mesh.nEdges, 1)
    )
    return OceanIce(model), RidgingState(ocean, ice)


def run(args):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    coupled, state = initial_case(args.cells, args.coast, args.patch)
    initial = state
    H = coupled.hamiltonian
    mesh = state.ocean.ocean.mesh
    area = mesh.cell_area[:, None]
    wet = H.model.energy.mass_area(state.ocean.ocean) > 0
    ocean_area = area[wet].sum()
    start = H.total_energy(state)
    water0 = (
        area
        * (
            state.ocean.ocean.column.rho0 * state.ocean.ocean.h.sum(-1, keepdim=True)
            + H.mass(state)
        )
    ).sum()
    salt0 = (area * state.ocean.ocean.salt_h).sum()
    # Smooth tangential eastward wind, tapering toward the poles.
    axis = torch.tensor([0.0, 0.0, 1.0]).expand_as(mesh.cell_xyz)
    wind = args.wind * torch.cross(axis, mesh.cell_xyz, dim=-1)
    q = torch.where(wet, torch.full_like(area, args.heat), 0.0)
    work = torch.zeros_like(start)
    t = 0.0
    rows = []
    clock = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    while t < args.hours * 3600.0 - 1e-7:
        dt = min(args.dt, args.hours * 3600.0 - t)
        state, external = coupled.step(
            state, dt, heat_flux=q, wind_velocity=wind if args.wind else None
        )
        work = work + external
        t += dt
        s = state.ocean.ocean
        water = (
            area * (s.column.rho0 * s.h.sum(-1, keepdim=True) + H.mass(state))
        ).sum()
        rows.append(
            dict(
                seconds=t,
                energy_error_J_m2=float(
                    (H.total_energy(state) - start - work) / ocean_area
                ),
                relative_water_error=float((water - water0) / water0),
                relative_salt_error=float(((area * s.salt_h).sum() - salt0) / salt0),
                max_ice_speed_m_s=float(state.ice.velocity.abs().max()),
                max_ocean_speed_m_s=float(s.u_e.abs().max()),
                min_coverage=float(state.ice.area.sum(-1)[wet[:, 0]].min()),
                max_coverage=float(state.ice.area.sum(-1).max()),
                total_ice_kg=float((area * H.mass(state)).sum()),
                ocean_tke_J=float(
                    (H.model.energy.mass_area(s) * state.ocean.tke_h).sum()
                ),
            )
        )
        if args.verbose and (len(rows) % 10 == 0 or t == args.hours * 3600.0):
            print(
                f'{t/3600:.2f} h | ice {rows[-1]["max_ice_speed_m_s"]:.3g} m/s | energy {rows[-1]["energy_error_J_m2"]:.3g} J/m2',
                flush=True,
            )
    wall = time.perf_counter() - clock
    with (args.output / "series.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        cells=args.cells,
        layers=3,
        dt=args.dt,
        hours=args.hours,
        wind_m_s=args.wind,
        heat_W_m2=args.heat,
        coast=args.coast,
        patch=args.patch,
        steps=len(rows),
        wall_seconds=wall,
        seconds_per_step=wall / len(rows),
        peak_energy_error_J_m2=max(abs(r["energy_error_J_m2"]) for r in rows),
        peak_relative_water_error=max(abs(r["relative_water_error"]) for r in rows),
        peak_relative_salt_error=max(abs(r["relative_salt_error"]) for r in rows),
        initial_ice_kg=float((area * H.mass(initial)).sum()),
        final=rows[-1],
        torch=torch.__version__,
        threads=1,
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
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
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--dt", type=float, default=120.0)
    p.add_argument("--wind", type=float, default=8.0)
    p.add_argument("--heat", type=float, default=-30.0)
    p.add_argument("--coast", action="store_true")
    p.add_argument("--patch", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if (
        args.cells < 16
        or any(
            not math.isfinite(x) for x in (args.hours, args.dt, args.wind, args.heat)
        )
        or min(args.hours, args.dt) <= 0
    ):
        p.error("finite forcing, positive duration/dt, cells>=16 required")
    run(args)


if __name__ == "__main__":
    main()
