"""Integrated Roquet-ocean evaluation; explicit cases, budgets and saved endpoints.

Run against an installed VFS package. Research modules are loaded explicitly
from this checkout. No retries, clipping, parameter tuning or hidden dt changes.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "validation/temperature_coordinate"),
    str(ROOT / "validation/ocean_column"),
]
from roquet import RoquetOcean, RoquetMixing, RoquetHeating, TemperatureBoundary
from eddy import GMRedi
from eddy_energy import EddyMixingLength, EddyDissipation
from backscatter import EddyBackscatter
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn, OceanDomain
import vfs

DAY = 86400.0


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Experiment:
    def __init__(self, args, variant):
        self.args, self.variant = args, variant
        radius = args.radius or (1e6 if args.case == "channel" else 6371220.0)
        xyz = None if args.mesh is None else np.load(args.mesh, allow_pickle=False)
        mesh = VoronoiMesh(
            n_cells=args.cells,
            radius=radius,
            generator_xyz=xyz,
            lloyd_iters=0 if xyz is not None else 2,
            rotation_scheme="compatible",
            verbose=False,
        )
        self.mesh = mesh
        lat = torch.asin(mesh.cell_xyz[:, 2:3])
        edge_lat = torch.asin(mesh.edge_xyz[:, 2])
        east = torch.stack(
            (-mesh.edge_xyz[:, 1], mesh.edge_xyz[:, 0], torch.zeros(mesh.nEdges)), -1
        )
        east /= torch.linalg.vector_norm(east, dim=-1, keepdim=True)
        projection = (east * mesh.edge_normal).sum(-1)
        depth = torch.full((args.cells, 1), 1000.0)
        if args.case in ("rest", "stratified-rest"):
            depth += 200 * mesh.cell_xyz[:, :1]
        wet = torch.ones(args.cells, dtype=torch.bool)
        if args.case == "channel":
            wet = (lat[:, 0] > math.radians(15)) & (lat[:, 0] < math.radians(65))
        domain = OceanDomain(mesh, wet, depth=depth)
        fractions = torch.expm1(2 * torch.linspace(0, 1, args.layers + 1)) / math.expm1(
            2
        )
        column = OceanColumn(
            args.layers, total_depth=depth, rest_thickness=1000 * fractions.diff()
        )
        forced = args.case == "forced"
        self.model = RoquetOcean(
            mesh,
            column,
            domain=domain,
            closure=(
                RoquetMixing(2.0, substeps=math.ceil(args.dt / 30)) if forced else None
            ),
            boundary=TemperatureBoundary(0.0025) if forced else None,
            heating=RoquetHeating(0.6, (0.5, 20.0)) if forced else None,
            backscatter=(
                EddyBackscatter(
                    args.viscosity,
                    args.backscatter_days * DAY,
                    args.fraction,
                    biharmonic_viscosity=args.biharmonic_viscosity,
                )
                if variant == "backscatter"
                else None
            ),
        )
        h = depth * fractions.diff()[None]
        z = -(h.cumsum(-1) - 0.5 * h)
        ct = torch.full_like(h, 10.0)
        sa = torch.full_like(h, 35.0)
        u = torch.zeros(mesh.nEdges, args.layers)
        self.forcing = {}
        if args.case == "stratified-rest":
            sa = 35.0 - 0.001 * z
        if args.case == "channel":
            y = radius * (lat - math.radians(40))
            sa = 35.0 - 0.001 * z + args.gradient * y + 0.001 * mesh.cell_xyz[:, :1]
            by = -column.gravity / column.rho0 * self.model.config.s * args.gradient
            f = 2 * 7.292115e-5 * torch.sin(edge_lat)
            f = torch.where(domain.active_edge, f, torch.ones_like(f))
            u = (
                projection[:, None]
                * (-by / f)[:, None]
                * (mesh.cell_to_edge(z) + 1000.0)
            )
        elif forced:
            ct = 4.0 + 16 * torch.exp(z / 300.0)
            sa = 35.0 - 0.2 * torch.exp(z / 200.0)
            heat = 20 * (1 - 3 * torch.sin(lat[:, 0]).square())
            heat -= (mesh.cell_area * heat).sum() / mesh.cell_area.sum()
            self.forcing = dict(
                surface_stress=args.wind * torch.sin(2 * edge_lat) * projection,
                shortwave=torch.full((args.cells,), 30.0),
                surface_heat=heat - 30.0,
            )
        self.initial = self.model.state(
            u_e=domain.apply_edge(u),
            h=h,
            conservative_temperature=ct,
            absolute_salinity=sa,
            tracer=(1 + 0.2 * mesh.cell_xyz[:, 2:3]).expand_as(h)[..., None],
            tke=1e-4 if forced else 0.0,
            eddy_energy=0.0 if args.case in ("rest", "stratified-rest") else 0.1,
        )
        self.eddy = (
            None
            if variant == "resolved"
            else GMRedi(
                kappa_redi=args.redi,
                energy_closure=EddyMixingLength(args.length, 0.1, 0.5),
            )
        )
        self.decay = (
            None
            if args.case in ("rest", "stratified-rest")
            else EddyDissipation(args.decay_days * DAY)
        )
        if args.compile:
            self.model.compile_thermodynamics()
        self.mass_area = self.model.energy.mass_area(self.initial.ocean)
        self.mass = float((self.mass_area * self.initial.ocean.h).sum())
        self.initial_energy = float(self.model.total_energy(self.initial))
        self.initial_volume = float((self.mass_area * self.initial.ocean.h).sum())
        self.initial_salt = float((self.mass_area * self.initial.ocean.salt_h).sum())
        self.initial_tracer = float(
            (self.mass_area * self.initial.ocean.tracer_h[..., 0]).sum()
        )
        self.cumulative = dict(
            external_J_kg=0.0,
            gm_exchange_J_kg=0.0,
            redi_exchange_J_kg=0.0,
            dissipated_J_kg=0.0,
            closure_residual_J_kg=0.0,
            backscatter_work_J_kg=0.0,
            biharmonic_gain_J_kg=0.0,
        )
        self.max_closure = self.max_power = 0.0
        self.min_vertex_scale = 1.0

    def advance(self, state, dt):
        state, report = self.model.step_with_report(state, dt, **self.forcing)
        external = 0.0
        if report.boundary is not None:
            external += float(report.boundary.surface_work)
        if report.heating is not None:
            external += float(report.heating.external_heat)
        self.cumulative["external_J_kg"] += external / self.mass
        mass = self.mass_area * state.ocean.h
        wet = mass > 0
        for weight, r in zip((1 / 6, 1 / 6, 2 / 3), getattr(report, "backscatter", ())):
            self.cumulative["backscatter_work_J_kg"] += (
                dt
                * weight
                * float(r.kinetic_power - r.biharmonic_kinetic_power)
                / self.mass
            )
            self.cumulative["biharmonic_gain_J_kg"] += (
                dt * weight * float(r.biharmonic_reservoir_power) / self.mass
            )
            self.max_power = max(
                self.max_power,
                float((r.local_power_residual[wet].abs() / mass[wet]).max()),
                float((r.kinetic_power + r.reservoir_power).abs()) / self.mass,
            )
            self.min_vertex_scale = min(
                self.min_vertex_scale, float(r.vertex_scale.min())
            )
        if self.eddy is not None:
            state, r = self.eddy.apply(self.model, state, dt)
            self.cumulative["gm_exchange_J_kg"] += (
                float(r.gm_reservoir_exchange) / self.mass
            )
            self.cumulative["redi_exchange_J_kg"] += (
                float(r.redi_reservoir_exchange) / self.mass
            )
            self.cumulative["closure_residual_J_kg"] += (
                float(r.energy_change) / self.mass
            )
            for residual in (r.gm_local_energy_residual, r.redi_local_energy_residual):
                self.max_closure = max(
                    self.max_closure, float((residual[wet].abs() / mass[wet]).max())
                )
        if self.decay is not None:
            state, r = self.decay.apply(self.model, state, dt)
            self.cumulative["dissipated_J_kg"] += float(r.dissipated_energy) / self.mass
            self.cumulative["closure_residual_J_kg"] += (
                float(r.local_energy_residual.sum()) / self.mass
            )
            self.max_closure = max(
                self.max_closure,
                float((r.local_energy_residual[wet].abs() / mass[wet]).max()),
            )
        return state

    def diagnostics(self, state, seconds):
        s = state.ocean
        wet = self.model.ocean.domain.wet_cell
        terms = self.model.hamiltonian.terms(state)
        density = self.model.config.polynomial(
            s.temperature(), s.salinity(), torch.zeros_like(s.h)
        )
        dz = 0.5 * (s.h[:, 1:] + s.h[:, :-1])
        n2 = s.column.gravity / s.column.rho0 * density.diff(dim=-1) / dz
        energy_error = (
            float(terms["total"]) - self.initial_energy
        ) / self.mass - self.cumulative["external_J_kg"]
        # Reconstructed non-zonal KE is a diagnostic proxy, not the native H.
        v = self.mesh.cell_wind_vector(s.u_e)
        xyz = self.mesh.cell_xyz
        east = torch.stack((-xyz[:, 1], xyz[:, 0], torch.zeros_like(xyz[:, 0])), -1)
        east /= torch.linalg.vector_norm(east, dim=-1, keepdim=True)
        north = torch.linalg.cross(xyz, east)
        uv = torch.stack(
            ((v * east[:, None]).sum(-1), (v * north[:, None]).sum(-1)), -1
        )
        bins = ((xyz[:, 2] + 1) * 6).long().clamp(max=11)
        weights = self.mass_area * s.h
        sums = torch.zeros(12, self.args.layers).index_add(0, bins, weights)
        means = torch.zeros(12, self.args.layers, 2).index_add(
            0, bins, weights[..., None] * uv
        )
        means /= sums.clamp_min(1)[..., None]
        anomaly = uv - means[bins]
        row = dict(
            time_days=seconds / DAY,
            energy_error_J_kg=energy_error,
            volume_relative_error=float((self.mass_area * s.h).sum())
            / self.initial_volume
            - 1,
            salt_relative_error=float((self.mass_area * s.salt_h).sum())
            / self.initial_salt
            - 1,
            tracer_relative_error=float((self.mass_area * s.tracer_h[..., 0]).sum())
            / self.initial_tracer
            - 1,
            kinetic_J_kg=float(terms["kinetic"]) / self.mass,
            eddy_J_kg=float(terms["mesoscale_eddy"]) / self.mass,
            tke_J_kg=float(terms["turbulent_kinetic"]) / self.mass,
            nonzonal_ke_proxy_J_kg=float((weights[..., None] * anomaly.square()).sum())
            / (2 * self.mass),
            max_speed_m_s=float(s.u_e.abs().max()),
            min_h_m=float(s.h[wet].min()),
            max_eta_m=float(s.column.eta(s.h)[wet].abs().max()),
            min_CT=float(s.temperature()[wet].min()),
            max_CT=float(s.temperature()[wet].max()),
            min_SA=float(s.salinity()[wet].min()),
            max_SA=float(s.salinity()[wet].max()),
            min_eddy_J_kg=float(state.specific_eddy_energy()[wet].min()),
            min_potential_N2_s2=float(n2[wet].min()),
            max_closure_residual_J_kg=self.max_closure,
            max_backscatter_power_residual_W_kg=self.max_power,
            min_vertex_scale=self.min_vertex_scale,
            **self.cumulative,
        )
        if not all(math.isfinite(v) for v in row.values()):
            raise FloatingPointError("nonfinite diagnostic")
        return row

    def save(self, state, path):
        s = state.ocean
        np.savez_compressed(
            path,
            cell_xyz=self.mesh.cell_xyz.numpy(),
            edge_xyz=self.mesh.edge_xyz.numpy(),
            wet_cell=self.model.ocean.domain.wet_cell.numpy(),
            u_e=s.u_e.numpy(),
            h=s.h.numpy(),
            CT=s.temperature().numpy(),
            SA=s.salinity().numpy(),
            eta=s.column.eta(s.h).numpy(),
            tracer_h=s.tracer_h.numpy(),
            eddy_h=state.eddy_h.numpy(),
            tke_h=state.tke_h.numpy(),
        )


def run(args, variant):
    directory = args.output / variant
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "summary.json").exists():
        raise FileExistsError(
            f"{directory} already contains results; choose a fresh --output"
        )
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    summary = dict(
        config=config, variant=variant, status="initializing", completed_steps=0
    )
    write_json(directory / "summary.json", summary)
    started = time.perf_counter()
    state = None
    experiment = None
    seconds = 0.0
    step = 0
    try:
        experiment = Experiment(args, variant)
        state = experiment.initial
        summary.update(
            min_edge_m=float(experiment.mesh.dc_edge.min()),
            wet_cells=int(experiment.model.ocean.domain.wet_cell.sum()),
            mesh_sha256=hashlib.sha256(
                experiment.mesh.cell_xyz.numpy().tobytes()
            ).hexdigest(),
        )
        experiment.save(state, directory / "initial.npz")
        np.save(directory / "mesh.npy", experiment.mesh.cell_xyz.numpy())
        target = args.days * DAY
        summary["status"] = "running"
        write_json(directory / "summary.json", summary)
        with (directory / "series.csv").open("w", newline="") as stream:
            initial = experiment.diagnostics(state, 0.0)
            writer = csv.DictWriter(
                stream, fieldnames=initial.keys(), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerow(initial)
            stream.flush()
            summary["initial"] = initial
            while seconds < target - 1e-8:
                if time.perf_counter() - started > args.max_wall_seconds:
                    raise TimeoutError("per-variant wall time limit reached")
                dt = min(args.dt, target - seconds)
                saved = (
                    experiment.cumulative.copy(),
                    experiment.max_closure,
                    experiment.max_power,
                    experiment.min_vertex_scale,
                )
                try:
                    trial = experiment.advance(state, dt)
                except (ValueError, FloatingPointError, RuntimeError):
                    (
                        experiment.cumulative,
                        experiment.max_closure,
                        experiment.max_power,
                        experiment.min_vertex_scale,
                    ) = saved
                    raise
                state = trial
                seconds += dt
                step += 1
                if step % args.log_every == 0 or seconds >= target - 1e-8:
                    row = experiment.diagnostics(state, seconds)
                    writer.writerow(row)
                    stream.flush()
                    summary.update(
                        completed_steps=step,
                        elapsed_seconds=seconds,
                        last=row,
                        wall_seconds=time.perf_counter() - started,
                    )
                    write_json(directory / "summary.json", summary)
                    print(
                        f'{variant}: day={seconds/DAY:.4f} step={step} u_max={row["max_speed_m_s"]:.3e} '
                        f'energy residual={row["energy_error_J_kg"]:.3e} J/kg',
                        flush=True,
                    )
            summary["status"] = "completed"
    except (ValueError, FloatingPointError, RuntimeError, TimeoutError) as error:
        summary.update(
            status="rejected" if not isinstance(error, TimeoutError) else "timeout",
            error=repr(error),
            failed_step=step + 1,
        )
        print(f'{variant}: {summary["status"]}: {error}', flush=True)
    finally:
        summary.update(
            completed_steps=step,
            elapsed_seconds=seconds,
            wall_seconds=time.perf_counter() - started,
        )
        if state is not None:
            experiment.save(state, directory / "last_accepted.npz")
            summary["last"] = experiment.diagnostics(state, seconds)
        write_json(directory / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case",
        choices=("rest", "stratified-rest", "channel", "forced"),
        required=True,
    )
    p.add_argument(
        "--variants",
        nargs="+",
        choices=("resolved", "gmredi", "backscatter"),
        default=["resolved", "gmredi", "backscatter"],
    )
    p.add_argument("--cells", type=int, default=128)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--days", type=float, default=2.0)
    p.add_argument("--dt", type=float, default=120.0)
    p.add_argument("--radius", type=float)
    p.add_argument("--mesh", type=Path)
    p.add_argument(
        "--gradient", type=float, default=1e-6, help="channel SA gradient, (g/kg)/m"
    )
    p.add_argument(
        "--wind", type=float, default=0.05, help="global wind stress amplitude, Pa"
    )
    p.add_argument("--redi", type=float, default=100.0)
    p.add_argument("--length", type=float, default=10000.0)
    p.add_argument("--viscosity", type=float, default=1e4)
    p.add_argument(
        "--biharmonic-viscosity",
        type=float,
        default=0.0,
        help="rotational biharmonic sink coefficient [m^4/s]",
    )
    p.add_argument("--backscatter-days", type=float, default=1.0)
    p.add_argument("--fraction", type=float, default=0.25)
    p.add_argument("--decay-days", type=float, default=30.0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--max-wall-seconds", type=float, default=7200.0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if (
        args.layers < 2
        or args.days <= 0
        or args.dt <= 0
        or args.log_every < 1
        or args.threads < 1
    ):
        p.error(
            "positive duration/dt/log interval/thread count and at least two layers required"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    sources = {
        str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
        for folder in (
            "validation/temperature_coordinate",
            "validation/ocean_evaluation",
        )
        for f in (ROOT / folder).glob("*.py")
    }
    write_json(
        args.output / "provenance.json",
        dict(
            revision=revision,
            source_sha256=sources,
            python=sys.version,
            torch=str(torch.__version__),
            platform=platform.platform(),
            vfs=vfs.__file__,
        ),
    )
    variants = (
        ["resolved"] if args.case in ("rest", "stratified-rest") else args.variants
    )
    results = [run(args, v) for v in variants]
    write_json(args.output / "campaign.json", results)
    return 0 if all(r["status"] == "completed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
