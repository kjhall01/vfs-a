"""Isolated eddy front relaxation and timestep refinement; no resolved dynamics.

Run from an installed VFS environment:
  python validation/temperature_coordinate/eddy_campaign.py --output eddy.json
A small-radius sphere makes lateral eddy transport measurable in two days.
This is an operator experiment, not an Earth climate or baroclinic-instability run.
"""

import argparse
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocean_column"))
from roquet import RoquetOcean
from eddy import GMRedi
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn


def make(cells=32, layers=8, radius=100000.0):
    mesh = VoronoiMesh(n_cells=cells, radius=radius, lloyd_iters=2, verbose=False)
    model = RoquetOcean(mesh, OceanColumn(layers, total_depth=1000.0))
    h = torch.full((cells, layers), 1000.0 / layers)
    z = -(torch.arange(layers) + 0.5) * 1000.0 / layers
    ct = torch.full_like(h, 10.0)
    sa = 35.0 - 0.001 * z[None, :] + 0.15 * mesh.cell_xyz[:, 1:2]
    tracer = (1.0 + 0.2 * mesh.cell_xyz[:, :1] + 0.1 * torch.cos(z / 1000.0))[..., None]
    state = model.state(
        u_e=torch.zeros(mesh.nEdges, layers),
        h=h,
        conservative_temperature=ct,
        absolute_salinity=sa,
        tracer=tracer,
        tke=1e-4,
    )
    return model, state


def run(model, initial, coefficients, dt, duration):
    eddy = GMRedi(*coefficients)
    state = initial
    n = round(duration / dt)
    if n < 1 or abs(n * dt - duration) > 1e-6:
        raise ValueError("each dt must divide duration exactly")
    energy = gm_energy = redi_energy = 0.0
    maximum_cfl = maximum_redi = maximum_div = 0.0
    for _ in range(n):
        state, report = eddy.apply(model, state, dt)
        energy += float(report.energy_change)
        gm_energy += float(report.gm_energy_change)
        redi_energy += float(report.redi_energy_change)
        maximum_cfl = max(maximum_cfl, float(report.gm_outflow_fraction.max()))
        maximum_redi = max(maximum_redi, float(report.redi_step_number.max()))
        maximum_div = max(maximum_div, float(report.volume_residual.abs().max()))
    s, t = initial.ocean, state.ocean
    mass = model.energy.mass_area(s) * s.h
    volume = s.mesh.cell_area[:, None] * s.h
    q0 = torch.stack((s.temperature(), s.salinity(), s.tracer_h[..., 0] / s.h), -1)
    q1 = torch.stack((t.temperature(), t.salinity(), t.tracer_h[..., 0] / t.h), -1)
    drift = (volume[..., None] * (q1 - q0)).sum((0, 1)) / (
        volume[..., None] * q0.abs()
    ).sum((0, 1))

    # Horizontal salinity contrast, removing the changing layer mean.
    def contrast(q):
        mean = (volume * q).sum(0) / volume.sum(0)
        return float((volume * (q - mean).square()).sum() / volume.sum())

    tracer_mean = (volume * q0[..., 2]).sum() / volume.sum()
    stats = dict(
        dt=dt,
        steps=n,
        relative_inventory_drift=drift.tolist(),
        gm_max_outflow_fraction=maximum_cfl,
        redi_max_step_number=maximum_redi,
        max_volume_residual_m3_s=maximum_div,
        energy_change_J_kg=energy / float(mass.sum()),
        gm_energy_change_J_kg=gm_energy / float(mass.sum()),
        redi_energy_change_J_kg=redi_energy / float(mass.sum()),
        salinity_contrast_initial=contrast(s.salinity()),
        salinity_contrast_final=contrast(t.salinity()),
        tracer_variance_initial=float(
            (volume * (q0[..., 2] - tracer_mean).square()).sum() / volume.sum()
        ),
        tracer_variance_final=float(
            (volume * (q1[..., 2] - tracer_mean).square()).sum() / volume.sum()
        ),
    )
    return q1, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells", type=int, default=32)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--radius", type=float, default=100000.0)
    p.add_argument("--days", type=float, default=2.0)
    p.add_argument(
        "--dt", type=float, nargs="+", default=[3600.0, 1800.0, 900.0, 450.0]
    )
    p.add_argument("--kappa", type=float, default=1000.0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    results = dict(
        config={**vars(args), "output": str(args.output)},
        torch=torch.__version__,
        dtype="float64",
        device="cpu",
        threads=1,
        cases={},
    )
    model, initial = make(args.cells, args.layers, args.radius)
    volume = initial.ocean.mesh.cell_area[:, None] * initial.ocean.h
    with torch.no_grad():
        for name, coefficients in [
            ("gm", (args.kappa, 0.0)),
            ("redi", (0.0, args.kappa)),
            ("gm_redi", (args.kappa, args.kappa)),
        ]:
            runs = [
                run(model, initial, coefficients, dt, args.days * 86400.0)
                for dt in sorted(set(args.dt), reverse=True)
            ]
            reference = runs[-1][0]
            for values, stats in runs:
                error = (
                    (volume[..., None] * (values - reference).square()).sum((0, 1))
                    / volume.sum()
                ).sqrt()
                stats["CT_SA_tracer_rms_vs_finest_dt"] = error.tolist()
            results["cases"][name] = [stats for _, stats in runs]
            print(name, json.dumps(results["cases"][name][0]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
