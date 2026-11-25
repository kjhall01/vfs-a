"""Finite budget checks and separate eager CPU timing of optional backscatter."""

import argparse
import gc
import json
import resource
import statistics
import sys
import time
from pathlib import Path
import torch

sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "ocean_column"),
]
from vfs.ocean.temperature.backscatter import EddyBackscatter
from vfs.ocean.temperature.eddy import GMRedi
from vfs.ocean.temperature.eddy_energy import EddyDissipation
from eddy_feedback_campaign import channel
from eddy_performance import wave


def closure_step(model, state, op, dt):
    """Isolated SSPRK3 verification, identical weights to the model integrator."""
    from dataclasses import replace

    s = state.ocean

    def blend(a, b, wa, wb):
        return replace(
            a,
            ocean=s.with_tensors(
                wa * a.ocean.u_e + wb * b.ocean.u_e, s.h, s.temp_h, s.salt_h, s.tracer_h
            ),
            eddy_h=wa * a.eddy_h + wb * b.eddy_h,
        )

    def euler(x):
        du, de, r = op.tendency(model, x)
        out = replace(
            x,
            ocean=s.with_tensors(
                x.ocean.u_e + dt * du, s.h, s.temp_h, s.salt_h, s.tracer_h
            ),
            eddy_h=x.eddy_h + dt * de,
        )
        model.check(out)
        return out, r

    first, r1 = euler(state)
    second, r2 = euler(first)
    third, r3 = euler(blend(state, second, 0.75, 0.25))
    return blend(state, third, 1 / 3, 2 / 3), (r1, r2, r3)


def budgets(biharmonic_viscosity=0.0):
    cases = []
    for dynamics, viscosity in ((False, 1e7), (True, 1e7), (False, 1e12), (True, 1e12)):
        m, initial = channel(128, 8)
        state = initial
        op = EddyBackscatter(viscosity, 86400.0, 0.25, biharmonic_viscosity)
        if dynamics:
            m.backscatter = op
        maxlocal = 0.0
        work = 0.0
        residual = 0.0
        for _ in range(20):
            if dynamics:
                state, report = m.step_with_report(state, 10.0)
                reports = report.backscatter
            else:
                state, reports = closure_step(m, state, op, 10.0)
            mass = m.energy.mass_area(state.ocean) * state.ocean.h
            wet = mass > 0
            for weight, r in zip((1 / 6, 1 / 6, 2 / 3), reports):
                work += float(10 * weight * r.kinetic_power / mass.sum())
                residual += float(
                    10 * weight * r.local_power_residual.sum() / mass.sum()
                )
                maxlocal = max(
                    maxlocal,
                    float((r.local_power_residual[wet].abs() / mass[wet]).max()),
                )
        cases.append(
            dict(
                dynamics=dynamics,
                viscosity=viscosity,
                steps=20,
                dt=10.0,
                integrated_backscatter_power_J_kg=work,
                integrated_pairing_residual_J_kg=residual,
                max_local_power_residual_W_kg=maxlocal,
                direct_total_energy_change_J_kg=float(
                    (m.total_energy(state) - m.total_energy(initial)) / mass.sum()
                ),
                minimum_eddy_energy_J_kg=float(state.specific_eddy_energy()[wet].min()),
                minimum_vertex_scale=float(reports[-1].vertex_scale.min()),
            )
        )
    return cases


def refinement(biharmonic_viscosity=0.0):
    m, initial = channel(64, 4)
    op = EddyBackscatter(1e7, 1000.0, 0.5, biharmonic_viscosity)
    mass = (m.energy.mass_area(initial.ocean) * initial.ocean.h).sum()

    def sector(x):
        return (
            m.hamiltonian.base.mechanical.terms(x.ocean)["kinetic"]
            + (m.energy.mass_area(x.ocean) * x.eddy_h).sum()
        )

    rows = []
    for dt in (100.0, 50.0, 25.0, 12.5):
        x = initial
        for _ in range(round(400 / dt)):
            x, _ = closure_step(m, x, op, dt)
        rows.append(
            dict(dt=dt, energy_error_J_kg=float((sector(x) - sector(initial)) / mass))
        )
    return rows


def timing(cells, layers, biharmonic_viscosity=0.0, *, full_step=False):
    m, state = wave(cells, layers)
    state = m.with_eddy_reservoir(state, 0.1)
    op = EddyBackscatter(
        viscosity=1e4,
        timescale=86400.0,
        fraction=0.25,
        biharmonic_viscosity=biharmonic_viscosity,
    )
    if full_step:
        m.backscatter = op
        evaluate = lambda: m.step_with_report(state, 1.0)
    else:
        evaluate = lambda: op.tendency(m, state)
    evaluate()
    samples = []
    for _ in range(3):
        gc.collect()
        start = time.perf_counter()
        result = evaluate()
        samples.append(time.perf_counter() - start)
        del result
    return dict(
        full_step=full_step,
        biharmonic_viscosity=biharmonic_viscosity,
        cells=cells,
        layers=layers,
        samples_s=samples,
        median_s=statistics.median(samples),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        threads=1,
        compiled=False,
        dtype="float64",
        torch=str(torch.__version__),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("budgets", "time", "step-time", "refinement"),
        default="budgets",
    )
    p.add_argument("--biharmonic-viscosity", type=float, default=0.0)
    p.add_argument("--cells", type=int, default=16384)
    p.add_argument("--layers", type=int, default=40)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    with torch.no_grad():
        result = (
            budgets(a.biharmonic_viscosity)
            if a.mode == "budgets"
            else (
                refinement(a.biharmonic_viscosity)
                if a.mode == "refinement"
                else timing(
                    a.cells,
                    a.layers,
                    a.biharmonic_viscosity,
                    full_step=a.mode == "step-time",
                )
            )
        )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
