"""Paired CPU FP64 ocean replay of halving and direct GM streamfunction funding.

Uses the original strict day-10 restart. Radiation and atmosphere are outside
this 1200 s pre-exchange ocean/ice window. Both solvers see identical input
for each timing pair. Per-step transport is compared against each solver's
own unrestricted proposal at that step. The diverging trajectories are kept.
"""
import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
import sys
from time import perf_counter
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parent
sys.path[:0] = [str(TOOLS), str(TOOLS.parent)]

import torch
from gm_parcel_limiter import make_experimental_gm_class
from run_gm_day9 import load_campaign, reserve_output
from vfs.ocean.transport import divergence

CASES = ("well_funded", "first_failure", "window", "sustained")
SOLVERS = ("halving", "direct")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_start(args, campaign):
    manifest = json.loads(args.manifest.read_text())
    if manifest["source_sha256"] != campaign.source_hashes():
        raise ValueError("Use the original strict donor-parcel manifest and matching VFS sources.")
    cfg = campaign.CampaignConfig(**manifest["configuration"])
    cfg = replace(cfg, compile=False, device="cpu", dtype="float64", threads=args.threads or cfg.threads)
    cfg, generators, mesh = campaign.load_mesh(cfg, args.mesh)
    if mesh["sha256"] != manifest["mesh"]["sha256"]:
        raise ValueError("Mesh and manifest differ.")
    payload = torch.load(args.resume, weights_only=True, map_location="cpu")
    if (payload["schema"] != 1 or payload["fingerprint"] != manifest["fingerprint"]
            or payload["air"]["time_seconds"] != 936000):
        raise ValueError("Expected the strict day-10 restart at second 936000.")
    sys.path.insert(0, str(args.source / "src"))
    _, _, receiver, template = campaign.build(
        cfg, args.source, None, smoke_no_radiation=True, generator_xyz=generators)
    ocean = template.coupled.ocean
    ocean = replace(ocean, ocean=ocean.ocean.with_tensors(**payload["ocean"]),
                    tke_h=payload["tke_h"], eddy_h=payload["eddy_h"])
    ice = replace(template.coupled.ice, **payload["ice"])
    sea = replace(template, coupled=replace(template.coupled, ocean=ocean, ice=ice),
                  snow_mass=payload["snow_mass"], cold_content=payload["cold_content"])
    receiver.snow.check(sea)
    return cfg, receiver, sea


def ocean_window(campaign, cfg, receiver, sea):
    for offset in range(0, cfg.physics_dt, cfg.ocean_dt):
        dt = min(cfg.ocean_dt, cfg.physics_dt - offset)
        sea, _ = receiver.snow.step(sea, dt)
        sea, _ = campaign.mesoscale_step(receiver, sea, dt, cfg)
    return sea.coupled.ocean


def find_inputs(campaign, cfg, receiver, sea):
    """Find the first native GM failure, retaining two immutable step inputs."""
    states = []
    steps = 0

    class Probe(campaign.GMRedi):
        def apply(self, model, state, dt):
            nonlocal steps
            steps += 1
            if steps == 1:
                states.append(state)
            try:
                return super().apply(model, state, dt)
            except FloatingPointError as error:
                if "GM requires unavailable local eddy energy" in str(error):
                    states.append(state)
                raise

    with patch.object(campaign, "GMRedi", Probe):
        try:
            ocean_window(campaign, cfg, receiver, sea)
        except FloatingPointError:
            if len(states) != 2:
                raise
    if len(states) != 2 or steps == 1:
        raise ValueError("Expected both a well-funded step and a later native GM funding failure.")
    return states[0], states[1], steps


def make_observer(native, solver, directory, rows):
    """Use the same proposal/acceptance reductions and per-step I/O in both arms."""
    from gm_direct_funding import make_direct_gm_class

    events = []
    event_log = directory / "limiter_events.jsonl"

    class ObserveNative(native):
        def _geometry(self, model, state, *, part="all", metrics=None, gm_diffusivity=None):
            geom = super()._geometry(model, state, part=part, metrics=metrics, gm_diffusivity=gm_diffusivity)
            if part == "gm" and self.proposed is None:
                self.proposed = geom[1:4]
                self.edge_length = metrics[6]
            return geom

    if solver == "halving":
        limited = make_experimental_gm_class(ObserveNative, divergence, event_log)
    else:
        limited = make_direct_gm_class(ObserveNative, divergence, events=events)

    class Measure(limited):
        def _settle(self, model, before, transported, energy_inflow, name):
            if name == "GM":
                self.calls += 1
            return super()._settle(model, before, transported, energy_inflow, name)

        def apply(self, model, state, dt):
            self.calls = 0
            self.proposed = None
            event_count = len(events)
            end, report = super().apply(model, state, dt)
            if solver == "direct" and len(events) > event_count:
                event = events[-1]
                with event_log.open("a") as handle:
                    handle.write(json.dumps(event) + "\n")
                print("DIRECT GM FUNDING: " + json.dumps(event), flush=True)
            psi, horizontal, vertical = self.proposed
            before = psi[:, 1:-1]
            after = report.streamfunction[:, 1:-1]
            raw = self.edge_length * before
            changed = before != after
            area = state.ocean.mesh.cell_area[:, None]
            row = dict(
                gm_attempts=self.calls,
                min_eddy_J_kg=float(end.specific_eddy_energy().min()),
                max_energy_residual_J_m2=max(float((x / area).abs().max()) for x in (
                    report.gm_local_energy_residual, report.redi_local_energy_residual)),
                max_volume_divergence_m3_s=float(report.volume_residual.abs().max()),
                proposed_m3=dt * float(raw.abs().sum()),
                proposed_changed_m3=dt * float(raw[changed].abs().sum()),
                cut_m3=dt * float((self.edge_length * (before - after)).abs().sum()),
            )
            for axis, original, final in (("horizontal", horizontal, report.horizontal_transport),
                                          ("vertical", vertical, report.vertical_transport)):
                row[axis + "_proposed_m3"] = dt * float(original.abs().sum())
                row[axis + "_change_m3"] = dt * float((final - original).abs().sum()) if self.calls > 1 else 0.
            rows.append(row)
            with (directory / "transport.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            self.proposed = None
            return end, report

    return Measure


def trial(campaign, cfg, receiver, sea, inputs, case, solver, directory, args):
    directory.mkdir(parents=True)
    rows = []
    op_class = make_observer(campaign.GMRedi, solver, directory, rows)
    model = receiver.H.model
    op = op_class(kappa_redi=cfg.redi_diffusivity_m2_s, positive_redi=cfg.redi_positive_transport,
                  energy_closure=campaign.EddyMixingLength(
                      cfg.gm_length_m, cfg.gm_efficiency, cfg.gm_kinetic_fraction))
    state = inputs[0 if case == "well_funded" else 1]
    model_dt = cfg.ocean_dt / cfg.eddy_substeps
    with (directory / "stdout.log").open("w") as log, redirect_stdout(log):
        start = perf_counter()
        if case == "window":
            with patch.object(campaign, "GMRedi", op_class):
                state = ocean_window(campaign, cfg, receiver, sea)
        elif case == "sustained":
            for _ in range(args.stress_steps):
                state, _ = op.apply(model, state, model_dt)
        else:
            initial = state
            for _ in range(args.fixed_steps):
                state, _ = op.apply(model, initial, model_dt)
        elapsed = perf_counter() - start
    path = directory / "limiter_events.jsonl"
    events = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    changed = sum(r["proposed_changed_m3"] for r in rows)
    cut = sum(r["cut_m3"] for r in rows)
    metrics = dict(
        seconds=elapsed,
        gm_steps=len(rows),
        gm_attempts=sum(r["gm_attempts"] for r in rows),
        limited_steps=len(events),
        fallback_events=sum(bool(e.get("fallback", False)) for e in events),
        gm_steps_per_second=len(rows) / elapsed,
        model_seconds_per_wall_second=len(rows) * model_dt / elapsed,
        min_eddy_J_kg=min(r["min_eddy_J_kg"] for r in rows),
        end_min_eddy_J_kg=float(state.specific_eddy_energy().min()),
        max_energy_residual_J_m2=max(r["max_energy_residual_J_m2"] for r in rows),
        max_volume_divergence_m3_s=max(r["max_volume_divergence_m3_s"] for r in rows),
        cut_percent_all=100 * cut / sum(r["proposed_m3"] for r in rows),
        cut_percent_changed=100 * cut / changed if changed else None,
    )
    for axis in ("horizontal", "vertical"):
        metrics[axis + "_change_percent_all"] = (
            100 * sum(r[axis + "_change_m3"] for r in rows) / sum(r[axis + "_proposed_m3"] for r in rows))
    return metrics, state


def state_equal(a, b):
    return (all(torch.equal(getattr(a.ocean, k), getattr(b.ocean, k)) for k in (
                "h", "u_e", "temp_h", "salt_h", "tracer_h"))
            and torch.equal(a.eddy_h, b.eddy_h) and torch.equal(a.tke_h, b.tke_h))


def summary(pairs):
    result = {}
    for name in SOLVERS:
        result[name] = dict(median_wall_s=median(p[name]["seconds"] for p in pairs),
                            median_GM_steps_s=median(p[name]["gm_steps_per_second"] for p in pairs))
    ratios = [p["halving"]["seconds"] / p["direct"]["seconds"] for p in pairs]
    return dict(**result, paired_speedup_median=median(ratios), paired_speedup_range=[min(ratios), max(ratios)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("manifest", "resume", "source", "mesh", "output"):
        parser.add_argument("--" + arg, required=True, type=Path)
    parser.add_argument("--threads", type=int, help="defaults to the checkpoint configuration")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1, help="untimed pairs per case (excluded)")
    parser.add_argument("--fixed-steps", type=int, default=4, help="identical frozen inputs per timed trial")
    parser.add_argument("--stress-steps", type=int, default=30)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    args = parser.parse_args()
    if min(args.repeats, args.fixed_steps, args.stress_steps) < 1 or args.warmups < 0:
        parser.error("Use positive repeat/step counts and nonnegative warmups.")
    campaign = load_campaign()
    cfg, receiver, sea = read_start(args, campaign)
    well, failed, step = find_inputs(campaign, cfg, receiver, sea)
    results = dict(
        cases={}, summary={}, strict_failure_step=step,
        torch=str(torch.__version__), platform=platform.platform(), threads=torch.get_num_threads(),
        mode="CPU float64 eager with identical per-step reductions and event/transport JSONL I/O",
        mesh_sha256=sha256(args.mesh), checkpoint_sha256=sha256(args.resume),
        sources={p: sha256(TOOLS / p) for p in (
            "benchmark_gm_funding.py", "gm_parcel_limiter.py", "gm_direct_funding.py")},
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    with reserve_output(args.output):
        for case in args.cases:
            pairs = results["cases"][case] = []
            for n in range(-args.warmups, args.repeats):
                pair, states = {}, {}
                order = SOLVERS if n % 2 == 0 else SOLVERS[::-1]
                for solver in order:
                    folder = args.output / case / f"pair-{n}-{solver}"
                    pair[solver], states[solver] = trial(
                        campaign, cfg, receiver, sea, (well, failed), case, solver, folder, args)
                    print(f"{case} pair {n} {solver}: {pair[solver]['seconds']:.4f} s", flush=True)
                pair["end_state_bitwise_equal"] = state_equal(*states.values())
                if case == "well_funded" and not pair["end_state_bitwise_equal"]:
                    raise AssertionError("Well-funded state differed.")
                if n >= 0:
                    pairs.append(pair)
                    results["summary"][case] = summary(pairs)
                    campaign.atomic_json(args.output / "benchmark.json", results)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
