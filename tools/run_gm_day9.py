"""Replay the day-8 coupled checkpoint to day 9 with the experimental GM parcel flux."""
import argparse
import csv
import hashlib
import inspect
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EDDY = "validation/temperature_coordinate/eddy.py"
PARENT_SHA256 = "cff94479234142d0274c8fc101b981ff7cfaa39561ce637a89665f8d7f503849"
PARCEL_SHA256 = "f522418332ac5dcd35bdae656f9efc8ee35194fd4694bad943101c236c1f35c8"
START, END = 692400, 777600


def verify_fork(old, new):
    """Admit exactly the tested GM source transition. The campaign checks both fingerprints."""
    changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    if changed != {EDDY} or (old[EDDY], new[EDDY]) != (PARENT_SHA256, PARCEL_SHA256):
        raise ValueError(f"Expected only the parcel-energy GM change. Changed sources: {sorted(changed)}")
    return {"experiment": "donor-parcel GM energy flux with strict local funding",
            "source_change": {EDDY: {"parent": PARENT_SHA256, "candidate": PARCEL_SHA256}}}


def load_campaign():
    sys.path.insert(0, str(ROOT))
    from workflows.coupled import campaign

    paths = [(Path(campaign.__file__), ROOT / "workflows/coupled/campaign.py"),
             (Path(inspect.getfile(campaign.GMRedi)), ROOT / EDDY)]
    for loaded, expected in paths:
        if not loaded.samefile(expected):
            raise RuntimeError(f"Imported {loaded.resolve()}. Expected {expected.resolve()}.")
    if hashlib.sha256((ROOT / EDDY).read_bytes()).hexdigest() != PARCEL_SHA256:
        raise ValueError(f"The GM implementation in {ROOT} differs from the tested parcel flux.")
    return campaign


@contextmanager
def reserve_output(output):
    """Reuse an empty directory left before startup, and keep concurrent runs apart on macOS/Linux."""
    import fcntl

    output.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another replay is using {output}.") from error
        if any(output.iterdir()):
            raise ValueError(f"Output contains files. Choose a new directory with --output: {output}")
        yield
    finally:
        os.close(fd)


def record_funding_failures(campaign, output):
    native = campaign.GMRedi

    class ReportingGM(native):
        @staticmethod
        def _settle(model, before, transported, energy_inflow, name):
            try:
                return native._settle(model, before, transported, energy_inflow, name)
            except FloatingPointError as error:
                if "requires unavailable local eddy energy" in str(error):
                    h = before.ocean.h
                    work = model.hamiltonian.finite_change(before, transported) - energy_inflow
                    debit = work / (model.hamiltonian.mass_area(before.ocean) * h)
                    available = transported.eddy_h / h
                    balance = available - debit
                    cell, layer = divmod(int(balance.argmin()), h.shape[1])
                    site = (cell, layer)
                    campaign.atomic_json(output / "gm_budget_failure.json", {
                        "operator": name, "cell": cell, "layer_zero_based": layer,
                        "available_J_kg": float(available[site]), "debit_J_kg": float(debit[site]),
                        "balance_J_kg": float(balance[site]), "unfunded_cell_layers": int((balance < 0).sum()),
                    })
                raise

    campaign.GMRedi = ReportingGM


def comparison_values(run):
    status = json.loads((run / "status.json").read_text())
    if status.get("status") != "segment_completed" or status["latest"]["seconds"] != END:
        raise ValueError(f"Expected a completed day-9 segment in {run}")
    with (run / "diagnostics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or (int(rows[0]["seconds"]), int(rows[-1]["seconds"])) != (START, END):
        raise ValueError(f"Unexpected comparison window in {run}")
    interval = ("numerical_energy_residual_J_m2", "gm_exchange_J_m2", "redi_exchange_J_m2")
    endpoint = ("min_eddy_energy_J_kg", "water_residual_kg_m2", "salt_relative")
    values = {k: float(rows[-1][k]) - float(rows[0][k]) for k in interval}
    values.update({k: float(rows[-1][k]) for k in endpoint})
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError(f"Nonfinite day-9 diagnostic in {run}")
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    desktop = Path.home() / "Desktop"
    parser.add_argument("--parent", type=Path, default=desktop / "vfs-a/run-4096-ocean600-30day")
    parser.add_argument("--limiter-run", type=Path, help="Completed limiter replay to compare with the candidate")
    parser.add_argument("--output", type=Path, default=desktop / "vfs-a/gm-parcel-day9")
    parser.add_argument("--source", type=Path, default=desktop / "PyAM21")
    parser.add_argument("--radiation-data", type=Path, default=desktop / "sea_esf")
    parser.add_argument("--mesh", type=Path)
    args = parser.parse_args(argv)
    config = args.parent / "configuration.json"
    manifest = args.parent / "manifest.json"
    restart = args.parent / "last_good_restart.pt"
    prior = json.loads(manifest.read_text())
    mesh = args.mesh or Path(prior["mesh"]["path"])
    for path in (config, restart, mesh, args.source, args.radiation_data):
        if not path.exists():
            parser.error(f"Missing input: {path}")

    import torch
    payload = torch.load(restart, weights_only=True, map_location="cpu")
    if (payload["fingerprint"] != prior["fingerprint"] or payload["air"]["time_seconds"] != START
            or payload["ledger"]["steps"] != 577):
        parser.error("Expected the original 692400-second checkpoint at step 577.")
    del payload
    limiter_run = args.limiter_run or desktop / "vfs-a-gm-limiter/gm-limiter-day8-replay"
    reference = None
    if args.limiter_run is not None or (limiter_run / "status.json").is_file():
        reference_manifest = json.loads((limiter_run / "manifest.json").read_text())
        if reference_manifest["fingerprint"] != prior["fingerprint"]:
            parser.error(f"Limiter run has a different parent model fingerprint: {limiter_run}")
        reference = comparison_values(limiter_run)
    else:
        print(f"Running without a reference comparison. No limiter results found at {limiter_run}.", flush=True)
    campaign = load_campaign()
    campaign.verify_compatible_source_delta = verify_fork
    output = args.output.resolve()
    record_funding_failures(campaign, output)
    campaign_args = [
        "--config", str(config), "--mesh", str(mesh), "--source", str(args.source),
        "--radiation-data", str(args.radiation_data), "--resume", str(restart),
        "--resume-compatible-manifest", str(manifest), "--stop-after-days", "9",
        "--output", str(output), "--verbose",
    ]
    with reserve_output(output):
        status = campaign.main(campaign_args)
        if status == 0 and reference is not None:
            candidate = comparison_values(output)
            metrics = {k: {"limiter": reference[k], "parcel": v, "parcel_minus_limiter": v - reference[k]}
                       for k, v in candidate.items()}
            report = {"start_seconds": START, "end_seconds": END,
                      "limiter_run": str(limiter_run.resolve()),
                      "interval_changes": ["numerical_energy_residual_J_m2", "gm_exchange_J_m2", "redi_exchange_J_m2"],
                      "metrics": metrics}
            campaign.atomic_json(output / "day9_comparison.json", report)
            print(f"Day-9 comparison: {output / 'day9_comparison.json'}", flush=True)
        return status


if __name__ == "__main__":
    raise SystemExit(main())
