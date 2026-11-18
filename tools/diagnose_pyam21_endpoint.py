"""Replay one unchanged campaign interval and capture a PyAM21 endpoint mismatch.

Run on the original device/environment using the failed segment's configuration,
manifest and last_good_restart.pt. This tool lives outside the campaign's hashed
source directories. It neither bypasses restart identity nor changes a solver
check. A captured failure is diagnostic evidence, never a successful model step.
"""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch


ERROR = "PyAM21 endpoint and tendencies disagree beyond roundoff"
TRACERS = ("vapor", "liquid", "ice", "cloud_fraction")


def number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def endpoint_report(column, out, dt):
    """Evaluate the existing guard, in its original device/dtype arithmetic."""
    T, R = column["T"], column["tracers"]
    Tend, Rend = out["endpoint_temperature"], out["endpoint_tracers"]
    eps = torch.finfo(T.dtype).eps
    water_scale = torch.maximum(R[..., :3].abs().sum(-1).amax(),
                                Rend[..., :3].abs().sum(-1).amax())
    reconstructed_T = T + dt * out["tdt"]
    reconstructed_R = R + dt * out["rdt"]
    cases = [("temperature", T, out["tdt"], Tend, reconstructed_T,
              128 * eps * T.abs().amax())]
    for i, name in enumerate(TRACERS):
        tolerance = (water_scale if i < 3 else water_scale.new_ones(())) * (128 * eps)
        cases.append((name, R[..., i], out["rdt"][..., i], Rend[..., i],
                      reconstructed_R[..., i], tolerance))
    fields, selected = {}, set()
    failing_columns = torch.zeros(T.shape[0], device=T.device, dtype=torch.bool)
    for name, initial, tendency, actual, reconstructed, tolerance in cases:
        difference = (actual - reconstructed).abs()
        violation = difference > tolerance
        failing_columns |= violation.any(-1)
        flat = int(difference.argmax())
        cell, level = divmod(flat, difference.shape[-1])
        selected.add(cell)
        error = float(difference[cell, level])
        allowed = float(tolerance)
        ratio = error / allowed if allowed else (math.inf if error else 0.0)
        fields[name] = dict(
            units="K" if name == "temperature" else ("1" if name == "cloud_fraction" else "kg/kg"),
            max_absolute_difference=number(error), tolerance=number(allowed),
            max_error_over_tolerance=number(ratio),
            violations=int(violation.sum()), cell=cell, level=level,
            initial=number(initial[cell, level]), tendency=number(tendency[cell, level]),
            dt_tendency=number(dt * tendency[cell, level]),
            reconstructed=number(reconstructed[cell, level]), endpoint=number(actual[cell, level]),
        )
    # Keep all five worst locations and at most 32 other failing columns. The
    # column physics can then be replayed without a full global mesh or optics.
    selected.update(failing_columns.nonzero().flatten()[:32].cpu().tolist())
    return dict(device=str(T.device), dtype=str(T.dtype), dt=dt,
                water_scale=number(water_scale), fields=fields,
                failing_columns=int(failing_columns.sum()), selected_columns=sorted(selected))


def column_subset(value, indices, cells):
    """Detached CPU evidence, preserving dtype and all levels of selected cells."""
    if isinstance(value, torch.Tensor):
        if value.ndim and value.shape[0] == cells:
            value = value.index_select(0, torch.tensor(indices, device=value.device))
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: column_subset(v, indices, cells) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(column_subset(v, indices, cells) for v in value)
    return value


@contextmanager
def capture_failure(adapter_class, moist_class, output):
    """Observe the actual call; always re-raise the original rejection."""
    latest = {}
    original_moist = moist_class.__call__
    original_apply = adapter_class.apply_result

    def moist(self, *args, **kwargs):
        result = original_moist(self, *args, **kwargs)
        latest.update(args=args, kwargs=kwargs, result=result)
        return result

    def apply(self, state, dt, column, geom, out, surf, mem, rad, **kwargs):
        try:
            return original_apply(self, state, dt, column, geom, out, surf, mem, rad, **kwargs)
        except ValueError as error:
            if ERROR not in str(error):
                raise
            report = endpoint_report(column, out, dt)
            report.update(status="endpoint_mismatch_captured", diagnostic_only=True,
                          error=str(error), start_seconds=state.time_seconds)
            indices, cells = report["selected_columns"], column["T"].shape[0]
            evidence = dict(
                schema=1, report=report, namelist=self.configuration["namelist"],
                column=column, geometry=geom,
                output={k: out[k] for k in ("tdt", "rdt", "udt", "vdt",
                                            "endpoint_temperature", "endpoint_tracers")},
                moist_call=latest,
            )
            torch.save(column_subset(evidence, indices, cells), output / "endpoint_failure.pt")
            (output / "endpoint_failure.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print("Endpoint mismatch details:\n" + json.dumps(report, indent=2), flush=True)
            raise

    with patch.object(moist_class, "__call__", moist), patch.object(adapter_class, "apply_result", apply):
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--failed-run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--radiation-data", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    # Match campaign.py's import behavior: use installed VFS, and import only
    # the research/workflow modules from the explicit checkout.
    sys.path.append(str(args.repo.resolve()))
    from workflows.coupled import campaign
    from workflows.pyam21.adapter import AM2Adapter
    from pyam21.physics.moist import MoistProcesses

    manifest = json.loads((args.failed_run / "manifest.json").read_text())
    if not manifest["radiation_enabled"]:
        parser.error("this diagnostic requires the original real-radiation campaign")
    restart = args.failed_run / "last_good_restart.pt"
    if not restart.is_file():
        parser.error("failed run has no last_good_restart.pt")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new/empty")
    # Fail before costly model construction if fetching this tool also changed
    # any fingerprinted model source. Normal campaign checks still run as well.
    if campaign.source_hashes() != manifest["source_sha256"]:
        parser.error("model sources changed: use the original checkout and installed VFS; no restart bypass is permitted")
    args.output.mkdir(parents=True, exist_ok=True)
    config = manifest["configuration"]
    config_path = args.output / "replay_configuration.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    # The preflight does exactly one interval and, if it succeeds, verifies its
    # restart replay. The target-day CLI does not affect the restart identity.
    command = ["--config", str(config_path), "--mesh", str(args.mesh),
               "--source", str(args.source), "--radiation-data", str(args.radiation_data),
               "--resume", str(restart), "--output", str(args.output / "replay"),
               "--preflight", "--verbose"]
    with capture_failure(AM2Adapter, MoistProcesses, args.output):
        result = campaign.main(command)
    if (args.output / "endpoint_failure.json").exists():
        print("Captured the rejected step; upload endpoint_failure.json and endpoint_failure.pt. "
              "No failed model step was accepted.", flush=True)
        return 0  # the diagnostic capture succeeded; replay/status.json still says failed
    print("Endpoint mismatch was not captured; inspect replay/status.json and replay/run.log.", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
