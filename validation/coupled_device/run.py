"""Compare full seasonal CPU and GPU campaigns on exactly the same mesh/config.

Real radiation data are mandatory. Run through at least a fresh/cache/fresh
radiation sequence; report field differences, independent budgets, completed
interval timings and same-device exact restart replay. This is a short
qualification report, not a climate-stability or cross-device bitwise claim.
"""
import argparse
from dataclasses import asdict, replace
import gc
from pathlib import Path
import statistics
import sys
import time
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from workflows.coupled import campaign as c
from workflows.coupled.device import execution_context, resolve_device, runtime_manifest, synchronize
from workflows.pyam21.state import copy_tree


def tensors(value, path=""):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensors(item, f"{path}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from tensors(item, f"{path}/{index}")


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def differences(left, right):
    """Report every tensor; metadata fingerprints intentionally differ by device."""
    a, b = dict(tensors(left)), dict(tensors(right))
    if a.keys() != b.keys():
        raise AssertionError("CPU/device state structures differ")
    result = {}
    for name, x in a.items():
        y = b[name]
        if x.shape != y.shape or x.is_floating_point() != y.is_floating_point():
            raise AssertionError(f"CPU/device shape or tensor kind differs: {name}")
        if not x.is_floating_point():
            if x.dtype != y.dtype:
                raise AssertionError(f"integer/bool dtype differs: {name}")
            result[name] = dict(unequal=int((x != y).sum()))
        else:
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError(f"nonfinite state: {name}")
            # Reporting boundary only: compare on CPU in FP64, including MPS
            # results. Never request unsupported doubles on the MPS device.
            dx, dy = x.to(device="cpu", dtype=torch.float64), y.to(device="cpu", dtype=torch.float64)
            error = float((dx - dy).abs().max()) if x.numel() else 0.0
            scale = float(dx.abs().max()) if x.numel() else 0.0
            result[name] = dict(max_absolute=error, max_scaled=error / max(1.0, scale),
                                reference_dtype=str(x.dtype), candidate_dtype=str(y.dtype))
    return result


def run_mode(config, source, data, xyz, steps, warmup, directory, progress):
    device = resolve_device(config.device)
    with execution_context(device):
        progress("building", dict(configuration=asdict(config)))
        start = time.perf_counter()
        adapter, air, receiver, sea = c.build(config, source, data, generator_xyz=xyz)
        synchronize(device)
        initial = c.inventories(adapter, air, receiver, sea)
        identity = c.fingerprint(config, adapter, c.source_hashes(), False)
        row = dict(configuration=asdict(config), execution=runtime_manifest(device, config.dtype),
                   compilation=adapter.compilation_manifest, radiation=adapter.radiation_manifest,
                   setup_seconds=time.perf_counter() - start, initial=initial, intervals=[])
        external = upstream = 0.0
        for step in range(steps):
            if step == steps - 1:
                c.checkpoint(directory / "replay_input.pt",
                             c.pack(adapter, air, receiver, sea, identity, {}))
            synchronize(device)
            start = time.perf_counter()
            air, sea, report = c.advance(adapter, air, receiver, sea, config.physics_dt, config)
            synchronize(device)
            elapsed = time.perf_counter() - start
            external += report["external"]
            upstream += report["upstream"]
            current = c.inventories(adapter, air, receiver, sea)
            row["intervals"].append(dict(seconds=air.time_seconds, wall_seconds=elapsed,
                report=report, inventories=current,
                numerical_energy_residual_J_m2=current["energy"] - initial["energy"] - external - upstream,
                water_residual_kg_m2=current["water"] - initial["water"],
                salt_relative=(current["salt"] - initial["salt"]) / initial["salt"]))
            for name, value in tensors(c.pack(adapter, air, receiver, sea, identity, {})):
                if value.device != device:
                    raise AssertionError(f"state tensor escaped {device}: {name} on {value.device}")
                if value.is_floating_point() and str(value.dtype).removeprefix("torch.") != config.dtype:
                    raise AssertionError(f"state tensor changed precision: {name} has {value.dtype}")
            progress(f"interval {step + 1}/{steps}: {elapsed:.3f} s", row)
        final = copy_tree(c.pack(adapter, air, receiver, sea, identity, {}), device="cpu")
        payload = torch.load(directory / "replay_input.pt", map_location="cpu", weights_only=True)
        ra, rs, _ = c.unpack(payload, adapter, receiver, sea, identity)
        ra, rs, rr = c.advance(adapter, ra, receiver, rs, config.physics_dt, config)
        synchronize(device)
        replay = copy_tree(c.pack(adapter, ra, receiver, rs, identity, {}), device="cpu")
        row["restart_bitwise"] = equal(final, replay) and equal(report, rr)
        if not row["restart_bitwise"]:
            raise AssertionError("same-device restart replay differs")
        median = statistics.median(x["wall_seconds"] for x in row["intervals"][warmup:])
        row.update(steady_median_seconds=median, steady_sypd=config.physics_dt / (365 * median))
        if device.type == "cuda":
            row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        elif device.type == "mps":
            row["current_allocated_bytes"] = torch.mps.current_allocated_memory()
            row["driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
        c.checkpoint(directory / "final.pt", final)
        c.snapshot(directory / "final.nc", adapter, air, receiver, sea)
        return row, final


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--radiation-data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "workflows/coupled/coupled_aquaplanet_1yr.yaml")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument("--reference-dtype", choices=("float32", "float64"),
                        help="CPU reference precision (default: same as --dtype/YAML)")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = c.CampaignConfig(**yaml.safe_load(args.config.read_text()))
    config = replace(config, device=args.device, **{name: getattr(args, name)
        for name in ("threads", "dtype") if getattr(args, name) is not None})
    config.validate()
    if args.warmup < 1 or args.steps <= args.warmup:
        parser.error("require steps > warmup >= 1")
    if args.steps < config.radiation_dt // config.physics_dt + 1:
        parser.error("need enough steps to include a second fresh radiation call")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new/empty")
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(status="running", torch=str(torch.__version__), modes={})
    try:
        device = resolve_device(args.device)
        config, xyz, mesh = c.load_mesh(config, args.mesh)
        result.update(configuration=asdict(config), mesh=mesh, source_sha256=c.source_hashes(),
                      steps=args.steps, warmup=args.warmup, radiation_enabled=True)
        payloads = {}
        for label, target, compiled in (("cpu", "cpu", False), ("device", str(device), args.compile)):
            directory = args.output / label
            directory.mkdir()
            if target.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            def progress(message, row):
                result["modes"][label] = row
                c.atomic_json(args.output / "comparison.json", result)
                print(f"{label}: {message}", flush=True)
            precision = args.reference_dtype if label == "cpu" and args.reference_dtype else config.dtype
            row, final = run_mode(replace(config, device=target, dtype=precision, compile=compiled),
                args.source, args.radiation_data, xyz, args.steps, args.warmup, directory, progress)
            result["modes"][label] = row
            payloads[label] = final
            gc.collect()
        result["state_differences"] = differences(payloads["cpu"], payloads["device"])
        result["steady_speedup"] = result["modes"]["cpu"]["steady_median_seconds"] / result["modes"]["device"]["steady_median_seconds"]
        result["status"] = "comparison_completed"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        c.atomic_json(args.output / "comparison.json", result)
        raise
    c.atomic_json(args.output / "comparison.json", result)
    print(f"Comparison completed; steady speedup {result['steady_speedup']:.3f}; inspect comparison.json for budgets and field differences", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
