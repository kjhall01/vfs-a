"""Time paired coupled advances from the same checkpoint, without writing a model restart."""
import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import json
import platform
from pathlib import Path
import statistics
import sys
import tempfile
import time
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))


def load_inputs(campaign, args, manifest):
    import torch
    from gm_parcel_limiter import experiment_sources

    cfg = campaign.CampaignConfig(**manifest["configuration"])
    mesh_file = args.mesh or Path(manifest["mesh"]["path"])
    cfg, xyz, mesh = campaign.load_mesh(cfg, mesh_file)
    if mesh["sha256"] != manifest["mesh"]["sha256"]:
        raise ValueError("Checkpoint mesh differs from supplied mesh")
    native = campaign.source_hashes()
    if manifest["source_sha256"] not in (native, experiment_sources(native)):
        raise ValueError("Expected the strict donor or day-11 limiter sources")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
    if checkpoint["fingerprint"] != manifest["fingerprint"] or checkpoint["schema"] != 1:
        raise ValueError("Checkpoint differs from supplied manifest")
    cfg = replace(cfg, compile=False) if args.radiation_off else cfg
    tables = None if args.radiation_off else args.radiation_data
    adapter, template_air, receiver, template = campaign.build(
        cfg, args.source, tables, smoke_no_radiation=args.radiation_off, generator_xyz=xyz
    )
    if args.radiation_off:
        # A timing-only ablation. Restore physical fields and column physics memory
        # onto the eager test owners. The real-radiation model's cache is inapplicable.
        p = campaign.copy_tree(checkpoint, device=adapter.model.mesh.cell_area.device)
        fields = p["air"]
        air = replace(template_air, atmosphere=replace(template_air.atmosphere, **fields["atmosphere"]),
                      cloud_fraction=fields["cloud_fraction"], memory=fields["memory"],
                      surface=fields["surface"], time_seconds=fields["time_seconds"])
        oc = template.coupled.ocean
        ocean = replace(oc, ocean=oc.ocean.with_tensors(**p["ocean"]),
                        tke_h=p["tke_h"], eddy_h=p["eddy_h"])
        coupled = replace(template.coupled, ocean=ocean, ice=replace(template.coupled.ice, **p["ice"]))
        sea = replace(template, coupled=coupled, snow_mass=p["snow_mass"], cold_content=p["cold_content"])
        adapter.check(air)
        receiver.snow.check(sea)
    else:
        if not manifest["radiation_enabled"]:
            raise ValueError("Sea ESF comparison needs a radiation-enabled checkpoint")
        identity = campaign.fingerprint(cfg, adapter, manifest["source_sha256"], False, mesh["sha256"])
        air, sea, _ = campaign.unpack(checkpoint, adapter, receiver, template, identity)
    return cfg, adapter, receiver, air, sea


def main():
    desktop = Path.home() / "Desktop"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, default=desktop / "vfs-a/gm-parcel-day30/last_good_restart.pt")
    parser.add_argument("--manifest", type=Path, help="Defaults to manifest.json beside the checkpoint")
    parser.add_argument("--source", type=Path, default=desktop / "PyAM21")
    parser.add_argument("--mesh", type=Path, help="Defaults to the saved generator path in the manifest")
    parser.add_argument("--radiation-data", type=Path, default=desktop / "sea_esf")
    parser.add_argument("--radiation-off", action="store_true",
                        help="Timing-only ablation with eager kernels, restored physical state and radiation disabled")
    parser.add_argument("--intervals", type=int, default=1, help="Consecutive coupled intervals per trial")
    parser.add_argument("--pairs", type=int, default=3, help="Paired trials, after one untimed pair")
    parser.add_argument("--report", type=Path, required=True, help="New JSON output file")
    args = parser.parse_args()
    if min(args.intervals, args.pairs) < 1:
        parser.error("--intervals and --pairs must both be positive")
    if args.report.exists():
        parser.error("--report must be a new file")
    import torch
    from run_gm_day9 import load_campaign
    sys.path.insert(0, str(args.source.resolve() / "src"))
    campaign = load_campaign()
    from benchmark_gm_funding import make_observer
    from workflows.coupled.device import execution_context, synchronize
    manifest = json.loads((args.manifest or args.resume.with_name("manifest.json")).read_text())
    device = manifest["configuration"]["device"]
    with torch.no_grad(), execution_context(device), tempfile.TemporaryDirectory() as tmp:
        cfg, adapter, receiver, initial_air, initial_sea = load_inputs(campaign, args, manifest)
        data = dict(torch=str(torch.__version__), platform=platform.platform(),
                    sources={name: hashlib.sha256((TOOLS / name).read_bytes()).hexdigest() for name in (
                        "benchmark_coupled_gm_funding.py", "benchmark_gm_funding.py",
                        "gm_direct_funding.py", "gm_parcel_limiter.py")},
                    checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
                    threads=cfg.threads, device=device, dtype=cfg.dtype,
                    mesh_sha256=manifest["mesh"]["sha256"], checkpoint_seconds=initial_air.time_seconds,
                    compiled=cfg.compile, radiation="off" if args.radiation_off else "Sea ESF",
                    intervals=args.intervals, interval_seconds=cfg.physics_dt, trials=[],
                    instrumentation="Matched per-substep transport and budget audit; no campaign snapshot I/O")
        simulated = cfg.physics_dt * args.intervals
        print(f"Benchmarking {args.intervals} interval(s) from {initial_air.time_seconds} s "
              f"with radiation {data['radiation']}", flush=True)
        for pair in range(args.pairs + 1):
            order = ["halving", "direct"] if pair % 2 == 0 else ["direct", "halving"]
            for name in order:
                trial_dir = Path(tmp) / f"{pair}-{name}"
                trial_dir.mkdir()
                rows = []
                cls = make_observer(campaign.GMRedi, name, trial_dir, rows)
                air, sea = initial_air, initial_sea
                local_residual = 0.0
                with patch.object(campaign, "GMRedi", cls), (Path(tmp) / "progress.log").open("w") as output, redirect_stdout(output):
                    synchronize(device)
                    start = time.perf_counter()
                    for _ in range(args.intervals):
                        air, sea, report = campaign.advance(adapter, air, receiver, sea, cfg.physics_dt, cfg)
                        local_residual = max(local_residual, report["gmredi_local_residual_J_m2"])
                    synchronize(device)
                    wall = time.perf_counter() - start
                if air.time_seconds != initial_air.time_seconds + simulated:
                    raise RuntimeError("Coupled run did not reach its target")
                log = trial_dir / "limiter_events.jsonl"
                events = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
                if pair:
                    data["trials"].append(dict(pair=pair, method=name, seconds=wall, events=len(events),
                                               gm_attempts=sum(row["gm_attempts"] for row in rows),
                                               fallbacks=sum(bool(e.get("fallback")) for e in events),
                                               min_eddy_J_kg=float(sea.coupled.ocean.specific_eddy_energy().min()),
                                               max_gmredi_residual_J_m2=local_residual))
                    print(f"Pair {pair} {name}: {wall:.3f} s", flush=True)
        medians = {method: statistics.median(t["seconds"] for t in data["trials"] if t["method"] == method)
                   for method in ("halving", "direct")}
        paired = [{t["method"]: t["seconds"] for t in data["trials"] if t["pair"] == pair}
                  for pair in range(1, args.pairs + 1)]
        ratios = [t["halving"] / t["direct"] for t in paired]
        data.update(median_seconds=medians, paired_speedup_median=statistics.median(ratios),
                    paired_speedup_range=[min(ratios), max(ratios)],
                    simulated_years_per_wall_day={k: simulated / (365 * t) for k, t in medians.items()})
    with args.report.open("x") as out:
        json.dump(data, out, indent=2, allow_nan=False)
        out.write("\n")
    print(f"Median paired halving/direct speedup {data['paired_speedup_median']:.3f}x. Results in {args.report}")


if __name__ == "__main__":
    main()
