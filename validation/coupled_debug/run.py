"""Radiation-OFF stage tracing and controls for the experimental campaign.

This is a debugging driver, not a seasonal integration or restart launcher.
Each run starts fresh and writes configuration, source hashes, stage extrema,
the last accepted checkpoint and a success/failure status. Exceptions fail the
process; no rejected interval is committed. Use campaign.py for normal runs.
"""

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
import traceback

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workflows.coupled import campaign as c
from workflows.coupled.driver import exchange_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--mode", choices=("coupled", "dynamics", "physics"), default="coupled")
    parser.add_argument("--config", type=Path, default=ROOT / "workflows/coupled/coupled_aquaplanet_1yr.yaml")
    args = parser.parse_args()
    config = c.CampaignConfig(**yaml.safe_load(args.config.read_text())).validate()
    config, xyz, mesh_info = c.load_mesh(config, args.mesh)
    end = args.hours * 3600
    if end <= 0 or end % config.physics_dt:
        parser.error("hours must be positive and divisible by the physics interval")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new/empty")
    args.output.mkdir(parents=True, exist_ok=True)
    adapter, air, receiver, sea = c.build(config, args.source, None, smoke_no_radiation=True, generator_xyz=xyz)
    hashes = c.source_hashes()
    identity = c.fingerprint(config, adapter, hashes, True, mesh_info.get("sha256"))
    c.atomic_json(args.output / "manifest.json", dict(
        mode=args.mode, hours=args.hours, radiation_enabled=False,
        configuration=asdict(config), mesh=mesh_info, fingerprint=identity,
        source_sha256=hashes, torch=str(torch.__version__),
    ))
    ledger = dict(
        **dict.fromkeys(c.OCEAN_SUMS + c.OCEAN_PEAKS, 0.0),
        initial=c.inventories(adapter, air, receiver, sea), external=0.0, upstream=0.0,
        radiation_energy_residual=0.0,
        surface_residual=0.0, air_residual=0.0, radiation_residual=0.0, steps=0,
    )
    start = time.perf_counter()
    with (args.output / "stages.jsonl").open("w") as trace, (args.output / "diagnostics.csv").open("w") as diagnostics, torch.no_grad():
        def record(stage, state, **extra):
            s = state.atmosphere
            T = adapter.model.physical_fields(s)["temperature"]
            index = int(s.u_e.abs().argmax())
            row = dict(seconds=air.time_seconds, stage=stage,
                wind=float(s.u_e.abs().max()), wind_edge=index // s.u_e.shape[1],
                wind_level=index % s.u_e.shape[1], Tmin=float(T.min()), Tmax=float(T.max()),
                T_curvature=float(T.diff(n=2, dim=-1).abs().max()), **extra)
            trace.write(json.dumps(row) + "\n")
            trace.flush()

        original = adapter.column_call

        def traced(state, dt, columns, geometry, **kwargs):
            record("before_physics", state, column_wind=float(torch.hypot(columns["u"], columns["v"]).max()))
            result = original(state, dt, columns, geometry, **kwargs)
            out = result[0]
            record("am2_output", state, **{k: float((dt * out[name]).abs().max()) for k, name in (("du", "udt"), ("dv", "vdt"), ("dT", "tdt"))})
            return result

        adapter.column_call = traced
        row = c.diagnostics(adapter, air, receiver, sea, ledger)
        writer = csv.DictWriter(diagnostics, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
        try:
            while air.time_seconds < end:
                record("before", air)
                dt = config.physics_dt
                if args.mode == "coupled":
                    candidate, water, report = c.advance(adapter, air, receiver, sea, dt, config)
                    c.accumulate_ocean(ledger, report)
                    ledger["external"] += report["external"]
                    ledger["upstream"] += report["upstream"]
                    ledger["radiation_energy_residual"] += report["radiation_energy_residual"]
                    for k in ("surface_residual", "air_residual", "radiation_residual"):
                        ledger[k] = max(ledger[k], report[k])
                elif args.mode == "dynamics":
                    candidate, water = air, sea
                    for _ in range(dt // config.atmosphere_dt):
                        candidate = c.advance_dynamics(adapter, candidate, config.atmosphere_dt)
                    candidate = replace(candidate, time_seconds=air.time_seconds + dt)
                else:
                    candidate, water, _, ar, _ = exchange_step(adapter, air, receiver, sea, dt)
                    A = adapter.model.mesh.cell_area
                    ledger["upstream"] += float((A * ar.energy_changes["am2_energy_residual"]).sum() / A.sum())
                record("end", candidate)
                air, sea = candidate, water
                ledger["steps"] += 1
                if air.time_seconds % 21600 == 0 or air.time_seconds == end:
                    row = c.diagnostics(adapter, air, receiver, sea, ledger)
                    writer.writerow(row)
                    diagnostics.flush()
                    print(json.dumps(row), flush=True)
            status = dict(status="completed", seconds=air.time_seconds)
        except Exception as exc:
            status = dict(status="failed", seconds=air.time_seconds, error=str(exc))
            traceback.print_exc()
        status.update(wall_seconds=time.perf_counter() - start, latest=c.diagnostics(adapter, air, receiver, sea, ledger))
        c.atomic_json(args.output / "status.json", status)
        c.checkpoint(args.output / "last_accepted.pt", c.pack(adapter, air, receiver, sea, identity, ledger))
        print(json.dumps(status), flush=True)
        return 0 if status["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
