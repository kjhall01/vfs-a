"""Fully coupled, non-invasive wall-time and operator profiler for VFS.

Profiles the REAL campaign.advance() path: bounded moist dynamics, moving
ocean/ice/snow, TKE mixing, GM/Redi, full PyAM21 column physics and reciprocal
surface exchange, optional SeaEsfRad optics, post-exchange ice and ALE.

This file lives under tools/, NOT workflows/, validation/ or vfs/, so adding
it does not alter the numerical source hash used by campaign restart validation.
Only this driver is instrumented; the reference model and physical ledgers are
not modified. For reproducible CPU timing prefer fixed threads and float64.

Examples:
  python -u -I tools/profile_coupled.py --config ... --mesh ... --source ... \\
      --radiation-data ... --output ... --set atmosphere_dt=800 \\
      --set ocean_dt=1200 --set ocean_dynamics_integrator=ssprk43 \\
      --set radiation_dt=4800 --set ale_dt=4800

  Add --resume /path/to/restart.pt ONLY for an exact matching source, config,
  runtime, radiation-data and mesh fingerprint.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, replace
import functools
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time
from threading import get_ident
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)] + [
    str(ROOT / "validation" / sub)
    for sub in ("sea_ice", "temperature_coordinate", "ocean_column")
]
import torch
import yaml
from workflows.coupled import campaign as c
from workflows.coupled.device import (
    execution_context, resolve_device, runtime_manifest, synchronize,
)


def _peak_rss_bytes():
    """ru_maxrss uses bytes on macOS and KiB on Linux."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _summary(values):
    if not values:
        return dict(calls=0, total_s=0., mean_ms=0., median_ms=0.,
                    min_ms=0., max_ms=0., p95_ms=0.)
    ordered = sorted(values)
    return dict(
        calls=len(ordered),
        total_s=sum(ordered),
        mean_ms=1e3 * statistics.mean(ordered),
        median_ms=1e3 * statistics.median(ordered),
        min_ms=1e3 * ordered[0],
        max_ms=1e3 * ordered[-1],
        p95_ms=1e3 * ordered[min(len(ordered) - 1, math.ceil(.95 * len(ordered)) - 1)],
    )


class SectionClock:
    """Nested wall-clock attribution: report both inclusive and exclusive work.

    Synchronizing each GPU section provides truthful elapsed GPU wall time but
    may cause major scheduling overhead. Normal GPU timing synchronizes only
    whole coupled intervals; its nested phase times are approximate CPU-side
    dispatch/stall attribution, not reliable GPU kernel execution durations.
    """

    def __init__(self, device, *, synchronize_stages=False, torch_ranges=False):
        self.device = torch.device(device)
        self.sync_stages = synchronize_stages and self.device.type != "cpu"
        self.torch_ranges = torch_ranges
        self.owner_thread = get_ident()
        self.stack = []
        self.samples = defaultdict(list)

    @contextmanager
    def section(self, label):
        # Concurrent worker time overlaps the caller and cannot be subtracted
        # from its nesting stack. Total interval time still includes the join.
        if get_ident() != self.owner_thread:
            yield
            return
        if self.sync_stages:
            synchronize(self.device)
        start = time.perf_counter()
        frame = [label, 0.]
        self.stack.append(frame)
        region = (torch.profiler.record_function("VFS." + label)
                  if self.torch_ranges else nullcontext())
        try:
            with region:
                yield
        finally:
            if self.sync_stages:
                synchronize(self.device)
            wall = time.perf_counter() - start
            popped = self.stack.pop()
            assert popped is frame
            self.samples[label].append(
                (wall, max(0., wall - frame[1]))
            )
            if self.stack:
                self.stack[-1][1] += wall

    def snapshot(self):
        return {
            label: dict(count=len(samples),
                        inclusive_s=sum(sample[0] for sample in samples),
                        exclusive_s=sum(sample[1] for sample in samples))
            for label, samples in self.samples.items()
        }

    def results(self):
        result = {}
        for label, samples in self.samples.items():
            inclusive, exclusive = zip(*samples)
            result[label] = {
                "inclusive": _summary(inclusive),
                "exclusive": _summary(exclusive),
            }
        total = result.get("coupled_interval", {}).get("inclusive", {}).get("total_s", 0.)
        for item in result.values():
            item["percent_exclusive_wall"] = (
                100 * item["exclusive"]["total_s"] / total if total else 0.
            )
        return result


class Instrument:
    """Temporary, transparent wrappers around the installed physical operators.

    All wrappers call the ORIGINAL callable exactly once with identical args.
    No production imports, class state, restart contents or solver source files
    are changed. Object __call__ must be patched on its class because Python's
    special-method lookup bypasses instance attributes.
    """

    def __init__(self, timer, adapter, receiver):
        self.t = timer
        self.adapter = adapter
        self.receiver = receiver
        self.stack = ExitStack()

    def _method(self, obj, name, label):
        old = getattr(obj, name)
        @functools.wraps(old)
        def timed(*args, **kwargs):
            with self.t.section(label):
                return old(*args, **kwargs)
        self.stack.enter_context(patch.object(obj, name, timed))

    def _class_method(self, cls, name, label, *, only=None):
        old = getattr(cls, name)
        @functools.wraps(old)
        def timed(instance, *args, **kwargs):
            if only is not None and instance is not only:
                return old(instance, *args, **kwargs)
            with self.t.section(label):
                return old(instance, *args, **kwargs)
        self.stack.enter_context(patch.object(cls, name, timed))

    def _callable(self, obj, label):
        if obj is not None:
            self._class_method(type(obj), "__call__", label, only=obj)

    def __enter__(self):
        self.stack.__enter__()
        try:
            self.install()
        except BaseException:
            self.stack.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.stack.__exit__(exc_type, exc, tb)

    def install(self):
        a, owner = self.adapter, self.receiver.snow
        # Top-level, mutually exclusive stages of campaign.advance().
        for name, label in (
            ("advance_dynamics", "atmosphere_dynamics"),
            ("mesoscale_step", "mesoscale"),
            ("exchange_step", "shared_surface_exchange"),
            ("pressure_velocity", "atmosphere_omega_diagnostic"),
            ("ale_step", "ocean_ALE"),
        ):
            self._method(c, name, label)

        # Bound and unbounded atmospheric tendency construction.
        if a.model.transport is not None:
            self._method(a.model.transport, "euler_with_info", "atmosphere_bounded_euler")
        self._method(a.model.dynamics, "tendency", "atmosphere_unlimited_rhs")
        self._method(a.model.dynamics.kernel, "_assemble", "atmosphere_centered_assembly")
        self._method(a.model.dynamics.kernel, "mixture_gradients", "atmosphere_mixture_gradients")

        # Ocean–ice mechanics, optional cold-content and snow, plus column mixing.
        for name, label in (
            ("step", "ocean_ice_snow_step"),
            ("dynamics_step", "ocean_ice_dynamics"),
            ("tendency", "ocean_ice_rhs"),
            ("mixing_step", "ocean_vertical_mixing"),
            ("phase_step", "ice_phase"),
            ("accumulate", "snow_accumulation"),
        ):
            self._method(owner, name, label)
        self._method(owner.coupled, "drag_step", "reciprocal_ice_drag")
        self._class_method(
            type(owner.coupled.ocean.closure), "_apply",
            "ocean_mixing_internal_substep"
        )
        self._class_method(c.GMRedi, "apply", "GM_Redi_internal_substep")
        self._class_method(c.EddyDissipation, "apply", "EKE_decay")
        if owner.coupled.ocean.backscatter is not None:
            self._class_method(c.EddyBackscatter, "tendency", "ocean_backscatter")
        self._method(
            owner.coupled.ocean, "remap_with_report", "ALE_conservative_remap"
        )

        # Atmospheric AM2 plus actual reciprocal ocean surface settlement.
        self._method(a, "column_call", "PyAM21_total")
        self._method(a, "apply_result", "atmosphere_conservative_settlement")
        self._method(self.receiver, "prepare", "ocean_surface_prepare")
        self._method(self.receiver, "finish", "ocean_surface_finish")
        if hasattr(self.receiver, "land_surface"):
            self._method(self.receiver.land_surface.column, "advance", "land_column")
            self._method(self.receiver.land_surface, "route", "runoff_routing")
        p = a.physics
        for name, label in (
            ("down", "PyAM21_down"),
            ("couple", "PyAM21_implicit_surface_coupling"),
            ("up", "PyAM21_up"),
        ):
            self._method(p, name, label)
        self._method(p.vdiff, "down", "atmosphere_vertical_diffusion_down")
        self._method(p.vdiff, "up", "atmosphere_vertical_diffusion_up")
        self._callable(p.vturb, "atmosphere_vertical_turbulence")
        self._callable(p.damping, "atmosphere_damping")
        self._callable(p.sflux, "surface_bulk_flux")
        self._callable(p.moist.ras, "RAS_convection")
        self._callable(p.moist.strat, "stratiform_cloud_microphysics")
        import workflows.pyam21.surface as surface
        self._method(surface, "surface_layer", "surface_tile_fluxes")

        if a.radiation is not None:
            self._method(a.radiation, "step", "radiation_driver_cache_and_remap")
            # Distinguishes expensive SeaEsfRad optics from cheap cache calls.
            self._method(a.radiation, "package", "radiation_optical_transfer")


def _update_ledger(ledger, report):
    ledger["steps"] += 1
    ledger["external"] += report["external"]
    ledger["upstream"] += report["upstream"]
    ledger["radiation_energy_residual"] += report["radiation_energy_residual"]
    c.accumulate_ocean(ledger, report)
    for key in ("surface_residual", "air_residual", "radiation_residual"):
        ledger[key] = max(ledger[key], report[key])


def _fresh_ledger(adapter, air, receiver, sea):
    return {
        **dict.fromkeys(c.OCEAN_SUMS + c.OCEAN_PEAKS, 0.),
        "initial": c.inventories(adapter, air, receiver, sea),
        "external": 0.,
        "upstream": 0.,
        "radiation_energy_residual": 0.,
        "surface_residual": 0.,
        "air_residual": 0.,
        "radiation_residual": 0.,
        "steps": 0,
    }


def _throughput(model_seconds, elapsed):
    return dict(
        model_seconds=model_seconds,
        simulated_days=model_seconds / c.DAY,
        wall_seconds=elapsed,
        simulated_days_per_wall_hour=(
            model_seconds / c.DAY * 3600 / elapsed if elapsed else 0.
        ),
        simulated_years_per_wall_day=(
            model_seconds / (365 * c.DAY) * c.DAY / elapsed if elapsed else 0.
        ),
        SYPD=model_seconds / (365 * elapsed) if elapsed else 0.,
    )


def _render_summary(report):
    t = report["throughput"]
    rows = [
        "Fully coupled VFS performance profile",
        "=====================================",
        f"Grid: {report['configuration']['cells']} cells / "
        f"{report['configuration']['atmosphere_levels']} atmosphere / "
        f"{report['configuration']['ocean_levels']} ocean levels",
        f"Device: {report['runtime']['device']}, dtype: "
        f"{report['runtime']['dtype']}, threads: {report['configuration']['threads']}, "
        f"torch.compile: {report['configuration']['compile']}",
        f"Warmup: {report['warmup_intervals']} coupled intervals "
        f"({report['warmup_wall_seconds']:.3f} s; excluded)",
        f"Measurement: {report['measure_intervals']} intervals, "
        f"{t['simulated_days']:.6f} simulated days, {t['wall_seconds']:.3f} s wall",
        f"Throughput: {t['simulated_days_per_wall_hour']:.3f} simulated days/hour; "
        f"{t['SYPD']:.4f} SYPD",
        "",
        ("Main-thread exclusive time (background air overlaps ocean; interval includes the join):"
         if report["configuration"].get("overlap_dynamics") else
         "Exclusive time by operator (nested rows do NOT double count):"),
        f"{'operator':39} {'calls':>7} {'exclusive s':>12} {'wall %':>9} "
        f"{'incl median ms':>16}",
        "-" * 92,
    ]
    for label, d in sorted(
        report["stages"].items(),
        key=lambda pair: pair[1]["exclusive"]["total_s"], reverse=True
    ):
        rows.append(
            f"{label[:39]:39} {d['inclusive']['calls']:7d} "
            f"{d['exclusive']['total_s']:12.3f} "
            f"{d['percent_exclusive_wall']:8.1f}% "
            f"{d['inclusive']['median_ms']:16.2f}"
        )
    rows.extend([
        "",
        "Inclusive stage durations overlap; sum EXCLUSIVE time to explain wall time.",
        "Startup/build and lazy compilation during warmup are excluded.",
        "On GPU, stage timings require --synchronize-stages for accurate device",
        "attribution, but synchronization changes the throughput itself.",
        "The main throughput excludes disk output. --sample-io times I/O separately.",
        "Optional --torch-trace-intervals produces a separate, expensive ATen trace.",
    ])
    return "\n".join(rows) + "\n"


def _sample_io(output, adapter, air, receiver, sea, ledger, identity, timer):
    # These samples are OUTSIDE the measured time: NetCDF and serialization
    # frequency is a separate experiment, not the model's pure stepping cost.
    from workflows.coupled import campaign
    with timer.section("sample_diagnostics_io"):
        diag = campaign.diagnostics(adapter, air, receiver, sea, ledger)
        (output / "sample_diagnostics.json").write_text(
            json.dumps(diag, indent=2, allow_nan=False) + "\n"
        )
    with timer.section("sample_netcdf_io"):
        campaign.snapshot(output / "sample_snapshot.nc", adapter, air, receiver, sea)
    with timer.section("sample_checkpoint_io"):
        payload = campaign.pack(adapter, air, receiver, sea, identity, ledger)
        campaign.checkpoint(output / "sample_restart.pt", payload)
    return {k: v["inclusive"]["total_s"] for k, v in timer.results().items()}


def _trace_aten(output, adapter, air, receiver, sea, config, device, n_intervals):
    from torch.profiler import ProfilerActivity
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    clock = SectionClock(device, torch_ranges=True)
    begin = time.perf_counter()
    with torch.no_grad():
        with torch.profiler.profile(
            activities=activities, record_shapes=False, profile_memory=True,
            with_stack=False
        ) as prof:
            with Instrument(clock, adapter, receiver):
                a, s = air, sea
                for _ in range(n_intervals):
                    with clock.section("coupled_interval"):
                        a, s, _ = c.advance(adapter, a, receiver, s,
                                           config.physics_dt, config)
                        synchronize(device)
                    prof.step()
    prof.export_chrome_trace(str(output / "torch_trace.json"))
    (output / "torch_ops_cpu.txt").write_text(
        prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=75)
        + "\n"
    )
    if device.type == "cuda":
        (output / "torch_ops_cuda.txt").write_text(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=75)
            + "\n"
        )
    return dict(wall_seconds=time.perf_counter() - begin,
                steps=n_intervals,
                chrome_trace=str(output / "torch_trace.json"),
                operator_table=str(output / "torch_ops_cpu.txt"))


def run_profile(
    adapter, air, receiver, sea, config, *, output, identity,
    ledger=None, warmup_intervals=2, measure_intervals=8,
    sample_io=False, trace_intervals=0, synchronize_stages=False,
):
    """Measure actual consecutive coupled advances, not a synthetic RHS loop.

    The caller owns construction/identity verification. No wall-time records
    are committed to a solver checkpoint. Unmodified VFS/PyAM21 state evolves.
    """
    if warmup_intervals < 0 or measure_intervals <= 0 or trace_intervals < 0:
        raise ValueError("nonnegative warmup/trace and positive measurement required")
    output = Path(output)
    device = adapter.model.mesh.cell_area.device
    ledger = _fresh_ledger(adapter, air, receiver, sea) if ledger is None else ledger
    initial_seconds = air.time_seconds
    t_warm = time.perf_counter()
    with torch.no_grad():
        for _ in range(warmup_intervals):
            air, sea, r = c.advance(adapter, air, receiver, sea,
                                    config.physics_dt, config)
            synchronize(device)
            _update_ledger(ledger, r)
    warmup_wall = time.perf_counter() - t_warm

    # Release prior CUDA maxima, but never reset/modify the model state.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    clock = SectionClock(device, synchronize_stages=synchronize_stages)
    rows = []
    begin_second = air.time_seconds
    with torch.no_grad(), Instrument(clock, adapter, receiver):
        for i in range(measure_intervals):
            before = clock.snapshot()
            t0 = air.time_seconds
            try:
                with clock.section("coupled_interval"):
                    air, sea, r = c.advance(
                        adapter, air, receiver, sea, config.physics_dt, config
                    )
                    synchronize(device)  # GPU completion BEFORE stopping wall clock
            except BaseException:
                failure = dict(
                    failed_interval=i + 1, last_completed_seconds=t0,
                    error=traceback.format_exc(), partial_stages=clock.results(),
                    configuration=asdict(config),
                )
                (output / "failure.json").write_text(
                    json.dumps(failure, indent=2, allow_nan=False) + "\n"
                )
                raise
            _update_ledger(ledger, r)
            after = clock.snapshot()
            region = {}
            for key, stats in after.items():
                prev = before.get(key, {})
                num = stats["count"] - prev.get("count", 0)
                if num:
                    region[key] = dict(
                        calls=num,
                        inclusive_seconds=stats["inclusive_s"] - prev.get("inclusive_s", 0),
                        exclusive_seconds=stats["exclusive_s"] - prev.get("exclusive_s", 0),
                    )
            rows.append(dict(
                interval=i + 1, start_seconds=t0,
                end_seconds=air.time_seconds,
                wall_seconds=region["coupled_interval"]["inclusive_seconds"],
                stages=region, ocean_report=r,
            ))
            print(f"Measured interval {i + 1}/{measure_intervals}: "
                  f"{rows[-1]['wall_seconds']:.3f} s; "
                  f"model day {air.time_seconds / c.DAY:.5f}", flush=True)

    wall = sum(row["wall_seconds"] for row in rows)
    runtime = runtime_manifest(device, config.dtype)
    result = dict(
        status="completed",
        configuration=asdict(config),
        runtime=runtime,
        host=dict(platform=platform.platform(), python=sys.version.split()[0],
                  cpu_count=os.cpu_count(), torch=str(torch.__version__)),
        compiled=adapter.compilation_manifest,
        started_model_seconds=initial_seconds,
        measurement_start_seconds=begin_second,
        measurement_end_seconds=air.time_seconds,
        warmup_intervals=warmup_intervals,
        measure_intervals=measure_intervals,
        warmup_wall_seconds=warmup_wall,
        throughput=_throughput(config.physics_dt * measure_intervals, wall),
        per_interval_wall=_summary([row["wall_seconds"] for row in rows]),
        stages=clock.results(),
        peak_rss_bytes=_peak_rss_bytes(),
        gpu_stage_synchronization=synchronize_stages,
    )
    if device.type == "cuda":
        result["cuda_memory"] = dict(
            allocated_bytes=torch.cuda.memory_allocated(device),
            reserved_bytes=torch.cuda.memory_reserved(device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        result["mps_allocated_bytes"] = torch.mps.current_allocated_memory()

    # A single report file contains every per-stage call, count and timing.
    (output / "profile.json").write_text(json.dumps(result, indent=2,
                                                   allow_nan=False) + "\n")
    with (output / "intervals.jsonl").open("w") as file:
        for row in rows:
            file.write(json.dumps(row, allow_nan=False) + "\n")
    import csv
    with (output / "intervals.csv").open("w", newline="") as file:
        w = csv.writer(file)
        w.writerow(("interval", "start_model_seconds", "end_model_seconds",
                    "wall_seconds", "radiative_transfer_calls",
                    "GM_Redi_calls", "mixing_substeps", "ALE_calls"))
        for row in rows:
            stages = row["stages"]
            w.writerow((
                row["interval"], row["start_seconds"], row["end_seconds"],
                row["wall_seconds"],
                stages.get("radiation_optical_transfer", {}).get("calls", 0),
                stages.get("GM_Redi_internal_substep", {}).get("calls", 0),
                stages.get("ocean_mixing_internal_substep", {}).get("calls", 0),
                stages.get("ocean_ALE", {}).get("calls", 0),
            ))
    (output / "summary.txt").write_text(_render_summary(result))
    print(_render_summary(result), flush=True)

    if sample_io:
        io_clock = SectionClock(device, synchronize_stages=True)
        result["sample_io_seconds"] = _sample_io(
            output, adapter, air, receiver, sea, ledger, identity, io_clock
        )
        (output / "profile.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
    if trace_intervals:
        result["operator_trace"] = _trace_aten(
            output, adapter, air, receiver, sea, config, device, trace_intervals
        )
        (output / "profile.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
    return result, air, sea


def _config(args):
    data = yaml.safe_load(args.config.read_text())
    if not isinstance(data, dict):
        raise ValueError("configuration must be a YAML/JSON dictionary")
    for setting in args.set:
        if "=" not in setting:
            raise ValueError(f"--set expects KEY=VALUE: {setting}")
        key, value = setting.split("=", 1)
        if key not in c.CampaignConfig.__dataclass_fields__:
            raise ValueError(f"unknown campaign field in --set: {key}")
        data[key] = yaml.safe_load(value)
    for key in ("device", "dtype", "threads"):
        opt = getattr(args, key)
        if opt is not None:
            data[key] = opt
    if args.compile is not None:
        data["compile"] = args.compile
    return c.CampaignConfig(**data).validate()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True,
                        help="Same YAML or configuration.json as campaign (without modifying it)")
    parser.add_argument("--source", type=Path, required=True, help="Pinned PyAM21 source")
    coast = parser.add_mutually_exclusive_group()
    coast.add_argument("--island", action="store_true", help="profile an island with active ocean and ice")
    coast.add_argument("--geography", type=Path, help="mesh-ordered .npz with land, height_m, depth_m")
    parser.add_argument("--mesh", type=Path, help="Exact saved 4096x3 Voronoi generators")
    parser.add_argument("--radiation-data", type=Path,
                        help="Real SeaEsfRad data directory; omit ONLY with --smoke-no-radiation")
    parser.add_argument("--smoke-no-radiation", action="store_true",
                        help="Explicit radiation-off test (NOT a coupled performance result)")
    parser.add_argument("--output", type=Path, required=True,
                        help="New/empty output folder (never overwrites campaign data)")
    parser.add_argument("--resume", type=Path,
                        help="Use exactly matching source/config/runtime checkpoint; fail closed")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override any CampaignConfig field; repeat for each change")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "cuda:0"))
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument("--threads", type=int)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--warmup-intervals", type=int, default=2)
    parser.add_argument("--measure-intervals", type=int, default=8)
    parser.add_argument("--sample-io", action="store_true",
                        help="Time one NetCDF, diagnostics and valid restart OUTSIDE throughput")
    parser.add_argument("--torch-trace-intervals", type=int, default=0,
                        help="Optional extra, slow ATen/Chrome trace after timed measurement")
    parser.add_argument("--synchronize-stages", action="store_true",
                        help="Accurate accelerator stage wall timings; slows throughput")
    args = parser.parse_args(argv)
    if args.radiation_data is None and not args.smoke_no_radiation:
        parser.error("pass --radiation-data for FULL physics or explicit --smoke-no-radiation")
    config = _config(args)
    if (args.measure_intervals <= 0 or args.warmup_intervals < 0
            or args.torch_trace_intervals < 0):
        parser.error("interval counts must be valid")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        device = resolve_device(config.device)
        config = replace(config, device=str(device))
        with execution_context(device):
            print("Preparing exact VFS coupled configuration...", flush=True)
            config, xyz, mesh = c.load_mesh(config, args.mesh)
            adapter, air, receiver, sea = c.build(
                config, args.source, args.radiation_data,
                smoke_no_radiation=args.smoke_no_radiation,
                generator_xyz=xyz, island=args.island, geography=c.load_geography(args.geography, mesh_sha256=mesh.get("sha256")),
            )
            fingerprint = c.fingerprint(
                config, adapter, c.source_hashes(),
                args.smoke_no_radiation, mesh.get("sha256"),
            )
            init_wall = time.perf_counter() - start
            ledger = None
            if args.resume is not None:
                payload = torch.load(args.resume, map_location="cpu", weights_only=True)
                air, sea, ledger = c.unpack(
                    payload, adapter, receiver, sea, fingerprint
                )
                print(f"Strictly matched restart at day {air.time_seconds / c.DAY:.5f}",
                      flush=True)
            print(f"Construction: {init_wall:.2f} s; device={device}; "
                  f"model start second={air.time_seconds}", flush=True)
            result, _, _ = run_profile(
                adapter, air, receiver, sea, config, output=args.output,
                identity=fingerprint, ledger=ledger,
                warmup_intervals=args.warmup_intervals,
                measure_intervals=args.measure_intervals,
                sample_io=args.sample_io,
                trace_intervals=args.torch_trace_intervals,
                synchronize_stages=args.synchronize_stages,
            )
            result["build_wall_seconds"] = init_wall
            result["source_fingerprint"] = fingerprint
            result["mesh"] = mesh
            result["land_cells"] = int(receiver.land_surface.land.sum()) if sea.land is not None else 0
            result["radiation_enabled"] = not args.smoke_no_radiation
            (args.output / "profile.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
            return 0
    except BaseException:
        failure = dict(error=traceback.format_exc(), source=str(args.source),
                       config=str(args.config), resume=str(args.resume))
        (args.output / "driver_failure.json").write_text(
            json.dumps(failure, indent=2) + "\n"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
