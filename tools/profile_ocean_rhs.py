"""Run the real fully coupled benchmark with a deeper ocean–ice RHS clock.

This is observational instrumentation only, layered on profile_coupled.py.
Every wrapped method delegates exactly once with original arguments. No
changes to prognostic equations, limiter, exchange or energy checks.

CLI is IDENTICAL to tools/profile_coupled.py. Run from the repository root:
    python -u -I tools/profile_ocean_rhs.py --config ... --source ... \
        --mesh ... --radiation-data ... --output ...
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import profile_coupled as base


class RHSInstrument(base.Instrument):
    """Observe pure nested regions of the real ocean+ice+snow RHS."""

    def install(self):
        super().install()
        coupled = self.receiver.ocean_ice.mechanics
        ocean = coupled.ocean
        core = ocean.dynamics

        # Time the joint mechanical RHS separately from transported snow and cold content.
        self._method(coupled, "tendency", "rhs_coupled_ocean_ice")

        # Temperature-coordinate baroclinic/ocean dynamics, with full Gibbs
        # thermal and bounded salt+tracer transport.
        self._method(core, "tendency_with_report", "rhs_ocean_temperature_tendency")
        self._method(core, "fields", "rhs_temperature_entropy_fields")
        # Distinguish Gibbs/temperature reconstruction from repeated checks.
        # All three wrappers are observational and retain every validation.
        self._method(ocean, "check", "rhs_temperature_ocean_check")
        self._method(coupled.ridge, "check", "rhs_ridge_validation")
        self._method(ocean.layers.reference, "evaluate", "rhs_caloric_reference_evaluate")
        from vfs.ocean.stratified import transport
        self._method(transport, "evaluate", "rhs_conservative_transport_total")
        # The Roquet path calls the shared entropy/Gibbs parent evaluator.
        # Its older standalone bounded_transport() helper is not invoked.
        self._method(transport.parent, "evaluate", "rhs_parent_gibbs_transport")
        self._method(transport, "local_target", "rhs_local_energy_target")
        self._method(transport, "_local_target_fields", "rhs_local_target_fields")
        # Split the expensive parent and composed evaluator into native
        # subroutines before resorting to line-level observations.
        eq = transport.parent
        for name, label in (
            ("bounded_transport", "rhs_parent_admissibility_limiter"),
            ("high_traces", "rhs_Gibbs_face_traces"),
            ("momentum", "rhs_Gibbs_pressure_rotation"),
            ("validate", "rhs_entropy_domain_validation"),
            ("divergence", "rhs_parent_flux_divergence"),
        ):
            self._method(eq, name, label)
        self._method(transport, "divergence", "rhs_local_flux_divergence")
        native = ocean.layers.base
        if hasattr(native, "_stratified_transport"):
            self._method(
                native._stratified_transport, "advance",
                "rhs_donor_transport_advance"
            )

        self._method(ocean.hamiltonian.base, "gradients", "rhs_ocean_energy_gradients")
        self._method(native.dynamics.mechanical, "rotation_acceleration", "rhs_ocean_rotation")

        # Sea-ice contributions on the same RHS stage; identify graph advection
        # versus ridging/compaction versus energy and drag work.
        self._method(coupled, "ice_transport", "rhs_ice_transport")
        self._method(coupled, "conjugates", "rhs_ice_conjugates")
        self._method(coupled.ridge, "tendency", "rhs_ice_ridging")
        self._method(coupled, "check", "rhs_state_checks")
        self._method(coupled.hamiltonian, "mass", "rhs_ice_mass")
        self._method(coupled.hamiltonian, "metric", "rhs_ice_metric")
        self._method(coupled.hamiltonian, "coefficient", "rhs_ice_coefficients")


def _profile_line_pass(kwargs, intervals):
    """Trace selected transport source lines *after* the timed benchmark.

    Values are line-inclusive wall time (the elapsed time between Python
    line events, including calls made by the line). They are not ATen kernel
    times and tracing significantly slows the model. Do not add them to
    the exclusive method timings.
    """
    import inspect
    import json
    import linecache
    import sys
    import time
    from collections import defaultdict
    from vfs.ocean.temperature.model import layer_fields
    from vfs.ocean import Ocean
    from tools.profile_coupled import synchronize
    from vfs.ocean.stratified import transport
    from vfs.ocean.stratified import equations

    adapter, air, receiver, sea, config = (
        kwargs["adapter"], kwargs["air"], kwargs["receiver"],
        kwargs["sea"], kwargs["config"]
    )
    out = kwargs["output"]
    device = adapter.model.mesh.cell_area.device
    selected = {
        (fn.__code__.co_filename, fn.__code__.co_firstlineno):
            f"{module}.{fn.__name__}"
        for module, fn in (
            ("local", transport.evaluate),
            ("local", transport._local_target_fields),
            ("parent", equations.evaluate),
            ("parent", equations.bounded_transport),
            ("temperature", layer_fields),
            ("temperature", Ocean.check),
        )
    }
    counts = defaultdict(int)
    durations = defaultdict(float)
    active = {}
    def trace(frame, event, arg):
        code = frame.f_code
        key = (code.co_filename, code.co_firstlineno)
        name = selected.get(key)
        if event == "call":
            if name is None:
                return None
            active[id(frame)] = (name, frame.f_lineno, time.perf_counter())
            return trace
        fid = id(frame)
        prior = active.get(fid)
        if prior is None:
            return None
        label, line, t0 = prior
        now = time.perf_counter()
        if event in ("line", "return", "exception"):
            # The current statement (including any called Torch ops) ends
            # upon arrival at the next Python line / return / exception.
            if event != "exception":
                durations[(label, code.co_filename, line)] += now - t0
                counts[(label, code.co_filename, line)] += 1
            if event == "return":
                del active[fid]
            else:
                active[fid] = (label, frame.f_lineno, now)
        return trace

    elapsed_start = air.time_seconds
    if sys.gettrace() is not None:
        raise RuntimeError("ocean source-line profiler cannot nest in sys.settrace")
    try:
        with __import__("torch").no_grad():
            sys.settrace(trace)
            for _ in range(intervals):
                air, sea, _ = __import__("workflows.coupled.campaign",
                    fromlist=["advance"]).advance(
                        adapter, air, receiver, sea, config.physics_dt, config)
            synchronize(device)
    finally:
        sys.settrace(None)
    lines = []
    for (name, filename, number), elapsed in sorted(
        durations.items(), key=lambda kv: kv[1], reverse=True
    ):
        source = linecache.getline(filename, number).strip()
        lines.append(dict(region=name, path=str(filename),
                          line=number, source=source,
                          elapsed_s=elapsed,
                          calls=counts[(name, filename, number)]))
    result = dict(
        description="Line-inclusive wall time; includes child function calls. "
                    "Do NOT sum with nested exclusive operator timers.",
        measured_intervals=intervals,
        model_start_seconds=elapsed_start,
        model_end_seconds=air.time_seconds,
        source_lines=lines,
    )
    (out / "ocean_transport_lines.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    head = [
        "Ocean transport statement costs (line-inclusive, heavily instrumented)",
        "====================================================================",
        f"Intervals: {intervals}, model seconds: {elapsed_start} to {air.time_seconds}",
        "These timings INCLUDE work called by each statement.",
        "Do not infer speedup or add to exclusive stage totals.",
        "",
        f"{'seconds':>10} {'calls':>7} {'region':32} {'line':>5} statement",
    ]
    for row in lines[:90]:
        head.append(
            f"{row['elapsed_s']:10.3f} {row['calls']:7d} "
            f"{row['region'][:32]:32} {row['line']:5d} {row['source']}"
        )
    (out / "ocean_transport_lines.txt").write_text("\n".join(head)+"\n")
    print("\n".join(head[:35]), flush=True)
    return result


if __name__ == "__main__":
    # Keep the ordinary coupled profiler's CLI and numerical fingerprint
    # unchanged. Source tracing is an additional, opt-in pass after all
    # warmup and 24-interval throughput measurements have completed.
    import argparse
    from pathlib import Path
    args = list(sys.argv[1:])
    parsed = argparse.ArgumentParser(add_help=False)
    parsed.add_argument("--ocean-line-intervals", type=int, default=0)
    options, remaining = parsed.parse_known_args(args)
    if options.ocean_line_intervals < 0:
        parsed.error("--ocean-line-intervals must be nonnegative")
    if options.ocean_line_intervals:
        original_run_profile = base.run_profile

        def run_then_trace(*pos, **kw):
            result, a, s = original_run_profile(*pos, **kw)
            saved = dict(kw, adapter=pos[0], air=a, receiver=pos[2],
                         sea=s, config=pos[4])
            tracing = _profile_line_pass(
                saved, options.ocean_line_intervals
            )
            result["source_line_trace"] = dict(
                path="ocean_transport_lines.json",
                intervals=options.ocean_line_intervals,
                start_seconds=tracing["model_start_seconds"],
                end_seconds=tracing["model_end_seconds"],
            )
            (Path(kw["output"]) / "profile.json").write_text(
                __import__("json").dumps(result, indent=2, allow_nan=False)+"\n"
            )
            return result, a, s

        base.run_profile = run_then_trace
    base.Instrument = RHSInstrument
    raise SystemExit(base.main(remaining))
