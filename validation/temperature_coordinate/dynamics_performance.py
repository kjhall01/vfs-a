"""Profile tensor lifetimes, capture exact results, or time installed-wheel dynamics.

Use separate processes/wheels for before/after; never profile during timing.
The Linux line profiler samples RSS at Python boundaries, not inside ATen.
"""

import argparse
import dataclasses
import gc
import hashlib
import json
import resource
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "validation/temperature_coordinate"),
    str(ROOT / "validation/ocean_column"),
]
import torch
from eddy_performance import wave
from eddy_campaign import make
from eddy_energy import EddyState
from roquet import RoquetOcean
from vfs.ocean import OceanDomain
from vfs.ocean.transport import HorizontalTransport
import vfs


def tensors(value, prefix, result):
    if isinstance(value, torch.Tensor):
        result[prefix] = value.detach().clone()
    elif hasattr(value, "u_e") and hasattr(value, "h"):
        for name in ("u_e", "h", "temp_h", "entropy_h", "salt_h", "tracer_h"):
            if hasattr(value, name):
                tensors(getattr(value, name), prefix + "/" + name, result)
    elif dataclasses.is_dataclass(value):
        for f in dataclasses.fields(value):
            if f.name not in ("mesh", "column"):
                tensors(getattr(value, f.name), prefix + "/" + f.name, result)
    elif isinstance(value, dict):
        for k, v in value.items():
            tensors(v, prefix + "/" + str(k), result)
    elif isinstance(value, (tuple, list)):
        for k, v in enumerate(value):
            tensors(v, prefix + "/" + str(k), result)


def capture():
    result = {}
    for front in ("density", "cold", "warm"):
        for coast in (False, True):
            m, state = make(front=front)
            s = state.ocean
            if coast:
                domain = OceanDomain(s.mesh, s.mesh.cell_xyz[:, 2] > -0.4)
                m = RoquetOcean(s.mesh, s.column, domain=domain)
                state = EddyState(
                    domain.project_state(s),
                    domain.apply_cell(state.tke_h),
                    domain.apply_cell(state.eddy_h),
                )
            s = state.ocean
            # Nonzero, spatially variable flow activates flux and work limiters.
            u = 0.05 * torch.sin(7 * s.mesh.edge_xyz[:, :1]).expand_as(s.u_e)
            if coast:
                u = m.ocean.domain.apply_edge(u)
            state = dataclasses.replace(
                state, ocean=s.with_tensors(u, s.h, s.temp_h, s.salt_h, s.tracer_h)
            )
            prefix = f"{front}/{coast}"
            tensors(
                m.dynamics.tendency_with_report(m.pack(m._carry_eddy(state)), 10.0),
                prefix + "/rhs",
                result,
            )
            end = state
            for step in range(3):
                end, report = m.step_with_report(end, 10.0)
                tensors((end, report), prefix + "/step" + str(step), result)
            for scheme in ("donor", "fct"):
                ledger = HorizontalTransport(
                    s.mesh, domain=m.ocean.domain, scheme=scheme
                ).advance(state.ocean, 10.0)
                tensors(ledger, prefix + "/" + scheme, result)
            with torch.enable_grad():
                names = ("u_e", "h", "temp_h", "salt_h", "tracer_h")
                leaves = [
                    getattr(state.ocean, n).detach().clone().requires_grad_()
                    for n in names
                ]
                changed = dataclasses.replace(
                    state, ocean=state.ocean.with_tensors(*leaves)
                )
                end, _ = m.step_with_report(changed, 10.0)
                objective = (
                    end.ocean.temperature().square().mean()
                    + end.ocean.u_e.square().mean()
                    + end.specific_eddy_energy().mean()
                )
                gradients = torch.autograd.grad(objective, leaves, allow_unused=True)
                tensors(gradients, prefix + "/backward", result)
    return result


def profile(operation):
    paths = (
        "vfs/ocean/transport.py",
        "vfs/ocean/stratified/equations.py",
        "vfs/ocean/stratified/transport.py",
        "temperature_coordinate/model.py",
    )
    peaks = {}
    maximum = {"rss_bytes": 0}

    def rss():
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024

    def sizes(value, seen):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = storage.data_ptr()
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if hasattr(value, "u_e") and hasattr(value, "h"):
            return sum(
                sizes(getattr(value, name), seen)
                for name in ("u_e", "h", "temp_h", "entropy_h", "salt_h", "tracer_h")
                if hasattr(value, name)
            )
        if dataclasses.is_dataclass(value):
            return sum(
                sizes(getattr(value, f.name), seen)
                for f in dataclasses.fields(value)
                if f.name not in ("mesh", "column")
            )
        if isinstance(value, dict):
            return sum(sizes(v, seen) for v in value.values())
        if isinstance(value, (tuple, list)):
            return sum(sizes(v, seen) for v in value)
        if type(value).__name__ == "_StageEvaluation":
            return sizes(vars(value), seen)
        return 0

    def trace(frame, event, arg):
        name = frame.f_code.co_filename
        if not any(name.endswith(p) for p in paths):
            return None
        if event == "line":
            current = rss()
            key = name.split("/site-packages/")[-1] + ":" + frame.f_code.co_name
            if current > peaks.get(key, {}).get("rss_bytes", 0):
                local = {k: sizes(v, set()) for k, v in frame.f_locals.items()}
                peaks[key] = dict(
                    rss_bytes=current,
                    line=frame.f_lineno,
                    local_tensor_storage_bytes={
                        k: v for k, v in sorted(local.items(), key=lambda x: -x[1]) if v
                    },
                )
            if current > maximum["rss_bytes"]:
                maximum.update(rss_bytes=current, function=key, line=frame.f_lineno)
        return trace

    sys.settrace(trace)
    try:
        operation()
    finally:
        sys.settrace(None)
    return dict(
        maximum=maximum,
        function_peaks=peaks,
        note="RSS includes allocator and interpreter; locals share storage and cannot be added across names. Python-boundary sampling may miss within-kernel peaks.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("check", "profile", "time"), required=True)
    p.add_argument("--cells", type=int, default=16384)
    p.add_argument("--layers", type=int, default=40)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "check":
        torch.save(capture(), a.output)
        return
    model, state = wave(a.cells, a.layers)
    state = model.with_eddy_reservoir(state, 0.1)
    operation = lambda: model.step_with_report(state, 120.0)
    if a.mode == "profile":
        result = profile(operation)
    else:
        operation()
        samples = []
        for _ in range(a.repeats):
            gc.collect()
            start = time.perf_counter()
            out = operation()
            samples.append(time.perf_counter() - start)
            del out
        result = dict(samples_s=samples, median_s=statistics.median(samples))
    root = Path(vfs.__file__).parent
    result.update(
        cells=a.cells,
        layers=a.layers,
        threads=1,
        compiled=False,
        torch=torch.__version__,
        package=str(root),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        source_sha256={
            str(f): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [
                root / "ocean/transport.py",
                root / "ocean/stratified/transport.py",
                root / "ocean/stratified/equations.py",
                Path(__file__).with_name("model.py"),
            ]
        },
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
