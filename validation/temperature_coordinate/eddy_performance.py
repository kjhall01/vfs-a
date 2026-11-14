"""Reproducible eager CPU timing or numerical comparison with a pinned revision.

Run modes in SEPARATE sequential processes (no concurrent timing jobs):
  python validation/temperature_coordinate/eddy_performance.py --revision SHA --mode check --output before.pt
  python validation/temperature_coordinate/eddy_performance.py --mode check --output after.pt
  python validation/temperature_coordinate/eddy_performance.py --mode time --output timing.json
Only three research modules are loaded from the reference revision; unchanged
VFS/dependencies and all other research modules come from this environment.
"""

import argparse
import gc
import hashlib
import resource
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]


def wave(cells, layers):
    import torch
    from roquet import RoquetOcean
    from vfs.geometry.mesh import VoronoiMesh
    from vfs.ocean import OceanColumn

    mesh = VoronoiMesh(
        n_cells=cells, lloyd_iters=0, rotation_scheme="compatible", verbose=False
    )
    z = 4000.0 * torch.expm1(4 * torch.linspace(0, 1, layers + 1)) / math.expm1(4)
    rest = z.diff()
    depth = 0.5 * (z[:-1] + z[1:])[None]
    model = RoquetOcean(
        mesh, OceanColumn(layers, total_depth=4000.0, rest_thickness=rest)
    )
    h = rest[None].expand(cells, -1).clone()
    pulse = torch.exp((mesh.cell_xyz[:, :1] - 1) / 0.015)
    pulse -= (mesh.cell_area[:, None] * pulse).sum() / mesh.cell_area.sum()
    h += 0.1 * pulse * rest[None] / 4000.0
    state = model.state(
        u_e=torch.zeros(mesh.nEdges, layers),
        h=h,
        conservative_temperature=(2 + 18 * torch.exp(-depth / 500))
        .expand_as(h)
        .clone(),
        absolute_salinity=(35 - 0.5 * torch.exp(-depth / 200)).expand_as(h).clone(),
        tracer=(1 + 0.2 * mesh.cell_xyz[:, 2:3]).expand_as(h)[..., None],
        tke=0.0,
    )
    return model, state


def checks():
    import torch
    from dataclasses import replace, fields
    from eddy import GMRedi
    from eddy_campaign import make
    from roquet import RoquetOcean
    from vfs.ocean import OceanDomain
    from vfs.ocean.stratified import TKEState

    result = {}
    for front in ("density", "cold", "warm"):
        for coast in (False, True):
            model, initial = make(front=front)
            s = initial.ocean
            if coast:
                domain = OceanDomain(s.mesh, s.mesh.cell_xyz[:, 2] > -0.4)
                model = RoquetOcean(s.mesh, s.column, domain=domain)
                s = domain.project_state(s)
                initial = model.with_eddy_reservoir(
                    TKEState(s, domain.apply_cell(initial.tke_h)), 0.1
                )
            for coefficients in ((1000.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0)):
                eddy = GMRedi(*coefficients)
                state = initial
                for _ in range(5):
                    state, report = eddy.apply(model, state, 60.0)
                prefix = f"{front}/{coast}/{coefficients}"
                for name in ("u_e", "h", "temp_h", "salt_h", "tracer_h"):
                    result[prefix + "/state/" + name] = getattr(state.ocean, name)
                result[prefix + "/state/tke_h"] = state.tke_h
                result[prefix + "/state/eddy_h"] = state.eddy_h
                for field in fields(report):
                    result[prefix + "/report/" + field.name] = getattr(
                        report, field.name
                    )
    # Backward on an active, non-clipped branch, compared across implementations.
    model, initial = make(front="cold")
    with torch.enable_grad():
        alpha = torch.tensor(0.0, requires_grad=True)
        s = initial.ocean
        x = s.mesh.cell_xyz[:, :1]
        changed = replace(
            initial,
            ocean=s.with_tensors(
                s.u_e, s.h, s.temp_h, s.salt_h + alpha * s.h * x, s.tracer_h
            ),
        )
        end, _ = GMRedi(1000.0, 1000.0).apply(model, changed, 60.0)
        result["gradient"] = torch.autograd.grad(
            (x * end.specific_eddy_energy()).mean(), alpha
        )[0]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision")
    parser.add_argument(
        "--closure-only",
        action="store_true",
        help="Time only the closure, isolating its process memory from dynamics",
    )
    parser.add_argument("--mode", choices=("check", "time"), required=True)
    parser.add_argument("--cells", type=int, default=16384)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # A fresh process prevents mixing reference and candidate class identities.
    with tempfile.TemporaryDirectory(prefix="vfs-eddy-reference-") as directory:
        sys.path[:0] = [
            str(ROOT / "validation/temperature_coordinate"),
            str(ROOT / "validation/ocean_column"),
        ]
        if args.revision:
            for name in ("roquet.py", "eddy.py", "eddy_energy.py"):
                content = subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(ROOT),
                        "show",
                        f"{args.revision}:validation/temperature_coordinate/{name}",
                    ]
                )
                Path(directory, name).write_bytes(content)
            sys.path.insert(0, directory)
        import torch
        from eddy import GMRedi

        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.mode == "check":
            torch.save(checks(), args.output)
            return
        model, state = wave(args.cells, args.layers)
        funded = model.with_eddy_reservoir(state, 0.1)
        eddy = GMRedi(1000.0, 1000.0)
        operations = dict(
            dynamics=lambda: model.step_with_report(state, 120.0),
            dynamics_reservoir=lambda: model.step_with_report(funded, 120.0),
            funded_eddy=lambda: eddy.apply(model, funded, 120.0),
        )
        if args.closure_only:
            operations = {"funded_eddy": operations["funded_eddy"]}
        samples = {name: [] for name in operations}
        for name, operation in operations.items():
            out = operation()
            del out
            print("warmed", name, flush=True)
        for repeat in range(args.repeats):
            for name, operation in operations.items():
                gc.collect()
                start = time.perf_counter()
                out = operation()
                samples[name].append(time.perf_counter() - start)
                del out
            print("repeat", repeat + 1, flush=True)
        from roquet import RoquetOcean
        from eddy_energy import RoquetHamiltonian
        import inspect

        result = dict(
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_sha256={
                Path(inspect.getfile(cls))
                .name: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes())
                .hexdigest()
                for cls in (GMRedi, RoquetOcean, RoquetHamiltonian)
            },
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            revision=args.revision or "working-tree",
            cells=args.cells,
            layers=args.layers,
            dt=120.0,
            compiled=False,
            threads=1,
            torch=torch.__version__,
            samples_s=samples,
            medians_s={
                name: statistics.median(values) for name, values in samples.items()
            },
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
