"""Measure C1 geometry identities against the frozen installed VFS baseline.

This is a measurement driver, not a scientific pass/fail threshold. No dycore
step is taken. Extend experiments separately; do not change the baseline here
to make a failed comparison disappear.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sysconfig

import torch
import vfs
from vfs import VoronoiMesh

BASELINE = "8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648"
# User-approved stage-sequence extraction; not a new numerical baseline.
INTERFACE_COMMIT = "ae132773e621671f70bfa6a698eebb1a82cffa94"
INTERFACE_PATH = "vfs/atmosphere/bounded_moist.py"
# Proposed numerical revision, NOT a replacement verification baseline.
NUMERICAL_COMMIT = "a7facc1bcdb88f7dc9c10f4fb6059f29b0a9bac7"
NUMERICAL_PATH = "vfs/atmosphere/gibbs_means.py"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify_package(repo, package, numerical_revision=None):
    """Compare every tracked runtime file, not just a version string or HEAD."""
    if numerical_revision not in (None, NUMERICAL_COMMIT):
        raise ValueError("Unknown proposed numerical revision")
    paths = git(repo, "ls-tree", "-r", "--name-only", BASELINE, "vfs").decode().splitlines()
    if not paths:
        raise ValueError("Baseline runtime tree is unavailable")
    hashes = {}
    for name in paths:
        expected = git(repo, "show", f"{BASELINE}:{name}")
        if numerical_revision is not None and name == NUMERICAL_PATH:
            expected = git(repo, "show", f"{numerical_revision}:{name}")
        installed = package / Path(name).relative_to("vfs")
        actual = installed.read_bytes() if installed.is_file() else None
        if actual != expected and name == INTERFACE_PATH:
            expected = git(repo, "show", f"{INTERFACE_COMMIT}:{name}")
        if actual != expected:
            raise ValueError(f"Installed runtime differs from baseline: {name}")
        hashes[name] = sha256(expected)
    expected_python = {name for name in paths if name.endswith(".py")}
    actual_python = {"vfs/" + str(p.relative_to(package)) for p in package.rglob("*.py")}
    if actual_python != expected_python:
        raise ValueError("Installed runtime contains unexpected Python modules")
    return hashes


def normalized_residual(residual, scale):
    """No hidden unit-dependent floor; a zero denominator is explicit."""
    if scale == 0:
        return None
    return residual / scale


def measure_geometry(mesh, seed):
    """One scalar per cell/edge, represented as [horizontal, 1].

    Random dimensionless fields probe the area-weighted grad/div pairing.
    z/R = sin(latitude) has an independent analytic surface gradient, in m^-1.
    Its edge-normal derivative is n_z/R because edge normals are tangential.
    """
    generator = torch.Generator().manual_seed(seed)
    scalar = torch.randn(mesh.nCells, 1, generator=generator, dtype=torch.float64)
    flux = torch.randn(mesh.nEdges, 1, generator=generator, dtype=torch.float64)
    cell_work = mesh.cell_area[:, None] * scalar * mesh.div(flux)
    edge_work = (mesh.dc_edge * mesh.dv_edge)[:, None] * flux * mesh.grad(scalar)
    divergence = mesh.cell_area[:, None] * mesh.div(flux)
    reference = mesh.edge_normal[:, 2:3] / mesh.radius
    error = mesh.grad(mesh.cell_xyz[:, 2:3]) - reference
    weights = (mesh.dc_edge * mesh.dv_edge)[:, None]
    metrics = {
        "area_relative_error": float(mesh.cell_area.sum() / (4 * math.pi * mesh.radius**2) - 1),
        "minimum_cell_area_m2": float(mesh.cell_area.min()),
        "constant_gradient_max_m_inverse": float(mesh.grad(torch.ones_like(scalar)).abs().max()),
        "global_divergence_residual_m": float(divergence.sum()),
        "global_divergence_normalized": normalized_residual(float(divergence.sum()), float(divergence.abs().sum())),
        "adjoint_residual_m": float(cell_work.sum() + edge_work.sum()),
        "adjoint_normalized": normalized_residual(float(cell_work.sum() + edge_work.sum()), float(cell_work.abs().sum() + edge_work.abs().sum())),
        "sin_latitude_gradient_rms_error_m_inverse": float(torch.sqrt((weights * error**2).sum() / weights.sum())),
        "sin_latitude_gradient_max_error_m_inverse": float(error.abs().max()),
    }
    # Invalid data are execution failures, not favorable scientific metrics.
    if not all(value is None or math.isfinite(value) for value in metrics.values()):
        raise ValueError("Nonfinite geometry metrics")
    return metrics, {"scalar": scalar, "flux": flux, "gradient_reference": reference}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="repository containing the pinned commit")
    parser.add_argument("--output", type=Path, required=True, help="new directory; never overwrite a run")
    parser.add_argument("--cells", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--lloyd-iters", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--numerical-revision", choices=[NUMERICAL_COMMIT])
    args = parser.parse_args()
    if any(n < 4 for n in args.cells) or len(set(args.cells)) != len(args.cells) or args.lloyd_iters < 0:
        parser.error("Use distinct cell counts >= 4 and nonnegative Lloyd iterations")
    configuration = {"cells": args.cells, "lloyd_iters": args.lloyd_iters,
                     "seed": args.seed, "dtype": "float64", "device": "cpu",
                     "threads": 1, "radius_m": 6371220.0,
                     "mesh": "Fibonacci", "stepper": None}
    with recorded_run(args.repo, args.output, "C1-GEO-001", configuration, __file__, args.numerical_revision):
        rows = []
        for cells in args.cells:
            mesh = VoronoiMesh(n_cells=cells, lloyd_iters=args.lloyd_iters, radius=6371220.0, verbose=False)
            metrics, inputs = measure_geometry(mesh, args.seed)
            rows.append({"cells": cells, "edges": mesh.nEdges, "metrics": metrics})
            torch.save({"mesh": mesh.state_dict(), "inputs": inputs}, args.output / f"geometry-{cells}.pt")
        write_json(args.output / "metrics.json", rows)


@contextmanager
def recorded_run(repo, output, experiment_id, configuration, driver, numerical_revision=None):
    """Share baseline checks and the run/artifact lifecycle between experiments."""
    package = Path(vfs.__file__).resolve().parent
    if not package.is_relative_to(Path(sysconfig.get_path("purelib")).resolve()):
        raise ValueError("Install a baseline wheel and run outside the checkout")
    hashes = (verify_package(repo, package) if numerical_revision is None
              else verify_package(repo, package, numerical_revision))
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    environment_path = Path(__file__).with_name("record_environment.py")
    spec = importlib.util.spec_from_file_location("verification_environment", environment_path)
    environment_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(environment_module)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1, "experiment_id": experiment_id, "status": "running",
        "baseline_commit": BASELINE, "baseline_pr": "https://github.com/kjhall01/vfs-a/pull/12",
        "driver_commit": git(repo, "rev-parse", "HEAD").decode().strip(),
        "checkout_status": git(repo, "status", "--porcelain").decode(),
        "driver_sha256": sha256(Path(driver).read_bytes()),
        "environment_driver_sha256": sha256(environment_path.read_bytes()),
        "runtime_sha256": hashes,
        "numerical_change_commit": numerical_revision,
        "numerical_change_path": NUMERICAL_PATH if numerical_revision else None,
        "interface_change_commit": (
            INTERFACE_COMMIT if hashes.get(INTERFACE_PATH) == sha256(
                git(repo, "show", f"{INTERFACE_COMMIT}:{INTERFACE_PATH}")
            ) else None
        ),
        "approved_interface_path": INTERFACE_PATH,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "environment": environment_module.environment(),
        "interpretation": "Measurements only; no scientific acceptance thresholds or dycore qualification",
    }
    write_json(output / "manifest.json", manifest)
    try:
        for source in {Path(driver), Path(__file__), environment_path}:
            shutil.copyfile(source, output / source.name)
        yield manifest
        manifest["status"] = "completed"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["artifacts_sha256"] = {p.name: sha256(p.read_bytes()) for p in sorted(output.iterdir()) if p.name != "manifest.json"}
        write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
