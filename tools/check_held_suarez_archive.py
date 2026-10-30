"""Compare standalone forcing with the pinned PR #8 archive, outside runtime.

Requires local Git history containing ARCHIVE_COMMIT; never fetches or installs
historical code. Run against an installed VFS wheel from outside its checkout.
"""
import argparse
import ast
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess

import torch
from torch import nn
from vfs import State, VoronoiMesh, CharneyPhillipsColumn, make_hybrid_coefficients
from vfs.atmosphere import DryHydrostaticConservativeDynamics
from vfs.atmosphere.held_suarez import HeldSuarezForcing
from vfs.core.state import interface_mass
from vfs.core.integrators import ssprk3

ARCHIVE_COMMIT = "849ccb829c7555ab8ecd1c11f0ae45993538f8ed"
ARCHIVE_PATH = "vfs/physics.py"


def archived_forcing(repository):
    source = subprocess.check_output(
        ["git", "-C", str(repository), "show", f"{ARCHIVE_COMMIT}:{ARCHIVE_PATH}"],
        text=True,
    )
    names = {"HeldSuarezConstants", "_lat_for_pressure",
             "held_suarez_equilibrium_temperature", "HeldSuarezPhysics"}
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in tree.body} == names
    namespace = dict(torch=torch, nn=nn, State=State, interface_mass=interface_mass,
                     CharneyPhillipsColumn=CharneyPhillipsColumn, dataclass=dataclass)
    # Only these four definitions from the pinned object execute in memory.
    # No old package import, registry or installed legacy module is involved.
    exec(compile(tree, f"{ARCHIVE_COMMIT}:{ARCHIVE_PATH}", "exec"), namespace)
    old_class = namespace["HeldSuarezPhysics"]
    return old_class, hashlib.sha256(source.encode()).hexdigest()


def method_ast(method):
    import textwrap
    node = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        node.body.pop(0)
    return ast.dump(node)


def initial_state(levels):
    mesh = VoronoiMesh(n_cells=16, lloyd_iters=0, verbose=False)
    a, b = make_hybrid_coefficients(levels, p_top=5000., sigma_power=1.4)
    column = CharneyPhillipsColumn(
        levels, a, b, phi_s=500*mesh.cell_xyz[:, 0],
        phi_full_mode="direct_u_gl2_adjoint",
    )
    generator = torch.Generator().manual_seed(18)
    def random(*shape):
        return torch.randn(*shape, generator=generator)
    state = State.from_theta(
        10*random(mesh.nEdges, levels), 300+3*random(mesh.nCells, levels+1),
        1e5+1000*random(mesh.nCells, 1), mesh=mesh, column=column,
    )
    return State(*(x.clone().requires_grad_() for x in
                   (state.u_e, state.mtheta_i, state.ps)), mesh=mesh, column=column)


def capture(forcing, state, dynamics):
    source = forcing(state)
    stages = []
    def rhs(stage):
        stages.append(stage)
        return dynamics.tendency(stage) + forcing(stage)
    result = ssprk3(rhs, state, 1.)
    energy = dynamics.boundary_total_energy(result)
    gradients = torch.autograd.grad(energy, (state.u_e, state.mtheta_i, state.ps))
    return [getattr(s, name) for s in (source, *stages, result)
            for name in ("u_e", "mtheta_i", "ps")] + [energy, *gradients]


def compare(repository):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    old_class, digest = archived_forcing(repository)
    # inspect.getsource cannot resolve in-memory Git filenames. Parse the pinned
    # definition once more for a transparent executable-method comparison.
    source = subprocess.check_output(
        ["git", "-C", str(repository), "show", f"{ARCHIVE_COMMIT}:{ARCHIVE_PATH}"], text=True)
    old_tree = next(n for n in ast.parse(source).body if getattr(n, "name", None) == "HeldSuarezPhysics")
    for method in old_tree.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        if isinstance(method.body[0], ast.Expr) and isinstance(method.body[0].value, ast.Constant):
            method.body.pop(0)
        assert ast.dump(method) == method_ast(getattr(HeldSuarezForcing, method.name)), method.name
    count = 0
    for levels in (1, 3, 12):
        for prescribed in (False, True):
            state = initial_state(levels)
            kwargs = (dict(Teq=torch.tensor(270.), kT=torch.tensor(2e-6),
                           kv=torch.full((state.mesh.nCells, levels), 3e-6)) if prescribed else {})
            old, new = old_class(**kwargs), HeldSuarezForcing(**kwargs)
            pressure = state.interface_pressure()
            for name in ("equilibrium_temperature", "thermal_relaxation_rate", "drag_rate"):
                argument = pressure if name == "equilibrium_temperature" else pressure/state.ps
                torch.testing.assert_close(getattr(new, name)(state, argument),
                                           getattr(old, name)(state, argument), rtol=0, atol=0)
                count += 1
            dynamics = DryHydrostaticConservativeDynamics()
            before, after = capture(old, state, dynamics), capture(new, state, dynamics)
            for expected, actual in zip(before, after, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                count += 1
    return dict(archive_commit=ARCHIVE_COMMIT, archive_path=ARCHIVE_PATH,
                source_sha256=digest, exact_tensors=count, method_ast="identical",
                platform=platform.platform(), torch=str(torch.__version__),
                dtype="float64", device="cpu", torch_config=torch.__config__.show())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-repo", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = compare(args.archive_repo)
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
