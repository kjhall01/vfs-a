"""Capture checkpoint-54 fixtures or compare an installed extraction against them.

Reference mode is developer-only and needs the unchanged old archive. Normal
comparison needs only the installed package and this repository's fixtures.
Run from outside the checkout to verify that imports resolve to the wheel.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

FIELDS = ("u_e", "ps", "entropy_content", "water_content")


def imports(reference=None):
    if reference:
        sys.path.insert(0, str(reference / "vfs_verified_source"))
        sys.path.insert(0, str(reference / "vfs_energy_audit/bounded_gibbs"))
        from bounded_pairing import BoundedGibbs
    else:
        from vfs.atmosphere import BoundedGibbs
    from vfs.atmosphere import MoistConservativeDynamics, MoistState
    from vfs.atmosphere import (
        DryHydrostaticConservativeDynamics,
    )
    from vfs.geometry.mesh import VoronoiMesh
    from vfs.geometry.column import CharneyPhillipsColumn, make_hybrid_coefficients
    from vfs.initial_conditions.jablonski_williamson import JWInitialConditions
    from vfs.core.integrators import ssprk3

    return locals()


def tensors(state):
    names = FIELDS if hasattr(state, "water_content") else ("u_e", "ps", "mtheta_i")
    # Serialization also supports the archived standalone Field boundary.
    return {k: (getattr(state, k).values if hasattr(getattr(state, k), "location")
                else getattr(state, k)).detach().clone() for k in names}


def capture(api):
    """Deterministic 64-cell, six-level tests; active sharp cloud in both winds."""
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    mesh = api["VoronoiMesh"](n_cells=64, lloyd_iters=0, verbose=False)
    A, B = api["make_hybrid_coefficients"](6, p_top=5000.0, sigma_power=1.0)
    column = api["CharneyPhillipsColumn"](
        6, A, B, phi_s=torch.zeros(64), phi_full_mode="direct_u_gl2_adjoint"
    )
    dt = 30.0
    result = {}
    with torch.no_grad():
        for case in ("smooth", "cloud", "cloud_reverse"):
            d = api["MoistConservativeDynamics"]()
            p = column.interface_pressure(torch.full((64, 1), 1e5))
            from vfs.atmosphere.moist_cp import interface_pressure_mass

            dm = interface_pressure_mass(p.diff())
            q = torch.zeros(64, 7, 3)
            q[..., 0] = 0.01
            if case == "smooth":
                q[..., 1] = 0.001
                q[..., 2] = 0.0001
            else:
                q[..., 1] = 0.002 * (mesh.cell_xyz[:, :1] > 0) * (p / 1e5) ** 2
            fractions = torch.cat([1 - q.sum(-1, keepdim=True), q], -1)
            temperature = 280 * (p / 1e5) ** 0.08
            axis = torch.tensor([0.0, 0.0, 1.0]).expand_as(mesh.edge_xyz)
            speed = -10.0 if case == "cloud_reverse" else 10.0
            wind = (
                speed
                * (torch.linalg.cross(axis, mesh.edge_xyz) * mesh.edge_normal)
                .sum(-1, keepdim=True)
                .expand(-1, 6)
                .clone()
            )
            state = api["MoistState"](
                wind,
                torch.full((64, 1), 1e5),
                dm * d.thermodynamics.entropy(temperature, p, fractions),
                dm[..., None] * q,
                mesh,
                column,
            )
            model = api["BoundedGibbs"](d, **({"release_as_heat": True} if api["reference"] else {}))
            rhs, info = model.evaluate(state, dt)
            euler, ei = model.euler_with_info(state, dt)
            final = state
            stages = []
            for _ in range(3):
                final, si = model.step_with_info(final, dt)
                stages.append([h["limited_faces"] for h in si])
            area = mesh.cell_area[:, None]

            def budgets(s):
                dm_s = interface_pressure_mass(
                    column.interface_pressure(s.ps).diff()
                )
                return {
                    "mass": (area * dm_s).sum() / d.kernel.gravity,
                    "species": (area[..., None] * s.water_content).sum((0, 1))
                    / d.kernel.gravity,
                    "energy": d.energy_terms(s),
                }

            result[case] = dict(
                initial=tensors(state),
                rhs=tensors(rhs),
                euler=tensors(euler),
                final=tensors(final),
                stages=stages,
                limiter={
                    k: info[k]
                    for k in (
                        "limited_faces",
                        "amounts",
                        "integrated_flux",
                        "limiter_force",
                        "heat",
                        "entropy_source",
                        "beta",
                    )
                },
                gradients=tensors(d.energy_gradients(state)),
                initial_budgets=budgets(state),
                final_budgets=budgets(final),
                energy_rate=d.boundary_energy_rate(state, rhs),
                cfl=model.cfl(state, dt),
            )
        ic = api["JWInitialConditions"]
        drycol = ic.make_column_for_mode(
            mesh,
            6,
            A,
            B,
            mode="constant_ps_surface_phi",
            phi_full_mode="direct_u_gl2_adjoint",
        )
        dry = api["DryHydrostaticConservativeDynamics"]()
        s0 = ic(mesh, drycol, mode="constant_ps_surface_phi").make_state(
            dry,
            perturb=True,
            project_mass_flux=False,
            theta_source="phi_derivative",
            wind_source="U_jw",
        )
        rhs = dry.tendency(s0)
        s = s0
        for _ in range(3):
            s = api["ssprk3"](dry, s, dt)
        result["dry"] = dict(
            initial=tensors(s0),
            rhs=tensors(rhs),
            final=tensors(s),
            initial_energy=dry.boundary_total_energy(s0),
            final_energy=dry.boundary_total_energy(s),
            initial_mass=dry.total_mass(s0),
            final_mass=dry.total_mass(s),
        )
    return result


def compare(actual, expected):
    checked = 0

    def walk(a, b, path):
        nonlocal checked
        if isinstance(b, torch.Tensor):
            torch.testing.assert_close(
                a, b, rtol=0, atol=0, msg=lambda msg: f"{path}: {msg}"
            )
            checked += 1
        elif isinstance(b, dict):
            assert a.keys() == b.keys(), path
            for k in b:
                walk(a[k], b[k], path + "/" + k)
        else:
            assert a == b, (path, a, b)

    walk(actual, expected, "baseline")
    assert actual["smooth"]["limiter"]["limited_faces"] == 0
    assert actual["cloud"]["limiter"]["limited_faces"] > 0
    assert actual["cloud_reverse"]["limiter"]["limited_faces"] > 0
    return checked


def check(actual, expected, mode="exact"):
    """Select a named policy without altering the historical exact comparator."""
    import importlib.util
    path = Path(__file__).with_name("fixture_regression.py")
    spec = importlib.util.spec_from_file_location("fixture_regression", path)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    return policy.compare_selected(actual, expected, "checkpoint54.pt", mode, compare)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/fixtures/checkpoint54.pt",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--mode", choices=("exact", "portable"), default="exact")
    args = parser.parse_args()
    if args.reference and args.mode != "exact":
        parser.error("reference capture cannot select portable comparison")
    api = imports(args.reference)
    actual = capture(api)
    import vfs

    if args.reference:
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        torch.save(actual, args.fixture)
        summary = dict(
            mode="reference_capture", torch=str(torch.__version__), source=vfs.__file__
        )
    else:
        expected = torch.load(args.fixture, map_location="cpu", weights_only=True)
        summary = dict(mode=args.mode + "_comparison", status="pass",
                       torch=str(torch.__version__), source=vfs.__file__)
        try:
            summary.update(check(actual, expected, args.mode))
        except AssertionError as exc:
            summary.update(status="fail", error=str(exc))
        assert not any(
            "vfs_energy_audit" in str(getattr(m, "__file__", ""))
            for m in list(sys.modules.values())
        )
    summary["limited_faces"] = {
        k: actual[k]["limiter"]["limited_faces"]
        for k in ("smooth", "cloud", "cloud_reverse")
    }
    summary["fixture_sha256"] = hashlib.sha256(args.fixture.read_bytes()).hexdigest()
    print(json.dumps(summary, indent=2))
    if args.report:
        args.report.write_text(json.dumps(summary, indent=2) + "\n")
    if summary.get("status") == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
