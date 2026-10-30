"""Migration checks, independent of the old checkout at test time."""

import importlib.util
from pathlib import Path
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint54_fixture_equivalence_without_runtime_autograd(fixture_mode):
    checker = load(ROOT / "tools/check_baseline.py")
    with patch("torch.autograd.grad", side_effect=AssertionError("runtime autodiff")):
        actual = checker.capture(checker.imports())
    expected = torch.load(ROOT / "tests/fixtures/checkpoint54.pt", weights_only=True)
    assert checker.check(actual, expected, fixture_mode)["tensor_comparisons"] == 142


def test_inventories_preserve_autograd():
    from vfs.atmosphere.moist_budgets import inventories

    example = load(ROOT / "examples/bounded_moist.py")
    base, s = example.initialize()
    s.water_content = s.water_content.clone().requires_grad_()
    totals = inventories(base, s)
    (derivative,) = torch.autograd.grad(totals["species"].sum(), s.water_content)
    expected = (s.mesh.cell_area[:, None, None] / base.kernel.gravity).expand_as(
        derivative
    )
    # Autograd multiplies by reciprocal gravity; the independent expression
    # divides by gravity. Their rounding can differ by one float64 ulp.
    torch.testing.assert_close(derivative, expected, rtol=2 * torch.finfo(derivative.dtype).eps, atol=0)


def test_all_neighbor_bounded_budget():
    from vfs.atmosphere.bounded_transport import limited_traces

    torch.set_default_dtype(torch.float64)
    i = torch.zeros(6, dtype=torch.long)
    j = torch.arange(1, 7)
    mass = torch.ones(7)
    q = torch.zeros(7, 4)
    q[:, 1] = 0.01
    q[0, 2] = 0.002
    q[1:, 2] = 0.05
    q[:, 0] = 1 - q[:, 1:].sum(-1)
    for amount in (0.04, 0.15):
        _, face, info = limited_traces(
            mass,
            q,
            torch.zeros(6),
            0.5 * (q[i] + q[j]),
            torch.zeros(6, 3),
            i,
            j,
            torch.full((6,), amount),
            1.0,
        )
        assert info["limited_faces"] == 6
        assert bool((info["amounts"] >= -1e-16).all())
        torch.testing.assert_close(info["amounts"].sum(0), (mass[:, None] * q).sum(0))
        torch.testing.assert_close(face.sum(-1), torch.ones(6))
        assert info["substeps"] > 1 if amount == 0.15 else info["substeps"] == 1
