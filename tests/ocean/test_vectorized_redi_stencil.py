"""Exact stencil regression for the vectorized three-point Redi gradient."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "validation/temperature_coordinate")
)
from eddy import GMRedi, _RediGradient


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_three_point_gradient_matches_original_scalar_loop(dtype):
    # Deliberately irregular layer spacing exercises the actual metric
    # coefficients, including the one-sided boundary rows.
    z = torch.tensor(
        [[0., -2., -5., -9.],
         [0., -3., -7., -13.],
         [0., -4., -6., -11.]], dtype=dtype
    )
    ids, weights = GMRedi._vertical_rows(z)
    i = torch.tensor([0, 1], dtype=torch.long)
    j = torch.tensor([1, 2], dtype=torch.long)
    dc = torch.tensor([[2.], [3.]], dtype=dtype)
    ell = torch.ones_like(dc)
    cell_ids = torch.arange(z.numel()).reshape_as(z)
    metrics = (z, ids, weights, i, j, dc, ell, cell_ids, (z[j]-z[i])/dc)
    relative = torch.ones((2, z.shape[-1]), dtype=dtype)
    gradient = _RediGradient(metrics, relative)

    values = torch.arange(z.numel()*4, dtype=dtype).reshape(*z.shape, 4) / 17.
    values.requires_grad_(True)
    result = gradient.vertical(values)
    flat = values.reshape(-1, values.shape[-1])
    original = torch.zeros_like(values)
    for k in range(3):
        original = original + weights[..., k, None] * flat[ids[..., k]]

    assert torch.equal(result, original)
    derivative = torch.autograd.grad(result.square().sum(), values, retain_graph=True)[0]
    original_derivative = torch.autograd.grad(original.square().sum(), values)[0]
    assert torch.equal(derivative, original_derivative)
