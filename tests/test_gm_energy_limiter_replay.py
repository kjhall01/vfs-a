"""Pure discrete-geometry tests for the opt-in, replay-only local GM limiter.

The replay wrapper deliberately changes no vfs/ or validation/ source files;
production GMRedi remains unchanged until checkpoint evidence qualifies it.
"""

import importlib.util
from pathlib import Path

import torch

TOOL = Path(__file__).resolve().parents[2] / "tools" / "replay_gm_energy_limiter.py"
spec = importlib.util.spec_from_file_location("replay_gm_energy_limiter", TOOL)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def incoming(i, j, carrier, size):
    return carrier.new_zeros((size, *carrier.shape[1:])).index_add(
        0, i, -carrier
    ).index_add(0, j, carrier)


def test_affected_gm_interfaces_are_strictly_local():
    i = torch.tensor([0, 1, 2, 3, 0, 1])
    j = torch.tensor([1, 2, 3, 0, 2, 3])
    bad = torch.zeros((4, 5), dtype=torch.bool)
    bad[1, 3] = True
    affected = module.incident_gm_interfaces(bad, i, j)
    expected = torch.zeros((6, 4), dtype=torch.bool)
    expected[[0, 1, 5], 2] = True
    expected[[0, 1, 5], 3] = True
    assert torch.equal(affected, expected)


def test_local_flux_limiting_preserves_three_dimensional_volume_divergence():
    torch.manual_seed(33)
    cells, levels = 4, 5
    i = torch.tensor([0, 1, 2, 3, 0, 1])
    j = torch.tensor([1, 2, 3, 0, 2, 3])
    ell = torch.rand(i.numel(), 1, dtype=torch.float64) + 0.1
    ids = torch.arange(cells * levels).reshape(cells, levels)
    h = torch.ones(cells, levels, dtype=torch.float64)
    psi = torch.randn(i.numel(), levels + 1, dtype=torch.float64)
    psi[:, [0, -1]] = 0
    original = psi.clone()
    bad = torch.zeros((cells, levels), dtype=torch.bool)
    bad[1, 3] = True
    factor = torch.ones((i.numel(), levels - 1), dtype=h.dtype)
    factor = torch.where(module.incident_gm_interfaces(bad, i, j), 0.125, factor)
    metrics = (None, None, None, i, j, None, ell, ids, None)
    limited, horizontal, vertical, ii, jj, carrier = module.limited_gm_fluxes(
        psi, factor, metrics, h, incoming
    )
    assert torch.equal(psi, original)
    assert torch.equal(limited[:, [0, -1]], original[:, [0, -1]])
    assert torch.count_nonzero(vertical[:, [0, -1]]) == 0
    assert torch.equal(horizontal, ell * limited.diff(dim=1))
    assert torch.equal(carrier[: horizontal.numel()], horizontal.flatten())
    net = incoming(ii, jj, carrier, h.numel()).reshape_as(h)
    assert float(net.abs().max()) < 1e-13
    assert float(limited[3, 1]) == float(psi[3, 1])  # nonincident face
    assert float(limited[0, 3]) == float(psi[0, 3] * 0.125)


def test_identity_factor_matches_original_carriers():
    cells, levels = 2, 3
    i = torch.tensor([0])
    j = torch.tensor([1])
    ids = torch.arange(cells * levels).reshape(cells, levels)
    psi = torch.tensor([[0., 1., 2., 0.]], dtype=torch.float64)
    h = torch.ones((cells, levels), dtype=psi.dtype)
    metrics = (None, None, None, i, j, None, psi.new_ones((1, 1)), ids, None)
    out = module.limited_gm_fluxes(psi, psi.new_ones((1, 2)), metrics, h, incoming)
    assert torch.equal(out[0], psi)
    assert torch.equal(out[1], psi.diff(dim=1))
    assert float(incoming(out[3], out[4], out[5], 6).abs().max()) == 0.0