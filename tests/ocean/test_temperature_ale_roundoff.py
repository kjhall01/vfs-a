"""Cold, weakly stratified ALE columns at the entropy acceptance boundary."""
import pytest
import torch

from vfs.ocean.model import Ocean as RoquetOcean
from vfs.ocean.temperature.ale import entropy_content
from vfs import VoronoiMesh
from vfs.ocean import OceanColumn
from vfs.ocean.stratified.energy_columns import column_terms


def cold_columns(*, seed=10, salinity_scale=1.7105959511064084):
    """Deterministic counterexample; no user checkpoint or radiation data needed.

    Seed 10 and the salinity contrast place a cold column within one rounding
    unit of the entropy threshold. The old solver accepts its temporary T at
    iteration three, then rejects the actual stored h*T in its final check.
    """
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    mesh = VoronoiMesh(n_cells=16, lloyd_iters=0, verbose=False)
    model = RoquetOcean(mesh, OceanColumn(12, total_depth=1000., gravity=9.8))
    tail = torch.logspace(0, 1, 11)
    target = torch.cat((torch.tensor([20.]), 980 * tail / tail.sum())).expand(16, -1).clone()
    generator = torch.Generator().manual_seed(seed)
    amplitude = .001 + torch.rand(16, 1, generator=generator) * .002
    theta = -1.836 + amplitude * torch.exp(-torch.arange(12)[None, :] * 7.)
    h = target * (1 + torch.randn(16, 12, generator=generator) * 2e-5)
    h = h / h.sum(-1, keepdim=True) * 1000
    salt = 34 + torch.rand(16, 12, generator=generator) * .01 * salinity_scale
    state = model.state(u_e=torch.zeros(mesh.nEdges, 12), h=h,
                        conservative_temperature=theta, absolute_salinity=salt,
                        tracer=torch.ones(16, 12, 1), tke=1e-4, eddy_energy=.1)
    return model, state, target


@pytest.mark.parametrize('threads', [1, 4])
@pytest.mark.parametrize('seed,salinity_scale', [
    (10, 1.7105959511064084),  # temporary/stored-temperature disagreement
    (26, 4.257426495909136),  # stored entropy stalls outside tolerance for 24 iterations
])
def test_cold_ale_accepts_the_stored_state_and_conserves_columns(threads, seed, salinity_scale):
    model, initial, target = cold_columns(seed=seed, salinity_scale=salinity_scale)
    torch.set_num_threads(threads)
    try:
        end, report = model.remap_with_report(initial, target)
    finally:
        torch.set_num_threads(1)
    before, after = initial.ocean, end.ocean
    mass = model.energy.mass_area(before)
    eps = torch.finfo(before.h.dtype).eps
    old, _ = column_terms(model.energy, before)
    new, _ = column_terms(model.energy, after)
    energy = sum(new[k] - old[k] for k in old)
    entropy = (mass * (entropy_content(model.energy, after)
                       - entropy_content(model.energy, before))).sum(-1)
    energy_tol = 256 * eps * sum(v.abs() for v in old.values()).clamp_min(1)
    entropy_tol = 512 * eps * (mass * entropy_content(model.energy, before).abs()).sum(-1).clamp_min(1)
    assert (energy.abs() <= energy_tol).all()
    assert (entropy >= -entropy_tol).all()
    torch.testing.assert_close(report.column_energy_residual, energy, rtol=0, atol=0)
    torch.testing.assert_close(report.column_entropy_change, entropy, rtol=0, atol=0)
    torch.testing.assert_close(after.h, target, rtol=0, atol=0)
    for left, right, axis in ((before.h, after.h, -1),
                              (before.salt_h, after.salt_h, -1),
                              (before.tracer_h, after.tracer_h, -2),
                              (initial.tke_h, end.tke_h, -1),
                              (initial.eddy_h, end.eddy_h, -1)):
        torch.testing.assert_close(left.sum(axis), right.sum(axis), rtol=128*eps, atol=0)
    assert (end.tke_h >= 0).all() and (end.eddy_h >= 0).all()
    assert torch.equal(before.u_e, after.u_e)


def test_unsolved_projection_still_rejects_with_column_diagnostics(monkeypatch):
    from vfs.ocean.temperature import ale

    model, initial, target = cold_columns()
    def no_step(displacement, *args, settled=None):
        return torch.zeros_like(displacement), torch.zeros(displacement.shape[0], dtype=torch.bool)
    monkeypatch.setattr(ale, 'minimum_change_step', no_step)
    with pytest.raises((FloatingPointError, ValueError), match=r'column=\d+, .*tolerance=.*iterations=24'):
        model.remap_with_report(initial, target)
