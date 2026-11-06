"""Conservation, stage bounds, independent work and continuity regressions."""
import importlib.util
from pathlib import Path

import pytest
import torch

from vfs.ocean.transport import HorizontalTransport, divergence


@pytest.fixture(scope='module')
def probe():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    path = Path(__file__).parents[2]/'tools/probe_ocean_bounded_transport.py'
    spec = importlib.util.spec_from_file_location('bounded_ocean_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('scheme', ['donor', 'fct'])
@pytest.mark.parametrize('coast', [False, True])
def test_bounds_contents_and_mesh_orientation(probe, scheme, coast):
    model, state = probe.active_case(coast, zero_salt=True)
    transport = probe.transport_for(model, scheme)
    dt = .5*float(transport.timestep_limit(state))
    ledger = transport.advance(state, dt)
    result = ledger.install(state)
    area = state.mesh.cell_area[:, None]
    for name in ('h', 'temp_h', 'salt_h', 'tracer_h'):
        a = area[..., None] if name == 'tracer_h' else area
        before, after = a*getattr(state, name), a*getattr(result, name)
        torch.testing.assert_close(after.sum(0), before.sum(0), rtol=2e-15, atol=1.)
    amounts = ledger.amounts.reshape(-1, ledger.amounts.shape[-1])
    scale = amounts.abs().max()
    assert (amounts-ledger.lower).min() >= -256*torch.finfo(state.h.dtype).eps*scale
    assert (ledger.upper-amounts).min() >= -256*torch.finfo(state.h.dtype).eps*scale
    assert (result.salt_h >= 0).all() and (result.tracer_h >= 0).all()
    assert (result.h > transport.min_thickness).all()
    rhs = model.tendency(state)
    actual = divergence(ledger.i, ledger.j, ledger.carrier, state.h.numel()).reshape_as(state.h)/area
    torch.testing.assert_close(actual, rhs.h, rtol=2e-14, atol=1e-20)
    if coast:
        dry = ~model.domain.wet_cell
        torch.testing.assert_close(result.salt_h[dry], state.salt_h[dry], rtol=2e-15, atol=0)
        wet_indices = model.domain.wet_cell[:, None].expand_as(state.h).flatten()
        assert wet_indices[ledger.i].all() and wet_indices[ledger.j].all()


@pytest.mark.parametrize('scheme', ['donor', 'fct'])
def test_uniform_concentrations_follow_volume_with_no_fraction_constraint(probe, scheme):
    _, state = probe.active_case()
    # Negative Celsius is valid; neither T nor S is a mass fraction.
    state = state.with_tensors(state.u_e, state.h, -2*state.h, 35*state.h,
                              7*state.h[..., None])
    transport = HorizontalTransport(state.mesh, scheme=scheme)
    result = transport.advance(state, .8*float(transport.timestep_limit(state))).install(state)
    for actual, expected in ((result.temperature(), -2.), (result.salinity(), 35.), (result.tracers(), 7.)):
        torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=2e-15, atol=0)


@pytest.mark.parametrize('scheme', ['donor', 'fct'])
@pytest.mark.parametrize('salt_scale', [1., 1e-30])
def test_ring_zero_and_near_zero_front_at_every_euler_stage(probe, scheme, salt_scale):
    state = probe.ring_state(32, sharp=True)
    state.salt_h = state.salt_h*salt_scale
    transport = HorizontalTransport(state.mesh, scheme=scheme)
    dt = .8/32
    for _ in range(12):
        ledger = transport.advance(state, dt)
        result = ledger.install(state)
        assert result.salt_h.min() >= 0
        assert result.salt_h.max() <= salt_scale*(1+1e-14)
        assert result.tracer_h.min() >= 0
        torch.testing.assert_close(result.salt_h.sum(), state.salt_h.sum(), rtol=2e-15, atol=0)
        state = result


def test_donor_matches_analytic_upwind_and_rejects_hidden_outflow(probe):
    state = probe.ring_state(32, sharp=True)
    transport = HorizontalTransport(state.mesh, scheme='donor')
    result = transport.advance(state, .2/32).install(state)
    expected = .8*state.salt_h+.2*torch.roll(state.salt_h, 1, 0)
    torch.testing.assert_close(result.salt_h, expected, rtol=2e-15, atol=0)
    # Equal incoming/outgoing volume leaves final h positive even for huge dt.
    # The full outgoing carrier still exceeds what is available to donate.
    volume, i, j, flux = transport.graph(state)
    assert torch.count_nonzero(divergence(i, j, flux, volume.numel())) == 0
    with pytest.raises(FloatingPointError, match='actual carrier outflow'):
        transport.advance(state, 2/32)
    assert transport.timestep_limit(state) < 1/32
    state.u_e = torch.zeros_like(state.u_e)
    assert torch.isinf(transport.timestep_limit(state))


def test_three_full_stage_budgets_and_second_stage_rejection(probe):
    model, state = probe.active_case(zero_salt=True)
    result, ledgers = probe.step(model, state, 10.)
    assert len(ledgers) == 3 and result.salt_h.min() >= 0
    for ledger in ledgers:
        assert ledger.amounts[..., 1].min() >= 0
        assert ledger.outflow_fraction.max() < .9
    state.u_e = torch.zeros_like(state.u_e)
    calls = []
    def acceleration(s):
        calls.append(s)
        zero = s.zeros_like()
        return zero.with_tensors(torch.full_like(s.u_e, 1e8), zero.h, zero.temp_h, zero.salt_h, zero.tracer_h)
    model.tendency = acceleration
    with pytest.raises(FloatingPointError, match='actual carrier outflow'):
        probe.step(model, state, 10.)
    assert len(calls) == 2


@pytest.mark.parametrize('coast', [False, True])
def test_limiter_energy_residual_against_independent_autograd(probe, coast):
    model, state = probe.active_case(coast)
    names = ('u_e', 'h', 'temp_h', 'salt_h')
    leaves = [getattr(state, k).clone().requires_grad_() for k in names]
    state = state.with_tensors(*leaves, state.tracer_h)
    gradients = torch.autograd.grad(model.dynamics.total_energy(state), leaves)
    _, ledger, rhs = probe.euler(model, state, 10.)
    local = [g*getattr(rhs, k) for g, k in zip(gradients, names)]
    independent = sum(x.sum() for x in local)
    scale = sum(x.abs().sum() for x in local)
    work = probe.work_diagnostics(model, state, ledger, 10., rhs)
    tolerance = 1024*torch.finfo(state.h.dtype).eps*scale
    assert abs(independent-work['limiter_work_W']) < tolerance
    # Detect the residual; do not pass a zeroed-out scalar correction.
    assert abs(work['limiter_work_W']) > 1e6*tolerance
    force_work = (gradients[0]*probe.paired_force(model, state, ledger)).sum()
    assert abs(independent+force_work) < tolerance


def test_zero_speed_flux_continuity_and_paired_force_jump(probe):
    rows = probe.zero_velocity_probe()
    for scheme in ('donor', 'fct'):
        selected = [r for r in rows if r['scheme'] == scheme]
        for a, b in zip(selected, selected[1:]):
            # Regression: centered force and transported flux approach zero
            # jump, while the counterfactual all-mechanical correction does not.
            assert b['centered_force_jump_max_m_s2'] < .002*a['centered_force_jump_max_m_s2']
            assert b['salt_rate_jump_max'] < .002*a['salt_rate_jump_max']
        assert selected[-1]['counterfactual_paired_jump_max_m_s2'] > 1e6*selected[-1]['centered_force_jump_max_m_s2']


def test_piecewise_gradient_and_no_forward_autograd(probe, monkeypatch):
    state = probe.ring_state(32)
    # Break the sine's equal extrema: this checks a fixed limiter branch,
    # not a nonexistent unique derivative at a min/max tie.
    offset = .017*torch.cos(torch.arange(32)*1.13+.4)[:, None]
    original = (state.salt_h+offset).requires_grad_()
    direction = torch.sin(torch.arange(32)*.73)[:, None]
    def objective(salt):
        s = state.with_tensors(state.u_e, state.h, state.temp_h, salt, state.tracer_h)
        ledger = HorizontalTransport(state.mesh).advance(s, .2/32)
        return ledger.amounts[..., 1].square().sum()
    grad, = torch.autograd.grad(objective(original), original)
    eps = 1e-6
    fd = (objective(original+eps*direction)-objective(original-eps*direction))/(2*eps)
    torch.testing.assert_close((grad*direction).sum(), fd, rtol=2e-6, atol=1e-11)
    def forbidden(*args, **kwargs):
        raise AssertionError('forward called autograd')
    monkeypatch.setattr(torch.autograd, 'grad', forbidden)
    with torch.no_grad():
        assert torch.isfinite(objective(original))


def test_geometry_mismatch_rejected_without_cast(probe):
    state = probe.ring_state(32)
    changed = state.to(dtype=torch.float32)
    with pytest.raises(ValueError, match='geometry must match'):
        HorizontalTransport(state.mesh).advance(changed, .001)
