"""Research first-law closure plus retained counterexamples to second-law claims."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


@pytest.fixture(scope='module')
def probe():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    path = Path(__file__).parents[2]/'tools/probe_ocean_thermal_work.py'
    spec = importlib.util.spec_from_file_location('thermal_work_research', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def completion(probe, model, cp=4000., k=.3):
    return probe.CaloricCompletion(model.dynamics.energy, cp=cp, salt_entropy_coefficient=k)


@pytest.mark.parametrize('coast', [False, True])
def test_caloric_and_entropy_gradients_against_autograd(probe, coast):
    model, original = probe.make_case(coast=coast)
    names = ('u_e', 'h', 'temp_h', 'salt_h')
    leaves = [getattr(original, k).clone().requires_grad_() for k in names]
    state = original.with_tensors(*leaves, original.tracer_h)
    caloric = completion(probe, model)
    expected = torch.autograd.grad(caloric.terms(state)['total'], leaves)
    actual = caloric.gradients(state)
    for key, reference in zip(names, expected):
        torch.testing.assert_close(actual[key], reference, rtol=2e-12, atol=.1)
    expected = torch.autograd.grad(caloric.entropy(state), leaves[1:])
    actual = caloric.entropy_gradients(state)
    for key, reference in zip(names[1:], expected):
        torch.testing.assert_close(actual[key], reference, rtol=2e-12, atol=2.)


def test_declared_thermodynamic_potential_derivatives(probe):
    model, _ = probe.make_case()
    caloric = completion(probe, model)
    cp, k = caloric.cp, caloric.k
    theta = torch.tensor([2., 10., 25.])
    salt = torch.tensor([1., 35., 40.], requires_grad=True)
    eta = (cp*torch.log((theta+273.15)/283.15)-k*salt*torch.log(salt/35.)).detach().requires_grad_()
    z = torch.tensor([-500., -2000., -4000.], requires_grad=True)
    recovered = 283.15*torch.exp((eta+k*salt*torch.log(salt/35.))/cp)-273.15
    psi, pt, ps, pz = caloric.base.parcel_potential(recovered, salt, z)
    enthalpy = cp*recovered+psi
    temperature, chemical, height = torch.autograd.grad(enthalpy.sum(), (eta, salt, z), create_graph=True)
    expected_temperature = (recovered+273.15)*(1+pt/cp)
    expected_chemical = ps+expected_temperature*k*(torch.log(salt/35.)+1)
    torch.testing.assert_close(temperature, expected_temperature, rtol=2e-13, atol=1e-12)
    torch.testing.assert_close(chemical, expected_chemical, rtol=2e-13, atol=1e-12)
    torch.testing.assert_close(height, pz, rtol=2e-13, atol=1e-12)
    # Maxwell symmetry belongs to this explicitly defined potential. It is
    # not evidence that these chosen calorics match empirical seawater.
    ts, = torch.autograd.grad(temperature.sum(), salt, retain_graph=True)
    me, = torch.autograd.grad(chemical.sum(), eta)
    torch.testing.assert_close(ts, me, rtol=2e-13, atol=1e-12)


@pytest.mark.parametrize('coast', [False, True])
@pytest.mark.parametrize('scheme', ['donor', 'fct'])
def test_first_law_independent_of_force_and_entropy_claim(probe, coast, scheme):
    model, original = probe.make_case(coast=coast)
    names = ('u_e', 'h', 'temp_h', 'salt_h')
    leaves = [getattr(original, k).clone().requires_grad_() for k in names]
    state = original.with_tensors(*leaves, original.tracer_h)
    caloric = completion(probe, model)
    derivatives = torch.autograd.grad(caloric.terms(state)['total'], leaves)
    rhs, ledger = probe.evaluate(model, state, 10., caloric, scheme)
    local = [g*getattr(rhs, key) for g, key in zip(derivatives, names)]
    residual, scale = sum(x.sum() for x in local), sum(x.abs().sum() for x in local)
    assert abs(residual) < 512*torch.finfo(state.h.dtype).eps*scale
    torch.testing.assert_close(ledger.thermal_power.sum(), -ledger.face_work.sum(), rtol=2e-15, atol=0)
    torch.testing.assert_close(caloric.gradients(state)['temp_h']*ledger.temperature_content_source,
                               ledger.thermal_power, rtol=2e-15, atol=0)
    assert abs(ledger.face_work.sum()) > 1e6*abs(residual)
    assert torch.equal(rhs.u_e, model.tendency(state).u_e)
    if coast:
        assert torch.count_nonzero(ledger.temperature_content_source[~model.domain.wet_cell]) == 0


@pytest.mark.parametrize('scheme', ['donor', 'fct'])
def test_full_stages_salt_inventory_and_explicit_temperature_conversion(probe, scheme):
    model, state = probe.make_case('zero_salt')
    caloric = completion(probe, model)
    result, stages = probe.step(model, state, 10., caloric, scheme)
    assert len(stages) == 3
    assert torch.isfinite(caloric.entropy(result))
    assert all((s.transport.amounts[..., 1] >= 0).all() for s in stages)
    assert result.salt_h.min() >= 0 and result.tracer_h.min() >= 0
    before, after = model.budgets(state), model.budgets(result)
    for k in ('volume', 'salinity_content', 'tracer_content'):
        torch.testing.assert_close(after[k], before[k], rtol=2e-15, atol=0)
    area = model.mesh.cell_area[:, None]
    expected = 10*sum(w*(area*l.temperature_content_source).sum() for w, l in zip((1/6, 1/6, 2/3), stages))
    actual = (area*(result.temp_h-state.temp_h)).sum()
    roundoff = 64*torch.finfo(state.h.dtype).eps*(area*state.temp_h.abs()).sum()
    assert abs(actual-expected) < roundoff
    assert abs(expected) > 10*roundoff
    with pytest.raises(ValueError, match='positive salinity'):
        caloric.entropy_gradients(state)


def test_continuity_and_exact_zero_source_at_rest(probe):
    rows = probe.zero_velocity()
    for scheme in ('donor', 'fct'):
        selected = [r for r in rows if r['scheme'] == scheme]
        for a, b in zip(selected[:-1], selected[1:-1]):
            assert b['force_jump_max_m_s2'] < .002*a['force_jump_max_m_s2']
            assert b['thermal_source_max_K_s'] < .003*a['thermal_source_max_K_s']
        assert selected[-1]['thermal_source_max_K_s'] == 0
        assert selected[-1]['force_jump_max_m_s2'] == 0


@pytest.mark.parametrize('scheme', ['donor', 'fct'])
def test_salt_work_can_require_cooling_and_composition_entropy_matters(probe, scheme):
    model, state = probe.make_case('salt_only')
    caloric = completion(probe, model)
    rhs, ledger = probe.evaluate(model, state, 10., caloric, scheme)
    assert ledger.face_work.sum() > 0
    assert ledger.thermal_power.sum() < 0
    plain = completion(probe, model, k=0.)
    transport, thermal = probe.stable_entropy_terms(plain, state, ledger)
    assert transport.sum() == 0 and thermal.sum() < 0
    transport, thermal = probe.stable_entropy_terms(caloric, state, ledger)
    assert transport.sum()+thermal.sum() > 0


@pytest.mark.parametrize('scheme', ['donor', 'fct'])
def test_tilted_column_entropy_counterexample_is_retained(probe, scheme):
    model, original = probe.make_case('tilted')
    leaves = [getattr(original, k).clone().requires_grad_() for k in ('h', 'temp_h', 'salt_h')]
    state = original.with_tensors(original.u_e, *leaves, original.tracer_h)
    caloric = completion(probe, model)
    derivatives = torch.autograd.grad(caloric.entropy(state), leaves)
    rhs, ledger = probe.evaluate(model, state, 10., caloric, scheme)
    local = [g*getattr(rhs, k) for g, k in zip(derivatives, ('h', 'temp_h', 'salt_h'))]
    independent = sum(x.sum() for x in local)
    transport, thermal = probe.stable_entropy_terms(caloric, state, ledger)
    stable = transport.sum()+thermal.sum()
    assert abs(stable-independent) < 64*torch.finfo(state.h.dtype).eps*sum(x.abs().sum() for x in local)
    assert stable < -1e-3*(transport.abs().sum()+thermal.abs().sum())
    # This is a falsifying regression, not an acceptance test for the scheme.
    assert transport.sum() > 0 and thermal.sum() < 0


def test_source_gradient_and_no_forward_ad(probe, monkeypatch):
    model, state = probe.make_case()
    caloric = completion(probe, model)
    th = state.temp_h.clone().requires_grad_()
    direction = torch.sin(torch.arange(th.numel()).reshape_as(th)*.37+.2)
    def objective(x):
        s = state.with_tensors(state.u_e, state.h, x, state.salt_h, state.tracer_h)
        _, ledger = probe.evaluate(model, s, 10., caloric, scheme='donor')
        return 1e12*ledger.temperature_content_source.sum()
    grad, = torch.autograd.grad(objective(th), th)
    eps = .01
    fd = (objective(th+eps*direction)-objective(th-eps*direction))/(2*eps)
    torch.testing.assert_close((grad*direction).sum(), fd, rtol=3e-5, atol=1e-5)
    def forbidden(*args, **kwargs):
        raise AssertionError('forward used autograd')
    monkeypatch.setattr(torch.autograd, 'grad', forbidden)
    with torch.no_grad():
        assert torch.isfinite(objective(th))


def test_caloric_assumptions_are_explicit_and_validated(probe):
    model, state = probe.make_case()
    with pytest.raises(ValueError, match='cp'):
        completion(probe, model, cp=0.)
    with pytest.raises(ValueError, match='salt entropy'):
        completion(probe, model, k=-.1)
    c1, c2 = completion(probe, model, cp=3800.), completion(probe, model, cp=4200.)
    _, l1 = probe.evaluate(model, state, 10., c1)
    _, l2 = probe.evaluate(model, state, 10., c2)
    assert torch.equal(l1.face_work, l2.face_work)
    assert not torch.equal(l1.temperature_content_source, l2.temperature_content_source)
    torch.testing.assert_close(c1.gradients(state)['temp_h']*l1.temperature_content_source,
                               c2.gradients(state)['temp_h']*l2.temperature_content_source, rtol=2e-15, atol=0)
    invalid = state.with_tensors(state.u_e, state.h, -274*state.h, state.salt_h, state.tracer_h)
    with pytest.raises(FloatingPointError, match='absolute potential temperature'):
        c1.terms(invalid)


@pytest.mark.parametrize('coast', [False, True])
@pytest.mark.parametrize('kind', ['salt_only', 'tilted'])
def test_separate_onsager_exchange_energy_entropy_and_salt_bounds(probe, coast, kind):
    model, original = probe.make_case(kind, coast)
    names = ('u_e', 'h', 'temp_h', 'salt_h')
    leaves = [getattr(original, k).clone().requires_grad_() for k in names]
    state = original.with_tensors(*leaves, original.tracer_h)
    caloric = completion(probe, model)
    energy_derivatives = torch.autograd.grad(caloric.terms(state)['total'], leaves)
    entropy_derivatives = torch.autograd.grad(caloric.entropy(state), leaves[1:])
    for mobility in (1e14, 1e24):
        rhs, info = probe.thermodynamic_exchange(model, state, 10., caloric,
                                               heat_conductance=1e9, salt_mobility=mobility)
        work = [g*getattr(rhs, k) for g, k in zip(energy_derivatives, names)]
        assert abs(sum(x.sum() for x in work)) < 512*torch.finfo(state.h.dtype).eps*sum(x.abs().sum() for x in work)
        entropy = [g*getattr(rhs, k) for g, k in zip(entropy_derivatives, names[1:])]
        independent = sum(x.sum() for x in entropy)
        assert info['entropy_production'].min() >= 0 and independent > 0
        tolerance = 512*torch.finfo(state.h.dtype).eps*sum(x.abs().sum() for x in entropy)
        assert abs(independent-info['entropy_production'].sum()) < tolerance
        salt = state.mesh.cell_area[:, None]*rhs.salt_h
        assert abs(salt.sum()) < 512*torch.finfo(state.h.dtype).eps*salt.abs().sum()
        assert (state.salt_h+10*rhs.salt_h > 0).all()
        if mobility == 1e24:
            assert info['salt_limiter'].min() < 1
        assert torch.count_nonzero(rhs.u_e) == 0 and torch.count_nonzero(rhs.h) == 0
        if coast:
            assert torch.count_nonzero(rhs.temp_h[~model.domain.wet_cell]) == 0


def test_exchange_requires_explicit_nonnegative_mobilities_and_positive_salt(probe):
    model, state = probe.make_case()
    caloric = completion(probe, model)
    with pytest.raises(ValueError, match='coefficients'):
        probe.thermodynamic_exchange(model, state, 10., caloric, heat_conductance=-1., salt_mobility=0.)
    zero, _ = probe.thermodynamic_exchange(model, state, 10., caloric, heat_conductance=0., salt_mobility=0.)
    assert torch.count_nonzero(zero.temp_h) == 0 and torch.count_nonzero(zero.salt_h) == 0
    _, fresh = probe.make_case('zero_salt')
    fresh = state.with_tensors(fresh.u_e, fresh.h, fresh.temp_h, fresh.salt_h, fresh.tracer_h)
    with pytest.raises(ValueError, match='positive salinity'):
        probe.thermodynamic_exchange(model, fresh, 10., caloric, heat_conductance=1., salt_mobility=1.)
