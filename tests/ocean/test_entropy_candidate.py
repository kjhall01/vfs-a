"""Gibbs identity, independent work, entropy ledger and admissibility checks."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


@pytest.fixture(scope='module')
def probe():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    path = Path(__file__).parents[2]/'tools/probe_ocean_entropy.py'
    spec = importlib.util.spec_from_file_location('entropy_candidate_probe', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('kind', ['general', 'equal_t', 'equal_s', 'equal_both', 'near_equal'])
def test_symmetric_gibbs_identity_and_coalescence(probe, kind):
    ref = probe.previous.reference()
    si, sj = torch.tensor([1e-10, 1., 35., 41.]), torch.tensor([.01, 20., 36., 40.])
    ti, tj = torch.tensor([-2., 1., 10., 35.]), torch.tensor([0., 25., 11., 38.])
    if kind in ('equal_t', 'equal_both'):
        tj = ti.clone()
    if kind in ('equal_s', 'equal_both'):
        sj = si.clone()
    if kind == 'near_equal':
        sj, tj = si*(1+1e-10), ti+1e-10
    eta, salt, audit = probe.faces.gibbs_face(ref, si, ti, sj, tj)
    reverse_eta, reverse_salt, _ = probe.faces.gibbs_face(ref, sj, tj, si, ti)
    torch.testing.assert_close(eta, reverse_eta, rtol=3e-14, atol=1e-11)
    torch.testing.assert_close(salt, reverse_salt, rtol=3e-14, atol=1e-12)
    vi, vj = ref.evaluate(si, ti, entropy_derivatives=True), ref.evaluate(sj, tj, entropy_derivatives=True)
    mui, muj = [v['h_s']-v['kelvin']*v['entropy_s'] for v in (vi, vj)]
    # Use the absolute terms from which jumps were subtracted; tiny jumps
    # cannot be assigned relative machine precision after that subtraction.
    scale = vi['gibbs'].abs()+vj['gibbs'].abs()+(si*mui).abs()+(sj*muj).abs()+1
    assert (audit['identity'].abs() < 512*torch.finfo(si.dtype).eps*scale).all()
    assert audit['chemical_susceptibility'].min() > 0
    if kind == 'equal_both':
        torch.testing.assert_close(eta, vi['entropy'], rtol=3e-13, atol=2e-11)
        torch.testing.assert_close(salt, si, rtol=3e-13, atol=1e-12)


def test_entropy_inverse_and_independent_implicit_derivatives(probe):
    ref = probe.previous.reference()
    sp = torch.tensor([1e-10, 1., 35., 41.], requires_grad=True)
    theta = torch.tensor([-2., 1., 10., 35.])
    entropy = ref.evaluate(sp, theta)['entropy'].detach().requires_grad_()
    recovered = probe.faces.temperature_from_entropy(ref, sp, entropy)
    torch.testing.assert_close(recovered, theta, rtol=0, atol=1e-13)
    ds, de = torch.autograd.grad(recovered.sum(), (sp, entropy))
    v = ref.evaluate(sp, recovered, entropy_derivatives=True)
    torch.testing.assert_close(ds, -v['entropy_s']/v['entropy_theta'], rtol=3e-12, atol=1e-13)
    torch.testing.assert_close(de, 1/v['entropy_theta'], rtol=3e-12, atol=1e-13)


@pytest.mark.parametrize('coast', [False, True])
def test_energy_coordinate_change_against_independent_ad(probe, coast):
    model, old, energy = probe.case('front', coast)
    names = ('u_e', 'h', 'entropy_h', 'salt_h')
    leaves = [getattr(old, k).clone().requires_grad_() for k in names]
    state = replace(old, **dict(zip(names, leaves)))
    independent = torch.autograd.grad(energy.terms(state)['total'], leaves)
    analytic = energy.gradients(state)
    for key, value in zip(names, independent):
        torch.testing.assert_close(analytic[key], value, rtol=3e-12, atol=4.)
    physical = state.physical_state(energy.reference)
    e = probe.previous.enthalpy.EnthalpyEnergy(energy.base, energy.reference)
    a = probe.previous.enthalpy.EnthalpyState.from_temperature(physical, energy.reference)
    torch.testing.assert_close(energy.terms(state)['total'], e.terms(a)['total'], rtol=3e-15, atol=0)
    torch.testing.assert_close(energy.entropy(state), e.entropy(a), rtol=3e-15, atol=0)


@pytest.mark.parametrize('kind', ['tilted', 'near_fresh', 'upper_salt'])
@pytest.mark.parametrize('coast', [False, True])
def test_energy_entropy_and_all_full_euler_ledgers(probe, kind, coast):
    model, original, energy = probe.case(kind, coast)
    names = ('u_e', 'h', 'entropy_h', 'salt_h')
    leaves = [getattr(original, k).clone().requires_grad_() for k in names]
    state = replace(original, **dict(zip(names, leaves)))
    independent = torch.autograd.grad(energy.terms(state)['total'], leaves)
    rhs, work = probe.candidate.evaluate(model, state, 10., energy)
    products = [v*getattr(rhs, k) for v, k in zip(independent, names)]
    assert abs(sum(x.sum() for x in products)) < 512*torch.finfo(state.h.dtype).eps*sum(x.abs().sum() for x in products)
    assert work.entropy_source.min() >= 0 and work.heat.min() >= 0
    expected_heat = (energy.gradients(state)['entropy_h']*work.entropy_source).sum()
    torch.testing.assert_close(expected_heat, work.heat.sum(), rtol=3e-15, atol=0)
    mechanical = (energy.gradients(state)['u_e']*work.force_correction).sum()
    torch.testing.assert_close(mechanical, -work.face_work.clamp_min(0).sum(), rtol=3e-14, atol=1e-20)
    m = energy.mass_area(state)
    residual = (m*(rhs.entropy_h-work.entropy_source)).sum()
    assert abs(residual) < 512*torch.finfo(state.h.dtype).eps*(m*rhs.entropy_h).abs().sum()
    end, stages = probe.candidate.step(model, original, 10., energy)
    area = state.mesh.cell_area[:, None]
    for key in ('h', 'salt_h', 'tracer_h'):
        a = area[..., None] if key == 'tracer_h' else area
        torch.testing.assert_close((a*getattr(end, key)).sum(0), (a*getattr(original, key)).sum(0), rtol=3e-15, atol=0)
    expected = 10*sum(w*(m*l.entropy_source).sum() for w, l in zip((1/6, 1/6, 2/3), stages))
    actual = (m*(end.entropy_h-original.entropy_h)).sum()
    assert expected >= 0
    assert abs(actual-expected) < 128*torch.finfo(state.h.dtype).eps*(m*state.entropy_h.abs()).sum()
    for l in stages:
        assert (l.transport.amounts[..., 1] > 0).all()
        assert (l.transport.amounts[..., 1] < 42*l.transport.volume).all()
        assert (l.transport.amounts[..., 2:] >= 0).all()
        assert l.entropy_source.min() >= 0
    if kind in ('near_fresh', 'upper_salt'):
        assert (work.transport.beta[:, 1] < 1).any()
        assert work.heat.sum() > 0
    if kind == 'upper_salt':
        assert work.face_work.clamp_min(0).sum() > 0
    if coast:
        assert work.entropy_source[~model.domain.wet_cell].count_nonzero() == 0


def test_zero_passive_does_not_change_active_limiter_or_force(probe):
    model, state, energy = probe.case()
    filled = replace(state, tracer_h=torch.ones_like(state.tracer_h)*state.h[..., None])
    a, wa = probe.candidate.evaluate(model, state, 10., energy)
    b, wb = probe.candidate.evaluate(model, filled, 10., energy)
    assert (wa.transport.beta[:, 2:] < 1).any()
    assert torch.equal(wa.transport.beta[:, :2], wb.transport.beta[:, :2])
    for name in ('u_e', 'h', 'entropy_h', 'salt_h'):
        assert torch.equal(getattr(a, name), getattr(b, name))


def test_limiter_switches_off_in_two_sided_zero_velocity_limit(probe):
    rows = probe.zero_speed()
    for kind in ('front', 'near_fresh', 'upper_salt'):
        subset = [r for r in rows if r['kind'] == kind]
        near_zero = [r for r in subset if abs(r['velocity_factor']) <= 1e-9]
        assert all(r['limited_active_faces'] == 0 for r in near_zero)
        assert all(r['maximum_force_correction_m_s2'] == 0 for r in near_zero)
        assert all(r['entropy_production_W_K'] == 0 for r in near_zero)
    assert any(r['maximum_force_correction_m_s2'] > 0 for r in rows)


def test_single_edge_crossing_while_other_faces_remain_limited(probe):
    rows = probe.single_edge_zero()
    near_zero = [r for r in rows if abs(r['velocity_m_s']) <= 1e-6]
    assert all(r['force_correction_m_s2'] == 0 for r in near_zero)
    assert all(r['limited_active_faces'] > 0 for r in near_zero)
    assert any(r['force_correction_m_s2'] != 0 for r in rows)


def test_no_forward_ad_and_source_directional_derivative(probe, monkeypatch):
    model, state, energy = probe.case('near_fresh')
    eta = state.entropy_h.clone().requires_grad_()
    direction = torch.sin(torch.arange(eta.numel()).reshape_as(eta)*.7)
    def objective(x):
        _, work = probe.candidate.evaluate(model, replace(state, entropy_h=x), 10., energy)
        return work.entropy_source.sum()
    gradient, = torch.autograd.grad(objective(eta), eta)
    delta = .1
    fd = (objective(eta+delta*direction)-objective(eta-delta*direction))/(2*delta)
    torch.testing.assert_close((gradient*direction).sum(), fd, rtol=2e-5, atol=1e-13)
    def forbidden(*args, **kwargs):
        raise AssertionError('forward used autograd')
    monkeypatch.setattr(torch.autograd, 'grad', forbidden)
    with torch.no_grad():
        assert torch.isfinite(objective(eta))


def test_domain_and_full_stage_rejection_are_explicit(probe):
    model, state, energy = probe.case()
    with pytest.raises(ValueError, match='no salt floor'):
        probe.candidate.evaluate(model, replace(state, salt_h=torch.zeros_like(state.salt_h)), 10., energy)
    with pytest.raises(ValueError, match='entropy outside'):
        probe.candidate.validate(model, replace(state, entropy_h=1e5*state.h), energy)
    with pytest.raises(FloatingPointError, match='outflow'):
        probe.candidate.step(model, state, 1e12, energy)
    cold = replace(state, entropy_h=-1e5*state.h)
    with pytest.raises(ValueError, match='entropy outside'):
        probe.candidate.validate(model, cold, energy)
    # A real Euler stage, with an initially interior hot salt front, can leave
    # the joint caloric domain even though salt remains positive. It is rejected
    # before an SSP convex combination can conceal the excursion.
    model, state, energy = probe.case('near_fresh')
    physical = state.physical_state(energy.reference)
    hot = physical.with_tensors(physical.u_e, physical.h,
        physical.h*(40-1e-9)/energy.reference.to_90, physical.salt_h, physical.tracer_h)
    start = probe.candidate.EntropyState.from_temperature(hot, energy.reference)
    with pytest.raises(ValueError, match='entropy outside'):
        probe.candidate.euler(model, start, 10., energy)


def test_reference_rest_and_caloric_pressure_cancellation(probe):
    from vfs.ocean import Ocean, OceanColumn
    from vfs.ocean.jmd95 import JMD95Dynamics
    original, _, _ = probe.case()
    mesh = original.mesh
    bottom = 3000+200*mesh.cell_xyz[:, :1]
    h = torch.cat([torch.full_like(bottom, 500.), torch.full_like(bottom, 1000.), bottom-1500.], -1)
    model = Ocean(mesh, OceanColumn(3, total_depth=bottom, rest_thickness=torch.tensor([500., 1000., 1500.])), dynamics=JMD95Dynamics())
    physical = model.state(u_e=torch.zeros(mesh.nEdges, 3), h=h, temperature=torch.full_like(h, 10.), salinity=torch.full_like(h, 35.))
    ref = probe.previous.reference()
    state = probe.candidate.EntropyState.from_temperature(physical, ref)
    energy = probe.candidate.EntropyEnergy(model.dynamics.energy, ref)
    rhs, work = probe.candidate.evaluate(model, state, 10., energy)
    eta, salt, _ = probe.candidate.high_traces(model, state.physical_state(ref), energy)
    g = energy.gradients(state, dynamic_only=True)
    mass = energy.mass_area(state)
    terms = [mesh.grad(g['h']/mass), eta*mesh.grad(g['entropy_h']/mass), salt*mesh.grad(g['salt_h']/mass)]
    assert rhs.u_e.abs().max() < 2048*torch.finfo(h.dtype).eps*sum(x.abs().max() for x in terms)
    assert rhs.h.count_nonzero() == 0 and rhs.entropy_h.count_nonzero() == 0
    assert work.heat.count_nonzero() == 0
