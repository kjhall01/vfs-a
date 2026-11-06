"""Research identities/regressions; empirical accuracy is reported separately."""
from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


@pytest.fixture(scope='module')
def probe():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    path = Path(__file__).parents[2]/'tools/probe_ocean_seawater.py'
    spec = importlib.util.spec_from_file_location('seawater_research', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_surface_single_potential_derivatives_and_explicit_scales(probe):
    ref = probe.reference()
    sp = torch.tensor([.001, 1., 35., 42.], requires_grad=True)
    theta = torch.tensor([-2., 1., 10., 39.], requires_grad=True)
    v = ref.evaluate(sp, theta, entropy_derivatives=True)
    for value, derivative in [('h0', 'h'), ('entropy', 'entropy')]:
        actual = torch.autograd.grad(v[value].sum(), (sp, theta), retain_graph=True)
        for a, name in zip(actual, (derivative+'_s', derivative+'_theta')):
            torch.testing.assert_close(a, v[name], rtol=3e-12, atol=2e-10)
    gt, = torch.autograd.grad(v['gibbs'].sum(), theta)
    torch.testing.assert_close(-gt/ref.to_90, v['entropy'], rtol=3e-12, atol=2e-10)
    torch.testing.assert_close(v['h_theta']/v['entropy_theta'], v['kelvin'], rtol=3e-15, atol=0)
    other = probe.surface.SurfaceGibbs(ref.data, temperature_scale='ITS-90')
    w = other.evaluate(sp, theta*ref.to_90)
    for k in ('h0', 'entropy', 'cp', 'kelvin'):
        assert torch.equal(w[k], v[k])
    torch.testing.assert_close(w['h_theta']*ref.to_90, v['h_theta'], rtol=0, atol=0)
    with pytest.raises(ValueError, match='temperature_scale'):
        probe.surface.SurfaceGibbs(ref.data, temperature_scale='unspecified')
    model, _, _ = probe.case()
    with pytest.raises(ValueError, match='IPTS-68'):
        probe.enthalpy.EnthalpyEnergy(model.dynamics.energy, other)


def test_finite_freshwater_values_and_declared_domain(probe):
    ref = probe.reference()
    sp = torch.tensor([0., 0.], requires_grad=True)
    theta = torch.tensor([0., 10.], requires_grad=True)
    v = ref.evaluate(sp, theta)
    for x in v.values():
        assert torch.isfinite(x).all()
    gs, gt = torch.autograd.grad(v['h0'].sum(), (sp, theta))
    torch.testing.assert_close(gs, v['h_s'], rtol=3e-12, atol=1e-10)
    torch.testing.assert_close(gt, v['h_theta'], rtol=3e-12, atol=1e-10)
    with pytest.raises(ValueError, match='positive salinity'):
        ref.evaluate(sp, theta, entropy_derivatives=True)
    for s, t in [(-.01, 10.), (42.01, 10.), (35., 41.), (35., -3.), (35., float('nan'))]:
        with pytest.raises(ValueError):
            ref.evaluate(torch.tensor(s), torch.tensor(t))
    with pytest.raises(ValueError, match='dtype/device'):
        ref.evaluate(sp.float(), theta)


def test_enthalpy_inverse_and_implicit_derivatives(probe):
    ref = probe.reference()
    sp, theta = torch.meshgrid(torch.tensor([0., .001, 5., 35., 42.]),
        torch.linspace(-2.5, 40, 25)/ref.to_90, indexing='ij')
    a = ref.evaluate(sp, theta)['h0']
    recovered = ref.temperature_from_enthalpy(sp, a)
    torch.testing.assert_close(recovered, theta, rtol=0, atol=4e-13)
    s = torch.tensor([.1, 20., 40.], requires_grad=True)
    a = ref.evaluate(s, torch.tensor([1., 10., 30.]))['h0'].detach().requires_grad_()
    t = ref.temperature_from_enthalpy(s, a)
    ds, da = torch.autograd.grad(t.sum(), (s, a))
    v = ref.evaluate(s, t)
    torch.testing.assert_close(ds, -v['h_s']/v['h_theta'], rtol=2e-11, atol=1e-13)
    torch.testing.assert_close(da, 1/v['h_theta'], rtol=2e-11, atol=1e-15)
    with pytest.raises(ValueError, match='enthalpy outside'):
        ref.temperature_from_enthalpy(torch.tensor([35.]), torch.tensor([1e10]))
    fitter = Path(probe.__file__).with_name('fit_surface_gibbs.py')
    assert ref.data['fitter_sha256'] == hashlib.sha256(fitter.read_bytes()).hexdigest()


@pytest.mark.parametrize('coast', [False, True])
def test_transformed_energy_entropy_and_density_against_independent_ad(probe, coast):
    model, original, energy = probe.case(coast=coast)
    names = ('u_e', 'h', 'enthalpy_h', 'salt_h')
    leaves = [getattr(original, k).clone().requires_grad_() for k in names]
    state = replace(original, **dict(zip(names, leaves)))
    physical = state.physical_state(energy.reference)
    expected = torch.autograd.grad(energy.terms(state)['total'], leaves)
    actual = energy.gradients(state)
    for key, v in zip(names, expected):
        torch.testing.assert_close(actual[key], v, rtol=2e-11, atol=64.)
    expected = torch.autograd.grad(energy.entropy(state), leaves[1:])
    actual = energy.entropy_gradients(state)
    for key, v in zip(names[1:], expected):
        torch.testing.assert_close(actual[key], v, rtol=2e-11, atol=64.)
    _, old = probe.previous.make_case(coast=coast)
    # Identical physical density EOS; only inverse-solve rounding differs.
    torch.testing.assert_close(energy.base.density(physical), energy.base.density(old), rtol=5e-15, atol=0)
    t, mu = energy.thermodynamic_fields(state)
    g, e = energy.gradients(state), energy.entropy_gradients(state)
    torch.testing.assert_close(g['enthalpy_h'], t*e['enthalpy_h'], rtol=3e-15, atol=0)
    torch.testing.assert_close(g['salt_h'], t*e['salt_h']+energy.mass_area(state)*mu, rtol=3e-13, atol=2.)


@pytest.mark.parametrize('coast', [False, True])
@pytest.mark.parametrize('scheme', ['donor', 'fct'])
def test_first_law_and_authoritative_stage_contents(probe, coast, scheme):
    model, original, energy = probe.case('zero_salt', coast)
    names = ('u_e', 'h', 'enthalpy_h', 'salt_h')
    leaves = [getattr(original, k).clone().requires_grad_() for k in names]
    state = replace(original, **dict(zip(names, leaves)))
    gradients = torch.autograd.grad(energy.terms(state)['total'], leaves)
    rhs, work = probe.enthalpy.evaluate(model, state, 10., energy, scheme)
    products = [g*getattr(rhs, k) for k, g in zip(names, gradients)]
    assert abs(sum(x.sum() for x in products)) < 512*torch.finfo(state.h.dtype).eps*sum(x.abs().sum() for x in products)
    torch.testing.assert_close(work.thermal_power.sum(), -work.face_work.sum(), rtol=3e-15, atol=0)
    end, stages = probe.enthalpy.step(model, original, 10., energy, scheme)
    assert len(stages) == 3 and end.salt_h.min() >= 0 and end.tracer_h.min() >= 0
    area = state.mesh.cell_area[:, None]
    for key in ('h', 'salt_h', 'tracer_h'):
        a = area[..., None] if key == 'tracer_h' else area
        torch.testing.assert_close((a*getattr(end, key)).sum(0), (a*getattr(original, key)).sum(0), rtol=3e-15, atol=0)
    expected = 10*sum(w*(area*l.enthalpy_content_source).sum() for w, l in zip((1/6, 1/6, 2/3), stages))
    actual = (area*(end.enthalpy_h-original.enthalpy_h)).sum()
    roundoff = 128*torch.finfo(state.h.dtype).eps*(area*state.enthalpy_h.abs()).sum()
    assert abs(actual-expected) < roundoff
    assert abs(expected) > roundoff
    for l in stages:
        assert l.transport.amounts[..., 1:].min() >= 0
    if coast:
        assert torch.count_nonzero(work.enthalpy_content_source[~model.domain.wet_cell]) == 0


@pytest.mark.parametrize('scheme', ['donor', 'fct'])
def test_tilted_entropy_counterexample_with_validated_calorics(probe, scheme):
    model, original, energy = probe.case('tilted')
    names = ('h', 'enthalpy_h', 'salt_h')
    leaves = [getattr(original, k).clone().requires_grad_() for k in names]
    state = replace(original, **dict(zip(names, leaves)))
    independent = torch.autograd.grad(energy.entropy(state), leaves)
    rhs, work = probe.enthalpy.evaluate(model, state, 10., energy, scheme)
    products = [g*getattr(rhs, k) for k, g in zip(names, independent)]
    advective, thermal = probe.enthalpy.entropy_terms(energy, state, work, 10.)
    total = advective.sum()+thermal.sum()
    assert abs(total-sum(x.sum() for x in products)) < 512*torch.finfo(state.h.dtype).eps*sum(x.abs().sum() for x in products)
    assert advective.sum() > 0 and thermal.sum() < 0 and total < 0


@pytest.mark.parametrize('scheme', ['donor', 'fct'])
def test_zero_velocity_continuity(probe, scheme):
    model, state, energy = probe.case()
    at_rest = replace(state, u_e=torch.zeros_like(state.u_e))
    rest_rhs, rest_work = probe.enthalpy.evaluate(model, at_rest, 10., energy, scheme)
    assert rest_work.enthalpy_content_source.count_nonzero() == 0
    for sign in (-1, 1):
        previous = None
        for epsilon in (1e-3, 1e-6, 1e-9):
            moved = replace(state, u_e=sign*epsilon*state.u_e)
            rhs, work = probe.enthalpy.evaluate(model, moved, 10., energy, scheme)
            sizes = ((rhs.u_e-rest_rhs.u_e).abs().max(), work.enthalpy_content_source.abs().max())
            if previous is not None:
                # Both vanish at least linearly with velocity; no finite jump.
                assert sizes[0] < .002*previous[0]
                assert sizes[1] < .002*previous[1]
            previous = sizes


@pytest.mark.parametrize('coast', [False, True])
def test_exchange_uses_same_first_and_second_laws(probe, coast):
    model, original, energy = probe.case('salt_only', coast)
    names = ('u_e', 'h', 'enthalpy_h', 'salt_h')
    leaves = [getattr(original, k).clone().requires_grad_() for k in names]
    state = replace(original, **dict(zip(names, leaves)))
    hg = torch.autograd.grad(energy.terms(state)['total'], leaves)
    eg = torch.autograd.grad(energy.entropy(state), leaves[1:])
    for mobility in (1e14, 1e24):
        rhs, info = probe.enthalpy.thermodynamic_exchange(model, state, 10., energy,
            heat_conductance=1e9, salt_mobility=mobility)
        terms = [g*getattr(rhs, k) for g, k in zip(hg, names)]
        assert abs(sum(x.sum() for x in terms)) < 512*torch.finfo(state.h.dtype).eps*sum(x.abs().sum() for x in terms)
        terms = [g*getattr(rhs, k) for g, k in zip(eg, names[1:])]
        assert info['entropy_production'].min() >= 0
        assert abs(sum(x.sum() for x in terms)-info['entropy_production'].sum()) < 512*torch.finfo(state.h.dtype).eps*sum(x.abs().sum() for x in terms)
        assert (state.salt_h+10*rhs.salt_h > 0).all()
        if mobility == 1e24:
            assert info['salt_limiter'].min() < 1
            with pytest.raises(ValueError, match='salinity outside'):
                probe.enthalpy.validate(model, state+10*rhs, energy)


def test_no_forward_ad_and_directional_derivative(probe, monkeypatch):
    model, state, energy = probe.case()
    a = state.enthalpy_h.clone().requires_grad_()
    direction = torch.sin(torch.arange(a.numel()).reshape_as(a)*.7)
    def objective(x):
        _, work = probe.enthalpy.evaluate(model, replace(state, enthalpy_h=x), 10., energy, 'donor')
        return work.enthalpy_content_source.sum()
    gradient, = torch.autograd.grad(objective(a), a)
    delta = 100.
    fd = (objective(a+delta*direction)-objective(a-delta*direction))/(2*delta)
    torch.testing.assert_close((gradient*direction).sum(), fd, rtol=2e-5, atol=1e-12)
    def forbidden(*args, **kwargs):
        raise AssertionError('forward used autograd')
    monkeypatch.setattr(torch.autograd, 'grad', forbidden)
    with torch.no_grad():
        assert torch.isfinite(objective(a))


def test_reference_composition_rest_over_topography(probe):
    from vfs.ocean import Ocean, OceanColumn
    from vfs.ocean.jmd95 import JMD95Dynamics
    original, _, _ = probe.case()
    mesh = original.mesh
    bottom = 3000+200*mesh.cell_xyz[:, :1]
    column = OceanColumn(3, total_depth=bottom, rest_thickness=torch.tensor([500., 1000., 1500.]))
    h = torch.cat([torch.full_like(bottom, 500.), torch.full_like(bottom, 1000.), bottom-1500.], -1)
    model = Ocean(mesh, column, dynamics=JMD95Dynamics())
    physical = model.state(u_e=torch.zeros(mesh.nEdges, 3), h=h,
                           temperature=torch.full_like(h, 10.), salinity=torch.full_like(h, 35.))
    ref = probe.reference()
    state = probe.enthalpy.EnthalpyState.from_temperature(physical, ref)
    energy = probe.enthalpy.EnthalpyEnergy(model.dynamics.energy, ref)
    rhs, work = probe.enthalpy.evaluate(model, state, 10., energy)
    # The pressure covectors cancel for the reference parcel. Bound only the
    # floating-point cancellation, using the actual constituent force scale.
    g = energy.gradients(state, dynamic_only=True)
    mass = model.dynamics.rho0*mesh.cell_area[:, None]
    terms = [mesh.grad(g['h']/mass),
             model.dynamics.centered_trace(mesh, state.enthalpy_h/state.h)*mesh.grad(g['enthalpy_h']/mass),
             model.dynamics.centered_trace(mesh, state.salt_h/state.h)*mesh.grad(g['salt_h']/mass)]
    scale = sum(x.abs().max() for x in terms)
    assert rhs.u_e.abs().max() < 1024*torch.finfo(h.dtype).eps*scale
    assert rhs.h.count_nonzero() == 0 and rhs.enthalpy_h.count_nonzero() == 0
    assert work.thermal_power.count_nonzero() == 0
