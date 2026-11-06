"""Independent overlap integrals, invariants, bounds, and actual ALE steps."""
import pytest
import torch

from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import Ocean, OceanColumn, OceanDomain
from vfs.ocean.ale import ALEStepper, ReferenceGrid, VerticalRemapper, remap_contents
from vfs.ocean.jmd95 import JMD95Dynamics, checked_ssprk3


@pytest.fixture(autouse=True)
def precision():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)


@pytest.fixture(scope='module')
def mesh():
    torch.set_default_dtype(torch.float64)
    return VoronoiMesh(n_cells=32, lloyd_iters=2, verbose=False)


def case(mesh, coast=False):
    domain = OceanDomain(mesh, mesh.cell_xyz[:, 2] > -.4) if coast else None
    column = OceanColumn(4, total_depth=3000., rest_thickness=torch.tensor([300., 600., 900., 1200.]))
    x = mesh.cell_xyz[:, :1]
    h = torch.tensor([250., 700., 800., 1250.])[None, :] + x*torch.tensor([50., -30., 20., -40.])
    t = torch.tensor([-1., 8., 6., 3.])[None, :] + .2*x
    s = torch.tensor([34., 34.5, 35., 35.5])[None, :] + .01*x
    u = .02*torch.sin(torch.arange(mesh.nEdges)[:, None]*.3 + torch.arange(4))
    model = Ocean(mesh, column, domain=domain, dynamics=JMD95Dynamics(domain=domain), stepper=ALEStepper())
    state = model.state(u_e=u, h=h, temperature=t, salinity=s,
                        tracer=torch.stack((torch.ones_like(h), t.square()), -1))
    if domain:
        state = domain.project_state(state)
    return model, state


@pytest.mark.parametrize('method', ['pcm', 'plm'])
def test_identity_uniform_and_independent_pcm_overlaps(method):
    h = torch.tensor([[1., 2., 3., 4.], [4., 1., 2., 3.]])
    q = torch.tensor([[-2., 0., 6., 1.], [0., 4., 2., 8.]])
    target = torch.tensor([[.5, 5., 1., 3.5], [6., 1., 2., 1.]])
    torch.testing.assert_close(remap_contents(h, h*q, h, method=method), h*q, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(remap_contents(h, h*3.7, target, method=method), target*3.7, rtol=1e-14, atol=1e-14)
    if method == 'pcm':
        # Independent nested intersections, without searchsorted/prefix logic.
        reference = torch.zeros_like(target)
        for c in range(2):
            old_edges = [0.] + h[c].cumsum(0).tolist()
            new_edges = [0.] + target[c].cumsum(0).tolist()
            for j in range(4):
                for i in range(4):
                    overlap = max(0., min(old_edges[i+1],new_edges[j+1])-max(old_edges[i],new_edges[j]))
                    reference[c,j] += overlap*q[c,i]
        torch.testing.assert_close(remap_contents(h, h*q, target, method=method), reference, rtol=1e-14, atol=1e-14)


def test_plm_integrates_linear_profile_on_nonuniform_interior():
    h = torch.tensor([1., 2., 3., 4., 2.])
    target = torch.tensor([1., 1.5, 4.5, 3., 2.])
    q = 3 + 2*(h.cumsum(0)-.5*h)
    expected = target*(3 + 2*(target.cumsum(0)-.5*target))
    torch.testing.assert_close(remap_contents(h, h*q, target), expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('method', ['pcm', 'plm'])
def test_many_column_large_displacements_conservation_and_bounds(method):
    generator = torch.Generator().manual_seed(601)
    h = .1 + torch.rand(11, 17, generator=generator)
    target = .01 + torch.rand(11, 9, generator=generator)**3
    target = target/target.sum(-1,keepdim=True)*h.sum(-1,keepdim=True)
    q = torch.rand(11,17,3,generator=generator)
    q[...,0] -= .5  # Signed temperatures are allowed.
    q[:,::3,1] = 0  # Zero salinity/passive concentration is supported.
    result = remap_contents(h,h[...,None]*q,target,method=method)
    torch.testing.assert_close(result.sum(-2),(h[...,None]*q).sum(-2),rtol=3e-14,atol=1e-14)
    concentration = result/target[...,None]
    assert (concentration >= q.amin(-2,keepdim=True)-2e-13).all()
    assert (concentration <= q.amax(-2,keepdim=True)+2e-13).all()
    assert (concentration[...,1:] >= 0).all()


def test_thin_single_layer_and_one_layer_remap():
    h = torch.tensor([1000., 1000., 1000.])
    target = torch.tensor([1000., .001, 1999.999])
    q = torch.tensor([1., 2., 3.])
    result = remap_contents(h,h*q,target,method='pcm')
    torch.testing.assert_close(result[1]/target[1],q[1],rtol=3e-11,atol=0)
    torch.testing.assert_close(remap_contents(torch.tensor([10.]),torch.tensor([70.]),torch.tensor([1.,3.,6.])),torch.tensor([7.,21.,42.]))


@pytest.mark.parametrize('target', [torch.tensor([0.,2.]),torch.tensor([1.,2.]),torch.tensor([float('nan'),1.])])
def test_invalid_target_is_rejected(target):
    with pytest.raises(ValueError):
        remap_contents(torch.tensor([1.,1.]),torch.tensor([1.,2.]),target)


def test_momentum_conservation_does_not_imply_energy_conservation():
    h, target = torch.tensor([1.,1.]),torch.tensor([.5,1.5])
    u = torch.tensor([-1.,1.])
    momentum = remap_contents(h,h*u,target)
    velocity = momentum/target
    torch.testing.assert_close(velocity,torch.tensor([-1.,1/3]))
    torch.testing.assert_close(momentum.sum(),torch.tensor(0.))
    torch.testing.assert_close(.5*(target*velocity.square()).sum(),torch.tensor(1/3))


@pytest.mark.parametrize('coast', [False, True])
def test_state_remap_edge_metric_budgets_bounds_and_land(mesh, coast):
    model,state = case(mesh,coast)
    result,report = VerticalRemapper().remap_with_report(model,state,ReferenceGrid()(state))
    for name in ('h','temp_h','salt_h'):
        torch.testing.assert_close(getattr(result,name).sum(-1),getattr(state,name).sum(-1),rtol=3e-14,atol=1e-10)
    torch.testing.assert_close(result.tracer_h.sum(-2),state.tracer_h.sum(-2),rtol=3e-14,atol=1e-10)
    expected_constant = result.h if not coast else model.domain.apply_cell(result.h)
    torch.testing.assert_close(result.tracer_h[...,0],expected_constant,rtol=3e-14,atol=1e-10)
    old_mass,new_mass = [model.dynamics.energy.mechanical.metric(s) for s in (state,result)]
    torch.testing.assert_close((new_mass*result.u_e).sum(-1),(old_mass*state.u_e).sum(-1),rtol=3e-14,atol=32.)
    for original,new in ((state.temperature(),result.temperature()),(state.salinity(),result.salinity()),(state.u_e,result.u_e)):
        assert (new >= original.amin(-1,keepdim=True)-2e-12).all()
        assert (new <= original.amax(-1,keepdim=True)+2e-12).all()
    before,after = [model.dynamics.energy.terms(s) for s in (state,result)]
    for name in before:
        torch.testing.assert_close(report.energy_change[name],after[name]-before[name],rtol=0,atol=0)
    assert report.energy_change['kinetic'].abs() > 0
    if coast:
        for name in ('h','temp_h','salt_h','tracer_h'):
            assert torch.equal(getattr(state,name)[~model.domain.wet_cell],getattr(result,name)[~model.domain.wet_cell])
        assert torch.equal(state.u_e[~model.domain.active_edge],result.u_e[~model.domain.active_edge])


def test_target_and_full_step_match_explicit_composition(mesh):
    model,state = case(mesh)
    moved = checked_ssprk3(model,state,10.)
    expected = VerticalRemapper()(model,moved,ReferenceGrid()(moved))
    actual,report = model.stepper.step_with_report(model,state,10.)
    standard = model.step(state,10.)
    for name in ('u_e','h','temp_h','salt_h','tracer_h'):
        assert torch.equal(getattr(actual,name),getattr(expected,name))
        assert torch.equal(getattr(actual,name),getattr(standard,name))
    for key in ('volume','temperature_content','salinity_content','tracer_content'):
        torch.testing.assert_close(model.budgets(actual)[key],model.budgets(state)[key],rtol=3e-14,atol=0)
    torch.testing.assert_close(report.energy_change['total'],model.dynamics.total_energy(actual)-model.dynamics.total_energy(moved),rtol=0,atol=0)
    with pytest.raises(ValueError,match='sum to one'):
        ReferenceGrid(torch.ones(4))(state)
    with pytest.raises(FloatingPointError,match='thickness'):
        VerticalRemapper()(model,state,torch.full_like(state.h,1e-4))


def test_bad_euler_stage_never_reaches_remap(mesh):
    model,state = case(mesh)
    called = []
    model.stepper.remapper = lambda *args: called.append(True)
    def invalid(s):
        return s.with_tensors(torch.zeros_like(s.u_e),torch.zeros_like(s.h),
            torch.zeros_like(s.h),-2*s.salt_h,torch.zeros_like(s.tracer_h))
    model.tendency = invalid
    with pytest.raises(FloatingPointError,match='salinity'):
        model.step(state,1.)
    assert not called


def test_content_and_geometry_gradients_away_from_switches():
    h = torch.tensor([.8,1.4,.7,1.1])
    content = torch.tensor([.3,.9,1.2,1.0],requires_grad=True)
    target = torch.tensor([.6,1.1,1.3,1.0],requires_grad=True)
    dc,dh = torch.tensor([.2,-.1,.3,.4]),torch.tensor([.1,-.2,.15,-.05])
    weight = torch.tensor([.3,.7,-.2,.4])
    def objective(c,t):
        return (remap_contents(h,c,t)*weight).sum()
    gc,gh = torch.autograd.grad(objective(content,target),(content,target))
    eps = 1e-6
    fd = (objective(content+eps*dc,target+eps*dh)-objective(content-eps*dc,target-eps*dh))/(2*eps)
    torch.testing.assert_close((gc*dc).sum()+(gh*dh).sum(),fd,rtol=2e-7,atol=1e-9)


def test_ale_forward_uses_no_autograd_and_report_retains_graph(mesh,monkeypatch):
    model,state = case(mesh)
    content = state.temp_h.clone().requires_grad_()
    state = state.with_tensors(state.u_e,state.h,content,state.salt_h,state.tracer_h)
    result,report = model.stepper.step_with_report(model,state,1.)
    gradient, = torch.autograd.grad(report.energy_change['dynamic_enthalpy'],content)
    assert torch.isfinite(gradient).all()
    def forbidden(*args,**kwargs):
        raise AssertionError('forward used autograd')
    monkeypatch.setattr(torch.autograd,'grad',forbidden)
    with torch.no_grad():
        assert torch.isfinite(model.step(state,1.).u_e).all()


def test_selected_full_ale_step_directional_derivative(mesh):
    model,state = case(mesh)
    content = state.temp_h.clone().requires_grad_()
    direction = torch.sin(torch.arange(content.numel()).reshape_as(content)*.37)
    def objective(value):
        candidate = state.with_tensors(state.u_e,state.h,value,state.salt_h,state.tracer_h)
        return model.step(candidate,10.).u_e.square().sum()
    gradient, = torch.autograd.grad(objective(content),content)
    eps = .01
    difference = (objective(content+eps*direction)-objective(content-eps*direction))/(2*eps)
    torch.testing.assert_close((gradient*direction).sum(),difference,rtol=2e-6,atol=1e-12)


def test_distortion_target_preserves_small_motion_and_resets_thin_columns(mesh):
    from vfs.ocean.ale import DistortionLimitedGrid
    model,state = case(mesh)
    target = DistortionLimitedGrid()
    assert torch.equal(target(state),state.h)
    result,report = VerticalRemapper().remap_with_report(model,state,target(state))
    assert result is state
    assert all(float(v) == 0 for v in report.energy_change.values())
    h = state.h.clone()
    removed = h[0,0]*.9
    h[0,0] -= removed
    h[0,1] += removed
    distorted = model.state(u_e=state.u_e,h=h,temperature=state.temperature(),salinity=state.salinity())
    new = target(distorted)
    assert torch.equal(new[1:],h[1:])
    torch.testing.assert_close(new[0],ReferenceGrid()(distorted)[0])
    mapped = VerticalRemapper()(model,distorted,new)
    torch.testing.assert_close(mapped.temp_h.sum(-1),distorted.temp_h.sum(-1),rtol=3e-14,atol=1e-10)
    assert mapped.h[0,0] > h[0,0]
