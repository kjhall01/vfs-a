"""Local energy ownership, independent derivatives and constrained ALE closure."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import pytest
import torch

@pytest.fixture(scope='module')
def probe():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('hamiltonian_ale_tests',Path(__file__).parents[2]/'tools/probe_ocean_hamiltonian_ale.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

@pytest.mark.parametrize('coast',[False,True])
def test_column_decomposition_and_independent_derivative(probe,coast):
    m,x,e=probe.p.p.p.q.p.case('front',coast)
    physical=x.physical_state(e.reference)
    alpha=torch.zeros(x.h.shape[0],requires_grad=True)
    perturbed=physical.with_tensors(physical.u_e,physical.h,physical.temp_h+physical.h*alpha[:,None],physical.salt_h,physical.tracer_h)
    columns,analytic=probe.h.column_terms(e,perturbed)
    native=e.base.terms(perturbed)
    native['reference_thermal']=(e.mass_area(x)*x.h*e.reference.evaluate(perturbed.salinity(),perturbed.temperature())['h0']).sum()
    native['total']=native['total']+native['reference_thermal']
    for k,v in columns.items():assert torch.allclose(v.sum(),native[k],rtol=2e-14,atol=1e-8)
    independent=torch.autograd.grad(native['total'],alpha)[0]
    assert torch.allclose(analytic,independent,rtol=2e-13,atol=1e-8)

@pytest.mark.parametrize('kind',['front','tilted','near_fresh','upper_salt'])
@pytest.mark.parametrize('coast',[False,True])
def test_local_closure_entropy_and_unchanged_mechanical_state(probe,kind,coast):
    m,x,e=probe.p.p.p.q.p.case(kind,coast)
    target=x.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5])
    provisional,_=probe.p.e.EnthalpyRemapper().remap_with_report(m,x,target,e)
    end,r=probe.h.HamiltonianRemapper().remap_with_report(m,x,target,e)
    for k in ['u_e','h','salt_h','tracer_h']:assert torch.equal(getattr(end,k),getattr(provisional,k))
    scale=sum(v.abs() for v in probe.h.column_terms(e,x.physical_state(e.reference))[0].values()).clamp_min(1)
    assert (r.column_energy_residual.abs()<=256*torch.finfo(x.h.dtype).eps*scale).all()
    entropy_scale=(e.mass_area(x)*x.entropy_h.abs()).sum(-1).clamp_min(1)
    assert (r.column_entropy_change>=-512*torch.finfo(x.h.dtype).eps*entropy_scale).all()
    assert torch.equal(r.entropy_change,r.column_entropy_change.sum())
    if coast:assert torch.equal(end.entropy_h[~m.domain.wet_cell],x.entropy_h[~m.domain.wet_cell])


def test_identity_and_absent_tracer(probe):
    m,x,e=probe.p.p.p.q.p.case('front');x=replace(x,tracer_h=None)
    remap=probe.h.HamiltonianRemapper();end,r=remap.remap_with_report(m,x,x.h,e)
    assert end is x and (r.column_energy_residual==0).all()
    end,_=remap.remap_with_report(m,x,x.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5]),e)
    assert end.tracer_h is None


def test_unrepresentable_thermal_target_rejected(probe):
    m,x,e=probe.p.p.p.q.p.case('front')
    x=replace(x,u_e=torch.tensor([1e5,-1e5,1e5])[None,:].expand_as(x.u_e))
    with pytest.raises(ValueError,match='outside thermal-domain bracket'):
        probe.h.HamiltonianRemapper().remap_with_report(m,x,x.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5]),e)


def test_complete_step_remap_does_not_erase_time_integration_error(probe):
    m,x,e=probe.p.p.p.q.p.case('front')
    target=probe.p.p.v.ReferenceGrid(torch.tensor([.2,.3,.5]))
    moved,stages=probe.p.p.p.front.step(m,x,10.,e)
    end,r=probe.h.HamiltonianRemapper().remap_with_report(m,moved,target(moved),e)
    mass=e.mass_area(x)
    produced=10*sum(w*(mass*l.entropy_source).sum() for w,l in zip((1/6,1/6,2/3),stages))
    residual=(mass*(end.entropy_h-x.entropy_h)).sum()-produced-r.entropy_change
    assert abs(residual)<1024*torch.finfo(x.h.dtype).eps*(mass*x.entropy_h.abs()).sum()
    delta=e.terms(end)['total']-e.terms(moved)['total']
    assert abs(delta)<1024*torch.finfo(x.h.dtype).eps*e.terms(moved)['total'].abs()


def test_no_forward_autograd(probe,monkeypatch):
    m,x,e=probe.p.p.p.q.p.case('front')
    def forbidden(*args,**kwargs):raise AssertionError('forward AD forbidden')
    monkeypatch.setattr(torch.autograd,'grad',forbidden)
    probe.h.HamiltonianRemapper().remap_with_report(m,x,x.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5]),e)


def test_entropy_losing_root_is_rejected_without_modifying_input(probe):
    m,x,e=probe.p.p.p.q.p.case('front');physical=x.physical_state(e.reference)
    physical=physical.with_tensors(torch.zeros_like(x.u_e),x.h,x.h*(10+.01*torch.tensor([1.,0.,-1.])),x.h*35.,x.tracer_h)
    x=probe.p.p.p.front.EntropyState.from_temperature(physical,e.reference)
    original=x.entropy_h.clone()
    with pytest.raises(ValueError,match='decreases column entropy'):
        probe.h.HamiltonianRemapper(method='pcm').remap_with_report(m,x,x.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5]),e)
    assert torch.equal(original,x.entropy_h)


def test_corrected_temperature_directional_derivative(probe):
    m,x,e=probe.p.p.p.q.p.case('front')
    target=x.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5])
    direction=x.h*torch.sin(torch.arange(x.h.numel()).reshape_as(x.h))
    def observable(alpha):
        trial=replace(x,entropy_h=x.entropy_h+alpha*direction)
        end,_=probe.h.HamiltonianRemapper(method='pcm').remap_with_report(m,trial,target,e)
        return end.physical_state(e.reference).temperature().square().mean()
    alpha=torch.tensor(0.,requires_grad=True)
    ad=torch.autograd.grad(observable(alpha),alpha)[0]
    delta=1e-3;fd=(observable(delta)-observable(-delta))/(2*delta)
    assert torch.allclose(ad,fd,rtol=3e-5,atol=1e-8)
