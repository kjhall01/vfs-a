"""Independent constrained projection algebra and nonlinear remap checks."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import torch
import pytest

@pytest.fixture(scope='module')
def probe():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    s=importlib.util.spec_from_file_location('adaptive_tests',Path(__file__).parents[2]/'tools/probe_ocean_adaptive_thermal.py')
    p=importlib.util.module_from_spec(s);sys.modules[s.name]=p;s.loader.exec_module(p);return p

@pytest.mark.parametrize('residual',[1.,-1.])
def test_linearized_minimum_change_against_independent_kkt(probe,residual):
    g=torch.tensor([[2.,3.,4.]]);s=torch.ones_like(g);w=torch.tensor([[.2,.3,.5]])
    step,active=probe.a.minimum_change_step(torch.zeros_like(g),g,s,torch.tensor([residual]),torch.tensor([0.]),w,torch.tensor([1e-14]))
    A=torch.cat([g,s],0) if active.item() else g
    b=torch.tensor([-residual,0.]) if active.item() else torch.tensor([-residual])
    Winv=torch.diag(1/w[0]);expected=Winv@A.T@torch.linalg.solve(A@Winv@A.T,b)
    assert torch.allclose(step[0],expected,rtol=1e-13,atol=1e-13)
    assert bool(active)==(residual>0)


def test_incompatible_rank_deficient_constraints_rejected(probe):
    one=torch.ones(1,3)
    with pytest.raises(ValueError,match='independent thermal directions'):
        probe.a.minimum_change_step(torch.zeros_like(one),one,one,torch.ones(1),torch.zeros(1),one,torch.zeros(1))

@pytest.mark.parametrize('amplitude',[1e-6,1e-4,.01])
def test_rescues_uniform_correction_entropy_failures(probe,amplitude):
    m,x,e=probe.p.p.p.p.q.p.case('front');physical=x.physical_state(e.reference)
    physical=physical.with_tensors(torch.zeros_like(x.u_e),x.h,x.h*(10+amplitude*torch.tensor([1.,0.,-1.])),x.h*35.,x.tracer_h)
    x=probe.p.p.p.p.front.EntropyState.from_temperature(physical,e.reference)
    target=x.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5])
    with pytest.raises(ValueError,match='decreases column entropy'):
        probe.p.h.HamiltonianRemapper(method='pcm').remap_with_report(m,x,target,e)
    end,r=probe.a.AdaptiveThermalRemapper(method='pcm').remap_with_report(m,x,target,e)
    scale=sum(v.abs() for v in probe.p.h.column_terms(e,x.physical_state(e.reference))[0].values()).clamp_min(1)
    assert (r.column_energy_residual.abs()<=256*torch.finfo(x.h.dtype).eps*scale).all()
    es=(e.mass_area(x)*x.entropy_h.abs()).sum(-1).clamp_min(1)
    assert (r.column_entropy_change>=-512*torch.finfo(x.h.dtype).eps*es).all()
    assert r.entropy_constraint_columns.all()
    assert (r.temperature_increment.amax(-1)-r.temperature_increment.amin(-1)>0).all()

@pytest.mark.parametrize('coast',[False,True])
@pytest.mark.parametrize('kind',['front','near_fresh','upper_salt','tilted'])
def test_conservative_fields_and_energy(probe,coast,kind):
    m,x,e=probe.p.p.p.p.q.p.case(kind,coast)
    target=x.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5])
    base,_=probe.p.p.e.EnthalpyRemapper().remap_with_report(m,x,target,e)
    end,r=probe.a.AdaptiveThermalRemapper().remap_with_report(m,x,target,e)
    for k in ['h','u_e','salt_h','tracer_h']:assert torch.equal(getattr(base,k),getattr(end,k))
    if coast:assert torch.equal(end.entropy_h[~m.domain.wet_cell],x.entropy_h[~m.domain.wet_cell])


def test_identity_missing_passives_and_domain_rejection(probe):
    m,x,e=probe.p.p.p.p.q.p.case('front');x=replace(x,tracer_h=None)
    remapper=probe.a.AdaptiveThermalRemapper();end,r=remapper.remap_with_report(m,x,x.h,e)
    assert end is x and r.iterations==0
    target=x.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5])
    end,r=remapper.remap_with_report(m,x,target,e);assert end.tracer_h is None
    bad=replace(x,u_e=torch.tensor([1e5,-1e5,1e5])[None,:].expand_as(x.u_e))
    with pytest.raises((ValueError,FloatingPointError)):remapper.remap_with_report(m,bad,target,e)


def test_no_forward_ad_and_combined_entropy_ledger(probe,monkeypatch):
    m,x,e=probe.p.p.p.p.q.p.case('front')
    def forbidden(*args,**kwargs):raise AssertionError('no forward AD')
    monkeypatch.setattr(torch.autograd,'grad',forbidden)
    moved,stages=probe.p.p.p.p.front.step(m,x,10.,e)
    end,r=probe.a.AdaptiveThermalRemapper().remap_with_report(m,moved,moved.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5]),e)
    mass=e.mass_area(x);production=10*sum(w*(mass*l.entropy_source).sum() for w,l in zip((1/6,1/6,2/3),stages))
    residual=(mass*(end.entropy_h-x.entropy_h)).sum()-production-r.entropy_change
    assert abs(residual)<1024*torch.finfo(x.h.dtype).eps*(mass*x.entropy_h.abs()).sum()


def test_directional_derivative_through_projection(probe):
    m,x,e=probe.p.p.p.p.q.p.case('front');target=x.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5])
    direction=x.h*torch.sin(torch.arange(x.h.numel()).reshape_as(x.h))
    def observable(alpha):
        trial=replace(x,entropy_h=x.entropy_h+alpha*direction)
        end,_=probe.a.AdaptiveThermalRemapper(method='pcm').remap_with_report(m,trial,target,e)
        return end.physical_state(e.reference).temperature().square().mean()
    alpha=torch.tensor(0.,requires_grad=True);ad=torch.autograd.grad(observable(alpha),alpha)[0]
    fd=(observable(.001)-observable(-.001))/.002
    assert torch.allclose(ad,fd,rtol=3e-5,atol=1e-8)
