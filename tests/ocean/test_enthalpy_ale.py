"""Potential-enthalpy remap invariants and explicit nonlinear energy accounting."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import pytest
import torch

@pytest.fixture(scope='module')
def probe():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('enthalpy_ale_tests',Path(__file__).parents[2]/'tools/probe_ocean_enthalpy_ale.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

@pytest.mark.parametrize('kind',['front','tilted','near_fresh','upper_salt'])
@pytest.mark.parametrize('coast',[False,True])
@pytest.mark.parametrize('method',['pcm','plm'])
def test_native_energy_inventory_and_domain(probe,kind,coast,method):
    m,x,e=probe.p.p.q.p.case(kind,coast)
    target=x.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5])
    end,r=probe.e.EnthalpyRemapper(method=method).remap_with_report(m,x,target,e)
    physical=x.physical_state(e.reference)
    h0=x.h*e.reference.evaluate(physical.salinity(),physical.temperature())['h0']
    eps=torch.finfo(x.h.dtype).eps
    assert (r.potential_enthalpy_residual.abs()<1024*eps*h0.abs().sum(-1).clamp_min(1)).all()
    for name in ['h','salt_h','tracer_h']:
        scale=getattr(x,name).abs().sum(-2 if name=='tracer_h' else -1).clamp_min(1)
        assert (r.content_residuals[name].abs()<1024*eps*scale).all()
    # Entropy is diagnosed, not artificially held fixed. For these test states
    # entropy must not fall beyond roundoff; this is not a universal PLM proof.
    mass=e.mass_area(x)
    change=(mass*(end.entropy_h-x.entropy_h)).sum(-1)
    scale=(mass*x.entropy_h.abs()).sum(-1)
    assert (change>=-1024*eps*scale).all()
    terms0,terms1=e.terms(x),e.terms(end)
    assert abs(r.energy_change['reference_thermal'])<1024*eps*terms0['reference_thermal'].abs()
    for key in terms0:assert torch.equal(r.energy_change[key],terms1[key]-terms0[key])
    if coast:
        assert torch.equal(end.entropy_h[~m.domain.wet_cell],x.entropy_h[~m.domain.wet_cell])
        assert torch.equal(end.u_e[~m.domain.active_edge],x.u_e[~m.domain.active_edge])


def test_identity_and_absent_tracer(probe):
    m,x,e=probe.p.p.q.p.case('front');x=replace(x,tracer_h=None)
    end,r=probe.e.EnthalpyRemapper().remap_with_report(m,x,x.h,e)
    assert end is x and r.changed_columns==0
    assert all(v==0 for v in r.energy_change.values())
    end,r=probe.e.EnthalpyRemapper().remap_with_report(m,x,x.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5]),e)
    assert end.tracer_h is None


def test_complete_step_entropy_ledger(probe):
    m,x,e=probe.p.p.q.p.case('front')
    target=probe.p.v.ReferenceGrid(torch.tensor([.2,.3,.5]))
    stepper=probe.p.a.EntropyALEStepper(probe.p.p.front,target,probe.e.EnthalpyRemapper())
    end,stages,r=stepper.step_with_report(m,x,10.,e)
    mass=e.mass_area(x)
    produced=10*sum(w*(mass*l.entropy_source).sum() for w,l in zip((1/6,1/6,2/3),stages))
    residual=(mass*(end.entropy_h-x.entropy_h)).sum()-produced-r.entropy_change
    assert abs(residual)<1024*torch.finfo(x.h.dtype).eps*(mass*x.entropy_h.abs()).sum()
    assert r.changed_columns>0 and r.entropy_change>0


def test_thermal_derivative_through_remap(probe):
    m,x,e=probe.p.p.q.p.case('front')
    target=x.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5])
    direction=x.h*torch.sin(torch.arange(x.h.numel()).reshape_as(x.h))
    def observable(alpha):
        state=replace(x,entropy_h=x.entropy_h+alpha*direction)
        end,_=probe.e.EnthalpyRemapper(method='pcm').remap_with_report(m,state,target,e)
        return end.physical_state(e.reference).temperature().square().mean()
    alpha=torch.tensor(0.,requires_grad=True)
    ad=torch.autograd.grad(observable(alpha),alpha)[0]
    delta=1e-3;fd=(observable(delta)-observable(-delta))/(2*delta)
    assert torch.allclose(ad,fd,rtol=2e-5,atol=1e-8)


def test_repeated_remaps_preserve_potential_enthalpy(probe):
    m,x,e=probe.p.p.q.p.case('front');start=x
    grid=x.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5])
    remapper=probe.e.EnthalpyRemapper()
    before=e.terms(start)['reference_thermal']
    for _ in range(10):
        x,_=remapper.remap_with_report(m,x,grid,e)
        x,_=remapper.remap_with_report(m,x,start.h,e)
    after=e.terms(x)['reference_thermal']
    assert abs(after-before)<4096*torch.finfo(x.h.dtype).eps*before.abs()
    assert e.entropy(x)>e.entropy(start)
