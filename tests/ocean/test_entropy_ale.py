"""Entropy ALE inventories, explicit energy ledger and checked composition."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import pytest
import torch

@pytest.fixture(scope='module')
def probe():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('entropy_ale_test_probe',Path(__file__).parents[2]/'tools/probe_ocean_entropy_ale.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

@pytest.mark.parametrize('coast',[False,True])
@pytest.mark.parametrize('method',['pcm','plm'])
def test_remap_conserves_native_contents_and_reports_energy(probe,coast,method):
    m,x,e=probe.p.q.p.case('tilted',coast)
    target=x.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5])
    end,r=probe.a.EntropyRemapper(method=method).remap_with_report(m,x,target,e)
    assert r.changed_columns>0
    for name,residual in r.content_residuals.items():
        original=getattr(x,name)
        scale=original.abs().sum(-2 if name=='tracer_h' else -1).clamp_min(1)
        assert (residual.abs()<1024*torch.finfo(x.h.dtype).eps*scale).all()
    scale=m.dynamics.rho0*m.mesh.dc_edge*m.mesh.dv_edge*(m.mesh.cell_to_edge(x.h)*x.u_e).abs().sum(-1)
    assert (r.momentum_residual.abs()<1024*torch.finfo(x.h.dtype).eps*scale.clamp_min(1)).all()
    for key in r.energy_change:
        assert torch.equal(r.energy_change[key],e.terms(end)[key]-e.terms(x)[key])
    for name in ['entropy_h','salt_h','tracer_h']:
        old=getattr(x,name);new=getattr(end,name)
        if old is None:continue
        h=x.h[...,None] if name=='tracer_h' else x.h
        hn=end.h[...,None] if name=='tracer_h' else end.h
        axis=-2 if name=='tracer_h' else -1
        q=old/h;qn=new/hn
        assert (qn>=q.amin(axis,keepdim=True)-1e-10).all()
        assert (qn<=q.amax(axis,keepdim=True)+1e-10).all()
    if coast:
        assert torch.equal(end.entropy_h[~m.domain.wet_cell],x.entropy_h[~m.domain.wet_cell])
        assert torch.equal(end.u_e[~m.domain.active_edge],x.u_e[~m.domain.active_edge])


def test_identity_and_missing_passives(probe):
    m,x,e=probe.p.q.p.case('front');x=replace(x,tracer_h=None)
    remap=probe.a.EntropyRemapper()
    end,r=remap.remap_with_report(m,x,x.h,e)
    assert end is x and r.changed_columns==0
    assert all(v==0 for v in r.energy_change.values())
    target=x.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5])
    end,_=remap.remap_with_report(m,x,target,e)
    assert end.tracer_h is None


def test_step_matches_explicit_composition_and_entropy_budget(probe):
    m,x,e=probe.p.q.p.case('tilted')
    target=probe.v.ReferenceGrid(torch.tensor([.2,.3,.5]))
    integrator=probe.a.EntropyALEStepper(probe.p.front,target)
    end,stages,r=integrator.step_with_report(m,x,10.,e)
    moved,_=probe.p.front.step(m,x,10.,e)
    expected,_=integrator.remapper.remap_with_report(m,moved,target(moved),e)
    for name in ['u_e','h','entropy_h','salt_h','tracer_h']:
        assert torch.equal(getattr(end,name),getattr(expected,name))
    produced=10*sum(w*(e.mass_area(x)*l.entropy_source).sum() for w,l in zip((1/6,1/6,2/3),stages))
    assert abs(e.entropy(end)-e.entropy(x)-produced)<1024*torch.finfo(x.h.dtype).eps*(e.mass_area(x)*x.entropy_h.abs()).sum()
    assert r.changed_columns>0


def test_invalid_geometry_rejected(probe):
    m,x,e=probe.p.q.p.case('front')
    with pytest.raises(ValueError):probe.a.EntropyRemapper().remap_with_report(m,x,x.h*1.01,e)
    with pytest.raises(FloatingPointError):probe.a.EntropyRemapper().remap_with_report(m,x,-x.h,e)


def test_actual_wave_remaps(probe):
    r=probe.wave(4,1,.001)
    assert r['actual_remap_events']==r['steps'] and r['changed_column_events']>0
    assert max(r['max_column_relative_residual'].values())<1e-12


def test_distortion_trigger_actually_resets_thin_column(probe):
    m,x,e=probe.p.q.p.case('tilted')
    # Transfer thickness and its native contents together before applying the
    # policy, preserving concentrations; this deliberately crosses its trigger.
    h=x.h.clone();h[:,0]=.1*x.h[:,0];h[:,1]=x.h[:,1]+.9*x.h[:,0]
    x=replace(x,h=h,entropy_h=x.entropy_h*h/x.h,salt_h=x.salt_h*h/x.h,
              tracer_h=x.tracer_h*(h/x.h)[...,None])
    target=probe.v.DistortionLimitedGrid()(x)
    end,r=probe.a.EntropyRemapper().remap_with_report(m,x,target,e)
    assert r.changed_columns==x.h.shape[0]
    assert not torch.equal(end.h,x.h)
    for name in ['h','entropy_h','salt_h']:
        scale=getattr(x,name).abs().sum(-1).clamp_min(1)
        assert (r.content_residuals[name].abs()<1e-12*scale).all()
