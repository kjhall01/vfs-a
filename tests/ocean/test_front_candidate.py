"""Local flux bounds, independent energy work and continuous force budgeting."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch


@pytest.fixture(scope='module')
def probe():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    path=Path(__file__).parents[2]/'tools/probe_ocean_front.py'
    spec=importlib.util.spec_from_file_location('front_tests',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('kind',['front','tilted','near_fresh','upper_salt'])
@pytest.mark.parametrize('coast',[False,True])
def test_independent_energy_stage_entropy_and_inventories(probe,kind,coast):
    f=probe.front;m,x,e=probe.q.p.case(kind,coast)
    names=['u_e','h','entropy_h','salt_h']
    leaves=[getattr(x,k).clone().requires_grad_() for k in names]
    x=replace(x,**dict(zip(names,leaves)))
    gradient=torch.autograd.grad(e.terms(x)['total'],leaves)
    rhs,w=f.evaluate(m,x,10.,e)
    terms=[a*getattr(rhs,k) for a,k in zip(gradient,names)]
    assert abs(sum(a.sum() for a in terms))<1024*torch.finfo(x.h.dtype).eps*sum(a.abs().sum() for a in terms)
    assert (w.heat>=0).all() and (w.entropy_source>=0).all()
    end,stages=f.step(m,x,10.,e)
    mass=e.mass_area(x)
    expected=10*sum(a*(mass*l.entropy_source).sum() for a,l in zip((1/6,1/6,2/3),stages))
    actual=(mass*(end.entropy_h-x.entropy_h)).sum()
    assert abs(actual-expected)<256*torch.finfo(x.h.dtype).eps*(mass*x.entropy_h.abs()).sum()
    for k in ['h','salt_h']:
        initial=(mass*getattr(x,k)).sum()
        assert abs((mass*(getattr(end,k)-getattr(x,k))).sum())<256*torch.finfo(x.h.dtype).eps*initial.abs()
    for l in stages:
        assert (l.transport.amounts[...,1]>0).all()
        assert (l.transport.amounts[...,1]<42*l.transport.volume).all()
        assert (l.transport.amounts[...,2:]>=0).all()
    # Each Euler passive update obeys the stage's local amount bounds, before
    # SSP combinations. These bounds deliberately do not include active heat.
    physical=x.physical_state(e.reference)
    eta,salt,face=f.parent.high_traces(m,physical,e)
    target,low,_=f.local_target(m,x,10.,eta,salt,face['composition_entropy'])
    tol=256*torch.finfo(x.h.dtype).eps*low.upper.abs().clamp_min(1)
    amounts=w.transport.amounts.reshape_as(low.upper)
    assert (amounts[:,2:]>=low.lower[:,2:]-tol[:,2:]).all()
    assert (amounts[:,2:]<=low.upper[:,2:]+tol[:,2:]).all()


def test_isolated_local_bounds_and_conservation(probe):
    original=probe.q.c
    try:
        probe.q.c=probe.front;r={}
        probe.q.transport(SimpleNamespace(cells=32,profile='sharp',circuits=1,cfl=.2),r,lambda:None)
    finally:probe.q.c=original
    for k,field in r['fields'].items():
        scale=max(abs(field['maximum']),1.)
        assert field['overshoot']<512*torch.finfo(torch.float64).eps*scale
        assert field['undershoot']<512*torch.finfo(torch.float64).eps*scale
        assert abs(field['inventory_change'])<512*torch.finfo(torch.float64).eps*scale


def test_single_edge_force_has_two_sided_zero_limit(probe):
    rows=probe.zero_edge()
    assert any(r['work_limited_faces']>0 for r in rows)
    tiny=[r for r in rows if abs(r['velocity_m_s'])<=1e-9]
    for row in tiny:
        assert abs(row['force_m_s2'])<=abs(row['velocity_m_s'])/20*(1+1e-10)
        assert row['other_salt_limited_faces']>0


def test_work_budget_is_active_and_piecewise_derivative(probe,monkeypatch):
    f=probe.front;m,x,e=probe.q.p.case('upper_salt')
    u=(x.u_e*1e-9).requires_grad_();x=replace(x,u_e=u)
    _,w=f.evaluate(m,x,10.,e)
    _,base=f.parent.evaluate(m,x,10.,e)
    index=int(torch.nonzero((w.work_fraction>0)&(w.work_fraction<1)&(base.face_work==0))[0])
    force=w.force_correction.flatten()[index]
    assert float(force.detach())==pytest.approx(float(-u.flatten()[index].detach()/20),rel=1e-10,abs=1e-25)
    grad,=torch.autograd.grad(force,u)
    assert float(grad.flatten()[index])==pytest.approx(-1/20,rel=1e-9)
    def forbidden(*args,**kwargs):raise AssertionError('forward used autograd')
    monkeypatch.setattr(torch.autograd,'grad',forbidden)
    with torch.no_grad():f.evaluate(m,x,10.,e)


def test_passive_contents_do_not_change_active_dynamics(probe):
    f=probe.front;m,x,e=probe.q.p.case('front')
    filled=replace(x,tracer_h=torch.ones_like(x.tracer_h)*x.h[...,None])
    a,_=f.evaluate(m,x,10.,e);b,_=f.evaluate(m,filled,10.,e)
    for k in ['u_e','h','entropy_h','salt_h']:assert torch.equal(getattr(a,k),getattr(b,k))


def test_uniform_passive_and_absent_passive(probe):
    f=probe.front;m,x,e=probe.q.p.case('front')
    uniform=replace(x,tracer_h=4.2*x.h[...,None])
    end,_=f.step(m,uniform,10.,e)
    torch.testing.assert_close(end.tracer_h,4.2*end.h[...,None],rtol=2e-14,atol=1e-12)
    empty=replace(x,tracer_h=None)
    end,_=f.step(m,empty,10.,e)
    assert end.tracer_h is None


def test_work_cap_and_rest_pressure(probe):
    f=probe.front;m,x,e=probe.q.p.case('upper_salt')
    for factor in [1.,1e-6,1e-9,0.]:
        s=replace(x,u_e=x.u_e*factor)
        rhs,w=f.evaluate(m,s,10.,e);base,b=f.parent.evaluate(m,s,10.,e)
        bound=torch.maximum(b.force_correction.abs(),s.u_e.abs()/20)
        assert (w.force_correction.abs()<=bound*(1+1e-10)+1e-25).all()
        if factor==0:
            assert w.heat.count_nonzero()==0 and w.force_correction.count_nonzero()==0
            assert torch.equal(rhs.u_e,base.u_e)


@pytest.mark.parametrize('kind',['front','tilted','near_fresh','upper_salt'])
@pytest.mark.parametrize('coast',[False,True])
def test_shared_gibbs_stage_traces_bitwise_and_autograd(probe,kind,coast,monkeypatch):
    """Compare new scratch reuse to old independent face-field reconstruction.

    Both paths use identical energy/work equations, positivity constraints, and
    differentiation. The old path is invoked by disabling scratch *only* in
    the local correction; parent Gibbs still constructs its original traces.
    """
    from dataclasses import fields, is_dataclass
    f=probe.front
    m, x, e=probe.q.p.case(kind,coast)

    def exact(a,b):
        if isinstance(a,torch.Tensor):
            assert torch.equal(a,b)
        elif is_dataclass(a):
            for field in fields(a):
                if field.name not in ('mesh','column'):
                    exact(getattr(a,field.name),getattr(b,field.name))
        elif isinstance(a,(tuple,list)):
            assert len(a)==len(b)
            for aa,bb in zip(a,b):exact(aa,bb)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:exact(a[k],b[k])
        else:
            assert a==b

    original=f._local_target_fields
    with torch.no_grad():
        new_rhs,new_report=f.evaluate(m,x,10.,e)
        def old_local(*args,**kwargs):
            assert kwargs.get('scratch') is not None
            return original(*args,**{**kwargs,'scratch':None})
        with monkeypatch.context() as p:
            p.setattr(f,'_local_target_fields',old_local)
            old_rhs,old_report=f.evaluate(m,x,10.,e)
        exact(new_rhs,old_rhs)
        exact(new_report,old_report)

    def grad(reuse):
        inputs=[getattr(x,k).clone().detach().requires_grad_(True)
                for k in ('u_e','h','entropy_h','salt_h')]
        state=replace(x,**dict(zip(('u_e','h','entropy_h','salt_h'),inputs)))
        if reuse:
            rhs,report=f.evaluate(m,state,10.,e)
        else:
            def old_local(*args,**kwargs):
                return original(*args,**{**kwargs,'scratch':None})
            with monkeypatch.context() as p:
                p.setattr(f,'_local_target_fields',old_local)
                rhs,report=f.evaluate(m,state,10.,e)
        loss=(rhs.u_e.square().sum()+rhs.h.square().sum()+
              rhs.entropy_h.square().sum()+rhs.salt_h.square().sum()+
              report.face_work.square().sum()*1e-8)
        return [v.detach() for v in torch.autograd.grad(loss,inputs)]
    # This is more stringent than approximate forward endpoint agreement.
    for a,b in zip(grad(True),grad(False)):
        assert torch.equal(a,b)


def test_gibbs_scratch_is_only_used_by_opt_in_front(probe,monkeypatch):
    f=probe.front
    m,x,e=probe.q.p.case('front',False)
    calls=[]
    old=f.parent.bounded_transport
    def track(*args,**kwargs):
        calls.append(kwargs.get('scratch'))
        return old(*args,**kwargs)
    with monkeypatch.context() as p:
        p.setattr(f.parent,'bounded_transport',track)
        # A direct parent RHS must not expose or retain scratch results.
        f.parent.evaluate(m,x,10.,e)
        assert calls[-1] is None
        # The composed front supplies a fresh dict and never shares it across
        # independent stages or states.
        f.evaluate(m,x,10.,e)
        f.evaluate(m,x,10.,e)
    assert isinstance(calls[-1],dict)
    assert calls[-1] is not calls[-2]
    assert all(k in calls[-1] for k in ('q','high','donor','anti'))
