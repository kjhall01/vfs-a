"""The diagnostic observes the solver; it must not change its trajectory."""
import sys
from types import SimpleNamespace
import torch
from test_linear_column import load,harness
from test_path_work import path,exchange
sys.modules.setdefault('path_work',path)
sys.modules.setdefault('run',harness)
diag=load('diagnose_budget')

def test_ledger_signed_terms_and_transport_cancel():
    q=torch.tensor([[1.,2.,3.]],dtype=torch.float64)
    v=dict(mass=q*2,k0=q,dt=2.,g=q[:,:-1]*0+1,diss=q*0+.1,production=q*.2,work=q*.3,self=SimpleNamespace(budget_atol=1e-13,budget_rtol=1e-9))
    k=q*.5;r=diag.ledger(v,k)
    torch.testing.assert_close(r['transport'].sum(),torch.tensor(0.,dtype=q.dtype))
    torch.testing.assert_close(r['residual'],2*q*(k-q)-.2*q+.3*q+.1*k-r['transport'])
    negative=diag.ledger(v,-k)
    assert (negative['k']<0).all()  # No diagnostic clipping.

def test_observer_preserves_successful_solve():
    o,p,_,_,_,_=exchange()
    state=o.state(u_e=p.u_e,h=p.h,temperature=p.temperature(),salinity=p.salinity())
    k=p.h*.001;c=path.PathWorkClosure(2.)
    _,expected,report=c.apply(o,state,k,30.)
    observer=diag.Observer();result=observer.call(c,o,state,k,30.)
    actual=diag.ledger(observer.last,observer.last['final'])
    torch.testing.assert_close(actual['residual'],report.local_tke_residual,rtol=0,atol=0)
    assert result['status']=='completed' and result['iterations']==report.iterations
    assert torch.equal(torch.tensor(result['tke_h'],dtype=k.dtype),expected)

def test_observer_retains_rejected_ledger_without_mutating_inputs():
    o,p,_,_,_,_=exchange()
    state=o.state(u_e=p.u_e,h=p.h,temperature=p.temperature(),salinity=p.salinity())
    k=p.h*.001;original=[v.clone() for v in (state.entropy_h,state.salt_h,state.u_e,k)]
    r=diag.Observer().call(path.PathWorkClosure(2.,max_iterations=1),o,state,k,30.)
    assert r['status']=='rejected' and r['iteration']==1 and len(r['history'])==1
    assert all(torch.equal(x,y) for x,y in zip(original,(state.entropy_h,state.salt_h,state.u_e,k)))
