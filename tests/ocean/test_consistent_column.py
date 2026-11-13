"""Fresh end-TKE physics must supply both the guard and the committed update."""
import sys
import pytest
import torch
from test_linear_column import load
from test_path_work import path,exchange
from test_budget_diagnostic import diag
sys.modules.setdefault('path_work',path)
consistent=load('consistent')

@pytest.mark.parametrize('budget_acceleration',[False,True])
def test_fresh_budget_and_returned_physics_agree(budget_acceleration):
    o,p,_,_,_,_=exchange()
    state=o.state(u_e=p.u_e,h=p.h,temperature=p.temperature(),salinity=p.salinity())
    c=consistent.ConsistentPathClosure(2.,budget_acceleration=budget_acceleration)
    observer=diag.Observer();out=observer.call(c,o,state,p.h*.001,30.)
    assert out['status']=='completed'
    v=observer.last
    mixed,production,work,flux,g,diss,*_=v['trial'](v['final'])
    fresh=diag.ledger(dict(v,production=production,work=work,g=g,diss=diss),v['final'])
    assert float(fresh['scaled_residual'].max())<=1
    torch.testing.assert_close(fresh['residual'],v['report'].local_tke_residual,rtol=0,atol=0)
    torch.testing.assert_close(work,v['report'].buoyancy_layer_work,rtol=0,atol=0)
    torch.testing.assert_close(mixed.u_e,v['end'].u_e,rtol=0,atol=0)
    assert (v['final']>=0).all() and v['report'].entropy_change.min()>=0
    assert abs(float(v['report'].global_energy_residual/v['mass'].sum()))<1e-8
    assert v['report'].mixing_limiter['trial_evaluations']>v['iteration']


def test_refreshed_rejection_preserves_inputs():
    o,p,_,_,_,_=exchange()
    state=o.state(u_e=p.u_e,h=p.h,temperature=p.temperature(),salinity=p.salinity())
    k=p.h*.001;original=[x.clone() for x in (state.u_e,state.h,state.salt_h,state.entropy_h,k)]
    with pytest.raises(FloatingPointError):consistent.ConsistentPathClosure(2.,max_iterations=1).apply(o,state,k,30.)
    assert all(torch.equal(x,y) for x,y in zip(original,(state.u_e,state.h,state.salt_h,state.entropy_h,k)))
