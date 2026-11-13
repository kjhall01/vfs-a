"""Independent banded solve, shared-edge budgets and semi-implicit decay."""
import sys
import torch
import pytest
from test_linear_column import load
from test_path_work import path
from test_coupled_column import ocean,state
sys.modules.setdefault('path_work',path)
lag=load('lagged')

def test_factor_multiple_rhs_against_dense():
    d=torch.tensor([[3.,4.,5.],[5.,2.,3.]],dtype=torch.float64);off=d[:,:-1]*0-.2
    rhs=torch.arange(18,dtype=d.dtype).reshape(2,3,3)/10
    A=torch.diag_embed(d)+torch.diag_embed(off,1)+torch.diag_embed(off,-1)
    f=lag.Factor(d,off)
    torch.testing.assert_close(f.solve(rhs),torch.linalg.solve(A,rhs),rtol=2e-15,atol=1e-15)
    torch.testing.assert_close(f.solve(rhs[...,0]),torch.linalg.solve(A,rhs)[...,0],rtol=2e-15,atol=1e-15)

@pytest.mark.parametrize('unstable',[False,True])
@pytest.mark.parametrize('substeps',[1,4])
def test_native_shared_edges_conservation(ocean,unstable,substeps):
    m,s=state(ocean,lag.SubcycledLaggedClosure(2.,substeps=substeps),unstable=unstable,unequal=True)
    out,r=m.mix_with_report(s,30.);mass=ocean.energy.mass_area(s.ocean)*s.ocean.h
    assert (out.tke_h>=0).all() and (out.ocean.tracer_h>=0).all() and r.budget_error<=1
    torch.testing.assert_close(out.ocean.salt_h.sum(-1),s.ocean.salt_h.sum(-1),rtol=2e-15,atol=0)
    torch.testing.assert_close(out.ocean.tracer_h.sum(-2),s.ocean.tracer_h.sum(-2),rtol=2e-15,atol=0)
    assert abs(float((m.budgets(out)['total']-m.budgets(s)['total'])/mass.sum()))<1e-8
    assert float((r.column_energy_residual.abs()/mass.sum(-1)).max())<1e-8
    assert abs(float(r.kinetic_column_exchange.sum()/mass.sum()))<1e-16
    assert r.mixing_limiter['semi_trials']<=substeps*(1+10)


def test_linearized_decay_and_zero_tke(ocean):
    m,s=state(ocean,lag.LaggedClosure(2.,c_mu=0.,predictor_steps=0),seed=.01)
    out,r=m.mix_with_report(s,100.)
    expected=.01/(1+100*.7*.01**.5/2)
    torch.testing.assert_close(out.specific_tke(),torch.full_like(s.tke_h,expected),rtol=2e-15,atol=0)
    m,s=state(ocean,lag.LaggedClosure(2.),seed=0.)
    out,r=m.mix_with_report(s,100.)
    assert torch.equal(out.tke_h,s.tke_h)
    for field in ('u_e','h','salt_h','entropy_h','tracer_h'):assert torch.equal(getattr(out.ocean,field),getattr(s.ocean,field))

def test_explicit_scalar_deferral_preserves_energy_and_inventory(ocean):
    from dataclasses import replace
    m,s=state(ocean,lag.LaggedClosure(2.,predictor_steps=0,max_reductions=0),seed=1e-12)
    s=replace(s,ocean=replace(s.ocean,u_e=torch.zeros_like(s.ocean.u_e)))
    before=s.ocean.salt_h.clone();out,r=m.mix_with_report(s,300.)
    assert r.mixing_limiter['deferred_columns']>0
    assert torch.equal(out.ocean.salt_h,before) and torch.equal(s.ocean.salt_h,before)
    assert torch.equal(r.buoyancy_layer_work,torch.zeros_like(r.buoyancy_layer_work))
    assert (out.tke_h>=0).all() and r.budget_error<=1
    mass=ocean.energy.mass_area(s.ocean)*s.ocean.h
    assert abs(float((m.budgets(out)['total']-m.budgets(s)['total'])/mass.sum()))<1e-8
