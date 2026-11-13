"""Research coupled-column budgets, independent solves and transactional failure."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import pytest
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean,TKEOcean

path=Path(__file__).resolve().parents[2]/'validation/ocean_column/coupled.py'
spec=importlib.util.spec_from_file_location('coupled_column_test',path)
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)

@pytest.fixture(scope='module')
def ocean():
    mesh=VoronoiMesh(n_cells=12,lloyd_iters=0,rotation_scheme='compatible')
    return StratifiedOcean(mesh,OceanColumn(4,total_depth=100.))


def state(o,closure,seed=.001,unstable=False,unequal=False):
    h=torch.full((o.mesh.nCells,4),25.,dtype=torch.float64)
    if unequal:
        h=h*(1+.15*o.mesh.cell_xyz[:,:1]*h.new_tensor([1.,-1.,1.,-1.]))
    t=h.new_tensor([15.,14.,13.,12.])
    if unstable:t=t.flip(0)
    u=h.new_tensor([.2,.1,-.1,-.2]).expand(o.mesh.nEdges,4).clone()
    m=TKEOcean(o,closure=closure)
    s=m.state(o.state(u_e=u,h=h,temperature=t.expand_as(h),salinity=h*0+35,
                     tracer=(h.new_tensor([1.,0.,0.,1.]).expand_as(h))[...,None]),tke=h*0+seed)
    return m,s


def test_banded_solver_against_independent_dense():
    diagonal=torch.tensor([[3.,4.,5.],[5.,2.,3.]],dtype=torch.float64)
    off=torch.tensor([[-.2,-.3],[-.1,-.4]],dtype=diagonal.dtype)
    rhs=torch.tensor([[1.,2.,3.],[3.,-1.,2.]],dtype=diagonal.dtype)
    matrices=torch.diag_embed(diagonal)+torch.diag_embed(off,offset=1)+torch.diag_embed(off,offset=-1)
    exact=torch.linalg.solve(matrices,rhs[...,None])[...,0]
    torch.testing.assert_close(c.tridiagonal(diagonal,off,rhs),exact,rtol=2e-15,atol=1e-15)


@pytest.mark.parametrize('unstable',[False,True])
def test_native_shared_edges_and_actual_local_budget(ocean,unstable):
    m,s=state(ocean,c.CoupledClosure(2.),unstable=unstable,unequal=True)
    out,r=m.mix_with_report(s,30.)
    mass=ocean.energy.mass_area(s.ocean)*s.ocean.h
    assert (out.tke_h>=0).all() and (out.ocean.tracer_h>=0).all()
    assert r.fixed_point_error<=1 and r.budget_error<=1
    torch.testing.assert_close(out.ocean.salt_h.sum(-1),s.ocean.salt_h.sum(-1),rtol=2e-15,atol=0)
    torch.testing.assert_close(out.ocean.tracer_h.sum(-2),s.ocean.tracer_h.sum(-2),rtol=2e-15,atol=0)
    assert float((r.local_tke_residual.abs()/mass).max())<1e-10
    assert float((r.column_energy_residual.abs()/mass.sum(-1)).max())<1e-8
    assert abs(float((m.budgets(out)['total']-m.budgets(s)['total'])/mass.sum()))<1e-8
    # Native shared-edge kinetic exchange must sum to zero across columns.
    assert abs(float(r.kinetic_column_exchange.sum()/mass.sum()))<1e-16
    assert (r.buoyancy_layer_work.sum()<0) if unstable else (r.buoyancy_layer_work.sum()>0)


def test_implicit_decay_against_scalar_equation(ocean):
    m,s=state(ocean,c.CoupledClosure(2.,c_mu=0.),seed=.01)
    out,r=m.mix_with_report(s,100.)
    lo,hi=0.,.01
    for _ in range(80):
        mid=.5*(lo+hi)
        if mid+100*.7/2*mid**1.5>.01:hi=mid
        else:lo=mid
    torch.testing.assert_close(out.specific_tke(),torch.full_like(s.tke_h,.5*(lo+hi)),rtol=2e-6,atol=1e-12)
    assert (r.dissipated_heat>0).all()


def test_zero_tke_remains_exact_and_has_no_floor(ocean):
    m,s=state(ocean,c.CoupledClosure(2.),seed=0.)
    out,r=m.mix_with_report(s,100.)
    assert torch.equal(out.tke_h,s.tke_h)
    for name in ('u_e','h','salt_h','entropy_h','tracer_h'):
        assert torch.equal(getattr(out.ocean,name),getattr(s.ocean,name))


def test_failed_solve_is_transactional(ocean):
    m,s=state(ocean,c.CoupledClosure(2.,max_iterations=1))
    before={name:getattr(s.ocean,name).clone() for name in ('u_e','h','salt_h','entropy_h','tracer_h')}
    before_k=s.tke_h.clone()
    with pytest.raises(FloatingPointError,match='failed to converge'):
        m.mix_with_report(s,300.)
    assert torch.equal(s.tke_h,before_k)
    for name,value in before.items():assert torch.equal(getattr(s.ocean,name),value)


def test_zero_layer_can_receive_tke_without_initial_floor(ocean):
    m,s=state(ocean,c.CoupledClosure(2.),seed=.001)
    k=s.tke_h.clone();k[:,1]=0
    s=replace(s,tke_h=k)
    out,r=m.mix_with_report(s,10.)
    assert (out.tke_h[:,1]>0).all()
    assert (r.tke_energy_flux.abs()>0).any()
    assert r.budget_error<=1
