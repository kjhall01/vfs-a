"""Independent derivative and guarded-integration checks for research predictor."""
import importlib.util
from pathlib import Path
import sys
from dataclasses import replace
import pytest
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean,TKEOcean

path=Path(__file__).resolve().parents[2]/'validation/ocean_column/budget_predictor.py'
spec=importlib.util.spec_from_file_location('budget_predictor_test',path)
predictor=importlib.util.module_from_spec(spec);sys.modules[spec.name]=predictor;spec.loader.exec_module(predictor)

@pytest.fixture(scope='module')
def ocean():
    mesh=VoronoiMesh(n_cells=12,lloyd_iters=0,rotation_scheme='compatible')
    return StratifiedOcean(mesh,OceanColumn(2,total_depth=100.))

@pytest.mark.parametrize('unstable',[False,True])
def test_analytic_pair_cost_against_autograd(ocean,unstable):
    h=torch.tensor([[30.,70.]],dtype=torch.float64).expand(ocean.mesh.nCells,2)
    t=h.new_tensor([15.,10.] if not unstable else [10.,15.]).expand_as(h)
    salt=h.new_tensor([34.5,35.]).expand_as(h)
    state=ocean.state(u_e=h.new_zeros(ocean.mesh.nEdges,2),h=h,temperature=t,salinity=salt)
    physical=ocean.check_state(state);ref=ocean.reference.evaluate(salt,t)
    slope=predictor.pair_work_slope(ocean,physical,ref)
    c=h.new_zeros(h.shape[0],1).requires_grad_()
    def exchange(q):
        flux=c*(q[:,1:]-q[:,:-1])
        return q+torch.cat((flux,-flux),-1)/h
    s=exchange(salt);enthalpy=exchange(ref['h0'])
    theta=ocean.reference.temperature_from_enthalpy(s,enthalpy)
    base=ocean.energy.base;node,weight=base._quadrature(h)
    z=base.height(physical)[...,None]+h[...,None]*(node-.5)
    psi=base.parcel_potential(theta[...,None],s[...,None],z)[0]
    energy=(ocean.energy.mass_area(physical)*h*(psi*weight).sum(-1)).sum()
    independent=torch.autograd.grad(energy,c)[0]
    torch.testing.assert_close(slope,independent,rtol=2e-9,atol=1e-4)
    assert (slope<0).all() if unstable else (slope>0).all()

@pytest.mark.parametrize('seed',[0.,1e-8,.01])
def test_predictor_conserves_and_keeps_local_tke_positive(ocean,seed):
    h=torch.full((ocean.mesh.nCells,2),50.,dtype=torch.float64)
    t=h.new_tensor([15.,10.]).expand_as(h)
    model=TKEOcean(ocean,closure=predictor.BudgetPredictor(2.))
    state=model.state(ocean.state(u_e=h.new_zeros(ocean.mesh.nEdges,2),h=h,
        temperature=t,salinity=h*0+35,tracer=torch.ones(*h.shape,1,dtype=h.dtype)),tke=h*0+seed)
    out,report=model.mix_with_report(state,300.)
    mass=(ocean.energy.mass_area(state.ocean)*h).sum()
    assert (out.tke_h>=0).all()
    assert abs(float((model.budgets(out)['total']-model.budgets(state)['total'])/mass))<1e-8
    assert abs(float(report.global_energy_residual/mass))<1e-8
    torch.testing.assert_close(out.ocean.salt_h.sum(-1),state.ocean.salt_h.sum(-1),rtol=2e-15,atol=0)
    torch.testing.assert_close(out.ocean.tracer_h,state.ocean.tracer_h,rtol=0,atol=0)


def test_invalid_research_proposal_rejected(ocean):
    class Bad(predictor.BudgetPredictor):
        def _mixing_proposal(self,ocean,physical,reference,reservoir,proposed):return 2*proposed
    h=torch.full((ocean.mesh.nCells,2),50.,dtype=torch.float64)
    model=TKEOcean(ocean,closure=Bad(2.))
    state=model.state(ocean.state(u_e=h.new_zeros(ocean.mesh.nEdges,2),h=h,
        temperature=h*0+12,salinity=h*0+35),tke=h*0+.01)
    with pytest.raises(ValueError,match='mixing proposal'):
        model.mix_with_report(state,100.)
