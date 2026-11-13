"""Independent affine-energy controls; research adapter stays outside VFS."""
import importlib.util,sys
from pathlib import Path
from dataclasses import replace
import torch
import pytest
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import TKEOcean

def load(name):
    path=Path(__file__).resolve().parents[2]/'validation/ocean_column'/f'{name}.py'
    spec=importlib.util.spec_from_file_location('linear_test_'+name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

coupled=load('coupled');sys.modules.setdefault('coupled',coupled)
linear=load('linear_eos');harness=load('run')

def setup(layers=2):
    o=linear.LinearColumnOcean(harness.ColumnMetric(),OceanColumn(layers,total_depth=100.))
    h=torch.full((1,layers),100/layers,dtype=torch.float64)
    t=torch.linspace(15.,10.,layers,dtype=h.dtype)[None]
    s=torch.linspace(34.5,35.,layers,dtype=h.dtype)[None]
    state=o.state(u_e=h.new_zeros(2,layers),h=h,temperature=t,salinity=s,tracer=h[...,None]*0+1)
    return o,state


def test_equal_volume_exchange_matches_independent_work_formula():
    o,state=setup();physical=o.check_state(state);h=state.h
    t=physical.temperature();s=physical.salinity();c=.3
    t1=t+c*(t[:,1:]-t[:,:1])*h.new_tensor([[1.,-1.]])/h
    s1=s+c*(s[:,1:]-s[:,:1])*h.new_tensor([[1.,-1.]])/h
    other=o.state(u_e=state.u_e,h=h,temperature=t1,salinity=s1,tracer=state.tracer_h/h[...,None])
    actual=o.energy.base.stratification(o.check_state(other))[0]-o.energy.base.stratification(physical)[0]
    e=o.energy.base;z=e.height(physical)
    density_contrast=e.rho0*(-e.alpha*(t[:,1]-t[:,0])+e.beta*(s[:,1]-s[:,0]))
    expected=e.gravity*c*o.mesh.cell_area*density_contrast*(z[:,0]-z[:,1])
    torch.testing.assert_close(actual,expected.sum(),rtol=2e-10,atol=1e-10)
    assert actual>0
    # Doubling exchange doubles exact cost (not just a small-increment limit).
    other2=o.state(u_e=state.u_e,h=h,temperature=2*t1-t,salinity=2*s1-s,tracer=state.tracer_h/h[...,None])
    double=o.energy.base.stratification(o.check_state(other2))[0]-o.energy.base.stratification(physical)[0]
    torch.testing.assert_close(double,2*actual,rtol=2e-10,atol=1e-10)


def test_affine_hamiltonian_derivatives_independently():
    e=linear.LinearDensityEnergy()
    t=torch.tensor([10.,15.],dtype=torch.float64,requires_grad=True)
    s=torch.tensor([34.,36.],dtype=t.dtype,requires_grad=True)
    z=torch.tensor([-20.,-60.],dtype=t.dtype,requires_grad=True)
    psi,pt,ps,pz=e.parcel_potential(t,s,z)
    actual=torch.autograd.grad(psi.sum(),(t,s,z))
    for x,y in zip(actual,(pt,ps,pz)):torch.testing.assert_close(x,y,rtol=2e-15,atol=1e-15)


@pytest.mark.parametrize('constant_cp',[False,True])
def test_same_coupled_solver_conserves_linear_control(constant_cp):
    o=linear.LinearColumnOcean(harness.ColumnMetric(),OceanColumn(4,total_depth=100.),constant_cp=constant_cp)
    h=torch.full((1,4),25.,dtype=torch.float64)
    state=o.state(u_e=h.new_tensor([[.2,.1,0.,-.1],[0.,0.,0.,0.]]),h=h,temperature=h.new_tensor([[15.,14.,13.,12.]]),salinity=h*0+35,tracer=h[...,None]*0+1)
    m=TKEOcean(o,closure=coupled.CoupledClosure(2.));s=m.state(state,tke=h*0+.001)
    out,r=m.mix_with_report(s,30.)
    mass=(o.energy.mass_area(state)*h).sum()
    assert (out.tke_h>=0).all() and (out.ocean.tracer_h>=0).all()
    assert abs(float((m.budgets(out)['total']-m.budgets(s)['total'])/mass))<1e-8
    assert r.budget_error<=1 and r.fixed_point_error<=1
    torch.testing.assert_close(out.ocean.salt_h.sum(),s.ocean.salt_h.sum(),rtol=2e-15,atol=0)
    with pytest.raises(NotImplementedError):o.step_with_report(state,30.)


def test_exact_interface_work_matches_finite_affine_energy():
    o,state=setup();p=o.check_state(state);h=p.h
    dh=torch.tensor([[200.]],dtype=h.dtype);ds=torch.tensor([[.02]],dtype=h.dtype)
    work=linear.ExactLinearWorkClosure(2.)._buoyancy_work(o,p,p,dh,ds,torch.zeros_like(h))
    e=o.energy.base;z=e.height(p)
    expected=e.rho0*e.gravity*(z[:,0]-z[:,1])*(-e.alpha*dh[:,0]/o.reference.cp+e.beta*ds[:,0])
    torch.testing.assert_close(work.sum(-1),expected,rtol=2e-15,atol=1e-15)
    torch.testing.assert_close(work[:,0],work[:,1],rtol=0,atol=0)
