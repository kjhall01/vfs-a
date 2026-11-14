"""Independent energy/work and complete-path checks for the linear configuration."""
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'validation/temperature_coordinate'),str(Path(__file__).resolve().parents[2]/'validation/ocean_column')]
from linear import LinearEOS,LinearEnergy,LinearCaloric,LinearWorkspace,LinearOcean,LinearMixing,LinearHeating,LinearExchange,TemperatureBoundary
from complete_campaign import make
from vfs.ocean.stratified import TKEState
from vfs.ocean.stratified.faces import gibbs_face,temperature_from_entropy
from test_jmd95_hamiltonian import mesh,case


@pytest.mark.parametrize('coast',[False,True])
def test_linear_derivatives_and_actual_rhs_work(mesh,coast):
    old,s=case(mesh,coast);model=LinearOcean(mesh,old.column,domain=old.domain)
    keys=('u_e','h','temp_h','salt_h')
    leaves=[getattr(s,k).clone().requires_grad_() for k in keys]
    s=s.with_tensors(*leaves,s.tracer_h)
    expected=torch.autograd.grad(model.total_energy(TKEState(s,s.h*0)),leaves)
    analytic=model.dynamics.gradients(s)
    for key,value in zip(keys,expected):torch.testing.assert_close(analytic[key],value,rtol=3e-12,atol=.1)
    rhs,_=model.dynamics.tendency_with_report(s,1.)
    work=[g*getattr(rhs,key) for key,g in zip(keys,expected)]
    assert abs(sum(x.sum() for x in work)) < 4096*torch.finfo(s.h.dtype).eps*sum(x.abs().sum() for x in work)


def test_caloric_means_and_inverse():
    torch.set_default_dtype(torch.float64);r=LinearCaloric(LinearEOS())
    a=torch.tensor([-2.,0.,10.,10.,39.]);b=torch.tensor([39.,0.,10.+1e-9,15.,-2.])
    salt=a*0+35
    eta,s,report=gibbs_face(r,salt,a,salt+1,b)
    reverse=gibbs_face(r,salt+1,b,salt,a)[0]
    torch.testing.assert_close(eta,reverse,rtol=0,atol=1e-12)
    assert report['identity'].abs().max()<2e-9
    for t in (a,b):
        v=r.evaluate(salt,t)
        torch.testing.assert_close(temperature_from_entropy(r,salt,v['entropy']),t,rtol=1e-13,atol=1e-13)
        torch.testing.assert_close(r.temperature_from_enthalpy(salt,v['h0']),t,rtol=1e-13,atol=1e-13)
    # Independent integration of entropy along the temperature path.
    import numpy as np
    n,w=np.polynomial.legendre.leggauss(24)
    reference=sum(float(weight/2)*r.evaluate(salt,a+float((node+1)/2)*(b-a))['entropy'] for node,weight in zip(n,w))
    torch.testing.assert_close(eta,reference,rtol=1e-12,atol=1e-10)


@pytest.fixture
def setup():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    old,e,s,k,f=make(16,4,False)
    model=LinearOcean(old.mesh,old.column,closure=LinearMixing(2.,substeps=2),boundary=TemperatureBoundary(.0025),heating=LinearHeating(.6,(.5,20.)),exchange=LinearExchange())
    return model,TKEState(s,s.h*k),f


def test_exact_finite_mixing_work(setup):
    model,state,_=setup;o=model.ocean;s=state.ocean;w=LinearWorkspace(o,s)
    h=s.h;fh=torch.full_like(h[:,:-1],100.);fs=torch.full_like(fh,.001)
    zero=torch.zeros_like(h[:,:1]);div=lambda f:torch.cat((f,zero),-1)-torch.cat((zero,f),-1)
    final=s.with_tensors(s.u_e,h,s.temp_h+div(fh)/o.reference.cp_theta,s.salt_h+div(fs),s.tracer_h)
    work=w.path_work(model.closure,s,None,fh,fs,w.conjugates(None,None),after=final)
    mass=o.energy.mass_area(s)
    measured=(mass*h*(w.parts(final)[0]-w.parts(s)[0])).sum()
    torch.testing.assert_close(work.sum(),measured,rtol=1e-9,atol=1.)
    # Linear work scales exactly with the proposed finite flux, independent of quadrature.
    half=w.path_work(model.closure,s,None,fh/2,fs/2,w.conjugates(None,None),after=final)
    torch.testing.assert_close(half,work/2,rtol=0,atol=0)


@pytest.mark.parametrize('seed',[0.,1e-12,1e-4])
@pytest.mark.parametrize('unstable',[False,True])
def test_mixing_budget_and_positivity(setup,seed,unstable):
    model,state,_=setup;s=state.ocean
    t=s.h.new_tensor([15.,14.,13.,12.]);t=t.flip(0) if unstable else t
    s=s.with_tensors(s.u_e,s.h,s.h*t,s.salt_h,s.tracer_h);state=TKEState(s,s.h*seed)
    before=model.total_energy(state);end,k,r=model.closure.apply(model.ocean,s,state.tke_h,30.)
    mass=(model.energy.mass_area(s)*s.h).sum()
    assert abs((model.total_energy(TKEState(end,k))-before)/mass)<1e-8
    assert r.budget_error<=1 and (k>=0).all() and (end.tracer_h>=0).all()


def test_complete_forced_remap_no_nonlinear_eos(setup):
    model,state,f=setup;s=state.ocean;before=model.total_energy(state)
    with patch('vfs.ocean.jmd95_energy.jmd95_partials',side_effect=AssertionError('JMD95')),patch('vfs.ocean.stratified.thermodynamics.SurfaceGibbs.evaluate',side_effect=AssertionError('Gibbs')),patch('fast_thermo.Caloric.inverse',side_effect=AssertionError('iterative inverse')):
        end,r=model.step_with_report(state,10.,**f)
        target=end.ocean.h.sum(-1,keepdim=True).expand_as(end.ocean.h)/4
        final,remap=model.remap_with_report(end,target)
    external=r.boundary.surface_work+r.heating.external_heat+r.exchange.external_energy
    mass=(model.energy.mass_area(s)*s.h).sum()
    assert abs((model.total_energy(final)-before-external)/mass)<1e-8
    area=model.energy.mass_area(s)
    torch.testing.assert_close((area*final.ocean.salt_h).sum(),(area*s.salt_h).sum(),rtol=3e-15,atol=0)
    torch.testing.assert_close((area[...,None]*final.ocean.tracer_h).sum(),(area[...,None]*s.tracer_h).sum(),rtol=3e-15,atol=0)
    assert final.tke_h.min()>=0


def test_backward_and_configuration_guards(setup):
    model,state,f=setup;s=state.ocean
    q=s.temp_h.clone().requires_grad_();state=TKEState(s.with_tensors(s.u_e,s.h,q,s.salt_h,s.tracer_h),state.tke_h)
    end,_=model.step_with_report(state,1.,**f)
    assert torch.isfinite(torch.autograd.grad(end.ocean.temp_h.sum(),q)[0]).all()
    with pytest.raises(ValueError):LinearEOS(cp=-1)
    from path import TemperaturePathMixing
    with pytest.raises(TypeError):LinearOcean(s.mesh,s.column,closure=TemperaturePathMixing(2.))


def test_deep_column_heat_and_custom_parameters(setup):
    from vfs.ocean import OceanColumn
    old,state,_=setup
    model=LinearOcean(old.ocean.mesh,OceanColumn(4,total_depth=6000.),config=LinearEOS(alpha=2e-4,beta=8e-4,cp=4000.))
    s=state.ocean;h=torch.full_like(s.h,1500.)
    s=model.state(u_e=s.u_e*0,h=h,temperature=h*0+10.,salinity=h*0+35.).ocean
    w=LinearWorkspace(model.ocean,s)
    heat=torch.full_like(h,100.)
    end,residual,_=w.heat(s,s,heat)
    area=model.energy.mass_area(s)
    measured=model.total_energy(TKEState(end,h*0))-model.total_energy(TKEState(s,h*0))
    torch.testing.assert_close(measured,(area*h*heat).sum(),rtol=3e-12,atol=0)
    assert residual.abs().max()<1e-12
    assert (w.capacity>model.ocean.reference.cp_theta).all()
    torch.testing.assert_close(model.ocean.energy.base.density(s),h*0+model.ocean.column.rho0)
    # Uniform resting state on the sphere has no acceleration or transport.
    rhs,_=model.dynamics.tendency_with_report(s,1.)
    for name in ('u_e','h','temp_h','salt_h'):assert getattr(rhs,name).abs().max()<1e-12
