"""Independent checks for the explicit CT/SA polynomial configuration."""
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'validation/temperature_coordinate'),str(Path(__file__).resolve().parents[2]/'validation/ocean_column')]
from roquet import RoquetFreezingEOS,RoquetOcean,RoquetMixing,RoquetHeating,RoquetExchange,RoquetWorkspace,convert_jmd95_initialization,TemperatureBoundary
from vfs.ocean.stratified import TKEState
from complete_campaign import make
from test_jmd95_hamiltonian import mesh,case


@pytest.mark.parametrize('coast',[False,True])
def test_energy_derivative_and_rhs(mesh,coast):
    old,s=case(mesh,coast);m=RoquetOcean(mesh,old.column,domain=old.domain)
    keys=('u_e','h','temp_h','salt_h');leaves=[getattr(s,k).clone().requires_grad_() for k in keys]
    s=s.with_tensors(*leaves,s.tracer_h)
    expected=torch.autograd.grad(m.total_energy(TKEState(s,s.h*0)),leaves)
    actual=m.dynamics.gradients(s)
    for key,g in zip(keys,expected):torch.testing.assert_close(actual[key],g,rtol=3e-12,atol=.1)
    rhs,_=m.dynamics.tendency_with_report(s,1.)
    work=[g*getattr(rhs,key) for key,g in zip(keys,expected)]
    assert abs(sum(v.sum() for v in work))<4096*torch.finfo(s.h.dtype).eps*sum(v.abs().sum() for v in work)


@pytest.fixture
def setup():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    old,e,s,k,f=make(16,4,False)
    m=RoquetOcean(old.mesh,old.column,closure=RoquetMixing(2.,substeps=2),boundary=TemperatureBoundary(.0025),heating=RoquetHeating(.6,(.5,20.)),exchange=RoquetExchange())
    return m,TKEState(s,s.h*k),f


def test_published_polynomial_sign_and_partials():
    # Exact decimal evaluation of published Freezing terms, positive depth D.
    from decimal import Decimal as D
    c=RoquetFreezingEOS();t=torch.tensor([-2.,10.,30.],dtype=torch.float64,requires_grad=True)
    s=torch.tensor([34.,35.,37.],dtype=torch.float64,requires_grad=True);z=torch.tensor([0.,-1000.,-5000.],dtype=torch.float64,requires_grad=True)
    expected=[float(D('.7718')*D(str(sa))-D('.0491')*D(str(ct))-D('.005027')*D(str(ct))**2-D('.000025681')*D(str(ct))*D(str(depth))) for ct,sa,depth in zip(t.detach().tolist(),s.detach().tolist(),(-z).detach().tolist())]
    value=c.polynomial(t,s,z);torch.testing.assert_close(value,t.new_tensor(expected),rtol=2e-15,atol=1e-14)
    gt,gs,gz=torch.autograd.grad(value.sum(),(t,s,z))
    torch.testing.assert_close(gt,c.t+2*c.tt*t-c.tz*z)
    torch.testing.assert_close(gs,s*0+c.s);torch.testing.assert_close(gz,-c.tz*t)
    assert gt[2]<gt[1]<gt[0]<0
    # Mixing equal parts warm/cold at equal SA makes denser water (cabbeling).
    warm=t.new_tensor(20.);cold=t.new_tensor(0.);sa=t.new_tensor(35.);zero=t.new_tensor(0.)
    assert c.polynomial((warm+cold)/2,sa,zero)>.5*(c.polynomial(warm,sa,zero)+c.polynomial(cold,sa,zero))


def test_layer_integral_against_independent_gauss(setup):
    import numpy as np
    m,state,f=setup;s=state.ocean;b=m.ocean.energy.base
    nodes,weights=np.polynomial.legendre.leggauss(12);z=b.height(s)
    mean=sum(float(w/2)*b.parcel_potential(s.temperature(),s.salinity(),z+float(n/2)*s.h)[0] for n,w in zip(nodes,weights))
    exact=RoquetWorkspace(m.ocean,s).parts(s)[0]
    torch.testing.assert_close(exact,mean,rtol=2e-13,atol=1e-13)


def test_finite_mixing_includes_two_interface_cross_terms(setup):
    m,state,_=setup;s=state.ocean;w=RoquetWorkspace(m.ocean,s);h=s.h
    # Opposite signs force both faces to contribute to middle-layer delta T.
    fh=torch.tensor([2e5,-3e5,1e5]).expand(h.shape[0],3);fs=torch.tensor([.2,-.3,.1]).expand_as(fh)
    zero=torch.zeros_like(h[:,:1]);div=lambda f:torch.cat((f,zero),-1)-torch.cat((zero,f),-1)
    dt=div(fh)/(h*m.ocean.reference.cp_theta);ds=div(fs)/h
    after=s.with_tensors(s.u_e,h,s.temp_h+h*dt,s.salt_h+h*ds,s.tracer_h)
    work=w.path_work(m.closure,s,None,fh,fs,None,after=after)
    mass=m.energy.mass_area(s)*h
    exact=mass*((2*w.a*s.temperature()+w.b)*dt+w.a*dt.square()+w.c*ds)
    torch.testing.assert_close(work.sum(),exact.sum(),rtol=3e-13,atol=1.)
    measured=(mass*(w.parts(after)[0]-w.parts(s)[0])).sum()
    torch.testing.assert_close(work.sum(),measured,rtol=1e-11,atol=1.)


@pytest.mark.parametrize('heat',[100.,-100.])
def test_quadratic_heating(setup,heat):
    m,state,_=setup;s=state.ocean;w=RoquetWorkspace(m.ocean,s)
    q=torch.full_like(s.h,heat);after,residual,_=w.heat(s,s,q,allow_cooling=True)
    mass=m.energy.mass_area(s)*s.h
    delta=m.total_energy(TKEState(after,state.tke_h))-m.total_energy(state)
    torch.testing.assert_close(delta,(mass*q).sum(),rtol=2e-12,atol=0)
    assert residual.abs().max()<1e-12


@pytest.mark.parametrize('seed',[0.,1e-12,1e-4])
@pytest.mark.parametrize('profile',[[-1.5,-1.,0.,1.],[1.,0.,-1.,-1.5],[30.,20.,10.,0.]])
def test_mixing_cold_unstable_and_strong_contrast(setup,seed,profile):
    m,state,_=setup;s=state.ocean;t=s.h.new_tensor(profile)
    s=s.with_tensors(s.u_e,s.h,s.h*t,s.h*35.,s.tracer_h);state=TKEState(s,s.h*seed)
    out,k,r=m.closure.apply(m.ocean,s,state.tke_h,30.)
    mass=(m.energy.mass_area(s)*s.h).sum()
    assert abs((m.total_energy(TKEState(out,k))-m.total_energy(state))/mass)<1e-8
    assert r.budget_error<=1 and (k>=0).all()


def test_complete_path_and_backward(setup):
    m,state,f=setup;s=state.ocean;q=s.temp_h.clone().requires_grad_();state=TKEState(s.with_tensors(s.u_e,s.h,q,s.salt_h,s.tracer_h),state.tke_h)
    with patch('vfs.ocean.jmd95_energy.jmd95_partials',side_effect=AssertionError('JMD95')),patch('fast_thermo.Caloric.inverse',side_effect=AssertionError('iterative inverse')):
        end,r=m.step_with_report(state,1.,**f)
        final,rr=m.remap_with_report(end,end.ocean.h.sum(-1,keepdim=True).expand_as(end.ocean.h)/4)
    external=r.boundary.surface_work+r.heating.external_heat+r.exchange.external_energy
    mass=(m.energy.mass_area(s)*s.h).sum()
    assert abs((m.total_energy(final)-m.total_energy(state)-external)/mass)<1e-8
    assert torch.isfinite(torch.autograd.grad(final.ocean.temp_h.sum(),q)[0]).all()


def test_explicit_initialization_conversion(setup):
    gsw=pytest.importorskip('gsw');m,state,_=setup;s=state.ocean
    kwargs=dict(potential_temperature_ipts68=s.temperature(),practical_salinity=s.salinity(),pressure_dbar=s.h*0+100.,longitude=s.h*0-30.,latitude=s.h*0+45.)
    converted=convert_jmd95_initialization(**kwargs)
    sa=gsw.SA_from_SP(s.salinity().numpy(),100.,-30.,45.);ct=gsw.CT_from_pt(sa,s.temperature().numpy()/1.00024)
    torch.testing.assert_close(converted['absolute_salinity'],torch.tensor(sa));torch.testing.assert_close(converted['conservative_temperature'],torch.tensor(ct))
    out=m.state(u_e=s.u_e,h=s.h,**converted)
    assert out.ocean.h.shape==s.h.shape
    with pytest.raises(TypeError):m.state(u_e=s.u_e,h=s.h,temperature=s.temperature(),salinity=s.salinity())
    with pytest.raises(ValueError):convert_jmd95_initialization(**{**kwargs,'potential_temperature_ipts68':s.temperature().requires_grad_()})
