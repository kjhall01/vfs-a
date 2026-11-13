"""Independent finite-energy and affine oracles for research path work."""
from test_linear_column import load, linear, harness
import torch
import pytest
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean,TKEOcean
path=load('path_work')

def exchange(linear_model=False,scale=1.):
    cls=linear.LinearColumnOcean if linear_model else StratifiedOcean
    o=cls(harness.ColumnMetric(),OceanColumn(3,total_depth=100.))
    h=torch.tensor([[20.,30.,50.]],dtype=torch.float64)
    t=h.new_tensor([[18.,12.,5.]]);s=h.new_tensor([[34.,35.,36.]])
    state=o.state(u_e=h.new_zeros(2,3),h=h,temperature=t,salinity=s)
    p=o.check_state(state);fh=h.new_tensor([[-20000.,-10000.]])*scale;fs=h.new_tensor([[.7,.4]])*scale
    zero=h[:,:1]*0
    div=lambda f:torch.cat((f,zero),-1)-torch.cat((zero,f),-1)
    ref=o.reference.evaluate(p.salinity(),p.temperature())
    s1=p.salinity()+div(fs)/h
    t1=o.reference.temperature_from_enthalpy(s1,ref['h0']+div(fh)/h)
    after=o.check_state(o.state(u_e=state.u_e,h=h,temperature=t1,salinity=s1))
    return o,p,after,fh,fs,div

@pytest.mark.parametrize('order',[1,3,5,9])
def test_affine_limit_unequal_layers(order):
    o,p,a,fh,fs,_=exchange(True)
    got=path.PathWorkClosure(2.,quadrature_order=order)._buoyancy_work(o,p,a,fh,fs,p.h*0)
    expected=linear.ExactLinearWorkClosure(2.)._buoyancy_work(o,p,a,fh,fs,p.h*0)
    torch.testing.assert_close(got,expected,rtol=3e-13,atol=1e-11)

def test_jmd95_layer_work_matches_independent_endpoint_energy():
    o,p,a,fh,fs,div=exchange();e=o.energy.base
    nodes,w=e._quadrature(p.h);z=e.height(p)[...,None]+p.h[...,None]*(nodes-.5)
    psi=lambda state:(e.parcel_potential(state.temperature()[...,None],state.salinity()[...,None],z)[0]*w).sum(-1)
    am,bm=path.conjugates(o,(p.temperature()+a.temperature())/2,(p.salinity()+a.salinity())/2,z,w)
    alpha=p.h[:,:-1]/(p.h[:,:-1]+p.h[:,1:]);face=lambda q:(1-alpha)*q[:,:-1]+alpha*q[:,1:]
    expected=o.energy.mass_area(p)*(p.h*(psi(a)-psi(p))-div(face(am)*fh+face(bm)*fs))
    got=path.PathWorkClosure(2.,quadrature_order=5)._buoyancy_work(o,p,a,fh,fs,p.h*0)
    torch.testing.assert_close(got,expected,rtol=2e-8,atol=2e-7)
    higher=path.PathWorkClosure(2.,quadrature_order=9)._buoyancy_work(o,p,a,fh,fs,p.h*0)
    torch.testing.assert_close(got,higher,rtol=1e-11,atol=1e-10)

def test_zero_and_tiny_exchange():
    o,p,a,fh,fs,_=exchange(scale=1e-12);c=path.PathWorkClosure(2.)
    tiny=c._buoyancy_work(o,p,a,fh,fs,p.h*0)
    assert torch.isfinite(tiny).all() and (tiny.abs()>0).all()
    zero=c._buoyancy_work(o,p,p,fh*0,fs*0,p.h*0)
    assert torch.equal(zero,p.h*0)
    twice=c._buoyancy_work(o,p,a,2*fh,2*fs,p.h*0)
    torch.testing.assert_close(twice,2*tiny,rtol=1e-10,atol=0)

def test_coupled_path_full_energy_and_positive_tke():
    o,p,_,_,_,_=exchange()
    state=o.state(u_e=p.u_e,h=p.h,temperature=p.temperature(),salinity=p.salinity())
    m=TKEOcean(o,closure=path.PathWorkClosure(2.));s=m.state(state,tke=p.h*0+.001)
    out,r=m.mix_with_report(s,30.)
    assert (out.tke_h>=0).all()
    assert r.budget_error<=1 and r.fixed_point_error<=1
    mass=(o.energy.mass_area(state)*p.h).sum()
    assert abs(float((m.budgets(out)['total']-m.budgets(s)['total'])/mass))<1e-8
    torch.testing.assert_close(out.ocean.salt_h.sum(),state.salt_h.sum(),rtol=2e-15,atol=0)
