"""Layer heat budgets, optical references, masks, cooling and derivatives."""
from dataclasses import replace
import pytest
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn,OceanDomain
from vfs.ocean.stratified import StratifiedOcean,TKEOcean,TKEClosure,OceanSurfaceHeating

@pytest.fixture(scope='module')
def mesh():return VoronoiMesh(n_cells=32,lloyd_iters=2,rotation_scheme='compatible')

def case(mesh, *, layers=4, coast=False, varying=False, bottom='export'):
    depth=40.+10*mesh.cell_xyz[:,:1] if varying else torch.full((mesh.nCells,1),40.,dtype=torch.float64)
    column=OceanColumn(layers,total_depth=depth,rest_thickness=torch.full((layers,),40./layers))
    wet=mesh.cell_xyz[:,0]>0 if coast else torch.ones(mesh.nCells,dtype=torch.bool)
    domain=OceanDomain(mesh,wet,depth=depth)
    ocean=StratifiedOcean(mesh,column,domain=domain)
    h=(depth/layers).expand(mesh.nCells,layers).clone()
    state=ocean.state(u_e=torch.zeros(mesh.nEdges,layers,dtype=h.dtype),h=h,
        temperature=torch.full_like(h,12.),salinity=torch.full_like(h,35.),tracer=torch.ones(*h.shape,1,dtype=h.dtype))
    heating=OceanSurfaceHeating(.6,(.5,20.),bottom=bottom)
    return ocean,state,heating

def field(s,x):return torch.full_like(s.h[:,0],x)

def check_energy(o,s,out,r):
    mass=(o.energy.mass_area(s)*s.h).sum()
    actual=o.budgets(out)['total']-o.budgets(s)['total']
    assert float((actual-r.external_heat).abs()/mass)<1e-8
    assert float(r.layer_energy_residual.abs().sum()/mass)<1e-8
    for name in ('u_e','h','salt_h','tracer_h'):
        assert torch.equal(getattr(out,name),getattr(s,name))

@pytest.mark.parametrize('bottom',['export','absorb'])
def test_analytic_layer_absorption(mesh,bottom):
    o,s,heat=case(mesh,bottom=bottom)
    out,r=heat.apply(o,s,3600.,shortwave=field(s,200.))
    z=torch.arange(5,dtype=s.h.dtype)*10
    flux=200*(.6*torch.exp(-z/.5)+.4*torch.exp(-z/20.))
    expected=flux[:-1]-flux[1:]
    if bottom=='absorb':expected[-1]+=flux[-1]
    energy=expected[None,:]*3600*o.energy.mass_area(s)/o.column.rho0
    torch.testing.assert_close(r.layer_heat,energy,rtol=3e-15,atol=0)
    assert (out.entropy_h>s.entropy_h).all()
    assert (r.bottom_exported_energy.sum()>0)==(bottom=='export')
    check_energy(o,s,out,r)

@pytest.mark.parametrize('flux',[-150.,150.])
def test_signed_surface_heat(mesh,flux):
    o,s,heat=case(mesh)
    out,r=heat.apply(o,s,3600.,surface_heat=field(s,flux))
    assert torch.equal(out.entropy_h[:,1:],s.entropy_h[:,1:])
    assert ((out.entropy_h[:,0]-s.entropy_h[:,0])*flux>0).all()
    check_energy(o,s,out,r)

@pytest.mark.parametrize('layers',[1,4,16])
def test_column_absorption_resolution_independent(mesh,layers):
    o,s,heat=case(mesh,layers=layers,varying=True)
    out,r=heat.apply(o,s,100.,shortwave=field(s,200.),surface_heat=field(s,-50.))
    depth=s.h.sum(-1)
    expected=100*(200*(1-.6*torch.exp(-depth/.5)-.4*torch.exp(-depth/20.))-50)*o.energy.mass_area(s)[:,0]/o.column.rho0
    torch.testing.assert_close(r.layer_heat.sum(-1),expected,rtol=3e-15,atol=0)
    check_energy(o,s,out,r)

def test_dry_mask_and_zero_forcing(mesh):
    o,s,heat=case(mesh,coast=True)
    out,r=heat.apply(o,s,100.,shortwave=field(s,200.),surface_heat=field(s,-30.))
    dry=~o.domain.wet_cell
    assert torch.equal(out.entropy_h[dry],s.entropy_h[dry])
    assert r.layer_heat[dry].count_nonzero()==0
    check_energy(o,s,out,r)
    out,r=heat.apply(o,s,100.)
    assert torch.equal(out.entropy_h,s.entropy_h)
    assert r.external_heat==0 and r.energy_residual==0

def test_single_layer_cancellation_and_gradient_at_zero(mesh):
    o,s,heat=case(mesh,layers=1,bottom='absorb')
    out,r=heat.apply(o,s,100.,shortwave=field(s,200.),surface_heat=field(s,-200.))
    torch.testing.assert_close(out.entropy_h,s.entropy_h,rtol=0,atol=1e-11)
    def loss(a):
        out,_=heat.apply(o,s,100.,surface_heat=field(s,1.)*a)
        return out.entropy_h.mean()
    a=torch.tensor(0.,dtype=s.h.dtype,requires_grad=True)
    grad=torch.autograd.grad(loss(a),a)[0]
    eps=.1;fd=(loss(a.detach()+eps)-loss(a.detach()-eps))/(2*eps)
    assert grad>0
    torch.testing.assert_close(grad,fd,rtol=2e-6,atol=1e-9)

def test_smooth_shortwave_derivative(mesh):
    o,s,heat=case(mesh,varying=True)
    def loss(a):
        out,_=heat.apply(o,s,3600.,shortwave=field(s,200.)*a)
        return out.entropy_h[:,0].mean()
    a=torch.tensor(1.,dtype=s.h.dtype,requires_grad=True)
    grad=torch.autograd.grad(loss(a),a)[0]
    eps=1e-3;fd=(loss(a.detach()+eps)-loss(a.detach()-eps))/(2*eps)
    torch.testing.assert_close(grad,fd,rtol=2e-6,atol=1e-8)

@pytest.mark.parametrize('args',[(1.1,(1.,2.),'export'),(.5,(0.,2.),'export'),(.5,(1.,),'export'),(.5,(1.,2.),'other'),(float('nan'),(1.,2.),'export')])
def test_invalid_optics(args):
    with pytest.raises(ValueError):OceanSurfaceHeating(*args)

def test_bad_flux_and_unaffordable_cooling(mesh):
    o,s,heat=case(mesh)
    for q in (field(s,-1.),field(s,float('nan')),field(s,1.).float(),torch.ones(mesh.nCells,1,dtype=s.h.dtype)):
        with pytest.raises(ValueError):heat.apply(o,s,100.,shortwave=q)
    saved=s.entropy_h.clone()
    with pytest.raises((ValueError,FloatingPointError)):
        heat.apply(o,s,1e8,surface_heat=field(s,-1e8))
    assert torch.equal(s.entropy_h,saved)
    with pytest.raises(ValueError):heat.apply(o,s,0.)

def test_tke_composition(mesh):
    o,s,heat=case(mesh)
    m=TKEOcean(o,closure=TKEClosure(2.),heating=heat)
    state=m.state(s,tke=torch.full_like(s.h,.001))
    heated,r=m.heat_with_report(state,10.,shortwave=field(s,200.))
    assert heated.tke_h is state.tke_h
    out,r=m.step_with_report(state,1.,shortwave=field(s,200.),surface_heat=field(s,-50.))
    assert r.heating.external_heat>0 and r.turbulence is not None
    assert (out.tke_h>=0).all()
    plain=TKEOcean(o)
    with pytest.raises(ValueError):plain.heat_with_report(state,1.,shortwave=field(s,0.))
