"""Prescribed freshwater, impulse and full-energy boundary budgets."""
from dataclasses import replace
import pytest
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn,OceanDomain
from vfs.ocean.stratified import (StratifiedOcean,TKEOcean,TKEClosure,
    OceanSurfaceExchange,OceanSurfaceFluxes,OceanSurfaceHeating,OceanBoundaryStress)

@pytest.fixture(scope='module')
def mesh():return VoronoiMesh(n_cells=32,lloyd_iters=2,rotation_scheme='compatible')

def case(mesh,coast=False,layers=4):
    depth=40.+10*mesh.cell_xyz[:,2:3]
    column=OceanColumn(layers,total_depth=depth,rest_thickness=torch.full((layers,),40./layers))
    wet=mesh.cell_xyz[:,0]>0 if coast else torch.ones(mesh.nCells,dtype=torch.bool)
    domain=OceanDomain(mesh,wet,depth=depth)
    o=StratifiedOcean(mesh,column,domain=domain)
    h=(depth/layers).expand(mesh.nCells,layers).clone()
    u=domain.apply_edge(torch.full((mesh.nEdges,layers),.2,dtype=h.dtype))
    s=o.state(u_e=u,h=h,temperature=torch.full_like(h,12.),salinity=torch.full_like(h,35.),tracer=torch.ones(*h.shape,2,dtype=h.dtype))
    return o,s,OceanSurfaceExchange()

def field(s,x):return torch.full_like(s.h[:,0],x)
def carriers(s):return dict(water_energy_flux=field(s,0.),water_momentum_flux=torch.zeros_like(s.u_e[:,0]))

def budget(o,s,out,r):
    mass=o.energy.mass_area(s)
    m=(mass*s.h).sum()
    assert float(r.mass_residual.abs().sum()/m)<1e-15
    metric=o.energy.base.mechanical.metric(o.check_state(s))
    scale=(metric[:,0]*s.u_e[:,0].abs()+r.water_impulse.abs()).clamp_min(1)
    assert float((r.momentum_residual.abs()/scale).max())<1e-15
    actual=o.budgets(out)['total']-o.budgets(s)['total']
    assert float((actual-r.external_energy).abs()/m)<1e-8
    assert float(r.column_energy_residual.abs().sum()/m)<1e-8
    assert torch.equal(out.salt_h,s.salt_h) and torch.equal(out.tracer_h,s.tracer_h)
    assert torch.equal(out.entropy_h[:,1:],s.entropy_h[:,1:])
    assert torch.equal(out.u_e[:,1:],s.u_e[:,1:])

@pytest.mark.parametrize('kind',['precipitation','evaporation'])
def test_mass_salt_energy_and_zero_impulse(mesh,kind):
    o,s,op=case(mesh)
    f=OceanSurfaceFluxes(**{kind:field(s,.01)},**carriers(s))
    out,r=op.apply(o,s,100.,f)
    sign=1 if kind=='precipitation' else -1
    torch.testing.assert_close(out.h[:,0],s.h[:,0]+sign/1025.,rtol=0,atol=0)
    assert ((out.salt_h[:,0]/out.h[:,0]-35)*sign<0).all()
    assert ((out.u_e[:,0]-s.u_e[:,0])*sign<0).all()
    budget(o,s,out,r)

@pytest.mark.parametrize('sensible,latent',[(20.,-70.),(-20.,70.),(20.,-20.)])
def test_heat_only_and_no_double_count(mesh,sensible,latent):
    o,s,op=case(mesh)
    out,r=op.apply(o,s,100.,OceanSurfaceFluxes(sensible_heat=field(s,sensible),latent_heat=field(s,latent)))
    expected=100*(o.energy.mass_area(s)[:,0]/1025)*(sensible+latent)
    torch.testing.assert_close(r.external_energy,expected.sum(),rtol=2e-15,atol=0)
    assert torch.equal(out.h,s.h) and torch.equal(out.u_e,s.u_e)
    budget(o,s,out,r)
    if sensible+latent==0:assert torch.equal(out.entropy_h,s.entropy_h)

def test_balanced_water_changes_only_prescribed_heat_and_impulse(mesh):
    o,s,op=case(mesh,layers=1)
    f=OceanSurfaceFluxes(precipitation=field(s,.01),evaporation=field(s,.01),
        sensible_heat=field(s,10.),latent_heat=field(s,-50.),water_energy_flux=field(s,5.),water_momentum_flux=torch.full_like(s.u_e[:,0],.001))
    out,r=op.apply(o,s,100.,f)
    assert torch.equal(out.h,s.h)
    budget(o,s,out,r)
    torch.testing.assert_close(r.external_energy,(100*o.energy.mass_area(s)[:,0]/1025*(-35)).sum(),rtol=2e-15,atol=0)

def test_water_momentum_reference(mesh):
    o,s,op=case(mesh)
    # Native momentum carried away at the old current: unchanged velocity.
    e=field(s,.01)
    flux=-s.u_e[:,0]*s.mesh.cell_to_edge(e[:,None])[:,0]
    out,r=op.apply(o,s,100.,OceanSurfaceFluxes(evaporation=e,water_energy_flux=field(s,-500.),water_momentum_flux=flux))
    torch.testing.assert_close(out.u_e,s.u_e,rtol=2e-15,atol=0)
    budget(o,s,out,r)

def test_coasts_and_nonuniform_flux(mesh):
    o,s,op=case(mesh,coast=True)
    p=field(s,.01)*(1+mesh.cell_xyz[:,1])
    out,r=op.apply(o,s,100.,OceanSurfaceFluxes(precipitation=p,**carriers(s)))
    dry=~o.domain.wet_cell
    for name in ('h','entropy_h','salt_h','tracer_h'):
        assert torch.equal(getattr(out,name)[dry],getattr(s,name)[dry])
    assert out.u_e[~o.domain.active_edge].count_nonzero()==0
    budget(o,s,out,r)

def test_empty_flux_exact_noop(mesh):
    o,s,op=case(mesh)
    out,r=op.apply(o,s,100.,OceanSurfaceFluxes())
    for name in ('u_e','h','entropy_h','salt_h','tracer_h'):
        assert torch.equal(getattr(out,name),getattr(s,name))
    assert r.external_energy==0 and r.energy_residual==0

def test_missing_carriers_invalid_layout_and_unphysical_rates(mesh):
    o,s,op=case(mesh)
    for f in (OceanSurfaceFluxes(precipitation=field(s,.01)),
              OceanSurfaceFluxes(precipitation=field(s,.01),evaporation=field(s,.01)),
              OceanSurfaceFluxes(water_energy_flux=field(s,1.)),
              OceanSurfaceFluxes(precipitation=field(s,-.01),**carriers(s)),
              OceanSurfaceFluxes(sensible_heat=field(s,float('nan'))),
              OceanSurfaceFluxes(sensible_heat=torch.ones(s.h.shape,dtype=s.h.dtype)),
              OceanSurfaceFluxes(sensible_heat=field(s,1.).float())):
        with pytest.raises(ValueError):op.apply(o,s,100.,f)
    with pytest.raises(ValueError):op.apply(o,s,0.,OceanSurfaceFluxes())

@pytest.mark.parametrize('rate',[1e6,20.])
def test_exhaustion_and_salinity_reject_without_mutation(mesh,rate):
    o,s,op=case(mesh)
    saved=s.h.clone()
    with pytest.raises(ValueError):op.apply(o,s,100.,OceanSurfaceFluxes(evaporation=field(s,rate),**carriers(s)))
    assert torch.equal(s.h,saved)

def test_smooth_water_and_heat_derivative(mesh):
    o,s,op=case(mesh)
    def loss(a):
        f=OceanSurfaceFluxes(precipitation=field(s,.01)*a,water_energy_flux=field(s,500.)*a,
            water_momentum_flux=torch.zeros_like(s.u_e[:,0]),latent_heat=field(s,-50.)*a)
        out,_=op.apply(o,s,100.,f)
        return out.entropy_h[:,0].mean()+out.u_e[:,0].square().mean()
    a=torch.tensor(1.,dtype=s.h.dtype,requires_grad=True)
    grad=torch.autograd.grad(loss(a),a)[0]
    eps=1e-3;fd=(loss(a.detach()+eps)-loss(a.detach()-eps))/(2*eps)
    torch.testing.assert_close(grad,fd,rtol=2e-5,atol=1e-7)

def test_combined_composition_and_extensive_tke(mesh):
    o,s,op=case(mesh)
    m=TKEOcean(o,exchange=op,closure=TKEClosure(2.),boundary=OceanBoundaryStress(.0025),heating=OceanSurfaceHeating(.6,(.5,20.)))
    state=m.state(s,tke=torch.full_like(s.h,.001))
    f=OceanSurfaceFluxes(precipitation=field(s,1e-5),water_energy_flux=field(s,.5),water_momentum_flux=torch.zeros_like(s.u_e[:,0]))
    exchanged,r=m.exchange_with_report(state,1.,surface_fluxes=f)
    assert exchanged.tke_h is state.tke_h
    out,r=m.step_with_report(state,1.,surface_fluxes=f,shortwave=field(s,200.))
    assert r.exchange is not None and r.heating is not None and r.boundary is not None
    assert (out.tke_h>=0).all()
    with pytest.raises(ValueError):TKEOcean(o).exchange_with_report(state,1.,surface_fluxes=f)


def test_manufactured_full_energy_and_momentum_target(mesh):
    from vfs.ocean.stratified.energy_columns import column_terms
    o,s,op=case(mesh)
    dt=100.;p=field(s,.01)
    h=s.h.clone();h[:,0]+=dt*p/1025
    physical=o.check_state(s)
    mechanical=o.energy.base.mechanical
    u=s.u_e.clone();momentum=torch.full_like(u[:,0],.002)
    trial=physical.with_tensors(u,h,h*physical.temperature(),s.salt_h,s.tracer_h)
    metric=mechanical.metric(trial)
    u[:,0]=(mechanical.metric(physical)[:,0]*u[:,0]+dt*(mesh.dc_edge*mesh.dv_edge)*momentum)/metric[:,0]
    temperature=physical.temperature().clone();temperature[:,0]+=.01
    target=physical.with_tensors(u,h,h*temperature,s.salt_h,s.tracer_h)
    a,_=column_terms(o.energy,physical);b,_=column_terms(o.energy,target)
    energy_flux=sum(b[key]-a[key] for key in a)/(dt*o.energy.mass_area(s)[:,0]/1025)
    out,r=op.apply(o,s,dt,OceanSurfaceFluxes(precipitation=p,water_energy_flux=energy_flux,water_momentum_flux=momentum))
    torch.testing.assert_close(o.check_state(out).temperature(),temperature,rtol=0,atol=1e-10)
    torch.testing.assert_close(out.u_e,u,rtol=3e-15,atol=0)
    budget(o,s,out,r)
