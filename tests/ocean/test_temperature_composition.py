"""Opt-in fast surfaces, T/S-path work, and full temperature-state composition."""
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'validation/temperature_coordinate'),str(Path(__file__).resolve().parents[2]/'validation/ocean_column')]
from resolution_mesh import setup
from vfs.ocean.stratified import OceanSurfaceHeating,OceanSurfaceExchange,OceanSurfaceFluxes,TKEOcean,OceanBoundaryStress
from vfs.ocean.stratified.tke_model import TKEState
from model import TemperatureOcean
from surface import FastSurfaceHeating,FastSurfaceExchange,TemperatureBoundary,TemperatureFacade
from path import TemperaturePathMixing
from physics import TemperatureMixing
from system import TemperatureSystem
from fast_thermo import Caloric

@pytest.fixture
def inputs():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    o,e,k=setup(16,4);s=o.check_state(e)
    cell=lambda x:torch.full_like(s.h[:,0],x)
    flux=OceanSurfaceFluxes(precipitation=cell(1e-5),evaporation=cell(5e-6),
        sensible_heat=cell(-10),latent_heat=cell(-12.5),water_energy_flux=cell(.25),
        water_momentum_flux=torch.zeros_like(s.u_e[:,0]))
    forcing=dict(surface_stress=torch.full_like(s.u_e[:,0],.05),shortwave=cell(200),surface_heat=cell(-50),surface_fluxes=flux)
    return o,e,s,k,forcing


def test_fast_surface_entropy_and_temperature_match(inputs):
    o,e,s,k,f=inputs
    native=TemperatureFacade(o)
    for old,new,kwargs in [(OceanSurfaceHeating(.6,(.5,20.)),FastSurfaceHeating(.6,(.5,20.)),dict(shortwave=f['shortwave'],surface_heat=f['surface_heat'])),
                           (OceanSurfaceExchange(),FastSurfaceExchange(),dict(fluxes=f['surface_fluxes']))]:
        a,ra=old.apply(o,e,60.,**kwargs)
        b,rb=new.apply(o,e,60.,**kwargs)
        with patch('vfs.ocean.stratified.equations.temperature_from_entropy',side_effect=AssertionError('inverse')):
            c,rc=new.apply(native,s,60.,_physical=s,**kwargs)
        for p in (o.check_state(b),c):
            torch.testing.assert_close(p.temperature(),o.check_state(a).temperature(),rtol=0,atol=2e-10)
        mass=(o.energy.mass_area(s)*s.h).sum()
        for r in (rb,rc):assert abs(r.energy_residual/mass)<1e-8


@pytest.mark.parametrize('seed',[0.,1e-12,1e-4])
@pytest.mark.parametrize('unstable',[False,True])
def test_ts_path_energy_and_no_path_inversions(inputs,seed,unstable):
    o,e,s,k,f=inputs
    t=s.h.new_tensor([15.,14.,13.,12.]);t=t.flip(0) if unstable else t
    s=s.with_tensors(s.u_e,s.h,s.h*t,s.salt_h,s.tracer_h)
    calls=[];original=Caloric.inverse
    def counted(self,target,guess,kind):calls.append(kind);return original(self,target,guess,kind)
    with patch.object(Caloric,'inverse',counted):
        end,kh,r=TemperaturePathMixing(2.,substeps=2).apply(o,s,s.h*seed,30.)
    assert len(calls)==int(r.mixing_limiter['semi_trials']) and set(calls)=={'h0'}
    assert (kh>=0).all() and r.budget_error<=1
    mass=o.energy.mass_area(s)*s.h
    assert (r.column_energy_residual.abs()/mass.sum(-1)).max()<1e-8


def test_complete_step_and_remap_without_entropy_inverse(inputs):
    o,e,s,k,f=inputs
    model=TemperatureSystem(o,closure=TemperaturePathMixing(2.,substeps=2),boundary=TemperatureBoundary(.0025),
                            heating=FastSurfaceHeating(.6,(.5,20.)),exchange=FastSurfaceExchange())
    initial=TKEState(s,s.h*k);before=model.total_energy(initial)
    with patch('vfs.ocean.stratified.equations.temperature_from_entropy',side_effect=AssertionError('inverse')):
        end,r=model.step_with_report(initial,10.,**f)
        target=end.ocean.h.sum(-1,keepdim=True).expand_as(end.ocean.h)/4
        final,remap=model.remap_with_report(end,target)
    external=r.boundary.surface_work+r.heating.external_heat+r.exchange.external_energy
    mass=(o.energy.mass_area(s)*s.h).sum()
    assert abs((model.total_energy(final)-before-external)/mass)<1e-7
    assert (final.tke_h>=0).all() and (final.ocean.tracer_h>=0).all()
    torch.testing.assert_close((o.energy.mass_area(s)*final.ocean.salt_h).sum(),(o.energy.mass_area(s)*s.salt_h).sum(),rtol=3e-15,atol=0)
    identity,report=model.remap_with_report(final,final.ocean.h)
    assert torch.equal(identity.ocean.temp_h,final.ocean.temp_h)


def test_native_remap_matches_supported(inputs):
    from vfs.ocean.stratified import AdaptiveThermalRemapper
    o,e,s,k,f=inputs;o.remapper=AdaptiveThermalRemapper()
    native=TemperatureSystem(o);old=TKEOcean(o)
    target=s.h.sum(-1,keepdim=True).expand_as(s.h)/4
    a,ar=old.remap_with_report(TKEState(e,s.h*k),target)
    b,br=native.remap_with_report(TKEState(s,s.h*k),target)
    torch.testing.assert_close(b.ocean.temperature(),o.check_state(a.ocean).temperature(),rtol=0,atol=2e-9)
    torch.testing.assert_close(b.tke_h,a.tke_h,rtol=0,atol=0)


def test_guarded_path_retry_is_explicit(inputs):
    from path import GuardedTemperaturePathMixing,TemperaturePath
    o,e,s,k,f=inputs;original=TemperaturePath.path_work;calls=[]
    def fail_once(self,closure,*args,**kwargs):
        calls.append(closure.quadrature_order)
        if len(calls)==1:raise FloatingPointError('coupled full-H closure failed')
        return original(self,closure,*args,**kwargs)
    with patch.object(TemperaturePath,'path_work',fail_once):
        end,kh,r=GuardedTemperaturePathMixing(2.,substeps=1).apply(o,s,s.h*k,10.)
    assert calls[0]==1 and all(n==3 for n in calls[1:])
    assert r.mixing_limiter['path_refinements']==1 and r.budget_error<=1


def test_complete_step_backward_smoke(inputs):
    o,e,s,k,f=inputs
    model=TemperatureSystem(o,closure=TemperaturePathMixing(2.,substeps=1),boundary=TemperatureBoundary(.0025),
        heating=FastSurfaceHeating(.6,(.5,20.)),exchange=FastSurfaceExchange())
    alpha=torch.tensor(0.,requires_grad=True)
    f=dict(f,surface_heat=f['surface_heat']+alpha)
    end,_=model.step_with_report(TKEState(s,s.h*k),1.,**f)
    target=end.ocean.h.sum(-1,keepdim=True).expand_as(end.ocean.h)/4
    end,_=model.remap_with_report(end,target)
    gradient=torch.autograd.grad(end.ocean.temp_h.sum(),alpha)[0]
    assert torch.isfinite(gradient) and gradient>0


def test_native_coastal_remap_and_domain_rejection():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    from test_jmd95_hamiltonian import case
    from vfs.geometry.mesh import VoronoiMesh
    from vfs.ocean.stratified import StratifiedOcean
    mesh=VoronoiMesh(n_cells=32,lloyd_iters=1,verbose=False)
    old,s=case(mesh,True);o=StratifiedOcean(mesh,old.column,domain=old.domain)
    model=TemperatureSystem(o);state=TKEState(s,torch.where(old.domain.wet_cell[:,None],s.h*1e-4,torch.zeros_like(s.h)))
    target=s.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5])[None]
    end,r=model.remap_with_report(state,target)
    torch.testing.assert_close(end.ocean.temp_h[~old.domain.wet_cell],s.temp_h[~old.domain.wet_cell],rtol=0,atol=0)
    with pytest.raises(ValueError):model.remap_with_report(state,target*1.1)


def test_deep_mixing_path_matches_energy():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    from vfs.geometry.mesh import VoronoiMesh
    from vfs.ocean import OceanColumn
    from vfs.ocean.stratified import StratifiedOcean
    mesh=VoronoiMesh(n_cells=16,lloyd_iters=0,rotation_scheme='compatible',verbose=False)
    o=StratifiedOcean(mesh,OceanColumn(32,total_depth=6000.))
    h=torch.full((16,32),187.5,dtype=torch.float64)
    t=torch.linspace(18.,2.,32,dtype=h.dtype)[None].expand_as(h)
    s=o.base.state(u_e=torch.zeros(mesh.nEdges,32,dtype=h.dtype),h=h,temperature=t,salinity=h*0+35,tracer=h[...,None]*0+1)
    k=h*1e-4;model=TemperatureSystem(o)
    end,kh,r=TemperaturePathMixing(2.,substeps=1).apply(o,s,k,60.)
    mass=(o.energy.mass_area(s)*h).sum()
    change=model.total_energy(TKEState(end,kh))-model.total_energy(TKEState(s,k))
    assert abs(change/mass)<1e-8 and r.budget_error<=1 and (kh>=0).all()
