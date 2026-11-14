"""Explicit dtype/device benchmark with float64 inventory/energy auditing.

No tolerance changes or clipping. Failed cases have no throughput claim.
Build geometry before model caches; prognostic states remain separate objects.
"""
import argparse,copy,json,statistics,time,traceback
from dataclasses import fields
from pathlib import Path
import torch
from complete_campaign import make
from model import TemperatureOcean
from system import TemperatureSystem
from path import TemperaturePathMixing
from surface import FastSurfaceHeating,FastSurfaceExchange,TemperatureBoundary
from vfs.ocean import OceanState
from vfs.ocean.stratified import StratifiedOcean,TKEState,OceanSurfaceFluxes


def bind(state,mesh,column,dtype,device):
    cast=lambda x:None if x is None else x.to(device=device,dtype=dtype)
    return OceanState(*(cast(getattr(state,n)) for n in ('u_e','h','temp_h','salt_h','tracer_h')),mesh=mesh,column=column)


def construct(source,dtype,compiled,device):
    old,_,physical,k,forcing=source
    mesh=copy.deepcopy(old.mesh).to(device=device,dtype=dtype)
    column=copy.deepcopy(old.column).to(device=device,dtype=dtype)
    ocean=StratifiedOcean(mesh,column,warm_start=True)
    if compiled:ocean.compile_thermodynamics()
    state=TKEState(bind(physical,mesh,column,dtype,device),(physical.h*k).to(device=device,dtype=dtype))
    flux={}
    for name,value in forcing.items():
        if isinstance(value,OceanSurfaceFluxes):
            value=OceanSurfaceFluxes(**{f.name:getattr(value,f.name).to(device=device,dtype=dtype) for f in fields(value)})
        else:value=value.to(device=device,dtype=dtype)
        flux[name]=value
    model=TemperatureSystem(ocean,closure=TemperaturePathMixing(2.,substeps=4),boundary=TemperatureBoundary(.0025),
        heating=FastSurfaceHeating(.6,(.5,20.)),exchange=FastSurfaceExchange())
    # Audit the SAME rounded geometry, promoted before arithmetic. Keep the
    # independent evaluator eager float64, even for a compiled candidate.
    om=copy.deepcopy(mesh).to(device='cpu',dtype=torch.float64)
    oc=copy.deepcopy(column).to(device='cpu',dtype=torch.float64)
    oracle=TemperatureSystem(StratifiedOcean(om,oc,warm_start=True))
    return model,state,flux,oracle


def audit(oracle,state):
    s=bind(state.ocean,oracle.ocean.mesh,oracle.ocean.column,torch.float64,'cpu')
    k=state.tke_h.to(device='cpu',dtype=torch.float64)
    area=oracle.ocean.mesh.cell_area[:,None];mass=oracle.energy.mass_area(s)
    terms=oracle.energy.terms(s)
    return dict(energy=float(terms['total']+(mass*k).sum()),mass=float((mass*s.h).sum()),
        salt=float((mass*s.salt_h).sum()),tracer=float((mass[...,None]*s.tracer_h).sum()),
        min_tke=float((k/s.h).min()),min_salt=float(s.salinity().min()),min_tracer=float(s.tracers().min()),
        temperature_min=float(s.temperature().min()),temperature_max=float(s.temperature().max()))


def advance(model,state,forcing,case,dt,phase=0):
    external=water=0.;local=None
    if case=='dynamics':
        end=model.unpack(model.dynamics.step(model.pack(state),dt),state)
    elif case=='mixing':
        s,k,r=model.closure.apply(model.ocean,state.ocean,state.tke_h,dt)
        end=TKEState(s,k);local=float(r.budget_error)
    elif case=='ale':
        weights=torch.ones_like(state.ocean.h[0]) if phase%2==0 else torch.linspace(.5,1.5,state.ocean.h.shape[-1],dtype=state.ocean.h.dtype,device=state.ocean.h.device)
        target=state.ocean.h.sum(-1,keepdim=True)*(weights/weights.sum())
        end,_=model.remap_with_report(state,target)
    else:
        end,r=model.step_with_report(state,dt,**forcing);local=None if r.turbulence is None else float(r.turbulence.budget_error)
        # Re-reduce layer/edge ledgers in float64. They are still calculated in
        # candidate precision; this is not an independent stress-work formula.
        total=lambda x:float(x.double().sum())
        external=total(r.boundary.surface_work_edge)+total(r.heating.surface_energy)+total(r.heating.shortwave_energy)-total(r.heating.bottom_exported_energy)
        external+=sum(total(x) for x in (r.exchange.sensible_energy,r.exchange.latent_energy,r.exchange.water_energy))
        # Independent prescribed freshwater input, not rounded state difference.
        f=forcing['surface_fluxes'];area=model.ocean.mesh.cell_area.double()
        water=float((area*(f.precipitation.double()-f.evaporation.double())*dt).sum())
        if case=='full_ale':
            target=end.ocean.h.sum(-1,keepdim=True).expand_as(end.ocean.h)/end.ocean.h.shape[-1]
            end,_=model.remap_with_report(end,target)
    return end,external,water,local


def sync(device):
    if device=='cuda':torch.cuda.synchronize()
    elif device=='mps':torch.mps.synchronize()


def main():
    p=argparse.ArgumentParser();p.add_argument('--cells',type=int,default=128);p.add_argument('--layers',type=int,default=16)
    p.add_argument('--dtypes',nargs='+',default=['float64','float32']);p.add_argument('--cases',nargs='+',default=['dynamics','mixing','ale','forcing','full','full_ale'])
    p.add_argument('--compile',action='store_true');p.add_argument('--steps',type=int,default=10);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--dt',type=float,default=60.);p.add_argument('--device',default='cpu',choices=['cpu','cuda','mps']);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    source=make(a.cells,a.layers,False)
    result=dict(config={**vars(a),'output':str(a.output)},torch=torch.__version__,threads=1,autograd=False,cases={})
    def save():a.output.write_text(json.dumps(result,indent=2)+'\n')
    with torch.no_grad():
        for name in a.dtypes:
            dtype=getattr(torch,name);torch.set_default_dtype(dtype)
            for case in a.cases:
                key=name+'-'+case;row=dict(status='starting',completed_steps=0,samples_s=[],trajectory=[]);result['cases'][key]=row
                try:
                    model,initial,forcing,oracle=construct(source,dtype,a.compile,a.device)
                    if case=='forcing':model.closure=None
                    row['state_dtype']=str(initial.ocean.h.dtype);row['mesh_float_dtypes']=sorted(set(str(x.dtype) for x in model.ocean.mesh.buffers() if x.is_floating_point()))
                    start=audit(oracle,initial);row['initial']=start
                    native_start=model.total_energy(initial)
                    # Audit each component on one step before timing it.
                    end,external,water,local=advance(model,initial,forcing,case,a.dt)
                    row['first_step_local_budget_ratio']=local
                    for _ in range(1):advance(model,initial,forcing,case,a.dt)
                    for _ in range(a.repeats):
                        sync(a.device);t=time.perf_counter();advance(model,initial,forcing,case,a.dt);sync(a.device)
                        row['samples_s'].append(time.perf_counter()-t)
                    row['median_step_s']=statistics.median(row['samples_s']);row['sypd']=a.dt/(365*row['median_step_s'])
                    end=initial;external=water=0.
                    for step in range(a.steps):
                        end,e,w,local=advance(model,end,forcing,case,a.dt,phase=step);external+=e;water+=w
                        now=audit(oracle,end);row['completed_steps']=step+1
                        row['trajectory'].append(dict(step=step+1,energy_residual_J_kg=(now['energy']-start['energy']-external)/start['mass'],
                            native_energy_residual_J_kg=(float(model.total_energy(end)-native_start)-external)/start['mass'],
                            external_energy_J_kg=external/start['mass'],mass_residual_relative=(now['mass']-start['mass']-water)/start['mass'],
                            salt_residual_relative=(now['salt']-start['salt'])/start['salt'],tracer_residual_relative=(now['tracer']-start['tracer'])/start['tracer'],
                            min_tke=now['min_tke'],min_salt=now['min_salt'],min_tracer=now['min_tracer'],local_budget_ratio=local))
                    row['status']='completed'
                except Exception as exc:
                    tb=exc.__traceback__
                    while tb:
                        loc=tb.tb_frame.f_locals
                        if all(n in loc for n in ('actual_residual','mass','budget_error','budget_scale')):
                            row['failure_local_budget']=dict(normalized_ratio=float(loc['budget_error']),
                                max_abs_residual_J_kg=float((loc['actual_residual']/loc['mass']).abs().max()),
                                max_tolerance_J_kg=float((loc['budget_scale']/loc['mass']).max()))
                        tb=tb.tb_next
                    row['status']='rejected';row['error']=str(exc);row['error_type']=type(exc).__name__;row['traceback']=traceback.format_exc()
                save();print(key,{k:v for k,v in row.items() if k not in ('trajectory','traceback','initial')},flush=True)
    save()

if __name__=='__main__':main()
