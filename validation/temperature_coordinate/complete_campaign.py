"""Compare complete compiled steps; explicit physics, budgets and failures."""
import argparse,json,sys,time,statistics
from pathlib import Path
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ocean_column'))
from resolution_mesh import setup
from lagged import SubcycledLaggedClosure
from physics import TemperatureMixing
from path import TemperaturePathMixing
from system import TemperatureSystem
from surface import FastSurfaceExchange,FastSurfaceHeating,TemperatureBoundary
from vfs.ocean.stratified import TKEOcean,TKEState,OceanBoundaryStress,OceanSurfaceHeating,OceanSurfaceExchange,OceanSurfaceFluxes,AdaptiveThermalRemapper
from vfs.ocean.stratified import equations
from fast_thermo import Caloric
from collections import Counter


def make(cells,layers,compiled):
    o,e,k=setup(cells,layers);o.warm_start=True;o.remapper=AdaptiveThermalRemapper()
    if compiled:o.compile_thermodynamics()
    s=o.check_state(e);cell=lambda x:torch.full_like(s.h[:,0],x)
    f=dict(surface_stress=.05*torch.ones_like(s.u_e[:,0]),shortwave=cell(200),surface_heat=cell(-50),
        surface_fluxes=OceanSurfaceFluxes(precipitation=cell(1e-5),evaporation=cell(5e-6),sensible_heat=cell(-10),latent_heat=cell(-12.5),water_energy_flux=cell(.25),water_momentum_flux=torch.zeros_like(s.u_e[:,0])))
    return o,e,s,k,f


def models(o):
    shared=dict(boundary=OceanBoundaryStress(.0025),heating=OceanSurfaceHeating(.6,(.5,20.)),exchange=OceanSurfaceExchange())
    yield 'entropy',TKEOcean(o,closure=SubcycledLaggedClosure(2.,substeps=4,fast_thermodynamics=True),**shared),False
    shared.update(heating=FastSurfaceHeating(.6,(.5,20.)),exchange=FastSurfaceExchange())
    yield 'fast_surface_entropy',TKEOcean(o,closure=SubcycledLaggedClosure(2.,substeps=4,fast_thermodynamics=True),**shared),False
    shared.update(boundary=TemperatureBoundary(.0025))
    for name,closure in [('temperature_h_path',TemperatureMixing(2.,substeps=4)),('temperature_ts3',TemperaturePathMixing(2.,substeps=4)),('temperature_ts1',TemperaturePathMixing(2.,substeps=4,quadrature_order=1))]:
        yield name,TemperatureSystem(o,closure=closure,**shared),True


def main():
    p=argparse.ArgumentParser();p.add_argument('--cells',type=int,default=128);p.add_argument('--layers',type=int,default=16)
    p.add_argument('--compile',action='store_true');p.add_argument('--ale',action='store_true');p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--variants',nargs='+');p.add_argument('--steps',type=int,default=1);p.add_argument('--dt',type=float,default=60.);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    o,e,s,k,f=make(args.cells,args.layers,args.compile)
    results=dict(config={**vars(args),'output':str(args.output)},torch=torch.__version__,threads=1,cases={})
    initial_mass=(o.energy.mass_area(s)*s.h).sum();reference=None
    with torch.no_grad():
        for name,model,native in models(o):
            if args.variants and name not in args.variants:continue
            initial=TKEState(s if native else e,s.h*k)
            energy=lambda state:model.total_energy(state) if native else model.budgets(state)['total']
            before=energy(initial)
            def advance(state):
                out,r=model.step_with_report(state,args.dt,**f)
                external=r.boundary.surface_work+r.heating.external_heat+r.exchange.external_energy
                if args.ale:
                    target=out.ocean.h.sum(-1,keepdim=True).expand_as(out.ocean.h)/args.layers
                    out,remap=model.remap_with_report(out,target)
                return out,external,r
            try:
                advance(initial);advance(initial) # include first-use guard changes
                times=[]
                for _ in range(args.repeats):
                    start=time.perf_counter();advance(initial);times.append(time.perf_counter()-start)
                count=Counter();orig=Caloric.inverse
                def counted(self,target,guess,kind):count[kind]+=1;return orig(self,target,guess,kind)
                with patch.object(Caloric,'inverse',counted),patch.object(equations,'temperature_from_entropy',wraps=equations.temperature_from_entropy) as inverse:
                    advance(initial)
                out=initial;external=0.;max_local=0.;rows=[]
                for step in range(args.steps):
                    out,added,r=advance(out);external+=added;max_local=max(max_local,float(r.turbulence.budget_error))
                    if step==args.steps-1 or step%max(1,args.steps//10)==0:
                        rows.append(dict(step=step+1,energy_minus_inputs_J_kg=float((energy(out)-before-external)/initial_mass),min_tke=float((out.tke_h/out.ocean.h).min())))
                physical=out.ocean if native else o.check_state(out.ocean)
                row=dict(status='completed',median_step_s=statistics.median(times),samples_s=times,
                    inverse_calls=dict(count),state_inverse_calls=inverse.call_count,max_local_budget_ratio=max_local,
                    min_salt=float(physical.salinity().min()),min_tracer=float(physical.tracer_h.min()),trajectory=rows)
                if reference is None:reference=(physical,out.tke_h)
                row['versus_entropy']=dict(temperature_rms_C=float((physical.temperature()-reference[0].temperature()).square().mean().sqrt()),
                    velocity_rms_m_s=float((physical.u_e-reference[0].u_e).square().mean().sqrt()),tke_max_abs=float((out.tke_h-reference[1]).abs().max()))
            except (ValueError,FloatingPointError) as error:row=dict(status='rejected',error=str(error))
            results['cases'][name]=row;args.output.write_text(json.dumps(results,indent=2)+'\n');print(name,row,flush=True)

if __name__=='__main__':main()
