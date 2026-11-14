"""Runnable linear-ocean example and compact budget/timing campaign.

Requires an installed VFS; no data downloads. Timing is CPU float64, one thread.
The optional JMD95 comparison changes physics, not just implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ocean_column'))
from complete_campaign import make
from linear import LinearOcean,LinearMixing,LinearHeating,LinearExchange,TemperatureBoundary
from system import TemperatureSystem
from path import TemperaturePathMixing
from surface import FastSurfaceHeating,FastSurfaceExchange
from vfs.ocean.stratified import TKEState


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cells',type=int,default=128);p.add_argument('--layers',type=int,default=16)
    p.add_argument('--dt',type=float,default=60.);p.add_argument('--steps',type=int,default=10)
    p.add_argument('--mixing-substeps',type=int,default=4);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--compare-jmd95',action='store_true');p.add_argument('--ale',action='store_true')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if min(a.cells,a.layers,a.steps,a.repeats,a.mixing_substeps)<=0 or a.dt<=0:p.error('positive sizes, counts and dt required')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    o,e,s,k,f=make(a.cells,a.layers,a.compare_jmd95)
    linear=LinearOcean(o.mesh,o.column,closure=LinearMixing(2.,substeps=a.mixing_substeps),boundary=TemperatureBoundary(.0025),heating=LinearHeating(.6,(.5,20.)),exchange=LinearExchange())
    candidates=[('linear',linear)]
    if a.compare_jmd95:
        candidates.insert(0,('compiled_jmd95',TemperatureSystem(o,closure=TemperaturePathMixing(2.,substeps=a.mixing_substeps),boundary=TemperatureBoundary(.0025),heating=FastSurfaceHeating(.6,(.5,20.)),exchange=FastSurfaceExchange())))
    results=dict(config={**vars(a),'output':str(a.output)},torch=torch.__version__,dtype='float64',threads=1,geometry='Fibonacci Voronoi, zero Lloyd iterations, compatible rotation',source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('linear.py','linear_campaign.py')},cases={})
    with torch.no_grad():
        for name,model in candidates:
            initial=TKEState(s,s.h*k);area=model.energy.mass_area(s);mass=(area*s.h).sum();energy=model.total_energy(initial)
            def advance(state):
                end,r=model.step_with_report(state,a.dt,**f)
                external=r.boundary.surface_work+r.heating.external_heat+r.exchange.external_energy
                if a.ale:end,_=model.remap_with_report(end,end.ocean.h.sum(-1,keepdim=True).expand_as(end.ocean.h)/a.layers)
                return end,external,r
            end=initial;external=0.;completed=0;ratio=0.;samples=[];failure=None
            try:
                advance(initial);advance(initial)
                for _ in range(a.repeats):
                    start=time.perf_counter();advance(initial);samples.append(time.perf_counter()-start)
                for _ in range(a.steps):
                    end,added,r=advance(end);external+=added;completed+=1
                    ratio=max(ratio,float(r.turbulence.budget_error))
            except (ValueError,FloatingPointError) as exc:failure=str(exc)
            mass_added=(a.dt*completed*(f['surface_fluxes'].precipitation-f['surface_fluxes'].evaporation)*s.mesh.cell_area).sum()
            row=dict(status='rejected' if failure else 'completed',failure=failure,completed_steps=completed,simulated_seconds=completed*a.dt,samples_s=samples,
                median_step_s=statistics.median(samples) if samples else None,
                energy_minus_inputs_J_kg=float((model.total_energy(end)-energy-external)/mass),
                mass_minus_inputs_relative=float(((area*(end.ocean.h-s.h)).sum()-mass_added)/mass),
                salt_relative_change=float((area*(end.ocean.salt_h-s.salt_h)).sum()/(area*s.salt_h).sum()),
                passive_relative_change=float((area[...,None]*(end.ocean.tracer_h-s.tracer_h)).sum()/(area[...,None]*s.tracer_h).sum()),
                min_tke=float((end.tke_h/end.ocean.h).min()),min_salt=float(end.ocean.salinity().min()),min_tracer=float(end.ocean.tracer_h.min()),max_local_budget_ratio=ratio)
            results['cases'][name]=row;a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(results,indent=2)+'\n');print(name,json.dumps(row),flush=True)
    return int(any(r['status']!='completed' for r in results['cases'].values()))

if __name__=='__main__':raise SystemExit(main())
