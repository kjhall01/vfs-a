"""Vertical-resolution and actual-remapping measurements; no accuracy gates."""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import torch
import numpy as np
import importlib.util
import sys


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

p=load('ale_front_probe','probe_ocean_front.py')
a=load('entropy_ale_candidate','ocean_entropy_ale.py')
v=load('legacy_ale_probe','verify_ocean_ale.py')


def wave(n,interval,periods):
    model,physical,rest,mode,period,max_dt=v.wave_case(n)
    reference=p.q.p.previous.reference()
    state=p.front.EntropyState.from_temperature(physical,reference)
    rest=p.front.EntropyState.from_temperature(rest,reference)
    start=state;energy=p.front.EntropyEnergy(model.dynamics.energy,reference)
    area=model.mesh.cell_area[:,None];mass=energy.mass_area(state)
    def projection(s):
        # Entropy column content is invariant under this remap, unlike T.
        anomaly=(s.entropy_h-rest.entropy_h).sum(-1,keepdim=True)
        return float((area*mode*anomaly).sum()/(area*mode.square()).sum())
    amplitude=projection(start);initial_energy=float(energy.terms(start)['total'])
    perturbation=abs(initial_energy-float(energy.terms(rest)['total']))
    steps=math.ceil(periods*period/max_dt);dt=periods*period/steps
    remapper=a.EntropyRemapper();target=v.ReferenceGrid()
    changes={k:0. for k in energy.terms(start)};columns=0;events=0;production=0.;samples=[]
    max_residual={k:0. for k in ['h','entropy_h','salt_h']}
    for k in range(steps+1):
        if k%max(1,steps//32)==0 or k==steps:
            samples.append(dict(time_over_period=k*dt/period,projection_over_initial=projection(state)/amplitude))
        if k==steps:break
        state,ledgers=p.front.step(model,state,dt,energy)
        production+=dt*sum(w*float((mass*l.entropy_source).sum()) for w,l in zip((1/6,1/6,2/3),ledgers))
        if interval and (k+1)%interval==0:
            before=state
            state,report=remapper.remap_with_report(model,state,target(state),energy)
            columns+=report.changed_columns;events+=int(report.changed_columns>0)
            for key,value in report.energy_change.items():changes[key]+=float(value)
            for key in max_residual:
                scale=getattr(before,key).abs().sum(-1).clamp_min(1)
                max_residual[key]=max(max_residual[key],float((report.content_residuals[key].abs()/scale).max()))
        if (k+1)%500==0:print(f'layers={n} interval={interval} step={k+1}/{steps}',flush=True)
    x=np.array([s['time_over_period'] for s in samples]);y=np.array([s['projection_over_initial'] for s in samples])
    fit=np.linalg.lstsq(np.stack([np.cos(2*np.pi*x),np.sin(2*np.pi*x),np.ones_like(x)],-1),y,rcond=None)[0]
    final=state.physical_state(reference)
    delta=float(energy.terms(state)['total'])-initial_energy
    return dict(layers=n,remap_every=interval,periods=periods,steps=steps,dt_s=dt,period_s=period,
        projection_rms_error=float(np.sqrt(np.mean((y-np.cos(2*np.pi*x))**2))),
        fitted_amplitude=float(np.hypot(fit[0],fit[1])),fitted_phase_rad=float(np.arctan2(-fit[1],fit[0])),
        initial_perturbation_energy_J=perturbation,total_energy_change_J=delta,
        remap_energy_change_J=changes,remaining_dynamics_change_J=delta-changes['total'],
        remap_energy_over_initial_perturbation=changes['total']/perturbation,
        changed_column_events=columns,actual_remap_events=events,max_column_relative_residual=max_residual,
        entropy_produced_J_K=production,entropy_budget_residual_J_K=float(energy.entropy(state)-energy.entropy(start))-production,
        salt_relative_change=float((mass*(state.salt_h-start.salt_h)).sum()/(mass*start.salt_h).sum()),
        min_thickness_m=float(state.h.min()),temperature_range=[float(final.temperature().min()),float(final.temperature().max())],samples=samples)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers',type=int,required=True);parser.add_argument('--interval',type=int,default=1)
    parser.add_argument('--periods',type=float,default=1.);parser.add_argument('--legacy',action='store_true')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    report=dict(status='running',scheme='temperature-legacy' if args.legacy else 'entropy-front',
        torch=str(torch.__version__),device='cpu',dtype='float64',source_sha256={Path(f).name:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [__file__,a.__file__,p.front.__file__,v.__file__]})
    try:
        with torch.no_grad():
            report['wave']=v.internal_wave(args.layers,args.interval,args.periods) if args.legacy else wave(args.layers,args.interval,args.periods)
        report['status']='completed'
    except Exception as error:
        report.update(status='rejected',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
