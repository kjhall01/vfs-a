"""Compare entropy and potential-enthalpy ALE on identical saline wave probes."""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import torch
import importlib.util
import sys
spec=importlib.util.spec_from_file_location('enthalpy_ale_parent',Path(__file__).with_name('probe_ocean_entropy_ale.py'))
p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)
e=p.load('enthalpy_ale_candidate','ocean_enthalpy_ale.py')


def audits():
    rows=[]
    for kind in ['front','tilted','near_fresh','upper_salt']:
        for coast in [False,True]:
            model,state,energy=p.p.q.p.case(kind,coast)
            target=state.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5])
            for method in ['pcm','plm']:
                for name,remapper in [('entropy',p.a.EntropyRemapper(method=method)),('enthalpy',e.EnthalpyRemapper(method=method))]:
                    end,r=remapper.remap_with_report(model,state,target,energy)
                    rows.append(dict(kind=kind,coast=coast,method=method,scheme=name,
                        energy_J_m2={k:float(v/model.mesh.cell_area.sum()) for k,v in r.energy_change.items()},
                        entropy_change_J_K=float(energy.entropy(end)-energy.entropy(state)),
                        min_column_entropy_change=float((energy.mass_area(state)*(end.entropy_h-state.entropy_h)).sum(-1).min()),
                        enthalpy_residual_max=None if name=='entropy' else float(r.potential_enthalpy_residual.abs().max()),
                        salt_residual_max=float(r.content_residuals['salt_h'].abs().max()),changed_columns=r.changed_columns))
    return rows


def wave(n,scheme,periods):
    model,physical,rest,mode,period,max_dt=p.v.wave_case(n)
    reference=p.p.q.p.previous.reference();state=p.p.front.EntropyState.from_temperature(physical,reference)
    rest=p.p.front.EntropyState.from_temperature(rest,reference);start=state
    energy=p.p.front.EntropyEnergy(model.dynamics.energy,reference)
    area=model.mesh.cell_area[:,None];mass=energy.mass_area(state);surface=area.sum()
    def projection(s):
        # Salt content is conserved by BOTH remaps, unlike entropy or T.
        anomaly=(s.salt_h-rest.salt_h).sum(-1,keepdim=True)
        return float((area*mode*anomaly).sum()/(area*mode.square()).sum())
    amplitude=projection(start);initial=energy.terms(start)
    steps=math.ceil(periods*period/max_dt);dt=periods*period/steps
    remapper=e.EnthalpyRemapper() if scheme=='enthalpy' else p.a.EntropyRemapper()
    target=p.v.ReferenceGrid();changes={k:0. for k in initial};events=0;production=0.;remap_entropy=0.;samples=[]
    max_salt_residual=0.;max_enthalpy_residual=0.
    for k in range(steps+1):
        if k%max(1,steps//32)==0 or k==steps:
            samples.append(dict(time_over_period=k*dt/period,projection_over_initial=projection(state)/amplitude))
        if k==steps:break
        state,ledgers=p.p.front.step(model,state,dt,energy)
        production+=dt*sum(w*float((mass*l.entropy_source).sum()) for w,l in zip((1/6,1/6,2/3),ledgers))
        if scheme!='none':
            before=state;state,r=remapper.remap_with_report(model,state,target(state),energy)
            events+=int(r.changed_columns>0)
            remap_entropy+=float((mass*(state.entropy_h-before.entropy_h)).sum())
            for key,value in r.energy_change.items():changes[key]+=float(value/surface)
            max_salt_residual=max(max_salt_residual,float(r.content_residuals['salt_h'].abs().max()))
            if scheme=='enthalpy':max_enthalpy_residual=max(max_enthalpy_residual,float(r.potential_enthalpy_residual.abs().max()))
    x=torch.tensor([s['time_over_period'] for s in samples]);y=torch.tensor([s['projection_over_initial'] for s in samples])
    return dict(layers=n,scheme=scheme,periods=periods,steps=steps,dt_s=dt,actual_remap_events=events,
        salinity_projection_rms_error=float((y-torch.cos(2*torch.pi*x)).square().mean().sqrt()),
        remap_energy_J_m2=changes,total_energy_change_J_m2=float((energy.terms(state)['total']-initial['total'])/surface),
        horizontal_entropy_production_J_K=production,remap_entropy_change_J_K=remap_entropy,
        entropy_budget_relative_residual=(float((mass*(state.entropy_h-start.entropy_h)).sum())-production-remap_entropy)/float((mass*start.entropy_h.abs()).sum()),
        max_salt_column_residual=max_salt_residual,max_enthalpy_column_residual=max_enthalpy_residual,samples=samples)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--layers',type=int,default=8);parser.add_argument('--periods',type=float,default=.05)
    parser.add_argument('--scheme',choices=['none','entropy','enthalpy'],default='enthalpy');parser.add_argument('--audit',action='store_true')
    args=parser.parse_args();torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    report=dict(status='running',device='cpu',dtype='float64',torch=str(torch.__version__),source_sha256={Path(f).name:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [__file__,e.__file__]})
    try:
        with torch.no_grad():report['audits' if args.audit else 'wave']=audits() if args.audit else wave(args.layers,args.scheme,args.periods)
        report['status']='completed'
    except Exception as error:
        report.update(status='rejected',error=str(error));raise
    finally:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
