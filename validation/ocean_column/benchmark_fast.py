"""Interleaved baseline/fast mixing on identical inputs, installed-wheel only."""
import argparse,json,time,statistics,hashlib
from pathlib import Path
import torch
from resolution_mesh import setup,difference
from lagged import SubcycledLaggedClosure
from vfs.ocean.stratified import TKEOcean


def main():
    p=argparse.ArgumentParser();p.add_argument('--cells',type=int,default=1024);p.add_argument('--layers',type=int,default=64);p.add_argument('--repeats',type=int,default=3);p.add_argument('--patch',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    result=dict(cells=a.cells,layers=a.layers,patch=a.patch,dt=300,substeps=4,torch=torch.__version__,threads=1,source_hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('lagged.py','fast_thermo.py','benchmark_fast.py')},samples=[],warmup_excluded=True)
    def save():a.output.write_text(json.dumps(result,indent=2)+'\n')
    with torch.no_grad():
        ocean,s,k=setup(a.cells,a.layers,a.patch)
        models={name:TKEOcean(ocean,closure=SubcycledLaggedClosure(2.,fast_thermodynamics=flag)) for name,flag in [('baseline',False),('fast',True)]}
        states={n:m.state(s,tke=k) for n,m in models.items()}
        outputs={}
        for name in models:outputs[name]=models[name].mix_with_report(states[name],300.)
        for i in range(a.repeats):
            for name in (['baseline','fast'] if i%2==0 else ['fast','baseline']):
                start=time.perf_counter();cpu=time.process_time();out,report=models[name].mix_with_report(states[name],300.)
                row=dict(method=name,repeat=i,cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-start)
                result['samples'].append(row);outputs[name]=(out,report);save();print(row,flush=True)
        result['median_cpu_s']={n:statistics.median(x['cpu_s'] for x in result['samples'] if x['method']==n) for n in models}
        result['speedup']=result['median_cpu_s']['baseline']/result['median_cpu_s']['fast']
        result['differences']=difference(ocean,outputs['fast'][0],outputs['baseline'][0])
        result['checks']={}
        area=ocean.energy.mass_area(s);mass=area*s.h
        for n,(out,r) in outputs.items():
            result['checks'][n]=dict(energy_error_J_kg=float((models[n].budgets(out)['total']-models[n].budgets(states[n])['total'])/mass.sum()),local_budget_ratio=float(r.budget_error),min_tke=float(out.specific_tke().min()),salt_relative_change=float((area*(out.ocean.salt_h-s.salt_h)).sum()/(area*s.salt_h).sum()),passive_relative_change=float((area[...,None]*(out.ocean.tracer_h-s.tracer_h)).sum()/(area[...,None]*s.tracer_h).sum()),limiter={k:float(v) for k,v in r.mixing_limiter.items()})
        save();print('speedup',result['speedup'],'difference',result['differences'],flush=True)

if __name__=='__main__':main()
