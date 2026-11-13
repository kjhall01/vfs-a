"""Interleaved frozen-input timing: legacy versus refreshed JMD95 acceptance and acceleration."""
import argparse,json,time,statistics
from pathlib import Path
import torch
from run import ColumnMetric
from path_work import PathWorkClosure
from consistent import ConsistentPathClosure
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean,TKEOcean,OceanBoundaryStress


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);models={};states={};samples={};cpu_samples={};iterations={};trials={}
    with torch.no_grad():
        for name in ('legacy','refresh','budget'):
            mesh=ColumnMetric();column=OceanColumn(16,total_depth=100.)
            o=StratifiedOcean(mesh,column)
            h=torch.full((1,16),6.25,dtype=torch.float64);z=h.cumsum(-1)-h/2
            upper=.5*(1-torch.tanh((z-20)/5))
            closure=PathWorkClosure(2.) if name=='legacy' else ConsistentPathClosure(2.,budget_acceleration=name=='budget')
            m=TKEOcean(o,closure=closure,boundary=OceanBoundaryStress())
            s=m.state(o.state(u_e=h.new_zeros(2,16),h=h,temperature=10+5*upper,salinity=35-.5*upper),tke=h*0+1e-4)
            s,_=m.boundary_with_report(s,150.,surface_stress=h.new_tensor([.1,0.]))
            models[name]=m;states[name]=s;samples[name]=[];cpu_samples[name]=[]
            _,r=m.mix_with_report(s,150.);iterations[name]=r.iterations;trials[name]=float(r.mixing_limiter.get("trial_evaluations",r.iterations))
        names=list(models)
        for rep in range(3):
            for name in names[rep:]+names[:rep]:
                start=time.perf_counter();cpu_start=time.process_time();_,r=models[name].mix_with_report(states[name],150.)
                samples[name].append(time.perf_counter()-start);cpu_samples[name].append(time.process_time()-cpu_start)
        result={name:dict(samples_s=v,median_s=statistics.median(v),iterations=iterations[name],trial_evaluations=trials[name],cpu_samples_s=cpu_samples[name],median_cpu_s=statistics.median(cpu_samples[name])) for name,v in samples.items()}
    a.output.write_text(json.dumps(dict(torch=torch.__version__,threads=1,layers=16,dt_s=150,results=result),indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
