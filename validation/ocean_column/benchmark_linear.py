"""Interleaved fixed-input CPU timing; same solver, three EOS/caloric choices."""
import argparse,json,time,statistics
from pathlib import Path
import torch
from run import ColumnMetric
from coupled import CoupledClosure
from linear_eos import LinearColumnOcean
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean,TKEOcean,OceanBoundaryStress


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);models={};states={};samples={};iterations={}
    with torch.no_grad():
        for name in ('jmd95','linear-gibbs','linear-cp'):
            mesh=ColumnMetric();column=OceanColumn(16,total_depth=100.)
            o=StratifiedOcean(mesh,column) if name=='jmd95' else LinearColumnOcean(mesh,column,constant_cp=name=='linear-cp')
            h=torch.full((1,16),6.25,dtype=torch.float64);z=h.cumsum(-1)-h/2
            upper=.5*(1-torch.tanh((z-20)/5))
            m=TKEOcean(o,closure=CoupledClosure(2.,accelerate=False,relaxation=.3,max_iterations=240),boundary=OceanBoundaryStress())
            s=m.state(o.state(u_e=h.new_zeros(2,16),h=h,temperature=10+5*upper,salinity=35-.5*upper),tke=h*0+1e-4)
            s,_=m.boundary_with_report(s,150.,surface_stress=h.new_tensor([.1,0.]))
            models[name]=m;states[name]=s;samples[name]=[]
            _,r=m.mix_with_report(s,150.);iterations[name]=r.iterations
        names=list(models)
        for rep in range(3):
            for name in names[rep:]+names[:rep]:
                start=time.perf_counter();_,r=models[name].mix_with_report(states[name],150.)
                samples[name].append(time.perf_counter()-start)
        result={name:dict(samples_s=v,median_s=statistics.median(v),iterations=iterations[name]) for name,v in samples.items()}
    a.output.write_text(json.dumps(dict(torch=torch.__version__,threads=1,layers=16,dt_s=150,results=result),indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
