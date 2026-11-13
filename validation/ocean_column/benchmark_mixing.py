"""Interleaved CPU closure benchmark on fixed input states; not a throughput forecast.

The optional baseline source should be exported from the parent git commit.
It is loaded under the installed package for relative imports, never installed.
All four methods use identical inputs. Initialization and one warmup per method
are excluded. Timings include the public mix_with_report validation and reports.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean,TKEOcean,TKEClosure
from budget_predictor import BudgetPredictor
from run import ColumnMetric


def compare_exact(a,b):
    if isinstance(a,torch.Tensor):return int(torch.equal(a,b)),1
    if isinstance(a,dict):
        pairs=[compare_exact(a[k],b[k]) for k in a]
        return sum(x for x,n in pairs),sum(n for x,n in pairs)
    return int(a==b),1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-module',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=5)
    args=p.parse_args()
    if args.repeats<1:raise ValueError('repeats must be positive')
    name='vfs.ocean.stratified._benchmark_baseline'
    spec=importlib.util.spec_from_file_location(name,args.baseline_module)
    old=importlib.util.module_from_spec(spec);sys.modules[name]=old;spec.loader.exec_module(old)
    # Preserve the old implementation's method resolution while satisfying the
    # public facade's nominal closure type check. No methods are monkeypatched.
    class ParentClosure(old.TKEClosure,TKEClosure):
        pass
    torch.set_num_threads(1)
    rows=[]
    with torch.no_grad():
        for cells in (1,128):
            mesh=ColumnMetric() if cells==1 else VoronoiMesh(n_cells=cells,lloyd_iters=0,rotation_scheme='compatible')
            ocean=StratifiedOcean(mesh,OceanColumn(32,total_depth=100.))
            h=torch.full((cells,32),100/32,dtype=torch.float64)
            z=h.cumsum(-1)-h/2;upper=.5*(1-torch.tanh((z-20)/5))
            u=torch.linspace(.2,0.,32,dtype=h.dtype).expand(mesh.nEdges,32).clone()
            initial=ocean.state(u_e=u,h=h,temperature=10+5*upper,salinity=35-.5*upper,tracer=upper[...,None])
            closures={'parent':ParentClosure(2.),'cached':TKEClosure(2.),
                      'predictor_half':BudgetPredictor(2.,margin=.5),'predictor_95':BudgetPredictor(2.,margin=.95)}
            models={k:TKEOcean(ocean,closure=c) for k,c in closures.items()}
            states={k:m.state(initial,tke=torch.full_like(h,1e-4)) for k,m in models.items()}
            def evaluate(k):return models[k].mix_with_report(states[k],150.)
            warm={k:evaluate(k) for k in models}
            original,report=warm['parent'];cached,cached_report=warm['cached']
            tensor_names=('u_e','h','salt_h','entropy_h','tracer_h')
            equal,total=compare_exact({k:getattr(original.ocean,k) for k in tensor_names},
                                     {k:getattr(cached.ocean,k) for k in tensor_names})
            eq,n=compare_exact(original.tke_h,cached.tke_h);equal+=eq;total+=n
            eq,n=compare_exact(asdict(report),asdict(cached_report));equal+=eq;total+=n
            samples={k:[] for k in models};names=list(models)
            for repeat in range(args.repeats):
                for k in names[repeat%len(names):]+names[:repeat%len(names)]:
                    start=time.perf_counter();evaluate(k);samples[k].append(time.perf_counter()-start)
            rows.append(dict(cells=cells,layers=32,dt=150,exact_matches=equal,comparisons=total,
                timing={k:dict(samples_s=v,median_s=statistics.median(v)) for k,v in samples.items()}))
    args.output.write_text(json.dumps(dict(torch=torch.__version__,dtype='float64',threads=1,
        baseline_sha256=hashlib.sha256(args.baseline_module.read_bytes()).hexdigest(),rows=rows),indent=2)+'\n')
    print(args.output.read_text())

if __name__=='__main__':main()
