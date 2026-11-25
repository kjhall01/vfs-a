"""Compare unchanged finite-work and path-work research closures."""
from vfs.ocean.mixing import coupled as coupled_source
from vfs.ocean.mixing import path_work as path_work_source
import argparse,json,hashlib
from pathlib import Path
import torch
from run import run
from vfs.ocean.mixing.coupled import CoupledClosure
from vfs.ocean.mixing.path_work import PathWorkClosure


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['finite','path'],default='path');p.add_argument('--order',type=int,default=3)
    p.add_argument('--layers',type=int,default=16);p.add_argument('--dt',type=float,default=300.);p.add_argument('--hours',type=float,default=1.)
    p.add_argument('--scenario',choices=['wind','cooling','rain_heating'],default='wind');p.add_argument('--cautious',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();kwargs=dict(accelerate=not a.cautious,relaxation=.3 if a.cautious else .7,max_iterations=240 if a.cautious else 80)
    closure=PathWorkClosure(2.,quadrature_order=a.order,**kwargs) if a.method=='path' else CoupledClosure(2.,**kwargs)
    r=run(a.layers,a.dt,a.hours,a.scenario,closure=closure)
    r.update(method=a.method,path_order=a.order,cautious=a.cautious,torch=torch.__version__,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(path_work_source.__file__), Path(coupled_source.__file__), Path(__file__).with_name('run.py'))})
    a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('method','path_order','layers','dt_s','status','completed_hours','failure','wall_s','energy_error_J_kg','limiter')}),flush=True)

if __name__=='__main__':main()
