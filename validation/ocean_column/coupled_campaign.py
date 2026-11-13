"""Bounded coupled-column experiment; failures are recorded, never retried at smaller dt."""
import argparse,json,hashlib,platform,time
from pathlib import Path
import torch
from run import run
from coupled import CoupledClosure


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['default','coupled'],default='coupled')
    p.add_argument('--layers',type=int,default=16);p.add_argument('--dt',type=float,default=300)
    p.add_argument('--hours',type=float,default=1.)
    p.add_argument('--rtol',type=float,default=1e-6);p.add_argument('--atol',type=float,default=1e-12)
    p.add_argument('--iterations',type=int,default=80);p.add_argument('--no-acceleration',action='store_true')
    p.add_argument('--scenario',choices=['wind','cooling','rain_heating'],default='wind')
    p.add_argument('--relaxation',type=float,default=.7)
    p.add_argument('--interface-scheme',choices=['arithmetic','resistance'],default='resistance')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    closure=CoupledClosure(2.,rtol=a.rtol,atol=a.atol,max_iterations=a.iterations,accelerate=not a.no_acceleration,interface_scheme=a.interface_scheme,relaxation=a.relaxation) if a.method=='coupled' else None
    r=run(a.layers,a.dt,a.hours,a.scenario,closure=closure)
    r.update(method=a.method,rtol=a.rtol,atol=a.atol,max_iterations=a.iterations,acceleration=not a.no_acceleration,interface_scheme=a.interface_scheme,relaxation=a.relaxation,
        python=platform.python_version(),torch=torch.__version__,source_sha256=hashlib.sha256(Path(__file__).with_name('coupled.py').read_bytes()).hexdigest())
    a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('method','layers','dt_s','status','completed_hours','failure','wall_s','energy_error_J_kg','limiter')}),flush=True)

if __name__=='__main__':main()
