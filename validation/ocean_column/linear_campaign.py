"""Matched EOS controls for the SAME research coupled-column solver."""
import argparse,json,hashlib,platform
from pathlib import Path
from functools import partial
import torch
from run import run
from coupled import CoupledClosure
from linear_eos import LinearColumnOcean,ExactLinearWorkClosure


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eos',choices=['jmd95','linear-gibbs','linear-cp'],required=True)
    p.add_argument('--exact-work',action='store_true')
    p.add_argument('--solver',choices=['accelerated','cautious','existing'],default='accelerated')
    p.add_argument('--hours',type=float,default=1.);p.add_argument('--dt',type=float,default=300.)
    p.add_argument('--layers',type=int,default=16);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    factory=None if a.eos=='jmd95' else partial(LinearColumnOcean,constant_cp=a.eos=='linear-cp')
    if a.exact_work and (a.eos!='linear-cp' or a.solver=='existing'):raise ValueError('exact-work requires coupled linear-cp')
    cls=ExactLinearWorkClosure if a.exact_work else CoupledClosure
    closure=None if a.solver=='existing' else cls(2.,accelerate=a.solver=='accelerated',relaxation=.7 if a.solver=='accelerated' else .3,max_iterations=80 if a.solver=='accelerated' else 240)
    r=run(a.layers,a.dt,a.hours,'wind',closure=closure,ocean_factory=factory)
    r.update(eos=a.eos,solver=a.solver,exact_work=a.exact_work,python=platform.python_version(),torch=torch.__version__,
        sources={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('coupled.py','linear_eos.py','run.py')})
    a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('eos','solver','dt_s','status','completed_hours','wall_s','failure','energy_error_J_kg','limiter')}),flush=True)

if __name__=='__main__':main()
