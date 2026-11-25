"""Column controls for refreshed-budget research acceptance and acceleration."""
from vfs.ocean.mixing import coupled as coupled_source
from vfs.ocean.mixing import path_work as path_work_source
import argparse,json,hashlib
from pathlib import Path
from run import run
from consistent import ConsistentPathClosure
from vfs.ocean.mixing.path_work import PathWorkClosure

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--solver',choices=['legacy','refresh','budget'],default='budget')
    p.add_argument('--layers',type=int,default=16);p.add_argument('--dt',type=float,default=300.)
    p.add_argument('--hours',type=float,default=1.);p.add_argument('--order',type=int,default=3)
    p.add_argument('--scenario',choices=['wind','cooling','rain_heating'],default='wind')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    c=PathWorkClosure(2.,quadrature_order=a.order) if a.solver=='legacy' else ConsistentPathClosure(2.,quadrature_order=a.order,budget_acceleration=a.solver=='budget')
    r=run(a.layers,a.dt,a.hours,a.scenario,closure=c,progress=True)
    r.update(solver=a.solver,order=a.order,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(coupled_source.__file__), Path(path_work_source.__file__), Path(__file__).with_name('consistent.py'), Path(__file__).with_name('run.py'))})
    a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('status','completed_hours','failure','wall_s','energy_error_J_kg','limiter')}),flush=True)
if __name__=='__main__':main()
