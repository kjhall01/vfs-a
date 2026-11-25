"""Semi-implicit column accuracy/cost controls; reference remains fully coupled."""
from vfs.ocean.mixing import coupled as coupled_source
from vfs.ocean.mixing import lagged as lagged_source
from vfs.ocean.mixing import path_work as path_work_source
import argparse,json,hashlib
from pathlib import Path
from run import run
from vfs.ocean.mixing.lagged import LaggedClosure,SubcycledLaggedClosure
from consistent import ConsistentPathClosure

def main():
 p=argparse.ArgumentParser();p.add_argument('--dt',type=float,default=300.);p.add_argument('--hours',type=float,default=1.);p.add_argument('--layers',type=int,default=16)
 p.add_argument('--predictors',type=int,default=1);p.add_argument('--substeps',type=int,default=1);p.add_argument('--funding',choices=['split','column','local'],default='split');p.add_argument('--reduction',type=float,default=.9);p.add_argument('--reference',action='store_true');p.add_argument('--scenario',choices=['wind','cooling','rain_heating'],default='wind');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(lagged_source.__file__), Path(path_work_source.__file__), Path(__file__).with_name('run.py'), Path(__file__).with_name('consistent.py'), Path(coupled_source.__file__))}
 closure=ConsistentPathClosure(2.) if a.reference else SubcycledLaggedClosure(2.,reduction=a.reduction,predictor_steps=a.predictors,substeps=a.substeps,funding=a.funding)
 r=run(a.layers,a.dt,a.hours,a.scenario,closure=closure,progress=True)
 r.update(method='reference' if a.reference else 'lagged',reduction=a.reduction,predictors=a.predictors,substeps=a.substeps,funding=a.funding,source_hashes=hashes)
 a.output.write_text(json.dumps(r,indent=2)+'\n');print(r['status'],r['failure'],flush=True)
if __name__=='__main__':main()
