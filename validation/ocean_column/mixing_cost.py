"""Short cost/accuracy experiment; run outside the checkout against a wheel.

python mixing_cost.py --method default --dt 150 --hours 2 --output result.json
python mixing_cost.py --method predictor --dt 150 --hours 2 --output result.json
Use --profile for call-cost evidence; profile timings are not benchmark timings.
"""
import argparse
import hashlib
import platform
import torch
import vfs.ocean.stratified.tke as runtime
import cProfile
import io
import json
import pstats
from pathlib import Path
from run import run
from budget_predictor import BudgetPredictor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['default','predictor'],default='default')
    p.add_argument('--dt',type=float,default=150.)
    p.add_argument('--hours',type=float,default=2.)
    p.add_argument('--margin',type=float,default=.95)
    p.add_argument('--scenario',choices=['wind','cooling','rain_heating'],default='wind')
    p.add_argument('--layers',type=int,default=32)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    closure=BudgetPredictor(2.,margin=a.margin) if a.method=='predictor' else None
    profiler=cProfile.Profile() if a.profile else None
    if profiler:profiler.enable()
    result=run(a.layers,a.dt,a.hours,a.scenario,closure=closure)
    if profiler:profiler.disable()
    result['method']=a.method;result['predictor_margin']=a.margin if closure else None
    result['profiled']=a.profile
    result['runtime_sha256']=hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest()
    result['predictor_sha256']=hashlib.sha256(Path(__file__).with_name('budget_predictor.py').read_bytes()).hexdigest()
    result['python']=platform.python_version();result['torch']=torch.__version__
    if profiler:
        out=io.StringIO();stats=pstats.Stats(profiler,stream=out).strip_dirs().sort_stats('cumtime')
        stats.print_stats(30)
        a.output.with_suffix('.txt').write_text(out.getvalue())
        result['candidate_calls']=sum(v[1] for k,v in stats.stats.items() if k[2]=='candidate')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('method','dt_s','status','wall_s','energy_error_J_kg')}),flush=True)

if __name__=='__main__':main()
