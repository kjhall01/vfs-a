"""Long-column qualification of the opt-in thermodynamic evaluation path."""
from vfs.ocean.mixing import fast_thermo as fast_thermo_source
from vfs.ocean.mixing import lagged as lagged_source
import argparse,json,hashlib
from pathlib import Path
from vfs.ocean.mixing.lagged import SubcycledLaggedClosure
from run import run


def main():
    p=argparse.ArgumentParser();p.add_argument('--layers',type=int,default=64);p.add_argument('--hours',type=float,default=24);p.add_argument('--scenario',choices=['wind','cooling','rain_heating'],default='wind');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=run(a.layers,300.,a.hours,a.scenario,closure=SubcycledLaggedClosure(2.,fast_thermodynamics=True),progress=True)
    r['method']='fast_thermodynamics';r['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(fast_thermo_source.__file__), Path(lagged_source.__file__), Path(__file__).with_name('run.py'))}
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(r['status'],r['failure'],flush=True)

if __name__=='__main__':main()
