"""Cost of five 60-second dynamics-only steps over the mixing benchmark interval."""
import argparse,json,time,statistics
from pathlib import Path
import torch
from resolution_mesh import setup
from vfs.ocean.stratified import TKEOcean


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cells',type=int,default=1024);p.add_argument('--layers',type=int,default=64);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    with torch.no_grad():
        o,s,k=setup(a.cells,a.layers);model=TKEOcean(o);initial=model.state(s,tke=k)
        def run():
            state=initial
            for _ in range(5):state=model.step(state,60.)
            return state
        run();samples=[]
        for _ in range(2):
            cpu=time.process_time();wall=time.perf_counter();run()
            samples.append(dict(cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall))
    r=dict(cells=a.cells,layers=a.layers,threads=1,torch=torch.__version__,dtype='float64',steps=5,dt_s=60.,samples=samples,median_cpu_s=statistics.median(x['cpu_s'] for x in samples),scope='dynamics and horizontal TKE transport only, no mixing/ALE/forcing')
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)

if __name__=='__main__':main()
