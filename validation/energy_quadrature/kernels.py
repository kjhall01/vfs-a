"""Isolated JMD95 analytic-partials kernel timings, including optional compilation.

This does not time a whole ocean step or establish full compiled-model safety.
"""
import argparse,json,time,statistics
from pathlib import Path
import torch
from vfs.ocean.jmd95_energy import jmd95_partials


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cells',type=int,default=1024);p.add_argument('--layers',type=int,default=64)
    p.add_argument('--compile',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    rows=[]
    with torch.no_grad():
        for q in [4,2,3]:
            t=torch.linspace(2,25,a.cells*a.layers).reshape(a.cells,a.layers,1,1)
            s=34+t/25
            pressure=torch.linspace(0,6000,a.cells*a.layers*q*q).reshape(a.cells,a.layers,q,q)
            expected=jmd95_partials(s,t,pressure)
            variants={'eager':jmd95_partials}
            if a.compile:variants['compiled']=torch.compile(jmd95_partials,fullgraph=True)
            for name,fn in variants.items():
                start=time.perf_counter()
                try:actual=fn(s,t,pressure)
                except Exception as error:
                    rows.append(dict(order=q,mode=name,error=str(error)));continue
                startup=time.perf_counter()-start
                samples=[]
                for _ in range(5):
                    start=time.perf_counter();fn(s,t,pressure);samples.append(time.perf_counter()-start)
                errors=[dict(max_abs=float((x-y).abs().max()),relative_l2=float((x-y).norm()/y.norm())) for x,y in zip(actual,expected)]
                row=dict(order=q,points=q*q,mode=name,startup_s=startup,timings_s=samples,median_s=statistics.median(samples),errors=errors)
                rows.append(row);print(json.dumps(row),flush=True)
                a.output.write_text(json.dumps(dict(torch=torch.__version__,cells=a.cells,layers=a.layers,rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
