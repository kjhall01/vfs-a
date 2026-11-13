"""Reproduce the coupled prototype's fixed one-hour and six-minute matrices.

Run outside the checkout against an installed wheel. Default serial execution
keeps timings easier to interpret; --workers explicitly enables concurrent jobs.
Scientific rejections are saved in each JSON, not replaced by smaller timesteps.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--short',action='store_true')
    p.add_argument('--interface-scheme',choices=['arithmetic','resistance'],default='resistance')
    a=p.parse_args()
    if a.workers<1:raise ValueError('positive worker count required')
    a.output.mkdir(parents=True,exist_ok=True)
    dts=(180,90,45,11.25) if a.short else (600,300,150,75,18.75)
    jobs=[(method,16,dt,'wind') for method in ('coupled','default') for dt in dts]
    if not a.short:jobs += [('coupled',32,300,'wind'),('coupled',16,300,'cooling'),('coupled',16,300,'rain_heating')]
    def job(args):
        method,layers,dt,scenario=args
        name=f'short-{method}-{dt}.json' if a.short else f'{method}-{layers}-{dt}-{scenario}.json'
        cmd=[sys.executable,str(Path(__file__).with_name('coupled_campaign.py')),
             '--method',method,'--layers',str(layers),'--dt',str(dt),'--scenario',scenario,
             '--hours',str(.1 if a.short else 1.),'--interface-scheme',a.interface_scheme,
             '--output',str(a.output/name)]
        result=subprocess.run(cmd,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True)
        (a.output/(name+'.log')).write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(f'process failed: {name}; see its log')
        print(name,result.stdout.strip(),flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(job,jobs))

if __name__=='__main__':main()
