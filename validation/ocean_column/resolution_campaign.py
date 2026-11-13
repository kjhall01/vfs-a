"""Higher-layer column matrix; fixed scientific controls, explicit raw outputs.

Run serially for isolated wall times, or --workers 3 for overlapping jobs.
Oversized raw JSON is not suitable for repository upload; use the summarizer.
"""
import argparse,subprocess,sys,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

CASES=[]
for layers in (32,64):
    for label,args in [('wind24',['--hours','24']),('base',[]),('sub8',['--substeps','8']),('dt150',['--dt','150']),('dt75',['--dt','75']),('factor05',['--reduction','.5']),('reference75',['--reference','--dt','75'])]:
        CASES.append((f'L{layers}-{label}',['--layers',str(layers),'--substeps','4',*args]))
for scenario in ('cooling','rain_heating'):
    CASES.append((f'L64-{scenario}12',['--layers','64','--substeps','4','--hours','12','--scenario',scenario]))

CASES.append(('L128-dt75',['--layers','128','--substeps','4','--dt','75']))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=1);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    def launch(case):
        name,args=case
        with (a.output/(name+'.log')).open('w') as f:
            r=subprocess.run([sys.executable,str(Path(__file__).with_name('lagged_campaign.py')),*args,'--output',str(a.output/(name+'.json'))],stdout=f,stderr=f,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        print(name,r.returncode,flush=True)
        if r.returncode:raise RuntimeError(f'{name}: process failed; inspect log')
    with ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(launch,CASES))
if __name__=='__main__':main()
