"""Reproduce matched EOS/solver controls; no retries or hidden substeps."""
import argparse,os,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=1);p.add_argument('--exact-work',action='store_true');a=p.parse_args()
    if a.workers<1:raise ValueError('positive worker count required')
    a.output.mkdir(parents=True,exist_ok=True)
    jobs=[(e,s,16,300) for e in ('jmd95','linear-gibbs','linear-cp') for s in ('accelerated','cautious','existing')]
    jobs += [(e,'accelerated',n,dt) for e in ('jmd95','linear-gibbs','linear-cp') for n,dt in ((16,75),(32,300))]
    if a.exact_work:
        jobs=[('linear-cp','accelerated',16,dt) for dt in (600,300,150,75,18.75)]+[('linear-cp','accelerated',32,300)]
    def job(args):
        e,s,n,dt=args;name=f'{e}-{s}-{n}-{dt}'
        cmd=[sys.executable,str(Path(__file__).with_name('linear_campaign.py')),'--eos',e,'--solver',s,'--layers',str(n),'--dt',str(dt),'--output',str(a.output/(name+'.json'))]
        if a.exact_work:cmd.append('--exact-work')
        r=subprocess.run(cmd,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True)
        (a.output/(name+'.log')).write_text(r.stdout+r.stderr)
        if r.returncode:raise RuntimeError(f'{name}: process failed; see log')
        print(name,r.stdout.strip(),flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(job,jobs))

if __name__=='__main__':main()
