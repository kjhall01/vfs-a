import subprocess,concurrent.futures,json
from pathlib import Path
import argparse,sys
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
root=parser.parse_args().output;root.mkdir(parents=True,exist_ok=True)
cases=[(n,d,s,1e-4) for n in (16,32,64) for d in (300,150) for s in ('wind','cooling','rain_heating')]+[(16,300,'wind',k) for k in (0,1e-12)]
def run(c):
 n,d,s,k=c;p=root/f'{s}-{n}-{d}-{k}.json'
 subprocess.run([sys.executable,'-I',str(Path(__file__).with_name('run.py')),'--layers',str(n),'--dt',str(d),'--scenario',s,'--seed',str(k),'--hours','1','--output',str(p)],check=True,stdout=subprocess.DEVNULL)
 r=json.loads(p.read_text());print({k:v for k,v in r.items() if k!='profiles'},flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,cases))
