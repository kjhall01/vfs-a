"""48-hour forcing/refinement matrix and labelled limiter/seed sensitivities.

Each case writes a progress log and full local JSON. Publication uses compact
summaries; never upload oversized raw outputs.
"""
import argparse,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--group',choices=('all','primary','sensitivity'),default='all')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cases=[(n,d,s,1e-4,.5) for n in (16,32,64) for d in (300,150) for s in ('wind','cooling','rain_heating')]
    cases += [(32,150,'wind',k,.5) for k in (1e-6,1e-3)]
    cases += [(32,150,'wind',1e-4,a) for a in (.25,.75)]
    cases += [(32,75,'wind',1e-4,.5)]
    if args.group=='primary':cases=cases[:18]
    elif args.group=='sensitivity':cases=cases[18:]
    cases.sort(key=lambda c:(c[1],-c[0],{'rain_heating':0,'cooling':1,'wind':2}[c[2]]))
    def run(c):
        n,d,s,k,a=c;name=f'{s}-{n}-{d}-k{k}-a{a}'
        output=args.output/(name+'.json')
        while not output.exists():
            with (args.output/(name+'.log')).open('a') as log:
                subprocess.run([sys.executable,'-I',str(Path(__file__).with_name('run.py')),
                    '--layers',str(n),'--dt',str(d),'--scenario',s,'--seed',str(k),'--safety',str(a),
                    '--hours','48','--output',str(output),'--progress','--checkpoint',str(args.output/(name+'.pt')),'--wall-limit','120'],check=True,stdout=log,stderr=log)
        r=json.loads(output.read_text());r.pop('profiles');return r
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,c) for c in cases]):
            r=future.result();rows.append(r);print(json.dumps(r),flush=True)
            (args.output/('summary.json' if args.group=='all' else args.group+'-summary.json')).write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
