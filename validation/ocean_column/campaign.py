"""Run the fixed matrix and two separately labelled seed sensitivities."""
import argparse,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cases=[(n,dt,s,False,1e-4) for n in (16,32,64) for dt in (300,150) for s in ('wind','cooling','rain_heating')]
    cases += [(n,dt,'wind',False,1e-4) for n in (16,32,64) for dt in (30,3)]
    cases += [(n,dt,'wind',True,1e-4) for n in (16,32,64) for dt in (300,150)]
    cases += [(32,3,'wind',False,seed) for seed in (.001,.01)]
    def job(case):
        n,dt,s,control,seed=case
        name=f'{s}-{n}-{dt}-{seed}-'+('momentum_control' if control else 'active')
        output=args.output/(name+'.json')
        command=[sys.executable,'-I',str(Path(__file__).with_name('run.py')),
            '--layers',str(n),'--dt',str(dt),'--hours','48','--scenario',s,'--seed',str(seed),'--output',str(output)]
        if control:command.append('--momentum-only')
        result=subprocess.run(command,check=True,capture_output=True,text=True)
        return json.loads(result.stdout)|{'file':output.name}
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(job,c) for c in cases]):
            row=future.result();rows.append(row);print(json.dumps(row),flush=True)
            (args.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
