"""Alternating complete-step timing; same setup/guards, no instrumentation."""
import argparse,json,time,statistics
from pathlib import Path
import torch
from complete_campaign import make,models
from vfs.ocean.stratified import TKEState


def main():
    p=argparse.ArgumentParser();p.add_argument('--cells',type=int,default=1024);p.add_argument('--layers',type=int,default=64)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    o,e,s,k,f=make(a.cells,a.layers,True)
    selected=[(name,model,TKEState(s if native else e,s.h*k)) for name,model,native in models(o) if name in ('entropy','temperature_ts3')]
    def advance(model,state):
        end,_=model.step_with_report(state,60.,**f)
        target=end.ocean.h.sum(-1,keepdim=True).expand_as(end.ocean.h)/a.layers
        return model.remap_with_report(end,target)
    result=dict(cells=a.cells,layers=a.layers,dt_s=60,compiled=True,threads=1,mixing_substeps=4,ale_every=1,sequence=[],samples={name:[] for name,_,_ in selected})
    with torch.no_grad():
        for _,model,state in selected:
            advance(model,state);advance(model,state)
        for repeat in range(a.repeats):
            for name,model,state in selected if repeat%2==0 else reversed(selected):
                start=time.perf_counter();advance(model,state);elapsed=time.perf_counter()-start
                result['samples'][name].append(elapsed);result['sequence'].append(dict(name=name,seconds=elapsed))
                a.output.write_text(json.dumps(result,indent=2)+'\n');print(name,elapsed,flush=True)
    result['median_s']={k:statistics.median(v) for k,v in result['samples'].items()}
    result['baseline_over_candidate']=result['median_s']['entropy']/result['median_s']['temperature_ts3']
    a.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
