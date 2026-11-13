"""Replay the trusted local six-hour rejection checkpoint with a fresh oracle."""
import argparse,json
from pathlib import Path
import torch
from consistent import ConsistentPathClosure
from diagnose_budget import Observer,ledger

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with torch.no_grad():
  torch.set_num_threads(1);o,s,k,dt=torch.load(a.checkpoint,weights_only=False);results={}
  for name,c in [('refresh',ConsistentPathClosure(2.,budget_acceleration=False)),('budget',ConsistentPathClosure(2.))]:
   obs=Observer();r=obs.call(c,o,s,k,dt);results[name]=r
   if r['status']=='completed':
    v=obs.last;_,p,w,_,g,d,*_=v['trial'](v['final'])
    fresh=ledger(dict(v,production=p,work=w,g=g,diss=d),v['final'])
    r['independent_refreshed_error']=float(fresh['scaled_residual'].max())
    r['trial_evaluations']=float(v['report'].mixing_limiter['trial_evaluations'])
   print(name,r['status'],r.get('iterations'),r.get('independent_refreshed_error'),flush=True)
  a.output.write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
