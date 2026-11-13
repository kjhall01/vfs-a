"""Diagnostic only: test a positive local candidate against frozen/refreshed budgets.

No candidate is committed or returned as a physical update. This is not a
replacement nonlinear solver and does not qualify a complete timestepping path.
"""
import argparse,json
from pathlib import Path
import torch
from diagnose_budget import Observer,ledger,serial
from path_work import PathWorkClosure

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    with torch.no_grad():
        o,s,k,dt=torch.load(a.checkpoint,weights_only=False)
        obs=Observer();r=obs.call(PathWorkClosure(2.),o,s,k,dt)
        if r['status']!='rejected':raise ValueError('Expected pinned rejection')
        v=obs.last;direct=v['direct'];predicted=v['predicted']
        # Retain the existing positive iterate only in inadmissible direct layers.
        # This is an uncommitted diagnostic candidate, not clipping to zero.
        candidate=torch.where(direct>=0,direct,predicted)
        frozen=ledger(v,candidate)
        mixed,production,work,flux,g,diss,diagonal,off,reconstruction=v['trial'](candidate)
        refreshed=dict(v,production=production,work=work,g=g,diss=diss)
        renewed=ledger(refreshed,candidate)
        result=dict(replaced_layers=serial(direct<0),candidate=serial(candidate),frozen=serial(frozen),refreshed=serial(renewed),frozen_max_budget_error=float(frozen['scaled_residual'].max()),refreshed_max_budget_error=float(renewed['scaled_residual'].max()),note='Uncommitted diagnostic only; no full-H/entropy update or long-run qualification.')
        result['accepted_replay_checks']={}
        for name,closure,step in [('iterations320',PathWorkClosure(2.,max_iterations=320),dt),('cautious',PathWorkClosure(2.,max_iterations=320,accelerate=False,relaxation=.3),dt),('order5',PathWorkClosure(2.,quadrature_order=5),dt),('half_dt',PathWorkClosure(2.),dt/2)]:
            observe=Observer();out=observe.call(closure,o,s,k,step)
            if out['status']!='completed':raise ValueError(name+' unexpectedly rejected')
            values=observe.last;final=values['final']
            m,p,w,f,g,d,di,off,re=values['trial'](final)
            fresh=ledger(dict(values,production=p,work=w,g=g,diss=d),final)
            result['accepted_replay_checks'][name]=dict(iterations=out['iterations'],accepted_budget_error=out['budget_error'],refreshed_max_budget_error=float(fresh['scaled_residual'].max()),refreshed=serial(fresh))
            print(name,result['accepted_replay_checks'][name]['refreshed_max_budget_error'],flush=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
