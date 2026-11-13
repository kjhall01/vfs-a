"""Observe rejected coupled iterations without modifying solver expressions.

The optional local checkpoint is a trusted Python/Torch object, never uploaded.
Replays use the identical pre-mixing input; half-dt is a mixing-only sensitivity,
not a complete alternative forced trajectory.
"""
import argparse,json,sys,linecache
from dataclasses import replace
from pathlib import Path
import torch
from path_work import PathWorkClosure
from coupled import CoupledClosure
from run import run


def ledger(v,k):
    mass=v['mass'];zero=torch.zeros_like(k[:,:1])
    flux=v['dt']*v['g']*(k[:,1:]-k[:,:-1])
    transport=torch.cat((flux,zero),-1)-torch.cat((zero,flux),-1)
    storage=mass*(k-v['k0']);heat=v['diss']*k
    residual=storage-v['production']+v['work']+heat-transport
    scale=mass*v['self'].budget_atol+v['self'].budget_rtol*(mass*v['k0']+v['production']+v['work'].abs())
    return dict(k=k,storage=storage,production=v['production'],buoyancy_work=v['work'],dissipation=heat,transport=transport,residual=residual,scale=scale,scaled_residual=residual.abs()/scale)


def serial(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().tolist()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [serial(v) for v in x]
    return x

class Observer:
    def __init__(self):self.history=[];self.last=None
    def trace(self,frame,event,arg):
        if frame.f_code is not CoupledClosure.apply.__code__:return None
        if event=='line' and linecache.getline(frame.f_code.co_filename,frame.f_lineno).lstrip().startswith('next_guess=(1-self.relaxation)'):
            v=frame.f_locals
            self.history.append(dict(iteration=v['iteration'],error=float(v['error']),predicted=serial(ledger(v,v['predicted']))))
            self.history=self.history[-12:]
        if event=='exception' and isinstance(arg[1],FloatingPointError):self.last=dict(frame.f_locals)
        if event=='return' and arg is not None:self.last=dict(frame.f_locals)
        return self.trace
    def call(self,closure,ocean,state,tke_h,dt):
        prior=sys.gettrace();sys.settrace(self.trace)
        try:
            out=closure.apply(ocean,state,tke_h,dt)
            r=out[-1]
            return dict(status='completed',iterations=r.iterations,budget_error=float(r.budget_error),fixed_point_error=float(r.fixed_point_error),tke_h=serial(out[1]),local_residual=serial(r.local_tke_residual))
        except FloatingPointError as exc:
            v=self.last
            if v is None:raise
            return dict(status='rejected',message=str(exc),iteration=v['iteration'],depth=serial(v['h'].cumsum(-1)-v['h']/2),k0=serial(v['k0']),guess=serial(v['guess']),predicted=serial(ledger(v,v['predicted'])),direct=serial(ledger(v,v['direct'])) if 'direct' in v else None,history=self.history)
        finally:sys.settrace(prior)

class CaptureClosure(PathWorkClosure):
    def apply(self,ocean,state,tke_h,dt):
        try:return super().apply(ocean,state,tke_h,dt)
        except FloatingPointError:
            torch.save((ocean,state,tke_h,dt),CHECKPOINT)
            raise


def main():
    global CHECKPOINT
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args();CHECKPOINT=a.checkpoint
    torch.set_num_threads(1)
    result={}
    with torch.no_grad():
        if not CHECKPOINT.exists():result['trajectory']=run(16,300.,6.1,closure=CaptureClosure(2.))
        ocean,state,k,dt=torch.load(CHECKPOINT,weights_only=False)
        original=[x.clone() for x in (state.u_e,state.h,state.salt_h,state.entropy_h,k)]
        base=PathWorkClosure(2.)
        cases=[('baseline',base,dt),('iterations320',replace(base,max_iterations=320),dt),('cautious',replace(base,accelerate=False,relaxation=.3,max_iterations=320),dt),('order5',replace(base,quadrature_order=5),dt),('half_dt',base,dt/2)]
        result['replays']={}
        for name,closure,step in cases:
            obs=Observer();result['replays'][name]=obs.call(closure,ocean,state,k,step)
            assert all(torch.equal(x,y) for x,y in zip(original,(state.u_e,state.h,state.salt_h,state.entropy_h,k)))
            print(name,result['replays'][name]['status'],flush=True)
        result['inputs_unchanged']=True
    a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
