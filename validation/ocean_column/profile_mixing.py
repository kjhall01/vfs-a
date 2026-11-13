"""Fixed-state CPU mixing/dynamics cost with reversible nested instrumentation.

Run outside the checkout against an installed wheel. No state advances between
samples. Exclusive times partition wrapped work; inclusive times overlap.
"""
import argparse, cProfile, functools, hashlib, json, pstats, statistics, time
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import torch
import lagged
from path_work import PathWorkClosure
from resolution_mesh import setup
from vfs.ocean.stratified import TKEOcean, StratifiedOcean
from vfs.ocean.stratified.thermodynamics import SurfaceGibbs
from vfs.ocean.jmd95_energy import JMD95Energy
from vfs.ocean.stratified import equations


class Timers:
    def __init__(self):
        self.stack=[]; self.rows={}; self.roots={}
    def wrap(self, name, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            start=time.perf_counter(); frame=[0.]; self.stack.append(frame)
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed=time.perf_counter()-start; self.stack.pop()
                row=self.rows.setdefault(name,dict(calls=0,inclusive_s=0.,exclusive_s=0.))
                row['calls']+=1; row['inclusive_s']+=elapsed; row['exclusive_s']+=elapsed-frame[0]
                if self.stack:self.stack[-1][0]+=elapsed
                else:self.roots[name]=self.roots.get(name,0.)+elapsed
        return wrapped
    def install(self, stack):
        sites=[(SurfaceGibbs,'evaluate','surface_gibbs'),
               (SurfaceGibbs,'temperature_from_enthalpy','enthalpy_inverse'),
               (equations,'temperature_from_entropy','entropy_inverse'),
               (JMD95Energy,'parcel_potential','jmd95_parcel_potential'),
               (PathWorkClosure,'_buoyancy_work','buoyancy_path_work'),
               (lagged,'apply_heat','dissipation_heat_inverse'),
               (lagged,'thermal_parts','thermal_energy_parts'),
               (lagged,'diffuse_many','scalar_diffusion'),
               (lagged,'diffuse','momentum_diffusion'),
               (lagged.Factor,'__init__','banded_factor'),
               (lagged.Factor,'solve','banded_solve'),
               (StratifiedOcean,'check_state','state_validation')]
        for obj, attr, name in sites:
            stack.enter_context(patch.object(obj,attr,self.wrap(name,getattr(obj,attr))))


def identical(a,b):
    names=('u_e','h','salt_h','entropy_h','tracer_h')
    return all(torch.equal(getattr(a.ocean,n),getattr(b.ocean,n)) for n in names) and torch.equal(a.tke_h,b.tke_h)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cells',type=int,default=1024);p.add_argument('--layers',type=int,default=64)
    p.add_argument('--repeats',type=int,default=2);p.add_argument('--dt',type=float,default=300.)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    result=dict(cells=a.cells,layers=a.layers,dt=a.dt,threads=1,torch=torch.__version__,dtype='float64',
        source_hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('profile_mixing.py','resolution_mesh.py','lagged.py','path_work.py')},
        scope='fixed-state vertical mixing versus SSPRK3 dynamics with transported TKE; no ALE or boundary forcing',timings={})
    def save():a.output.write_text(json.dumps(result,indent=2)+'\n')
    with torch.no_grad():
        ocean,initial,k=setup(a.cells,a.layers)
        mixed=TKEOcean(ocean,closure=lagged.SubcycledLaggedClosure(2.,substeps=4))
        dynamics=TKEOcean(ocean)
        state=mixed.state(initial,tke=k)
        operations={'mixing4':lambda:mixed.mix_with_report(state,a.dt),
                    'dynamics':lambda:dynamics.step_with_report(state,a.dt)}
        baseline={}
        for name,fn in operations.items():
            try:
                baseline[name]=fn()[0]  # warmup excluded
                samples=[]
                for _ in range(a.repeats):
                    cpu=time.process_time();wall=time.perf_counter();out=fn()[0]
                    samples.append(dict(cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall))
                    assert identical(out,baseline[name])
                result['timings'][name]=dict(status='completed',samples=samples,median_cpu_s=statistics.median(s['cpu_s'] for s in samples))
            except (ValueError,FloatingPointError) as e:
                result['timings'][name]=dict(status='rejected',failure=str(e))
            save();print(name,result['timings'][name],flush=True)
        timer=Timers();profile=cProfile.Profile()
        with ExitStack() as stack:
            timer.install(stack);start=time.perf_counter();profile.enable()
            out,report=operations['mixing4']()
            profile.disable();elapsed=time.perf_counter()-start
        result['profile']=dict(wall_s=elapsed,bitwise_state_match=identical(out,baseline['mixing4']),
            categories=timer.rows,outermost_categories_s=timer.roots,
            unwrapped_s=elapsed-sum(timer.roots.values()),limiter={k:float(v) for k,v in report.mixing_limiter.items()})
        stats=pstats.Stats(profile)
        rows=[]
        for (filename,line,name),(primitive,calls,self_s,cumulative,callers) in stats.stats.items():
            rows.append(dict(file=filename,line=line,function=name,calls=calls,self_s=self_s,cumulative_s=cumulative))
        result['profile']['top_self']=sorted(rows,key=lambda x:x['self_s'],reverse=True)[:50]
        mass=ocean.energy.mass_area(initial)*initial.h
        result['energy_error_J_kg']=float((mixed.budgets(out)['total']-mixed.budgets(state)['total'])/mass.sum())
        if result['timings'].get('dynamics',{}).get('status')=='completed':
            d=result['timings']['dynamics']['median_cpu_s'];m=result['timings']['mixing4']['median_cpu_s']
            result['cost_budget']=dict(mixing_to_dynamics=m/d,mixing_fraction=m/(m+d),
                scenarios=[dict(target_mixing_fraction=f,allowed_mixing_s=f/(1-f)*d,required_speedup=m/(f/(1-f)*d)) for f in (.1,.2,.3)])
        save();print('profile',elapsed,'exact',result['profile']['bitwise_state_match'],flush=True)

if __name__=='__main__':main()
