"""Interleaved parent/candidate timings and exact state/report comparisons.

Run against an installed candidate wheel. --baseline is an exported parent
checkout; only its two unmodified transport modules are loaded as a reference.
The remaining package must match that parent except for this optimization.
"""
import argparse
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/'ocean_column'))
from cases import make_case
from vfs.ocean.stratified import equations,transport
from vfs.ocean.stratified import model as model_module


def reference(root):
    package='vfs.ocean.stratified'
    hashes={}
    for name in ('equations','transport'):
        path=root/'vfs/ocean/stratified'/f'{name}.py'
        source=path.read_text();hashes[name]=hashlib.sha256(source.encode()).hexdigest()
        source=source.replace('from .equations import','from ._baseline_equations import')
        source=source.replace('from . import equations as parent','from . import _baseline_equations as parent')
        key=package+'._baseline_'+name
        spec=importlib.util.spec_from_loader(key,loader=None)
        module=importlib.util.module_from_spec(spec);sys.modules[key]=module
        exec(compile(source,str(path),'exec'),module.__dict__)
    return module,hashes


def tensors(obj,prefix=''):
    if isinstance(obj,torch.Tensor):return {prefix:obj.detach().clone()}
    if is_dataclass(obj):
        return {k:v for f in fields(obj) if f.name not in ('mesh','column')
                for k,v in tensors(getattr(obj,f.name),prefix+'.'+f.name).items()}
    if isinstance(obj,dict):return {k:v for a,b in obj.items() for k,v in tensors(b,prefix+'.'+str(a)).items()}
    if isinstance(obj,(tuple,list)):return {k:v for i,b in enumerate(obj) for k,v in tensors(b,prefix+'.'+str(i)).items()}
    return {}


@contextmanager
def counters():
    from vfs.ocean.jmd95_energy import JMD95Energy
    from vfs.ocean.stratified.thermodynamics import SurfaceGibbs
    count={}
    targets=[(equations.EntropyState,'physical_state'),(JMD95Energy,'gradients'),
             (SurfaceGibbs,'evaluate')]
    patches=[]
    for cls,name in targets:
        original=getattr(cls,name);key=cls.__name__+'.'+name
        def wrapper(*args,_fn=original,_key=key,**kw):
            count[_key]=count.get(_key,0)+1
            return _fn(*args,**kw)
        patches.append(patch.object(cls,name,wrapper))
    try:
        for p in patches:p.start()
        yield count
    finally:
        for p in reversed(patches):p.stop()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cells',type=int,default=128);p.add_argument('--layers',type=int,default=16)
    p.add_argument('--cases',nargs='+',choices=['dry','moist','ocean','full','full_ale'],default=['ocean','full','full_ale'])
    p.add_argument('--dt',type=float,default=60.);p.add_argument('--steps',type=int,default=2)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--substeps',type=int,default=4)
    p.add_argument('--profile',action='store_true',help='separate cProfile samples after uninstrumented timings')
    a=p.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    old,hashes=reference(a.baseline)
    result=dict(baseline_hashes=hashes,torch=torch.__version__,threads=1,rows=[])
    with torch.no_grad():
        for case in a.cases:
            config=dict(case=case,closure='fast' if case.startswith('full') else 'none',cells=a.cells,
                layers=a.layers,dt=a.dt,mixing_substeps=a.substeps,remap_every=1)
            initial,advance,diagnose=make_case(config)
            def run(module):
                with patch.object(model_module,'transport',module):
                    state=initial;reports=[]
                    for i in range(a.steps):state,report=advance(state,i);reports.append(report)
                return state,reports
            expected=run(old);actual=run(transport)
            x,y=tensors(expected),tensors(actual)
            if x.keys()!=y.keys():raise AssertionError('report/state structure changed')
            different=[key for key in x if not torch.equal(x[key],y[key])]
            if different:raise AssertionError('nonidentical tensors: '+str(different))
            timings={'parent':[],'candidate':[]}
            for repeat in range(a.repeats):
                order=[('parent',old),('candidate',transport)]
                if repeat%2:order.reverse()
                for name,module in order:
                    start=time.perf_counter();run(module)
                    timings[name].append((time.perf_counter()-start)/a.steps)
            counts={}
            for name,module in [('parent',old),('candidate',transport)]:
                with counters() as c:run(module)
                counts[name]=c
            medians={k:statistics.median(v) for k,v in timings.items()}
            row=dict(config=config,steps=a.steps,repeats=a.repeats,tensor_comparisons=len(x),
                bitwise_equal=True,timings_s_per_step=timings,median_s_per_step=medians,
                speedup=medians['parent']/medians['candidate'],calls_per_trajectory=counts,
                endpoint=diagnose(actual[0]))
            if a.profile:
                import cProfile,pstats
                profiles={}
                for name,module in [('parent',old),('candidate',transport)]:
                    profiler=cProfile.Profile();profiler.enable();run(module);profiler.disable()
                    stats=pstats.Stats(profiler)
                    profiles[name]=[dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],calls=v[1],self_s=v[2],cumulative_s=v[3])
                        for k,v in sorted(stats.stats.items(),key=lambda kv:kv[1][3],reverse=True)[:30]]
                row['instrumented_profiles']=profiles
            result['rows'].append(row);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(row),flush=True)


if __name__=='__main__':main()
