"""Heating/mixing comparison with inverse attribution; CPU short samples."""
import argparse,json,sys,time,statistics
from pathlib import Path
from unittest.mock import patch
from collections import Counter
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ocean_column'))
from resolution_mesh import setup
from lagged import SubcycledLaggedClosure
from fast_thermo import MixingThermodynamics,Caloric
from model import TemperatureOcean
from physics import TemperatureMixing,heat
from vfs.ocean.stratified import equations


def main():
    p=argparse.ArgumentParser();p.add_argument('--cells',type=int,default=128);p.add_argument('--layers',type=int,default=16)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--compile',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    o,e,k=setup(a.cells,a.layers)
    if a.compile:o.compile_thermodynamics()
    s=o.check_state(e);c=TemperatureOcean(o);kh=s.h*k
    old=SubcycledLaggedClosure(2.,substeps=4,fast_thermodynamics=True)
    new=TemperatureMixing(2.,substeps=4)
    q=torch.full_like(s.h,.1)
    def before():
        ph=o.check_state(e);w=MixingThermodynamics(o,ph)
        warmed,_,_=w.heat(e,ph,q)
        return old.apply(o,warmed,kh,60.)
    def after():
        warmed,_=heat(o,s,q)
        return new.apply(o,warmed,kh,60.)
    out=dict(cells=a.cells,layers=a.layers,threads=1,torch=torch.__version__,compiled=a.compile,
        dt=60.,mixing_substeps=4,heat_J_kg=.1,scope='layer heating then mixing only; no dynamics/forcing/ALE',cases={})
    endpoints={}
    with torch.no_grad():
        for name,fn in [('entropy',before),('temperature',after)]:
            fn();times=[]
            for _ in range(a.repeats):
                start=time.perf_counter();fn();times.append(time.perf_counter()-start)
            counts=Counter();original=Caloric.inverse
            def count(self,target,guess,kind):
                counts[kind]+=1;return original(self,target,guess,kind)
            with patch.object(Caloric,'inverse',count),patch.object(equations,'temperature_from_entropy',wraps=equations.temperature_from_entropy) as entropy_count:
                end,tk,r=fn()
            endpoints[name]=(end,tk,r)
            ph=o.check_state(end) if name=='entropy' else end
            total=c.energy(ph)+(o.energy.mass_area(ph)*tk).sum()
            initial=c.energy(s)+(o.energy.mass_area(s)*kh).sum()
            mass=o.energy.mass_area(s)*s.h
            out['cases'][name]=dict(median_s=statistics.median(times),samples_s=times,
                caloric_inverse_calls=dict(counts),state_entropy_inverse_calls=entropy_count.call_count,
                energy_minus_heat_J_kg=float((total-initial-(mass*q).sum())/mass.sum()),
                max_local_budget_ratio=float(r.budget_error),min_tke=float((tk/ph.h).min()),
                min_tracer=float(ph.tracer_h.min()),
                limiter={key:float(value) for key,value in r.mixing_limiter.items()})
        ae,ak,_=endpoints['entropy'];b,bk,_=endpoints['temperature'];ap=o.check_state(ae)
        out['difference']=dict(temperature_rms_C=float((b.temperature()-ap.temperature()).square().mean().sqrt()),
            tke_max_abs=float((bk-ak).abs().max()),velocity_max_abs=float((b.u_e-ap.u_e).abs().max()),
            salt_max_abs=float((b.salt_h-ap.salt_h).abs().max()))
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
