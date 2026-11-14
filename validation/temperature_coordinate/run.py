"""Short coordinate-comparison campaign; no global-model qualification."""
import argparse,json,statistics,sys,time
from pathlib import Path
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ocean_column'))
from resolution_mesh import setup
from model import TemperatureOcean
from vfs.ocean.stratified import equations


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cells',type=int,default=128);p.add_argument('--layers',type=int,default=16)
    p.add_argument('--compile',action='store_true');p.add_argument('--front',action='store_true');a=p.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    o,entropy,_=setup(a.cells,a.layers);o.warm_start=True
    if a.compile:o.compile_thermodynamics()
    candidate=TemperatureOcean(o);physical=o.check_state(entropy)
    if a.front:
        front=o.mesh.cell_xyz[:,:1]>0
        t=torch.where(front,torch.full_like(physical.h,18.),torch.full_like(physical.h,4.))
        s=torch.where(front,torch.full_like(physical.h,34.),torch.full_like(physical.h,36.))
        physical=physical.with_tensors(physical.u_e*10000,physical.h,physical.h*t,physical.h*s,physical.tracer_h)
        entropy=o.from_temperature(physical)
    out=dict(cells=a.cells,layers=a.layers,threads=1,torch=torch.__version__,compiled=a.compile,
             quadrature=4,front=a.front,scope='unforced dynamics only',timings={},comparisons=[])
    with torch.no_grad():
        for name,step,initial in [('entropy',o.step,entropy),('temperature',candidate.step,physical)]:
            step(initial,60.)
            times=[]
            for _ in range(3):
                start=time.perf_counter();step(initial,60.);times.append(time.perf_counter()-start)
            with patch.object(equations,'temperature_from_entropy',wraps=equations.temperature_from_entropy) as count:
                step(initial,60.)
            out['timings'][name]=dict(samples_s=times,median_s=statistics.median(times),inverse_calls=count.call_count)
        initial_energy=candidate.energy(physical);mass=(o.energy.mass_area(physical)*physical.h).sum()
        for dt in (60.,30.,15.):
            es,ts=entropy,physical
            for _ in range(round(120/dt)):
                es=o.step(es,dt);ts=candidate.step(ts,dt)
            ep=o.check_state(es)
            out['comparisons'].append(dict(dt=dt,duration=120.,
                temperature_rms_C=float((ts.temperature()-ep.temperature()).square().mean().sqrt()),
                velocity_rms_m_s=float((ts.u_e-ep.u_e).square().mean().sqrt()),
                temperature_energy_change_J_kg=float((candidate.energy(ts)-initial_energy)/mass),
                entropy_energy_change_J_kg=float((o.energy.terms(es)['total']-initial_energy)/mass),
                min_salt=float(ts.salinity().min()),min_tracer=float(ts.tracer_h.min())))
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
