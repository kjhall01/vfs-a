"""Wave/steady-flow coordinate comparison; continuum reference only for wave."""
import argparse,importlib.util,json,math,sys
from pathlib import Path
import torch
import numpy as np
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean
from model import TemperatureOcean
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['wave','rest','jet']);p.add_argument('--cells',type=int,default=32)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    if a.kind=='wave':
        old=load('temperature_wave_initial',ROOT/'tools/qualify_ocean_entropy.py')
        m,e,energy,h0,mode,vector,omega,fast=old.wave_setup(a.cells)
        initial=e.physical_state(energy.reference);o=StratifiedOcean(m.mesh,m.column,warm_start=True)
        o.base.dynamics.mechanical.rotation_rate=0.
        period=2*math.pi/omega;duration=period;limit=.3*float(m.mesh.dc_edge.min())/fast
        steps=math.ceil(duration/limit);dt=duration/steps
    elif a.kind=='jet':
        old=load('temperature_jet_initial',ROOT/'tools/verify_ocean_williamson.py')
        mesh=VoronoiMesh(n_cells=a.cells,lloyd_iters=10,rotation_scheme='compatible',verbose=False)
        m=old.make_model(mesh);initial=old.initial_state(m)
        o=StratifiedOcean(mesh,m.column,warm_start=True);o.base.dynamics.mechanical.rotation_rate=old.OMEGA
        steps=144;dt=600.;duration=steps*dt
    else:
        mesh=VoronoiMesh(n_cells=a.cells,lloyd_iters=2,rotation_scheme='compatible',verbose=False)
        depth=3000+200*mesh.cell_xyz[:,:1];o=StratifiedOcean(mesh,OceanColumn(3,total_depth=depth,rest_thickness=torch.tensor([600.,900.,1500.])),warm_start=True)
        h=depth*torch.tensor([.2,.3,.5])[None]
        initial=o.base.state(u_e=torch.zeros(mesh.nEdges,3),h=h,temperature=h*0+10,salinity=h*0+35)
        steps=144;dt=600.;duration=steps*dt
    candidate=TemperatureOcean(o);entropy=o.from_temperature(initial)
    area=o.mesh.cell_area[:,None];mass=(o.energy.mass_area(initial)*initial.h).sum();energy0=candidate.energy(initial)
    report=dict(kind=a.kind,cells=a.cells,steps=steps,dt_s=dt,duration_s=duration,scope='unforced dynamics; no mixing/forcing/ALE',cases={})
    if a.kind=='wave':report.update(period_s=period,reference='continuum two-layer hydrostatic linearization',periods=1.)
    outputs={}
    with torch.no_grad():
        for name,state,step in [('entropy',entropy,o.step),('temperature',initial,candidate.step)]:
            samples=[]
            try:
                for n in range(steps+1):
                    if n%max(1,steps//96)==0 or n==steps:
                        s=o.check_state(state) if name=='entropy' else state
                        row=dict(time_s=n*dt,max_velocity=float(s.u_e.abs().max()),energy_drift_J_kg=float((candidate.energy(s)-energy0)/mass))
                        if a.kind=='wave':
                            norm=(area*mode.square()).sum()*vector.square().sum()
                            rhs,_=candidate.tendency_with_report(s,dt)
                            amplitude=(area*mode*(s.h-h0)*vector).sum()/norm
                            rate=(area*mode*rhs.h*vector).sum()/norm
                            row.update(amplitude_m=float(amplitude),phase=float(torch.atan2(-rate/omega,amplitude)),envelope=float(torch.sqrt(amplitude**2+(rate/omega)**2)/.1))
                        samples.append(row)
                    if n<steps:state=step(state,dt)
                s=o.check_state(state) if name=='entropy' else state;outputs[name]=s
                item=dict(status='completed',samples=samples,min_salt=float(s.salinity().min()),min_h=float(s.h.min()),
                    velocity_change_rms=float((s.u_e-initial.u_e).square().mean().sqrt()),height_change_rms=float((s.h-initial.h).square().mean().sqrt()))
                if a.kind=='wave':
                    phases=np.unwrap([r['phase'] for r in samples]);times=np.array([r['time_s'] for r in samples])
                    item.update(frequency_ratio=float(np.polyfit(times,phases,1)[0]/omega),final_envelope=samples[-1]['envelope'])
            except (ValueError,FloatingPointError) as error:item=dict(status='rejected',error=str(error),samples=samples)
            report['cases'][name]=item;a.output.write_text(json.dumps(report,indent=2)+'\n');print(name,item['status'],flush=True)
    if len(outputs)==2:
        x,y=outputs.values();report['coordinate_difference']=dict(temperature_rms_C=float((x.temperature()-y.temperature()).square().mean().sqrt()),velocity_rms_m_s=float((x.u_e-y.u_e).square().mean().sqrt()),height_rms_m=float((x.h-y.h).square().mean().sqrt()))
    a.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
