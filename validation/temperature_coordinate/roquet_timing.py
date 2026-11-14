"""Alternating complete-step timings: compiled JMD95, eager linear, eager Roquet.

Same geometry, forcing and substeps. Roquet initial CT/SA is explicitly converted
with offline GSW; models have different EOS/calorics, not equal physical accuracy.
"""
import argparse,json,math,statistics,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ocean_column'))
from complete_campaign import make
from system import TemperatureSystem
from path import TemperaturePathMixing
from surface import TemperatureBoundary,FastSurfaceHeating,FastSurfaceExchange
from linear import LinearOcean,LinearMixing,LinearHeating,LinearExchange
from roquet import RoquetOcean,RoquetMixing,RoquetHeating,RoquetExchange,convert_jmd95_initialization
from vfs.ocean.stratified import TKEState


def main():
    p=argparse.ArgumentParser();p.add_argument('--cells',type=int,default=1024);p.add_argument('--layers',type=int,default=64);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    o,e,s,k,f=make(a.cells,a.layers,True)
    j=TemperatureSystem(o,closure=TemperaturePathMixing(2.,substeps=4),boundary=TemperatureBoundary(.0025),heating=FastSurfaceHeating(.6,(.5,20.)),exchange=FastSurfaceExchange())
    l=LinearOcean(o.mesh,o.column,closure=LinearMixing(2.,substeps=4),boundary=TemperatureBoundary(.0025),heating=LinearHeating(.6,(.5,20.)),exchange=LinearExchange())
    r=RoquetOcean(o.mesh,o.column,closure=RoquetMixing(2.,substeps=4),boundary=TemperatureBoundary(.0025),heating=RoquetHeating(.6,(.5,20.)),exchange=RoquetExchange())
    xyz=s.mesh.cell_xyz;z=o.energy.base.height(s)
    converted=convert_jmd95_initialization(potential_temperature_ipts68=s.temperature(),practical_salinity=s.salinity(),pressure_dbar=-s.column.rho0*s.column.gravity*z/1e4,longitude=torch.full_like(xyz[:,:1],-30.),latitude=torch.full_like(xyz[:,2:3],45.))
    sr=r.state(u_e=s.u_e,h=s.h,**converted,tracer=s.tracer_h/s.h[...,None],tke=k)
    cases=[('compiled_jmd95',j,TKEState(s,s.h*k)),('linear',l,TKEState(s,s.h*k)),('roquet',r,sr)]
    results=dict(conversion_location='uniform lon=-30, lat=45; idealized reference water',cells=a.cells,layers=a.layers,dt=60.,mixing_substeps=4,ale_every=1,threads=1,dtype='float64',torch=torch.__version__,cases={n:dict(samples_s=[]) for n,_,_ in cases})
    def advance(m,s):
        end,report=m.step_with_report(s,60.,**f)
        return m.remap_with_report(end,end.ocean.h.sum(-1,keepdim=True).expand_as(end.ocean.h)/a.layers)[0]
    with torch.no_grad():
        for _,m,s in cases:advance(m,s);advance(m,s)
        for repeat in range(3):
            # Rotate ordering so one candidate is not always first/last.
            order=cases[repeat:]+cases[:repeat]
            for name,m,s in order:
                clock=time.perf_counter();out=advance(m,s);elapsed=time.perf_counter()-clock
                results['cases'][name]['samples_s'].append(elapsed)
                print(name,elapsed,flush=True)
                a.output.write_text(json.dumps(results,indent=2)+'\n')
    for row in results['cases'].values():row['median_step_s']=statistics.median(row['samples_s'])
    a.output.write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
