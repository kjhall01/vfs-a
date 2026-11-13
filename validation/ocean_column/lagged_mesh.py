"""Native spherical C-grid vertical-mixing cost/accuracy, no horizontal stepping."""
import argparse,json,time
from pathlib import Path
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean,TKEOcean
from lagged import SubcycledLaggedClosure
from consistent import ConsistentPathClosure

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=float,default=.001);p.add_argument('--speed',type=float,default=.2);p.add_argument('--duration',type=float,default=120.);p.add_argument('--cells',type=int,nargs='+',default=[42,162]);p.add_argument('--methods',nargs='+');a=p.parse_args();torch.set_num_threads(1);rows=[]
 with torch.no_grad():
  for cells in a.cells:
   mesh=VoronoiMesh(n_cells=cells,lloyd_iters=0,rotation_scheme='compatible');o=StratifiedOcean(mesh,OceanColumn(8,total_depth=100.))
   h=torch.full((cells,8),12.5,dtype=torch.float64)*(1+.15*mesh.cell_xyz[:,:1]*torch.cos(torch.arange(8,dtype=torch.float64))[None]);h=h*100/h.sum(-1,keepdim=True)
   z=h.cumsum(-1)-h/2;t=10+5*.5*(1-torch.tanh((z-20)/5));s=35-.1*(t-10)
   u=a.speed*torch.exp(-torch.arange(8,dtype=h.dtype)[None]/2)*(1+.2*mesh.edge_xyz[:,2:3])
   initial=o.state(u_e=u,h=h,temperature=t,salinity=s,tracer=(1+torch.sin(z/30))[...,None])
   mass=o.energy.mass_area(initial)*h;metric=o.energy.base.mechanical.metric(o.check_state(initial));outputs={}
   for name,closure,steps in [('reference',ConsistentPathClosure(2.),16),('coupled',ConsistentPathClosure(2.),1),('coupled2',ConsistentPathClosure(2.),2),('lagged1',SubcycledLaggedClosure(2.,substeps=1),1),('lagged4',SubcycledLaggedClosure(2.,substeps=4),1),('lagged8',SubcycledLaggedClosure(2.,substeps=8),1),('lagged16',SubcycledLaggedClosure(2.,substeps=16),1)]:
    if a.methods and name!='reference' and name not in a.methods:continue
    m=TKEOcean(o,closure=closure);q=m.state(initial,tke=h*0+a.seed);energy0=m.budgets(q)['total'];trials=0;limited=0;deferred=0;start=time.perf_counter();cpu=time.process_time()
    for _ in range(steps):
     q,r=m.mix_with_report(q,a.duration/steps);d=r.mixing_limiter
     trials+=float(d.get('semi_trials',d.get('trial_evaluations',r.iterations)));limited+=float(d.get('limited_interfaces',0));deferred+=float(d.get('deferred_columns',0))
    cpu=time.process_time()-cpu;wall=time.perf_counter()-start;outputs[name]=q
    row=dict(cells=cells,layers=8,method=name,cpu_s=cpu,wall_s=wall,trials=trials,limited_interfaces=limited,deferred_columns=deferred,energy_error_J_kg=float((m.budgets(q)['total']-energy0)/mass.sum()),min_tke=float(q.specific_tke().min()),min_tracer=float(q.ocean.tracer_h.min()))
    if name!='reference':
     ref=outputs['reference'];p=o.check_state(q.ocean);pr=o.check_state(ref.ocean)
     row.update(velocity_relative_l2=float(torch.sqrt((metric*(q.ocean.u_e-ref.ocean.u_e)**2).sum()/(metric*ref.ocean.u_e**2).sum())),temperature_rms_C=float(torch.sqrt((mass*(p.temperature()-pr.temperature())**2).sum()/mass.sum())),tke_relative_l2=float(torch.sqrt((mass*(q.specific_tke()-ref.specific_tke())**2).sum()/(mass*ref.specific_tke()**2).sum())))
    rows.append(row);print(json.dumps(row),flush=True)
 a.output.write_text(json.dumps(dict(cases=rows,seed=a.seed,speed=a.speed,duration=a.duration,note='Vertical-mixing-only comparison. Numerical reference uses 16 coupled substeps. No horizontal dycore evolution. Timings include concurrent jobs; CPU and wall both reported.'),indent=2)+'\n')
if __name__=='__main__':main()
