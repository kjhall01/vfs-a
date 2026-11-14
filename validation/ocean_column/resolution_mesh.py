"""Native-mesh resolution campaign; fixed physical profiles and explicit failures.

Only vertical mixing advances. Spherical mesh generation is outside the timer.
Results contain compact metrics, not enormous state-array uploads.
"""
import argparse,hashlib,json,time,resource
from pathlib import Path
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean,TKEOcean
from lagged import SubcycledLaggedClosure
from consistent import ConsistentPathClosure

def setup(cells,layers,patch=False,quadrature_order=4):
    mesh=VoronoiMesh(n_cells=cells,lloyd_iters=0,rotation_scheme='compatible')
    options={} if quadrature_order==4 else {'quadrature_order':quadrature_order}
    ocean=StratifiedOcean(mesh,OceanColumn(layers,total_depth=100.),**options)
    r=torch.linspace(0,1,layers+1,dtype=torch.float64)[None]
    interfaces=100*(r+.15*mesh.cell_xyz[:,:1]*torch.sin(2*torch.pi*r)/(2*torch.pi))
    h=interfaces[:,1:]-interfaces[:,:-1];z=(interfaces[:,1:]+interfaces[:,:-1])/2
    he=mesh.cell_to_edge(h);ze=he.cumsum(-1)-he/2
    t=10+2.5*(1-torch.tanh((z-20)/5));s=35-.1*(t-10)
    u=.004*torch.exp(-ze/25)*(1+.2*mesh.edge_xyz[:,2:3])
    state=ocean.state(u_e=u,h=h,temperature=t,salinity=s,tracer=(1+torch.sin(z/30))[...,None])
    k=torch.full_like(h,1e-4)
    if patch:k=torch.where(mesh.cell_xyz[:,2:3]>.95,k*1e-6,k)
    return ocean,state,k

def simulate(ocean,initial,k,method,duration):
    if method.startswith('coupled'):
        steps=int(method[len('coupled'):]);closure=ConsistentPathClosure(2.)
    else:steps=1;closure=SubcycledLaggedClosure(2.,substeps=int(method[len('semi'):]))
    model=TKEOcean(ocean,closure=closure);state=model.state(initial,tke=k)
    area=ocean.energy.mass_area(initial);mass=area*initial.h;start_energy=model.budgets(state)['total']
    counts={};accepted=0;failure=None;local_max=0.;start=time.perf_counter();cpu=time.process_time()
    for _ in range(steps):
        try:out,report=model.mix_with_report(state,duration/steps)
        except (ValueError,FloatingPointError) as exc:failure=str(exc);break
        state=out;accepted+=1;local_max=max(local_max,float(report.budget_error))
        for name,value in report.mixing_limiter.items():
            v=float(value);counts[name]=min(counts.get(name,1.),v) if name=='minimum_fraction' else counts.get(name,0.)+v
    cpu=time.process_time()-cpu;wall=time.perf_counter()-start
    row=dict(method=method,status='rejected' if failure else 'completed',failure=failure,completed_seconds=accepted*duration/steps,requested_seconds=duration,cpu_s=cpu,wall_s=wall,counts=counts,max_local_budget_ratio=local_max,energy_error_J_kg=float((model.budgets(state)['total']-start_energy)/mass.sum()),min_tke=float(state.specific_tke().min()),min_tracer=float(state.ocean.tracer_h.min()),salt_relative_change=float((area*(state.ocean.salt_h-initial.salt_h)).sum()/(area*initial.salt_h).sum()),passive_relative_change=float((area[...,None]*(state.ocean.tracer_h-initial.tracer_h)).sum()/(area[...,None]*initial.tracer_h).sum()))
    return row,state

def difference(ocean,state,ref):
    mass=ocean.energy.mass_area(state.ocean)*state.ocean.h
    metric=ocean.energy.base.mechanical.metric(ocean.check_state(state.ocean))
    a=ocean.check_state(state.ocean);b=ocean.check_state(ref.ocean)
    return dict(velocity_relative_l2=float(torch.sqrt((metric*(state.ocean.u_e-ref.ocean.u_e)**2).sum()/(metric*ref.ocean.u_e**2).sum())),tke_relative_l2=float(torch.sqrt((mass*(state.specific_tke()-ref.specific_tke())**2).sum()/(mass*ref.specific_tke()**2).sum())),temperature_rms_C=float(torch.sqrt((mass*(a.temperature()-b.temperature())**2).sum()/mass.sum())))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cells',type=int,default=128);p.add_argument('--layers',type=int,default=32);p.add_argument('--duration',type=float,default=300.);p.add_argument('--patch',action='store_true');p.add_argument('--methods',nargs='+',default=['coupled16','coupled1','semi4','semi8']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);results=dict(cells=a.cells,layers=a.layers,patch=a.patch,duration=a.duration,torch=torch.__version__,threads=1,geometry='Fibonacci Voronoi, zero Lloyd iterations, compatible rotation geometry',profiles='fixed functions of physical depth; no horizontal stepping',source_hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('resolution_mesh.py','lagged.py','coupled.py','path_work.py')},cases=[])
    with torch.no_grad():
        start=time.perf_counter();o,initial,k=setup(a.cells,a.layers,a.patch);results['setup_wall_s']=time.perf_counter()-start
        ref=None
        for method in a.methods:
            row,state=simulate(o,initial,k,method,a.duration)
            if method=='coupled16' and row['status']=='completed':ref=state
            elif ref is not None and row['status']=='completed':row['versus_coupled16']=difference(o,state,ref)
            results['cases'].append(row);results['peak_rss_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            a.output.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(row),flush=True)
if __name__=='__main__':main()
