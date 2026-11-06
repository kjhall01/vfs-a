"""Sharp-front measurements; preserve negative results, no accuracy thresholds."""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

s=importlib.util.spec_from_file_location('front_qualification',Path(__file__).with_name('qualify_ocean_entropy.py'))
q=importlib.util.module_from_spec(s);sys.modules[s.name]=q;s.loader.exec_module(q)
front=q.p.load('ocean_front_candidate','ocean_front_research.py')


def audit(kind, coast=False):
    model,start,energy=q.p.case(kind,coast)
    rhs,w=front.evaluate(model,start,10.,energy)
    rate,scale=front.contraction(energy.gradients(start),rhs)
    end,stages=front.step(model,start,10.,energy)
    mass=energy.mass_area(start)
    produced=10*sum(a*(mass*l.entropy_source).sum() for a,l in zip((1/6,1/6,2/3),stages))
    entropy=(mass*(end.entropy_h-start.entropy_h)).sum()
    return dict(kind=kind,coast=coast,normalized_spatial_work=float(rate/scale),
        entropy_produced_J_K=float(produced),entropy_budget_relative_residual=float((entropy-produced)/(mass*start.entropy_h.abs()).sum()),
        work_limited_faces=int((w.work_fraction<1).sum()),**q.budget(energy,start,end))


def zero_edge():
    model,state,energy=q.p.case('upper_salt')
    _,w=front.evaluate(model,state,10.,energy)
    index=int(w.force_correction.abs().flatten().argmax())
    rows=[]
    for speed in [1e-3,1e-6,1e-9,1e-12,0.]:
        for sign in [-1,1]:
            u=state.u_e.clone();u.flatten()[index]=sign*speed
            rhs,w=front.evaluate(model,replace(state,u_e=u),10.,energy)
            rows.append(dict(velocity_m_s=sign*speed,force_m_s2=float(w.force_correction.flatten()[index]),
                work_limited_faces=int((w.work_fraction<1).sum()),other_salt_limited_faces=int((w.transport.beta[:,1]<1).sum())))
    return rows


def trajectory(scheme,dt,duration):
    mesh=q.VoronoiMesh(n_cells=32,radius=1e5,lloyd_iters=2,verbose=False)
    model=q.Ocean(mesh,q.OceanColumn(3,total_depth=3000.),dynamics=q.JMD95Dynamics())
    x,y=mesh.cell_xyz[:,:1],mesh.cell_xyz[:,1:2]
    h=torch.tensor([500.,1000.,1500.])[None,:].expand(32,3).clone()
    physical=model.state(u_e=.03*torch.sin(torch.arange(mesh.nEdges)[:,None]*.3+torch.arange(3)[None,:]),h=h,
        temperature=torch.tensor([17.,10.,4.])[None,:]+torch.where(y>0,2.,-2.),
        salinity=torch.full_like(h,35.)+torch.where(x>0,1.,-1.),tracer=(x>0).to(h.dtype).expand_as(h)[...,None])
    reference=q.p.previous.reference();state=front.EntropyState.from_temperature(physical,reference);start=state
    energy=front.EntropyEnergy(model.dynamics.energy,reference)
    solver=front if scheme=='front' else q.c
    count=round(duration/dt);assert count*dt==duration
    production=0.;guarded=0;rows=[]
    for step in range(count+1):
        if step%max(1,count//12)==0 or step==count:
            physical=state.physical_state(reference)
            rows.append(dict(time_s=step*dt,temperature_min=float(physical.temperature().min()),temperature_max=float(physical.temperature().max()),
                passive_min=float(state.tracer_h.div(state.h[...,None]).min()),passive_max=float(state.tracer_h.div(state.h[...,None]).max()),**q.budget(energy,start,state)))
        if step<count:
            state,ledgers=solver.step(model,state,dt,energy)
            production+=dt*sum(a*float((energy.mass_area(state)*l.entropy_source).sum()) for a,l in zip((1/6,1/6,2/3),ledgers))
            if scheme=='front':guarded+=sum(int((l.work_fraction<1).sum()) for l in ledgers)
    actual=float((energy.mass_area(start)*(state.entropy_h-start.entropy_h)).sum())
    return dict(scheme=scheme,cells=32,radius_m=1e5,dt_s=dt,duration_s=duration,steps=count,samples=rows,
        scope='fully coupled active three-layer front; no exact trajectory reference; no long-run qualification',
        guarded_face_stages=guarded,entropy_produced_J_K=production,
        entropy_budget_relative_residual=(actual-production)/float((energy.mass_area(start)*start.entropy_h.abs()).sum()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trajectory',action='store_true')
    parser.add_argument('--scheme',choices=['front','parent'],default='front')
    parser.add_argument('--dt',type=float,default=30.)
    parser.add_argument('--duration',type=float,default=7200.)
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args();torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    report=dict(status='running',parent_commit='bf5ad5cb8887ec05d26e0566dde0c636daf0676a',device='cpu',dtype='float64',torch=torch.__version__,
        source_sha256={Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [__file__,front.__file__]})
    try:
        with torch.no_grad():
            if args.trajectory:report['trajectory']=trajectory(args.scheme,args.dt,args.duration)
            else:
                report['audits']=[audit(k,c) for k in ['front','tilted','near_fresh','upper_salt'] for c in [False,True]]
                report['single_edge_zero']=zero_edge();report['transport']=[]
                q.c=front
                for profile in ['smooth','sharp']:
                    for n in ([32] if args.quick else [32,64,128]):
                        row={};q.transport(SimpleNamespace(cells=n,profile=profile,circuits=1,cfl=.2),row,lambda:None)
                        row['cells']=n;report['transport'].append(row)
            report['status']='completed'
    except Exception as error:
        report.update(status='rejected',error_type=type(error).__name__,error=str(error));raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
