"""Qualification measurements for the frozen #36 candidate; no accuracy gates.

Full nonlinear wave/balance trajectories and isolated prescribed-carrier
transport. Each case writes a small JSON, including partial failures and
provenance. No field arrays are uploaded. CPU float64 only.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import Ocean, OceanColumn, OceanState
from vfs.ocean.jmd95 import JMD95Dynamics
from vfs.ocean.eos import jmd95_seawater_density

spec=importlib.util.spec_from_file_location('qualification_parent',Path(__file__).with_name('probe_ocean_entropy.py'))
p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)
c=p.candidate


def rms(x,w):
    return float(((w*x.square()).sum()/(w.expand_as(x).sum())).sqrt())


def wave_setup(n):
    mesh=VoronoiMesh(n_cells=n,radius=1e5,lloyd_iters=10,verbose=False)
    h0=torch.tensor([500.,2500.]);t=torch.tensor([17.,4.]);s=torch.tensor([34.,35.5])
    model=Ocean(mesh,OceanColumn(2,total_depth=3000.,rest_thickness=h0),dynamics=JMD95Dynamics(omega=0.))
    g,rho=model.dynamics.gravity,model.dynamics.rho0
    def f(k,z):
        pressure=torch.tensor(-rho*g*z/1e4)
        return g/rho*(jmd95_seawater_density(s[k],t[k],pressure)-jmd95_seawater_density(torch.tensor(35.),torch.tensor(10.),pressure))
    a=g+f(0,0.);b=a-f(0,-500.)+f(1,-500.)
    matrix=torch.stack([torch.stack([a,a]),torch.stack([a,b])])
    values,vectors=torch.linalg.eig(torch.diag(h0)@matrix)
    index=values.real.argmin();vector=vectors[:,index].real;vector=vector/vector.abs().max()
    omega=math.sqrt(2*float(values[index].real))/mesh.radius
    area=mesh.cell_area[:,None];mode=mesh.cell_xyz[:,2:3];mode=mode-(area*mode).sum()/area.sum()
    physical=model.state(u_e=torch.zeros(mesh.nEdges,2),h=h0[None,:]+.1*mode*vector[None,:],temperature=t.expand(n,2),salinity=s.expand(n,2))
    ref=p.previous.reference();state=c.EntropyState.from_temperature(physical,ref)
    return model,state,c.EntropyEnergy(model.dynamics.energy,ref),h0,mode,vector,omega,math.sqrt(float(values.real.max()))


def budget(energy,start,state):
    area=state.mesh.cell_area[:,None];mass=energy.mass_area(state)
    return dict(energy_change_J_m2=float(p.energy_change(energy,start,state)/area.sum()),
        relative_volume_change=float((area*(state.h-start.h)).sum()/(area*start.h).sum()),
        relative_salt_change=float((area*(state.salt_h-start.salt_h)).sum()/(area*start.salt_h).sum()),
        entropy_change_J_K_m2=float((mass*(state.entropy_h-start.entropy_h)).sum()/area.sum()),
        minimum_h_m=float(state.h.min()),salinity_min=float((state.salt_h/state.h).min()),salinity_max=float((state.salt_h/state.h).max()))


def wave(args,report,save):
    model,start,energy,h0,mode,vector,omega,fast=wave_setup(args.cells)
    period=2*math.pi/omega;end=args.periods*period
    dt=args.cfl*float(model.mesh.dc_edge.min())/fast
    steps=math.ceil(end/dt);dt=end/steps
    interval=max(1,round(steps/(48*args.periods)))
    area=model.mesh.cell_area[:,None];norm=(area*mode.square()).sum()*vector.square().sum()
    report.update(period_s=period,dt_s=dt,planned_steps=steps,periods=args.periods,external_cfl=args.cfl,samples=[])
    state=start
    for k in range(steps+1):
        if k%interval==0 or k==steps:
            rhs,work=c.evaluate(model,state,dt,energy)
            amplitude=(area*mode*(state.h-h0)*vector).sum()/norm
            speed=(area*mode*rhs.h*vector).sum()/norm
            phase=math.atan2(float(-speed/omega),float(amplitude))
            expected=.1*mode*vector*math.cos(omega*k*dt)
            report['samples'].append(dict(time_s=k*dt,modal_amplitude_m=float(amplitude),modal_velocity_m_s=float(speed),
                phase_rad=phase,envelope_over_initial=float(torch.sqrt(amplitude.square()+(speed/omega).square())/.1),
                h_error_rms_m=rms(state.h-h0-expected,area),
                off_mode_h_rms_m=rms(state.h-h0-amplitude*mode*vector,area),**budget(energy,start,state)))
            report['completed_steps']=k;save()
            if len(report['samples'])%12==0: print(json.dumps(dict(kind='wave',cells=args.cells,completed_periods=k*dt/period)),flush=True)
        if k<steps: state,_=c.step(model,state,dt,energy)
    times=np.array([r['time_s'] for r in report['samples']]);phase=np.unwrap([r['phase_rad'] for r in report['samples']])
    envelope=np.array([r['envelope_over_initial'] for r in report['samples']])
    report['measured_frequency_over_reference']=float(np.polyfit(times,phase,1)[0]/omega)
    report['envelope_log_slope_per_period']=float(np.polyfit(times/period,np.log(envelope),1)[0])
    report['final_phase_error_rad']=float(phase[-1]-omega*end)


def balance(args,report,save):
    mesh=VoronoiMesh(n_cells=args.cells,lloyd_iters=20,verbose=False)
    if args.profile=='jet':
        w=p.load('qualification_williamson','verify_ocean_williamson.py')
        model=Ocean(mesh,OceanColumn(1,total_depth=w.H0,gravity=w.GRAVITY),dynamics=JMD95Dynamics(gravity=w.GRAVITY,omega=w.OMEGA))
        physical=w.initial_state(model)
        reference_terms=w.initial_balance(w.make_model(mesh),physical)[0]
        report['constant_density_operator_diagnosis']=reference_terms
    else:
        bottom=3000+200*mesh.cell_xyz[:,:1] if args.profile=='topography' else torch.full((mesh.nCells,1),3000.)
        h=torch.cat([torch.full_like(bottom,500.),torch.full_like(bottom,1000.),bottom-1500.],-1)
        model=Ocean(mesh,OceanColumn(3,total_depth=bottom,rest_thickness=torch.tensor([500.,1000.,1500.])),dynamics=JMD95Dynamics())
        physical=model.state(u_e=torch.zeros(mesh.nEdges,3),h=h,temperature=torch.tensor([17.,10.,4.]).expand_as(h),salinity=torch.tensor([34.,35.,35.5]).expand_as(h))
    ref=p.previous.reference();start=c.EntropyState.from_temperature(physical,ref);energy=c.EntropyEnergy(model.dynamics.energy,ref)
    steps=math.ceil(args.days*86400/args.dt);dt=args.days*86400/steps;state=start
    area=mesh.cell_area[:,None];edge=(mesh.dc_edge*mesh.dv_edge)[:,None]
    rhs,_=c.evaluate(model,start,dt,energy)
    report.update(dt_s=dt,planned_steps=steps,days=args.days,profile=args.profile,initial_momentum_rms_m_s2=rms(rhs.u_e,edge),initial_h_rate_rms_m_s=rms(rhs.h,area),samples=[])
    interval=max(1,steps//28)
    for k in range(steps+1):
        if k%interval==0 or k==steps:
            report['samples'].append(dict(time_s=k*dt,u_error_rms_m_s=rms(state.u_e-start.u_e,edge),h_error_rms_m=rms(state.h-start.h,area),max_velocity_m_s=float(state.u_e.abs().max()),**budget(energy,start,state)))
            report['completed_steps']=k;save()
            print(json.dumps(dict(kind='balance',profile=args.profile,cells=args.cells,day=k*dt/86400)),flush=True)
        if k<steps:state,_=c.step(model,state,dt,energy)


def transport(args,report,save):
    ring=p.previous.previous.transport_probe.RingMesh(args.cells)
    model=SimpleNamespace(mesh=ring,domain=None,dynamics=SimpleNamespace(min_thickness=1e-3))
    column=OceanColumn(1,total_depth=1.);ref=p.previous.reference()
    x=(torch.arange(args.cells)+.5)/args.cells
    nodes,weights=np.polynomial.legendre.leggauss(12);nodes=torch.tensor(nodes);weights=torch.tensor(weights/2)
    def exact(time):
        points=(x[:,None]+nodes[None,:]/(2*args.cells)-time)%1
        q=.5+.4*torch.sin(2*math.pi*points) if args.profile=='smooth' else (points<.5).to(points.dtype)
        salt=34+2*q;theta=10+2*q
        eta=ref.evaluate(salt,theta)['entropy']
        return (eta*weights).sum(-1),(salt*weights).sum(-1),(q*weights).sum(-1)
    eta,salt,q=exact(0);h=torch.ones(args.cells,1)
    start=c.EntropyState(torch.ones(args.cells,1),h,eta[:,None],salt[:,None],q[:,None,None],ring,column)
    steps=math.ceil(args.circuits*args.cells/args.cfl);dt=args.circuits/steps;state=start
    def euler(s):
        physical=s.physical_state(ref)
        i,j=ring.cellsOnEdge.unbind(1)
        eh,sh,info=p.faces.gibbs_face(ref,physical.salinity()[i],physical.temperature()[i],physical.salinity()[j],physical.temperature()[j])
        ledger=c.bounded_transport(model,s,dt,eh,sh,info['composition_entropy'])
        values=ledger.amounts/ring.cell_area[:,None,None]
        return replace(s,h=ledger.volume/ring.cell_area[:,None],entropy_h=values[...,0],salt_h=values[...,1],tracer_h=values[...,2:])
    for _ in range(steps):
        a=euler(state);b=.75*state+.25*euler(a);state=(1/3)*state+(2/3)*euler(b)
    exact_eta,exact_salt,exact_q=exact(args.circuits)
    report.update(completed_steps=steps,dt_s=dt,circuits=args.circuits,profile=args.profile,scope='isolated prescribed-carrier scalar fluxes; momentum/heating disabled')
    report['fields']={}
    for name,value,target,initial in [('entropy',state.entropy_h[:,0],exact_eta,eta),('salinity',state.salt_h[:,0],exact_salt,salt),('passive',state.tracer_h[:,0,0],exact_q,q)]:
        report['fields'][name]=dict(l1=float((value-target).abs().mean()),rms=float((value-target).square().mean().sqrt()),minimum=float(value.min()),maximum=float(value.max()),
            undershoot=float((initial.min()-value.min()).clamp_min(0)),overshoot=float((value.max()-initial.max()).clamp_min(0)),inventory_change=float(value.mean()-initial.mean()),total_variation=float((value-torch.roll(value,1)).abs().sum()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('kind',choices=['wave','balance','transport']);parser.add_argument('--cells',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--periods',type=float,default=3);parser.add_argument('--cfl',type=float,default=.2);parser.add_argument('--profile',default='jet');parser.add_argument('--days',type=float,default=7);parser.add_argument('--dt',type=float,default=600);parser.add_argument('--circuits',type=float,default=1)
    args=parser.parse_args();torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    report=dict(candidate_commit='705720c79e38c2f07de39d45a33c9ede50707b87',kind=args.kind,cells=args.cells,status='running',torch=torch.__version__,dtype='float64',device='cpu',
        source_sha256={Path(f).name:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [__file__,c.__file__,p.faces.__file__]})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    start=time.monotonic()
    try:
        with torch.no_grad(): globals()[args.kind](args,report,save)
        report['status']='completed'
    except Exception as error:
        report.update(status='rejected',error_type=type(error).__name__,error=str(error));raise
    finally:
        report['wall_seconds']=time.monotonic()-start;save()

if __name__=='__main__':main()
