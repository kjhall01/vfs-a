"""Roquet full-step example, cold/contrast budgets, timing and matched wave.

GSW is used only when --wave matches the previous JMD95 initial buoyancy;
ordinary warm/cold/contrast cases initialize native CT/SA without GSW.
"""
import argparse,json,math,statistics,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ocean_column'))
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import TKEState,OceanSurfaceFluxes
from roquet import RoquetOcean,RoquetMixing,RoquetHeating,RoquetExchange,TemperatureBoundary,convert_jmd95_initialization


def model(mesh,column,substeps,physics=True):
    return RoquetOcean(mesh,column,**(dict(closure=RoquetMixing(2.,substeps=substeps),boundary=TemperatureBoundary(.0025),heating=RoquetHeating(.6,(.5,20.)),exchange=RoquetExchange()) if physics else {}))


def setup(cells,layers,profile,substeps):
    mesh=VoronoiMesh(n_cells=cells,lloyd_iters=0,rotation_scheme='compatible',verbose=False)
    m=model(mesh,OceanColumn(layers,total_depth=100.),substeps)
    r=torch.linspace(0,1,layers+1)[None];interfaces=100*(r+.15*mesh.cell_xyz[:,:1]*torch.sin(2*torch.pi*r)/(2*torch.pi))
    h=interfaces[:,1:]-interfaces[:,:-1];depth=(interfaces[:,1:]+interfaces[:,:-1])/2
    amplitude,offset={'warm':(5.,10.),'cold':(2.,-1.5),'contrast':(30.,0.)}[profile]
    t=offset+amplitude*(1-torch.tanh((depth-20)/5))/2
    sa=35-.1*(t-offset);he=mesh.cell_to_edge(h);ze=he.cumsum(-1)-he/2
    u=.004*torch.exp(-ze/25)*(1+.2*mesh.edge_xyz[:,2:3])
    s=m.state(u_e=u,h=h,conservative_temperature=t,absolute_salinity=sa,tracer=(1+torch.sin(depth/30))[...,None],tke=1e-4)
    cell=lambda x:torch.full_like(h[:,0],x)
    forcing=dict(surface_stress=.05*torch.ones_like(u[:,0]),shortwave=cell(200),surface_heat=cell(-50),surface_fluxes=OceanSurfaceFluxes(precipitation=cell(1e-5),evaporation=cell(5e-6),sensible_heat=cell(-10),latent_heat=cell(-12.5),water_energy_flux=cell(.25),water_momentum_flux=torch.zeros_like(u[:,0])))
    return m,s,forcing,{}


def wave_setup(cells):
    from matched_eos import build,anomaly
    j,_,state,_,h0,meta,_=build('wave',cells,2,False);s=state.ocean
    m=model(s.mesh,s.column,1,False);m.ocean.base.dynamics.mechanical.rotation_rate=0.
    z=m.ocean.energy.base.height(s);xyz=s.mesh.cell_xyz
    # A 100-km idealized sphere has no real-world geographic composition map.
    # Use one declared reference-water location, preserving horizontal uniformity.
    lon=torch.full_like(xyz[:,:1],-30.);lat=torch.full_like(xyz[:,2:3],45.)
    converted=convert_jmd95_initialization(potential_temperature_ipts68=s.temperature(),practical_salinity=s.salinity(),pressure_dbar=-s.column.rho0*s.column.gravity*z/1e4,longitude=lon,latitude=lat)
    sa=converted['absolute_salinity'];c=m.config;target=anomaly(j,s)
    d0=c.t+2*c.tt*c.CT0-c.tz*z;rhs=target-c.s*(sa-c.SA0)
    disc=d0.square()+4*c.tt*rhs
    if (disc<=0).any():raise ValueError('no matching density root')
    t=c.CT0+2*rhs/(d0-torch.sqrt(disc))
    out=m.state(u_e=s.u_e,h=s.h,conservative_temperature=t,absolute_salinity=sa)
    meta.update(conversion_location='uniform lon=-30 deg, lat=45 deg; idealized reference water',initial_density_anomaly_mismatch=float((m.config.anomaly(t,sa,z)-target).abs().max()),initial_CT_adjustment_from_GSW_rms=float((t-converted['conservative_temperature']).square().mean().sqrt()),rest_thickness=h0.tolist())
    return m,out,{},meta


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cells',type=int,default=128);p.add_argument('--layers',type=int,default=16)
    p.add_argument('--profile',choices=['warm','cold','contrast'],default='warm');p.add_argument('--steps',type=int,default=10);p.add_argument('--dt',type=float,default=60.);p.add_argument('--mixing-substeps',type=int,default=4)
    p.add_argument('--wave',action='store_true');p.add_argument('--ale',action='store_true');p.add_argument('--repeats',type=int,default=3);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if min(a.cells,a.layers,a.steps,a.dt,a.repeats,a.mixing_substeps)<=0:p.error('positive counts and dt required')
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    m,initial,f,meta=wave_setup(a.cells) if a.wave else setup(a.cells,a.layers,a.profile,a.mixing_substeps)
    if a.wave:
        limit=.3*float(m.ocean.mesh.dc_edge.min())/meta['fast_speed'];a.steps=math.ceil(meta['period_s']/limit);a.dt=meta['period_s']/a.steps
    s=initial.ocean;area=m.energy.mass_area(s);mass=(area*s.h).sum();before=m.total_energy(initial)
    report=dict(config={**vars(a),'output':str(a.output)},metadata=meta,torch=torch.__version__,source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('roquet.py','roquet_campaign.py')},status='running',samples=[])
    def save():a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def advance(state):
        out,r=m.step_with_report(state,a.dt,**f)
        external=0. if a.wave else r.boundary.surface_work+r.heating.external_heat+r.exchange.external_energy
        if a.ale:out,_=m.remap_with_report(out,out.ocean.h.sum(-1,keepdim=True).expand_as(out.ocean.h)/out.ocean.h.shape[1])
        return out,external,r
    state=initial;external=0.;completed=0;budget=0.;times=[]
    with torch.no_grad():
        try:
            if not a.wave:
                advance(initial);advance(initial)
                for _ in range(a.repeats):
                    clock=time.perf_counter();advance(initial);times.append(time.perf_counter()-clock)
            for n in range(a.steps+1):
                if n%max(1,a.steps//48)==0 or n==a.steps:
                    st=state.ocean;row=dict(time_s=n*a.dt,energy_minus_inputs_J_kg=float((m.total_energy(state)-before-external)/mass),min_tke=float((state.tke_h/st.h).min()))
                    if a.wave:
                        weights=st.mesh.cell_area[:,None];mode=st.mesh.cell_xyz[:,2:3];mode=mode-(weights*mode).sum()/weights.sum()
                        v=st.h.new_tensor(meta['vector']);h0=st.h.new_tensor(meta['rest_thickness']);norm=(weights*mode.square()).sum()*v.square().sum()
                        rhs,_=m.dynamics.tendency_with_report(st,a.dt);amp=(weights*mode*(st.h-h0)*v).sum()/norm;rate=(weights*mode*rhs.h*v).sum()/norm
                        row.update(amplitude_m=float(amp),phase=float(torch.atan2(-rate/meta['omega'],amp)),envelope=float(torch.sqrt(amp.square()+(rate/meta['omega']).square())/.1))
                    report['samples'].append(row)
                if n==a.steps:break
                state,added,r=advance(state);external+=added;completed=n+1
                if r.turbulence is not None:budget=max(budget,float(r.turbulence.budget_error))
            report['status']='completed'
        except (ValueError,FloatingPointError) as exc:report.update(status='rejected',error=str(exc))
        st=state.ocean
        report.update(completed_steps=completed,median_step_s=statistics.median(times) if times else None,timings_s=times,max_local_budget_ratio=budget,energy_minus_inputs_J_kg=float((m.total_energy(state)-before-external)/mass),salt_relative_change=float((area*(st.salt_h-s.salt_h)).sum()/(area*s.salt_h).sum()),min_CT=float(st.temperature().min()),max_CT=float(st.temperature().max()),min_SA=float(st.salinity().min()),min_tke=float((state.tke_h/st.h).min()))
        if a.wave and completed==a.steps:report['frequency_over_jmd95_reference']=float(np.polyfit([r['time_s'] for r in report['samples']],np.unwrap([r['phase'] for r in report['samples']]),1)[0]/meta['omega'])
        save();print(json.dumps({k:v for k,v in report.items() if k not in ('samples','source_sha256','metadata')}),flush=True)
    return int(report['status']!='completed')
if __name__=='__main__':raise SystemExit(main())
