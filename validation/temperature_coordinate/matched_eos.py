"""Matched buoyancy-anomaly EOS comparisons; CPU float64, no acceptance tuning.

JMD95 is a model comparison, not observational truth. Initialization preserves
h, velocity and salinity, adjusting linear T to match rho-rho_ref at midpoints.
The pressure dependence between those points and calorics remain different.
"""
import argparse,json,math,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ocean_column'))
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn
from vfs.ocean.jmd95_energy import jmd95_partials
from vfs.ocean.stratified import StratifiedOcean,TKEState
from linear import LinearOcean,LinearMixing,LinearHeating,LinearExchange,TemperatureBoundary
from system import TemperatureSystem
from path import TemperaturePathMixing
from surface import FastSurfaceHeating,FastSurfaceExchange


def anomaly(model,s):
    base=model.ocean.energy.base
    if isinstance(model,LinearOcean):return base.density(s)-base.rho0
    z=base.height(s);p=-base.rho0*base.gravity*z/1e4
    return jmd95_partials(s.salinity(),s.temperature(),p)[0]-jmd95_partials(s.salinity()*0+base.S0,s.temperature()*0+base.T0,p)[0]


def matched(linear,jmd,state):
    s=state.ocean;c=linear.config
    t=c.T0+(c.beta*(s.salinity()-c.S0)-anomaly(jmd,s)/linear.ocean.column.rho0)/c.alpha
    out=TKEState(s.with_tensors(s.u_e,s.h,s.h*t,s.salt_h,s.tracer_h),state.tke_h.clone())
    linear.check(out);return out


def rms(x,w):return float(((w*x.square()).sum()/w.expand_as(x).sum()).sqrt())


def build(kind,cells,layers,compiled):
    radius=1e5 if kind=='wave' else 6371220.
    mesh=VoronoiMesh(n_cells=cells,radius=radius,lloyd_iters=10,rotation_scheme='compatible',verbose=False)
    layers=2 if kind=='wave' else layers
    depth=3000. if kind in ('wave','jet') else 100.
    column=OceanColumn(layers,total_depth=depth)
    o=StratifiedOcean(mesh,column,warm_start=True)
    if compiled:o.compile_thermodynamics()
    physics={} if kind!='mixing' else dict(closure=TemperaturePathMixing(2.,substeps=8),boundary=TemperatureBoundary(.0025),heating=FastSurfaceHeating(.6,(.5,20.)))
    jmd=TemperatureSystem(o,**physics)
    physics={} if kind!='mixing' else dict(closure=LinearMixing(2.,substeps=8),boundary=TemperatureBoundary(.0025),heating=LinearHeating(.6,(.5,20.)))
    lin=LinearOcean(mesh,column,**physics)
    omega=0. if kind in ('wave','mixing') else 7.292e-5
    for m in (jmd,lin):m.ocean.base.dynamics.mechanical.rotation_rate=omega
    h0=torch.full((layers,),depth/layers);h=h0[None].expand(cells,layers).clone()
    t=torch.linspace(17.,4.,layers).expand_as(h);salt=torch.linspace(34.,35.5,layers).expand_as(h)
    u=torch.zeros(mesh.nEdges,layers);meta={};forcing={}
    if kind=='wave':
        h0=torch.tensor([500.,2500.]);h=h0[None].expand(cells,2).clone()
        g,rho=column.gravity,column.rho0
        def f(k,z):
            p=torch.tensor(-rho*g*z/1e4)
            return g/rho*(jmd95_partials(salt[0,k],t[0,k],p)[0]-jmd95_partials(torch.tensor(35.),torch.tensor(10.),p)[0])
        a=g+f(0,0.);b=a-f(0,-500.)+f(1,-500.)
        mat=torch.stack((torch.stack((a,a)),torch.stack((a,b))))
        values,vectors=torch.linalg.eig(torch.diag(h0)@mat);ix=values.real.argmin()
        vector=vectors[:,ix].real;vector/=vector.abs().max();frequency=math.sqrt(2*float(values[ix].real))/radius
        mode=mesh.cell_xyz[:,2:3];area=mesh.cell_area[:,None];mode=mode-(area*mode).sum()/area.sum()
        h=h+.1*mode*vector
        meta=dict(period_s=2*math.pi/frequency,omega=frequency,fast_speed=math.sqrt(float(values.real.max())),vector=vector.tolist())
    elif kind=='jet':
        # Solid-body barotropic flow with flat internal interfaces; continuum
        # pressure balance uses upper-layer surface buoyancy. No discrete balancing.
        speed=5.;g=column.gravity;rho=column.rho0
        f=g/rho*(jmd95_partials(salt[0,0],t[0,0],torch.tensor(0.))[0]-jmd95_partials(torch.tensor(35.),torch.tensor(10.),torch.tensor(0.))[0])
        eta=-(radius*omega*speed+.5*speed**2)/(g+f)*mesh.cell_xyz[:,2:3].square()
        h[:,0]+=eta[:,0]
        pole=mesh.edge_xyz.new_tensor([0.,0.,1.]).expand_as(mesh.edge_xyz)
        u=((speed*torch.cross(pole,mesh.edge_xyz,dim=-1))*mesh.edge_normal).sum(-1,keepdim=True).expand(mesh.nEdges,layers).clone()
        meta=dict(speed_m_s=speed,reference='continuum solid-body gradient-wind balance with flat internal interfaces')
    else:
        z=h.cumsum(-1)-h/2;t=10+5*(1-torch.tanh((z-20)/5))/2;salt=35-.1*(t-10)
        # Uniform horizontally; edge-normal projection of an eastward shear.
        e=mesh.edge_xyz;east=torch.stack((-e[:,1],e[:,0],e[:,0]*0),-1);east/=east.norm(dim=-1,keepdim=True)
        u=.004*torch.exp(-(torch.arange(layers)+.5)[None]*depth/layers/25)*(east*mesh.edge_normal).sum(-1,keepdim=True)
        forcing=dict(surface_stress=.05*(east*mesh.edge_normal).sum(-1),shortwave=torch.full((cells,),200.),surface_heat=torch.full((cells,),-250.))
    physical=o.base.state(u_e=u,h=h,temperature=t,salinity=salt,tracer=torch.ones_like(h)[...,None])
    start=TKEState(physical,h*(1e-4 if kind=='mixing' else 0.))
    return jmd,lin,start,matched(lin,jmd,start),h0,meta,forcing


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=['wave','jet','mixing']);p.add_argument('--cells',type=int,default=64);p.add_argument('--layers',type=int,default=16)
    p.add_argument('--dt',type=float,default=600.);p.add_argument('--hours',type=float,default=24.);p.add_argument('--periods',type=float,default=1.);p.add_argument('--compile',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    jmd,lin,sj,sl,h0,meta,forcing=build(a.kind,a.cells,a.layers,a.compile)
    area=sj.ocean.mesh.cell_area[:,None];edge=(sj.ocean.mesh.dc_edge*sj.ocean.mesh.dv_edge)[:,None];mass=area*jmd.ocean.column.rho0
    duration=meta['period_s']*a.periods if a.kind=='wave' else a.hours*3600
    limit=.3*float(sj.ocean.mesh.dc_edge.min())/meta['fast_speed'] if a.kind=='wave' else a.dt
    steps=math.ceil(duration/min(limit,a.dt));dt=duration/steps
    report=dict(config={**vars(a),'output':str(a.output)},metadata=meta,steps=steps,dt_s=dt,torch=torch.__version__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),initial_anomaly_mismatch_kg_m3=float((anomaly(jmd,sj.ocean)-anomaly(lin,sl.ocean)).abs().max()),initial_temperature_difference_rms_C=rms(sl.ocean.temperature()-sj.ocean.temperature(),area),cases={})
    outputs={}
    def save():a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with torch.no_grad():
        for name,model,start in [('jmd95',jmd,sj),('linear',lin,sl)]:
            state=start;energy=model.total_energy(start);totalmass=(mass*start.ocean.h).sum();external=0.;rows=[];failure=None;completed=0;clock=time.perf_counter()
            rhs,_=model.dynamics.tendency_with_report(start.ocean,dt)
            initial_accel=rms(rhs.u_e,edge)
            for n in range(steps+1):
                if n%max(1,steps//64)==0 or n==steps:
                    s=state.ocean;row=dict(time_s=n*dt,energy_minus_inputs_J_kg=float((model.total_energy(state)-energy-external)/totalmass),velocity_change_rms=rms(s.u_e-start.ocean.u_e,edge))
                    if a.kind=='wave':
                        mode=s.mesh.cell_xyz[:,2:3];mode=mode-(area*mode).sum()/area.sum();v=s.h.new_tensor(meta['vector']);norm=(area*mode.square()).sum()*v.square().sum()
                        rhs,_=model.dynamics.tendency_with_report(s,dt);amp=(area*mode*(s.h-h0)*v).sum()/norm;rate=(area*mode*rhs.h*v).sum()/norm
                        row.update(amplitude_m=float(amp),phase=float(torch.atan2(-rate/meta['omega'],amp)),envelope=float(torch.sqrt(amp.square()+(rate/meta['omega']).square())/.1))
                    if a.kind=='mixing':
                        # This is an isolated column-physics comparison: no horizontal dynamics/ALE.
                        density=anomaly(model,s);contrast=density-density[:,:1];z=s.h.cumsum(-1)-s.h/2
                        mld=torch.where(contrast>.03,z,torch.full_like(z,100.)).min(-1).values
                        row.update(mld_density_003_mean_m=float((area[:,0]*mld).sum()/area.sum()),surface_temperature_C=float(s.temperature()[:,0].mean()),min_tke=float((state.tke_h/s.h).min()))
                    rows.append(row)
                if n==steps:break
                try:
                    if a.kind=='mixing':
                        s,k,br=model.boundary.apply(model.facade,state.ocean,state.tke_h,dt,surface_stress=forcing['surface_stress'],_physical=state.ocean)
                        s,hr=model.heating.apply(model.facade,s,dt,shortwave=forcing['shortwave'],surface_heat=forcing['surface_heat'],_physical=s)
                        s,k,mr=model.closure.apply(model.ocean,s,k,dt);state=TKEState(s,k);external+=br.surface_work+hr.external_heat
                    else:state,_=model.step_with_report(state,dt)
                    completed=n+1
                except (ValueError,FloatingPointError) as exc:failure=str(exc);break
            s=state.ocean
            item=dict(status='rejected' if failure else 'completed',failure=failure,completed_steps=completed,wall_s=time.perf_counter()-clock,initial_acceleration_rms_m_s2=initial_accel,samples=rows,energy_minus_inputs_J_kg=float((model.total_energy(state)-energy-external)/totalmass),salt_relative_change=float((mass*(s.salt_h-start.ocean.salt_h)).sum()/(mass*start.ocean.salt_h).sum()),min_tke=float((state.tke_h/s.h).min()))
            if a.kind=='wave' and completed==steps:
                item['frequency_over_reference']=float(np.polyfit([r['time_s'] for r in rows],np.unwrap([r['phase'] for r in rows]),1)[0]/meta['omega'])
            if a.kind=='mixing':item['final_profiles']={key:value.mean(0).tolist() for key,value in dict(temperature=s.temperature(),salinity=s.salinity(),density_anomaly=anomaly(model,s),tke=state.tke_h/s.h).items()}
            report['cases'][name]=item;outputs[name]=state;save();print(name,item['status'],completed,flush=True)
        if all(x['status']=='completed' for x in report['cases'].values()):
            x,y=outputs['jmd95'].ocean,outputs['linear'].ocean
            report['final_difference']=dict(velocity_rms_m_s=rms(x.u_e-y.u_e,edge),thickness_rms_m=rms(x.h-y.h,area),density_anomaly_rms_kg_m3=rms(anomaly(jmd,x)-anomaly(lin,y),area),temperature_rms_C=rms(x.temperature()-y.temperature(),area))
        save()
    return int(any(x['status']!='completed' for x in report['cases'].values()))
if __name__=='__main__':raise SystemExit(main())
