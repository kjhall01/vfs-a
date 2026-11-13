"""Offline f-plane column campaign using installed VFS production physics.

The two-component metric adapter is ONLY a column harness, not a dycore grid.
No horizontal transport/pressure gradient or production rotation is exercised.
"""
import argparse,json,math,time,sys,hashlib
from dataclasses import replace
from pathlib import Path
import torch
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import (StratifiedOcean,TKEOcean,TKEClosure,
    OceanBoundaryStress,OceanSurfaceHeating,OceanSurfaceExchange,OceanSurfaceFluxes)

class ColumnMetric:
    """One m² column: two native velocity DOFs u,v with metric rho*h each."""
    def __init__(self):
        self.nCells=1;self.nEdges=2
        self.cell_xyz=torch.tensor([[0.,0.,1.]],dtype=torch.float64)
        self.edge_xyz=self.cell_xyz.expand(2,3)
        self.edge_normal=torch.tensor([[1.,0.,0.],[0.,1.,0.]],dtype=torch.float64)
        self.cell_area=torch.ones(1,dtype=torch.float64)
        self.dc_edge=self.dv_edge=torch.ones(2,dtype=torch.float64)
        self.cellsOnEdge=torch.zeros(2,2,dtype=torch.int64)
        self.edge_side_area=torch.full((2,2),.5,dtype=torch.float64)
    def cell_to_edge(self,x):return x.expand(2,*x.shape[1:])
    def cell_wind_vector(self,u):return torch.stack((u[0],u[1],torch.zeros_like(u[0])), -1)[None]

def rotate(state,angle):
    u,v=state.ocean.u_e
    c,s=math.cos(angle),math.sin(angle)
    return replace(state,ocean=replace(state.ocean,u_e=torch.stack((c*u+s*v,-s*u+c*v))))

def run(layers=16,dt=300.,hours=48.,scenario='wind',seed=1e-4,mix_scalars=True,safety=.5,progress=False,checkpoint=None,wall_limit=None):
    if not math.isfinite(dt) or not math.isfinite(hours) or dt<=0 or hours<=0:
        raise ValueError("dt and requested hours must be finite and positive")
    if wall_limit is not None and (checkpoint is None or not math.isfinite(wall_limit) or wall_limit<=0):
        raise ValueError('wall_limit requires a checkpoint and a finite positive duration')
    torch.set_num_threads(1)
    mesh=ColumnMetric();depth=100.;rho=1025.;f=1e-4
    ocean=StratifiedOcean(mesh,OceanColumn(layers,total_depth=depth))
    model=TKEOcean(ocean,closure=TKEClosure(2.,mix_scalars=mix_scalars,mixing_backtrack_safety=safety),boundary=OceanBoundaryStress(),
        heating=OceanSurfaceHeating(.6,(.5,20.)),exchange=OceanSurfaceExchange())
    h=torch.full((1,layers),depth/layers,dtype=torch.float64)
    z=h.cumsum(-1)-h/2
    upper=.5*(1-torch.tanh((z-20.)/5.))
    temperature=10+5*upper;salt=35-.5*upper
    state=model.state(ocean.state(u_e=torch.zeros(2,layers,dtype=h.dtype),h=h,
        temperature=temperature,salinity=salt,tracer=upper[...,None]),tke=torch.full_like(h,seed))
    initial=model.budgets(state);mass=float((ocean.energy.mass_area(state.ocean)*h).sum())
    external=0.;elapsed=0.;logs=[];status='completed';failure=None;wall=time.monotonic()
    limiter_totals={}; limited_steps=0; accepted_steps=0; prior_wall=0.
    parameters=dict(layers=layers,dt=dt,hours=hours,scenario=scenario,seed=seed,mix_scalars=mix_scalars,safety=safety)
    import vfs.ocean.stratified.tke as tke_module
    runtime_hash=hashlib.sha256(Path(tke_module.__file__).read_bytes()).hexdigest()
    checkpoint=Path(checkpoint) if checkpoint is not None else None
    if checkpoint is not None and checkpoint.exists():
        saved=torch.load(checkpoint,weights_only=True)
        if saved['parameters']!=parameters or saved['runtime_hash']!=runtime_hash:
            raise ValueError('checkpoint parameters or runtime differ')
        state=replace(state,ocean=replace(state.ocean,**saved['ocean']),tke_h=saved['tke_h'])
        elapsed=saved['elapsed'];external=saved['external'];logs=saved['logs']
        limiter_totals=saved['limiter'];limited_steps=saved['limited_steps'];accepted_steps=saved['accepted_steps'];prior_wall=saved['wall_s']
    def save_checkpoint():
        if checkpoint is None:return
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        payload=dict(parameters=parameters,runtime_hash=runtime_hash,
            ocean={k:getattr(state.ocean,k) for k in ('u_e','h','salt_h','entropy_h','tracer_h')},
            tke_h=state.tke_h,elapsed=elapsed,external=external,logs=logs,limiter=limiter_totals,
            limited_steps=limited_steps,accepted_steps=accepted_steps,wall_s=prior_wall+time.monotonic()-wall)
        temp=checkpoint.with_suffix('.tmp');torch.save(payload,temp);temp.replace(checkpoint)
    def profile():
        p=ocean.check_state(state.ocean);k=state.specific_tke();t=p.temperature();sp=p.salinity()
        # Explicit temperature-threshold diagnostic; not a density-based MLD.
        d=state.ocean.h.cumsum(-1)-state.ocean.h/2
        valid=(t>=t[:,:1]-.2)[0]
        # Contiguous from surface, never include a disconnected deeper layer.
        count=int(valid.to(torch.int64).cumprod(0).sum())
        return dict(time_s=elapsed,limiter=dict(limiter_totals),u=state.ocean.u_e[0].tolist(),v=state.ocean.u_e[1].tolist(),
            temperature=t[0].tolist(),salinity=sp[0].tolist(),tke=k[0].tolist(),
            diffusivity=(.1*2*torch.sqrt(k))[0].tolist(),depth=d[0].tolist(),
            temperature_mld_m=float(state.ocean.h[0,:count].sum()),
            energy_minus_input_J_kg=float((model.budgets(state)['total']-initial['total']-external)/mass))
    if not logs:logs.append(profile())
    next_log=(math.floor(elapsed/3600)+1)*3600.
    with torch.no_grad():
        while elapsed<hours*3600-1e-9:
            step=min(dt,hours*3600-elapsed)
            # Each attempt is transactional: only commit after all operators pass.
            try:
                trial=rotate(state,f*step/2)
                trial,b=model.boundary_with_report(trial,step,surface_stress=h.new_tensor([.1,0.]))
                heat=0.;water=0.
                if scenario=='cooling':
                    trial,r=model.heat_with_report(trial,step,surface_heat=h.new_tensor([-100.]))
                    heat=float(r.external_heat)
                elif scenario=='rain_heating':
                    # Rain 3.6 mm/hr for first 6h, thereafter no rain; radiation persists.
                    rate=.001 if elapsed<6*3600 else 0.
                    flux=OceanSurfaceFluxes(precipitation=h.new_tensor([rate]),
                        water_energy_flux=h.new_tensor([rate*50000.]),water_momentum_flux=h.new_zeros(2))
                    trial,r=model.exchange_with_report(trial,step,surface_fluxes=flux)
                    water=float(r.external_energy)
                    trial,r=model.heat_with_report(trial,step,shortwave=h.new_tensor([200.]),surface_heat=h.new_tensor([-50.]))
                    heat=float(r.external_heat)
                trial,mix=model.mix_with_report(trial,step)
                trial=rotate(trial,f*step/2)
            except (ValueError,FloatingPointError) as exc:
                status='rejected';failure=dict(time_s=elapsed,attempt_dt_s=step,message=str(exc));break
            state=trial;elapsed+=step;external+=float(b.surface_work)+heat+water
            accepted_steps+=1
            limited_steps+=int(float(mix.mixing_limiter['limited_pairs'])>0)
            for key,value in mix.mixing_limiter.items():
                v=float(value)
                limiter_totals[key]=min(limiter_totals.get(key,1.),v) if key=='minimum_fraction' else limiter_totals.get(key,0.)+v
            if elapsed>=next_log-1e-9:
                logs.append(profile());next_log+=3600.
                save_checkpoint()
                if progress: print(json.dumps(dict(hours=elapsed/3600,limited_steps=limited_steps,limiter=limiter_totals)),file=sys.stderr,flush=True)
                if wall_limit is not None and time.monotonic()-wall >= wall_limit and elapsed<hours*3600:
                    status='checkpointed';break
    if logs[-1]['time_s']!=elapsed:logs.append(profile())
    final=model.budgets(state)
    transport=(state.ocean.h*state.ocean.u_e).sum(-1)
    # Wind-only, closed stress-free bottom: viscosity cannot change depth integral.
    exact=.1/rho*(1-complex(math.cos(f*elapsed),-math.sin(f*elapsed)))/(1j*f)
    return dict(layers=layers,dt_s=dt,requested_hours=hours,completed_hours=elapsed/3600,
        limiter=limiter_totals,limited_steps=limited_steps,accepted_steps=accepted_steps,backtrack_safety=safety,
        scenario=scenario,seed_tke=seed,mix_scalars=mix_scalars,mixing_length_m=2.,status=status,failure=failure,
        wall_s=prior_wall+time.monotonic()-wall,external_energy_J_m2=external,
        energy_error_J_kg=float((final['total']-initial['total']-external)/mass),
        mass_residual_kg_m2=float(rho*(final['volume']-initial['volume']))-(min(elapsed,21600)*.001 if scenario=='rain_heating' else 0.),
        salt_relative_change=float((final['salt']-initial['salt'])/initial['salt']),
        passive_relative_change=float(((final['tracers']-initial['tracers'])/initial['tracers']).abs().max()),
        transport=[float(x) for x in transport],
        analytic_wind_transport=[exact.real,exact.imag] if scenario!='rain_heating' else None,
        profiles=logs)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--layers',type=int,default=16);p.add_argument('--dt',type=float,default=300)
    p.add_argument('--hours',type=float,default=48);p.add_argument('--scenario',choices=['wind','cooling','rain_heating'],default='wind');p.add_argument('--output',required=True);p.add_argument('--seed',type=float,default=1e-4);p.add_argument('--momentum-only',action='store_true');p.add_argument('--safety',type=float,default=.5);p.add_argument('--progress',action='store_true');p.add_argument('--checkpoint');p.add_argument('--wall-limit',type=float)
    a=p.parse_args();r=run(a.layers,a.dt,a.hours,a.scenario,seed=a.seed,mix_scalars=not a.momentum_only,safety=a.safety,progress=a.progress,checkpoint=a.checkpoint,wall_limit=a.wall_limit)
    if r['status']!='checkpointed':
        output=Path(a.output);temp=output.with_suffix(output.suffix+'.tmp')
        temp.write_text(json.dumps(r,indent=2)+'\n');temp.replace(output)
    print(json.dumps({k:v for k,v in r.items() if k!='profiles'}))
