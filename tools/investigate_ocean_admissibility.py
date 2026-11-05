"""Actual-carrier transport restrictions; no limiter or timestep policy change.

The two manufactured cases exercise the production centered thickness flux and
its donor hT/hS/hq fluxes. Celsius T is signed. A positive h alone is insufficient
for admissible salinity or tracer bounds. All failures are retained as evidence.
"""
import argparse
import importlib.util
from pathlib import Path

import torch
from vfs.ocean import Ocean, OceanColumn
from vfs.ocean.conservative import ConstantDensityDynamics
from vfs.integrators.explicit import forward_euler, ssprk3

_spec=importlib.util.spec_from_file_location('williamson',Path(__file__).with_name('verify_ocean_williamson.py'))
w=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)
common=w.common


def carrier_rates(state):
    """Incoming/outgoing volume rates [cell, layer], m³/s, for actual I(h)u.

    Positive edge flux goes from cellsOnEdge[:,0] to [:,1]. These graph sums
    are an independent incidence audit of the production mesh divergence.
    They are not a replacement flux or an advective |u| dt/dx estimate.
    """
    mesh=state.mesh
    rate=mesh.dv_edge[:,None]*mesh.cell_to_edge(state.h)*state.u_e
    c0,c1=mesh.cellsOnEdge.unbind(1)
    positive=rate.clamp_min(0);negative=(-rate).clamp_min(0)
    outgoing=torch.zeros_like(state.h).index_add(0,c0,positive).index_add(0,c1,negative)
    incoming=torch.zeros_like(state.h).index_add(0,c1,positive).index_add(0,c0,negative)
    amount=mesh.cell_area[:,None]*state.h
    # A zero outgoing rate imposes no transport restriction; no artificial floor.
    bound=torch.where(outgoing>0,amount/torch.where(outgoing>0,outgoing,torch.ones_like(outgoing)),torch.inf)
    return incoming,outgoing,bound


def make_case(mesh, *, throughflow, stepper=forward_euler):
    """A 0.1 m donor amid 1000 m cells; one outflow, optional equal inflow.

    Incoming water has T=10 degC, S=0, q=0. Donor has T=-2 degC, S=35, q=1.
    This stress case is intentionally not an equilibrated ocean initial state.
    """
    column=OceanColumn(1,total_depth=1000.)
    model=Ocean(mesh,column,dynamics=ConstantDensityDynamics(omega=0.),stepper=stepper)
    h=torch.full((mesh.nCells,1),1000.);h[0]=.1
    temperature=torch.full_like(h,10.);temperature[0]=-2.
    salinity=torch.zeros_like(h);salinity[0]=35.
    tracer=torch.zeros_like(h);tracer[0]=1.
    u=torch.zeros(mesh.nEdges,1)
    e0,e1=map(int,mesh.edgesOnCell[0,:2])
    u[e0]=mesh.edgeSignOnCell[0,0]
    flux=mesh.dv_edge[e0]*mesh.cell_to_edge(h)[e0]
    if throughflow:
        u[e1]=-mesh.edgeSignOnCell[0,1]*flux/(mesh.dv_edge[e1]*mesh.cell_to_edge(h)[e1])
    state=model.state(u_e=u,h=h,temperature=temperature,salinity=salinity,tracer=tracer[...,None])
    return model,state


def stage_report(state,rhs,dt):
    """Inspect the actual Euler substep; no state copies/conversions in dynamics."""
    incoming,outgoing,bound=carrier_rates(state)
    area=state.mesh.cell_area[:,None]
    output=state+dt*rhs
    fraction=dt*outgoing/(area*state.h)
    residual=area*rhs.h-(incoming-outgoing)
    inventories={}
    for name in ('h','temp_h','salt_h','tracer_h'):
        value=getattr(rhs,name)
        local=value*(area if value.ndim==2 else area[...,None])
        inventories[name]=dict(rate=float(local.sum()),absolute_local_rate_scale=float(local.abs().sum()))
    ratios={}
    if bool((output.h>0).all()):
        for name,value in [('temperature_degC',output.temperature()),('salinity',output.salinity()),('tracer',output.tracers())]:
            ratios[name]=dict(minimum=float(value.min()),maximum=float(value.max()))
    return dict(dt_s=dt,transport_bound_s=None if bool(torch.isinf(bound.min())) else float(bound.min()),
        maximum_carrier_outflow_fraction=float(fraction.max()),
        minimum_euler_h_m=float(output.h.min()),all_finite=all(bool(torch.isfinite(getattr(output,k)).all()) for k in ('u_e','h','temp_h','salt_h','tracer_h')),
        incidence_residual_max_m3_s=float(residual.abs().max()),
        incidence_local_scale_m3_s=float((incoming+outgoing).max()),
        content_rates=inventories,euler_concentrations=ratios)


class RecordedSSPRK3:
    """Verification-only observer of the existing sequence, not a new integrator.

    Each provider call records its RHS and actual dt Euler substep. SSPRK3 still
    owns all stage/convex-combination arithmetic. No stage state is altered.
    """
    def __init__(self):
        self.stages=[]

    def __call__(self,provider,state,dt):
        records=self.stages
        class Observer:
            def tendency(self,value):
                rhs=provider.tendency(value)
                records.append(stage_report(value,rhs,dt))
                return rhs
        return ssprk3(Observer(),state,dt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=dict(mesh='archived ocean fixture',layers=1,thin_layer_m=.1,other_thickness_m=1000.,
                initial_temperature_degC=[-2.,10.],initial_salinity=[0.,35.],initial_tracer=[0.,1.],
                dt_fractions_of_initial_carrier_bound=[.1,.5,.9,1.1],scope='transport admissibility, not equilibrium or stability qualification')
    with common.recorded_run(args.repo,args.output,'OCEAN-TRANSPORT-ADMISSIBILITY',config,__file__,
        numerical_revision=common.NUMERICAL_COMMIT,structural_revision=common.STRUCTURAL_COMMIT,ocean_revision=common.OCEAN_COMMIT):
        for name in ('verify_ocean_williamson.py','capture_ocean.py'):
            (args.output/name).write_bytes(Path(__file__).with_name(name).read_bytes())
        inputs=Path(__file__).resolve().parents[1]/'tests/fixtures/ocean/inputs.npz'
        mesh,_=w.capture.load_inputs(inputs)
        common.write_json(args.output/'inputs.json',dict(fixture_sha256=common.sha256(inputs.read_bytes()),configuration=config))
        results=[]
        for throughflow in (False,True):
            for fraction in config['dt_fractions_of_initial_carrier_bound']:
                for method in ('Euler','SSPRK3'):
                    recorder=RecordedSSPRK3() if method=='SSPRK3' else None
                    model,state=make_case(mesh,throughflow=throughflow,stepper=recorder or forward_euler)
                    _,_,bound=carrier_rates(state)
                    dt=fraction*float(bound.min())
                    row=dict(throughflow=throughflow,method=method,initial_bound_fraction=fraction,
                             initial=stage_report(state,model.tendency(state),dt))
                    try:
                        final=model.step(state,dt)
                        row.update(status='completed',minimum_final_h_m=float(final.h.min()),
                            minimum_final_salinity=float(final.salinity().min()),minimum_final_tracer=float(final.tracers().min()),
                            final_temperature_range_degC=[float(final.temperature().min()),float(final.temperature().max())],
                            relative_energy_change=float((model.dynamics.total_energy(final)-model.dynamics.total_energy(state))/model.dynamics.total_energy(state)))
                    except (FloatingPointError,ValueError) as error:
                        row.update(status='rejected',error=str(error))
                    row['stages']=recorder.stages if recorder else [row['initial']]
                    results.append(row)
                    common.write_json(args.output/'results.json',results)
                    print(throughflow,method,fraction,row['status'],row.get('minimum_final_salinity'),flush=True)


if __name__=='__main__':
    main()
