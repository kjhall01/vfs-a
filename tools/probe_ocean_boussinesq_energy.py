"""Design probe only: linear-Boussinesq layer energy and matched content work.

No runtime model or timestep is added. Existing production tendencies supply
continuity, passive-content transport, and constant-density mechanical work.
The proposed extra force is analytic; AD/FD only check independent derivatives.
"""
import argparse
import importlib.util
from pathlib import Path

import torch
from vfs.ocean import Ocean, OceanColumn, OceanState
from vfs.ocean.conservative import ConstantDensityDynamics
from vfs.ocean.dynamics import MultilayerOceanDynamics
from vfs.ocean.constants import OCEAN_ALPHA_T, OCEAN_BETA_S, OCEAN_T0, OCEAN_S0

_spec=importlib.util.spec_from_file_location('williamson',Path(__file__).with_name('verify_ocean_williamson.py'))
w=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(w)
common=w.common


def anomaly_energy(state, alpha=OCEAN_ALPHA_T, beta=OCEAN_BETA_S):
    """g sum A (delta-rho h) zbar, J; z is upward from fixed mean sea level.

    Layers run top to bottom. Theta=hT and Sigma=hS are held fixed for h
    derivatives. Linear EOS makes anomaly amount affine in these three contents.
    No pressure dependence, mixing, remap or thermodynamic heat budget is implied.
    """
    rho0,g=state.column.rho0,state.column.gravity
    aT,aS=-rho0*alpha,rho0*beta
    a0=-aT*OCEAN_T0-aS*OCEAN_S0
    amount=a0*state.h+aT*state.temp_h+aS*state.salt_h  # kg/m²
    lower_and_self=state.h.flip(-1).cumsum(-1).flip(-1)
    zbar=-state.column.total_depth_like(state.h)+lower_and_self-.5*state.h
    area=state.mesh.cell_area[:,None]
    energy=(g*area*amount*zbar).sum()
    # Increasing h_k lifts every overlying layer j<k by dh_k, and its own
    # midpoint by dh_k/2. This is the fixed-content derivative, not fixed T/S.
    above=amount.cumsum(-1)-amount
    gradients={'h':g*area*(a0*zbar+above+.5*amount),
               'temp_h':g*area*aT*zbar,'salt_h':g*area*aS*zbar}
    return energy,gradients


def paired_force(state, gradients):
    """Analytic extra edge acceleration, m/s², matched to actual donor traces.

    H_dot_scalar = rho0 sum A_e F [G b_h + T_up G b_T + S_up G b_S].
    Since H_u=rho0 A_e F, minus that bracket cancels the extra scalar work.
    This is a pairing proposal, not evidence of hydrostatic accuracy or PV closure.
    """
    mesh=state.mesh
    normalization=state.column.rho0*mesh.cell_area[:,None]
    c0,c1=mesh.cellsOnEdge.unbind(1)
    upwind=lambda q:torch.where(state.u_e>=0,q[c0],q[c1])
    return (-mesh.grad(gradients['h']/normalization)
            -upwind(state.temperature())*mesh.grad(gradients['temp_h']/normalization)
            -upwind(state.salinity())*mesh.grad(gradients['salt_h']/normalization))


def active_case(mesh,fields):
    column=OceanColumn(3,total_depth=3000.)
    h=fields['h'];u=fields['u_e']
    layer=torch.arange(3)[None,:]
    T=12.-3*layer+.5*mesh.cell_xyz[:,:1]
    S=34.+.4*layer+.1*mesh.cell_xyz[:,1:2]
    model=Ocean(mesh,column,dynamics=ConstantDensityDynamics())
    state=model.state(u_e=u,h=h,temperature=T,salinity=S,tracer=torch.ones_like(h)[...,None])
    return model,state


def work_probe(model,state):
    fields=('u_e','h','temp_h','salt_h')
    leaves=[getattr(state,k).clone().requires_grad_() for k in fields]
    state=state.with_tensors(*leaves,state.tracer_h)
    anomaly,grad=anomaly_energy(state)
    total=model.dynamics.total_energy(state)+anomaly
    ad=torch.autograd.grad(total,leaves)
    analytic=model.dynamics.energy.gradients(state)
    for name in grad:analytic[name]=analytic[name]+grad[name]
    rhs=model.tendency(state)
    extra=paired_force(state,grad)
    proposed_u=rhs.u_e+extra
    local=[ad[0]*proposed_u]+[derivative*getattr(rhs,name) for name,derivative in zip(fields[1:],ad[1:])]
    rate=sum(x.sum() for x in local);scale=sum(x.abs().sum() for x in local)
    # Quantify why simply attaching the old baroclinic pressure is not justified.
    legacy=MultilayerOceanDynamics()
    graft=-legacy.pressure_gradient(state)+legacy.gravity*state.mesh.grad(state.eta())
    wrong=[ad[0]*(rhs.u_e+graft)]+local[1:]
    wrong_rate=sum(x.sum() for x in wrong);wrong_scale=sum(x.abs().sum() for x in wrong)
    report=dict(anomaly_energy_J=float(anomaly),total_mechanical_energy_J=float(total),
        derivative_relative_errors={name:float((analytic[name]-value).abs().max()/value.abs().max()) for name,value in zip(fields,ad)},
        proposed_work_W=float(rate),absolute_local_work_scale_W=float(scale),normalized_proposed_work=float(rate/scale),
        archived_pressure_graft_work_W=float(wrong_rate),archived_pressure_graft_normalized_work=float(wrong_rate/wrong_scale),
        interpretation='proposal algebra only; no new runtime or timestep')
    fd=[]
    for name,native_scale in [('h',1.),('temp_h',10.),('salt_h',10.)]:
        source=getattr(state,name)
        direction=torch.sin(torch.arange(source.numel()).reshape_as(source)*.37+.2)*native_scale
        predicted=(grad[name]*direction).sum()
        for epsilon in (1e-1,1e-2,1e-3,1e-4):
            values=[]
            for sign in (-1,1):
                arrays={k:getattr(state,k) for k in w.capture.FIELDS}
                arrays[name]=source+sign*epsilon*direction
                changed=state.with_tensors(*(arrays[k] for k in w.capture.FIELDS))
                values.append(anomaly_energy(changed)[0])
            derivative=(values[1]-values[0])/(2*epsilon)
            fd.append(dict(field=name,direction_scale_native=native_scale,epsilon=epsilon,
                           analytic_J=float(predicted),fd_J=float(derivative),relative_error=float(abs((derivative-predicted)/predicted))))
    report['anomaly_finite_differences']=fd
    return report


def rest_probe(mesh, *, topography):
    depth=3000.+(200*mesh.cell_xyz[:,:1] if topography else torch.zeros(mesh.nCells,1))
    column=OceanColumn(3,total_depth=depth,rest_thickness=torch.tensor([500.,700.,1800.]))
    h=torch.cat([torch.full_like(depth,500.),torch.full_like(depth,700.),depth-1200.],-1)
    model=Ocean(mesh,column,dynamics=ConstantDensityDynamics())
    state=model.state(u_e=torch.zeros(mesh.nEdges,3),h=h,
        temperature=torch.tensor([15.,10.,5.]).expand_as(h),salinity=torch.full_like(h,35.))
    _,grad=anomaly_energy(state)
    force=model.tendency(state).u_e+paired_force(state,grad)
    weights=(mesh.dc_edge*mesh.dv_edge)[:,None].expand_as(force)
    return dict(topography=topography,force_m_s2=w.norms(force,weights),
                maximum_eta_m=float(state.eta().abs().max()),
                construction='horizontal upper interfaces, topographic thickness only in bottom constant-density layer')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=dict(scope='unimplemented active-energy design; no timestep',layers=3,alpha_K_inverse=OCEAN_ALPHA_T,beta_inverse_salinity=OCEAN_BETA_S,T_reference_degC=OCEAN_T0,S_reference=OCEAN_S0)
    with common.recorded_run(args.repo,args.output,'OCEAN-BOUSSINESQ-DESIGN',config,__file__,numerical_revision=common.NUMERICAL_COMMIT,structural_revision=common.STRUCTURAL_COMMIT,ocean_revision=common.OCEAN_COMMIT):
        for name in ('verify_ocean_williamson.py','capture_ocean.py'):
            (args.output/name).write_bytes(Path(__file__).with_name(name).read_bytes())
        inputs=Path(__file__).resolve().parents[1]/'tests/fixtures/ocean/inputs.npz'
        mesh,fields=w.capture.load_inputs(inputs)
        model,state=active_case(mesh,fields)
        zero,zero_gradient=anomaly_energy(state,alpha=0.,beta=0.)
        result=dict(input_fixture_sha256=common.sha256(inputs.read_bytes()),configuration=config,
                    work=work_probe(model,state),rest=[rest_probe(mesh,topography=x) for x in (False,True)],
                    constant_density_limit=dict(energy_J=float(zero),force_max_m_s2=float(paired_force(state,zero_gradient).abs().max())))
        common.write_json(args.output/'results.json',result)
        print(result['work']['normalized_proposed_work'],result['rest'],flush=True)


if __name__=='__main__':
    main()
