"""JW-A t=0 rotation-only comparisons on an unchanged dry atmospheric state.

No production buffers are replaced and no timestep is taken. Matrix variants
are diagnostic ablations of the rotation stencil with the same fixed-vorticity
energy-work pairing as the production dynamics.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import sparse
import torch
from vfs import VoronoiMesh, make_hybrid_coefficients
from vfs.atmosphere.conservative import DryHydrostaticConservativeDynamics
from vfs.atmosphere.initial_conditions.jablonski_williamson import JWInitialConditions

spec=importlib.util.spec_from_file_location('rotation_pv_audit',Path(__file__).with_name('audit_rotation_pv.py'))
a=importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def paired_rotation(J,u,q,metric):
    """Apply fixed-vorticity skew pairing independently, one vertical level at a time."""
    result=np.empty_like(u)
    for k in range(u.shape[1]):
        Q=sparse.diags(q[:,k])
        A=-.5*(Q@J+J@Q)
        result[:,k]=.5*(A@u[:,k]-(A.T@(metric[:,k]*u[:,k]))/metric[:,k])
    return result


def measure(mesh,levels=32,p_top=5000.,sigma_power=2.):
    dyn=DryHydrostaticConservativeDynamics()
    A,B=make_hybrid_coefficients(levels,p_top=p_top,sigma_power=sigma_power)
    column=JWInitialConditions.make_column_for_mode(mesh,levels,A,B,
        mode='constant_ps_surface_phi',phi_full_mode='direct_u_gl2_adjoint')
    state,info=JWInitialConditions(mesh,column,mode='constant_ps_surface_phi').make_state(
        dyn,perturb=False,project_mass_flux=False,theta_source='phi_derivative',
        wind_source='U_jw',theta_ic_mode='point',return_info=True)
    with torch.no_grad():
        terms=dyn.tendency_terms(state)
        Hu=a.array(dyn.kinetic_energy_gradient_u(state))
        gm,gp=dyn.thermal_gradients(state)
        db=state.column.B_interface[1:]-state.column.B_interface[:-1]
        gp=gp+(mesh.cell_area[:,None]*db*dyn.cell_kinetic_energy(state)/dyn.gravity).sum(-1,keepdim=True)
        scalar_work=float((gm*terms['mtheta_i']['total']).sum()+(gp*terms['ps']['total']).sum())
        scalar_abs=float((gm*terms['mtheta_i']['total']).abs().sum()+(gp*terms['ps']['total']).abs().sum())
    momentum={k:a.array(v) for k,v in terms['u_e'].items()}
    velocity=a.array(state.u_e)
    edgearea=a.array(mesh.dc_edge*mesh.dv_edge)[:,None]
    dpedge=a.array(mesh.cell_to_edge(state.layer_dp()))
    metric=edgearea*dpedge/dyn.gravity
    vort=a.array(mesh.curl(state.u_e)+dyn.rotation_rate*mesh.coriolis_unit_vertex[:,None])
    q=vort[a.array(mesh.verticesOnEdge)].mean(1)
    _,C,raw,stored=a.operators(mesh)
    kite,P,_,_=a.transfers(mesh)
    index=a.path_rotation(mesh,a.array(mesh.thuburn_corner_fraction),shift=-1)
    geometric=a.path_rotation(mesh,kite/a.array(mesh.cell_area)[:,None],shift=-1)
    W=sparse.diags(a.array(mesh.dc_edge*mesh.dv_edge))
    Winv=sparse.diags(1/a.array(mesh.dc_edge*mesh.dv_edge))
    geometric_skew=.5*(geometric-Winv@geometric.T@W)
    rotations={'production':momentum['rotation']}
    for label,J in [('stored_reassembled',stored),('raw_port',raw),('index_only',index),
                    ('index_and_intersection',geometric),('index_intersection_skew',geometric_skew)]:
        rotations[label]=paired_rotation(J,velocity,q,metric)
    def norms(x):
        return {'mass_weighted_rms_m_s2':float(np.sqrt(np.sum(metric*x*x)/np.sum(metric))),
                'edge_area_weighted_rms_m_s2':float(np.sqrt(np.sum(edgearea*x*x)/(edgearea.sum()*levels))),
                'unweighted_rms_m_s2':float(np.sqrt(np.mean(x*x))),
                'max_abs_m_s2':float(np.max(np.abs(x))),
                'per_level_area_rms_m_s2':np.sqrt(np.sum(edgearea*x*x,axis=0)/edgearea.sum()).tolist()}
    fixed_balance=momentum['pgf']+momentum['bernoulli']
    fixed_full=fixed_balance+momentum['vertical']
    results={}
    for label,rot in rotations.items():
        total=fixed_full+rot
        work=Hu*rot
        allwork=Hu*total
        results[label]={'horizontal_balance_residual':norms(fixed_balance+rot),
                        'full_momentum_residual':norms(total),'rotation':norms(rot),
                        'rotation_change':norms(rot-momentum['rotation']),
                        'rotation_work_W':float(work.sum()),
                        'rotation_work_relative':float(work.sum()/max(np.abs(work).sum(),1e-300)),
                        'closed_energy_rate_W':float(allwork.sum()+scalar_work),
                        'closed_energy_rate_relative':float((allwork.sum()+scalar_work)/(np.abs(allwork).sum()+scalar_abs))}
    base=results['production']['full_momentum_residual']['mass_weighted_rms_m_s2']
    for r in results.values():
        r['full_mass_rms_change_fraction']=r['full_momentum_residual']['mass_weighted_rms_m_s2']/base-1
    # Record max-residual location and strong-flow relative scale explicitly.
    loc=np.unravel_index(np.abs(momentum['total']).argmax(),velocity.shape)
    return {'cells':mesh.nCells,'levels':levels,'radius_m':mesh.radius,
            'p_top_Pa':p_top,'sigma_power':sigma_power,'initial_condition':asdict(info),
            'state_sha256':{key:hashlib.sha256(a.array(value).tobytes()).hexdigest()
                            for key,value in [('u_e',state.u_e),('ps',state.ps),('mtheta_i',state.mtheta_i)]},
            'initial_ranges':{'wind_max_m_s':float(state.u_e.abs().max()),
                              'theta_min_K':float(state.theta().min()),'theta_max_K':float(state.theta().max()),
                              'ps_min_Pa':float(state.ps.min()),'ps_max_Pa':float(state.ps.max())},
            'current_max_residual_location':{'edge':int(loc[0]),'level_zero_based':int(loc[1]),
                'latitude_deg':float(torch.rad2deg(torch.asin(mesh.edge_xyz[loc[0],2]))),
                'full_pressure_Pa':float(state.full_pressure()[0,loc[1]])},
            'term_norms':{k:norms(momentum[k]) for k in ('pgf','bernoulli','vertical')},
            'unchanged_ps_tendency_rms_Pa_s':float(terms['ps']['total'].square().mean().sqrt()),
            'variants':results}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--levels',type=int,default=32)
    p.add_argument('--p-top',type=float,default=5000.)
    p.add_argument('--sigma-power',type=float,default=2.)
    args=p.parse_args()
    xyz=np.load(args.mesh,allow_pickle=False)
    mesh=VoronoiMesh(n_cells=len(xyz),lloyd_iters=0,generator_xyz=xyz)
    result=measure(mesh,args.levels,args.p_top,args.sigma_power)
    result.update(torch=torch.__version__,dtype='float64',device='cpu',
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  geometry_audit_driver_sha256=hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest(),
                  input_npy_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for name,r in result['variants'].items():
        print(name,r['full_momentum_residual']['mass_weighted_rms_m_s2'],r['full_momentum_residual']['max_abs_m_s2'],r['rotation_work_relative'],flush=True)

if __name__=='__main__':main()
