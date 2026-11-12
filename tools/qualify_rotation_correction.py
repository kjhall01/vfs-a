"""Runtime correction checks: frozen-mesh JW-A, uniform PV and short evolution."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from vfs import VoronoiMesh, make_hybrid_coefficients
from vfs.atmosphere.conservative import DryHydrostaticConservativeDynamics
from vfs.atmosphere.initial_conditions.jablonski_williamson import JWInitialConditions
from vfs.integrators.explicit import ssprk3


def load(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def main():
    p=argparse.ArgumentParser();p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=10)
    p.add_argument('--dt',type=float,default=60.);args=p.parse_args()
    pv=load('audit_rotation_pv');jwa=load('audit_jwa_rotation_balance')
    result={'torch':torch.__version__,'dtype':'float64','device':'cpu',
            'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_npy_sha256':hashlib.sha256(args.mesh.read_bytes()).hexdigest(),'cases':{}}
    xyz=np.load(args.mesh,allow_pickle=False)
    for scheme in ('legacy','compatible'):
        mesh=VoronoiMesh(n_cells=len(xyz),lloyd_iters=0,generator_xyz=xyz,rotation_scheme=scheme)
        D,C,_,J=pv.operators(mesh);_,P,_,_=pv.transfers(mesh)
        A,B=make_hybrid_coefficients(32,p_top=5000.,sigma_power=2.)
        col=JWInitialConditions.make_column_for_mode(mesh,32,A,B,mode='constant_ps_surface_phi',phi_full_mode='direct_u_gl2_adjoint')
        dyn=DryHydrostaticConservativeDynamics()
        initial,info=JWInitialConditions(mesh,col,mode='constant_ps_surface_phi').make_state(dyn,perturb=False,project_mass_flux=False,return_info=True)
        state=initial
        metric=((mesh.dc_edge*mesh.dv_edge)[:,None]*mesh.cell_to_edge(state.layer_dp())/dyn.gravity)
        def diagnostic(s):
            terms=dyn.tendency_terms(s);u=terms['u_e']['total']
            hu=dyn.kinetic_energy_gradient_u(s);gm,gp=dyn.thermal_gradients(s)
            db=col.B_interface.diff();gp=gp+(mesh.cell_area[:,None]*db*dyn.cell_kinetic_energy(s)/dyn.gravity).sum(-1,keepdim=True)
            works=[hu*u,gm*terms['mtheta_i']['total'],gp*terms['ps']['total']]
            rot=hu*terms['u_e']['rotation']
            return {'momentum_mass_rms_m_s2':float(((metric*u*u).sum()/metric.sum()).sqrt()),
                    'momentum_max_m_s2':float(u.abs().max()),
                    'rotation_relative_work':float(rot.sum()/rot.abs().sum()),
                    'energy_relative_work':float(sum(w.sum() for w in works)/sum(w.abs().sum() for w in works))}
        with torch.no_grad():
            start=diagnostic(initial)
            for _ in range(args.steps): state=ssprk3(dyn,state,args.dt)
            end=diagnostic(state)
        case={'initial_condition':asdict(info),'initial':start,'after':end,'steps':args.steps,'dt_s':args.dt,
              'u_change_mass_rms_m_s':float(((metric*(state.u_e-initial.u_e)**2).sum()/metric.sum()).sqrt()),
              'ps_max_change_Pa':float((state.ps-initial.ps).abs().max()),
              'mass_relative_change':float(((mesh.cell_area[:,None]*(state.ps-initial.ps)).sum())/(mesh.cell_area[:,None]*initial.ps).sum()),
              'state_initial_sha256':{k:hashlib.sha256(pv.array(getattr(initial,k)).tobytes()).hexdigest() for k in ('u_e','ps','mtheta_i')},
              'curl_div_relative':pv.relative(C@J,P@D),
              'uniform_pv':pv.uniform_pv(mesh,P,J,J,.2)}
        result['cases'][scheme]=case
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(scheme,start,end,flush=True)

if __name__=='__main__':main()
