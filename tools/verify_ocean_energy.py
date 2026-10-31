"""Quantitative constant-density checks. Reports metrics; no scientific PASS cutoffs.

Run from a copied tools/tests tree using an installed wheel. Outputs are new
directories, never refreshed reference fixtures. The production RHS/integrators
are used unchanged; only independent diagnostics use autograd.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import platform
from pathlib import Path
import numpy as np
import torch
import vfs
from vfs.ocean import Ocean, OceanColumn, OceanState, OceanDomain
from vfs.ocean.conservative import ConstantDensityDynamics
from vfs.ocean.dynamics import MultilayerOceanDynamics
from vfs.geometry.mesh import VoronoiMesh

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('capture_ocean',ROOT/'tools/capture_ocean.py')
capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture)


def relative(value,scale):
    return float(abs(value)/scale) if scale else (0. if value==0 else None)


def work_probe(state,dynamics):
    u=state.u_e.detach().requires_grad_();h=state.h.detach().requires_grad_()
    state=state.with_tensors(u,h,state.temp_h,state.salt_h,state.tracer_h)
    energy=dynamics.total_energy(state)
    gu,gh=torch.autograd.grad(energy,(u,h))
    rhs=dynamics.tendency(state)
    local_u,local_h=gu*rhs.u_e,gh*rhs.h
    rate=local_u.sum()+local_h.sum();scale=local_u.abs().sum()+local_h.abs().sum()
    pressure=-dynamics.gravity*state.mesh.grad(state.eta())
    if dynamics.domain is not None:pressure=dynamics.domain.apply_edge(pressure)
    kinetic_work=(gu*pressure).sum()
    surface_local=dynamics.rho0*dynamics.gravity*state.mesh.cell_area[:,None]*state.eta()*rhs.h
    surface_rate=surface_local.sum()
    out={'energy_J':float(energy.detach()),'rate_W':float(rate),
        'cancellation_normalized_rate':relative(rate,scale),
        'free_surface_kinetic_work_W':float(kinetic_work),
        'free_surface_potential_rate_W':float(surface_rate),
        'free_surface_exchange_relative':relative(kinetic_work+surface_rate,max(abs(kinetic_work),abs(surface_rate)))}
    if isinstance(dynamics,ConstantDensityDynamics):
        analytic=dynamics.energy.gradients(state)
        out['gradient_max_relative']={key:relative((analytic[key]-g).abs().max(),g.abs().max())
                                      for key,g in [('u_e',gu),('h',gh)]}
        rotation=gu*dynamics.rotation_acceleration(state)
        out['rotation_work_W']=float(rotation.sum())
        out['rotation_relative']=relative(rotation.sum(),rotation.abs().sum())
    out['directional_finite_difference']=[]
    for dt in (1.,.1,.01):
        def shifted(sign):
            s=state.with_tensors(u+sign*dt*rhs.u_e,h+sign*dt*rhs.h,state.temp_h,state.salt_h,state.tracer_h)
            return dynamics.total_energy(s)
        fd=(shifted(1)-shifted(-1))/(2*dt)
        out['directional_finite_difference'].append({'increment_s':dt,'rate_W':float(fd.detach()),
            'error_normalized_by_local_work':relative(fd.detach()-rate,scale)})
    # A direction with nonzero energy derivative checks the gradient without
    # trying to resolve a tiny difference between almost equal global energies.
    du=.1*u.detach()
    dh=100*state.eta().detach().expand_as(h)
    directional=(gu*du).sum()+(gh*dh).sum()
    out['nonzero_direction_gradient']={'AD_J':float(directional),'rows':[]}
    for epsilon in (1e-2,1e-3,1e-4,1e-5):
        def energy_at(sign):
            return dynamics.total_energy(state.with_tensors(u+sign*epsilon*du,h+sign*epsilon*dh,
                state.temp_h,state.salt_h,state.tracer_h))
        fd=(energy_at(1)-energy_at(-1))/(2*epsilon)
        out['nonzero_direction_gradient']['rows'].append({'epsilon':epsilon,
            'relative_error':relative(fd.detach()-directional,abs(directional))})
    return out


def save_mesh(mesh,path):
    arrays={'mesh.'+name:value.detach().cpu().numpy() for name,value in mesh.named_buffers()}
    arrays['mesh_metadata']=np.array(json.dumps({k:getattr(mesh,k) for k in
        ('nCells','nEdges','nVertices','maxEdges','radius')}))
    np.savez_compressed(path,**arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def waves(out):
    rows=[]
    hashes={}
    for n in (32,64,128,256):
        mesh=VoronoiMesh(n_cells=n,lloyd_iters=1,verbose=False)
        filename=f'wave_mesh_{n}.npz'
        hashes[filename]=save_mesh(mesh,out/filename)
        for degree in (1,2):
            column=OceanColumn(1,total_depth=4000.)
            model=Ocean(mesh,column,dynamics=ConstantDensityDynamics(omega=0.))
            z=mesh.cell_xyz[:,2:3];mode=z if degree==1 else .5*(3*z*z-1)
            # Remove the mesh-area mean; a constant free surface has zero force.
            area=mesh.cell_area[:,None];mode=mode-(area*mode).sum()/area.sum()
            h=4000+.1*mode
            state=model.state(u_e=torch.zeros(mesh.nEdges,1),h=h,temperature=torch.full_like(h,10.),salinity=torch.full_like(h,35.))
            frequency=math.sqrt(column.gravity*4000*degree*(degree+1))/mesh.radius
            target=.2*2*math.pi/frequency
            steps=math.ceil(target/300.);dt=target/steps
            with torch.no_grad():
                for _ in range(steps):state=model.step(state,dt)
            amplitude=float((area*mode*state.eta()).sum()/(area*mode.square()).sum())
            rows.append({'cells':n,'degree':degree,'steps':steps,'dt_s':dt,
                'endpoint_projection_error_over_initial_amplitude':abs(amplitude-.1*math.cos(frequency*target))/.1})
    fits={str(l):float(np.polyfit(np.log([math.sqrt(4*math.pi/r['cells']) for r in rows if r['degree']==l]),
              np.log([r['endpoint_projection_error_over_initial_amplitude'] for r in rows if r['degree']==l]),1)[0]) for l in (1,2)}
    return {'rows':rows,'mesh_sha256':hashes,'fitted_horizontal_orders':fits,
            'limitation':'single endpoint projection, 0.2 period, no independent time extrapolation; damping and phase mixed'}


def temporal_convergence(out):
    mesh=VoronoiMesh(n_cells=32,lloyd_iters=1,radius=1e5,verbose=False)
    mesh_hash=save_mesh(mesh,out/'temporal_mesh.npz')
    column=OceanColumn(1,total_depth=1000.)
    model=Ocean(mesh,column,dynamics=ConstantDensityDynamics(omega=0.))
    h=1000+.1*mesh.cell_xyz[:,2:3]
    initial=model.state(u_e=torch.zeros(mesh.nEdges,1),h=h,temperature=torch.full_like(h,10.),salinity=torch.full_like(h,35.))
    def run(dt):
        state=initial
        with torch.no_grad():
            for _ in range(round(400/dt)):state=model.step(state,dt)
        return state
    reference=run(.625);finer=run(.3125);rows=[]
    speed=math.sqrt(column.gravity*1000.)
    reference_error=torch.cat([((reference.h-finer.h)/1000).flatten(),
        ((reference.u_e-finer.u_e)/speed).flatten()]).square().mean().sqrt()
    for dt in (40.,20.,10.,5.):
        state=run(dt)
        error=torch.cat([((state.h-reference.h)/1000).flatten(),((state.u_e-reference.u_e)/speed).flatten()]).square().mean().sqrt()
        rows.append({'dt_s':dt,'nondimensional_rms_error':float(error),
            'relative_energy_change':float((model.dynamics.total_energy(state)-model.dynamics.total_energy(initial))/model.dynamics.total_energy(initial))})
    orders=[math.log(rows[i]['nondimensional_rms_error']/rows[i+1]['nondimensional_rms_error'],2) for i in range(len(rows)-1)]
    return {'radius_m':1e5,'end_time_s':400.,'reference_dt_s':.625,
            'reference_refinement_dt_s':.3125,'reference_refinement_rms':float(reference_error),
            'mesh_sha256':mesh_hash,'rows':rows,'pairwise_orders':orders,
            'limitation':'same spatial operator and fine-step reference; not an analytic solution or spatial accuracy result'}


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    inputs=ROOT/'tests/fixtures/ocean/inputs.npz'
    mesh,fields=capture.load_inputs(inputs);column=OceanColumn(3,total_depth=3000.)
    fields['temp_h']=fields['h']*10;fields['salt_h']=fields['h']*35
    state=OceanState(**fields,mesh=mesh,column=column)
    result={'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
        'dtype':'float64','device':'cpu','threads':1,
        'input_sha256':hashlib.sha256(inputs.read_bytes()).hexdigest(),'cases':{}}
    for name,domain in [('global',None),('coast',OceanDomain(mesh,mesh.cell_xyz[:,2]>-.4))]:
        s=state if domain is None else domain.project_state(state)
        result['cases'][name]={'reference':work_probe(s,MultilayerOceanDynamics(domain=domain)),
                             'candidate':work_probe(s,ConstantDensityDynamics(domain=domain))}
    h=1000+.1*mesh.cell_xyz[:,:1].expand(mesh.nCells,3)/3
    pg=mesh.grad(h.sum(-1,keepdim=True)-3000.)
    u=(-.001*pg/pg.abs().max()).expand(mesh.nEdges,3).clone()
    aligned=OceanState.from_temperature(u,h,torch.full_like(h,10.),torch.full_like(h,35.),mesh=mesh,column=column)
    result['cases']['aligned']={'reference':work_probe(aligned,MultilayerOceanDynamics()),
                                'candidate':work_probe(aligned,ConstantDensityDynamics())}
    result['external_waves']=waves(out);result['time_refinement']=temporal_convergence(out)
    package=Path(vfs.__file__).parent
    result['source_sha256']={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*sorted((package/'ocean').glob('*.py')),package/'geometry/mesh/grid.py',package/'integrators/explicit.py']}
    result['verification_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
