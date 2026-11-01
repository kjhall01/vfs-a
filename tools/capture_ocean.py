"""Capture actual ocean kernels in separate donor/extracted Python processes.

Create inputs ONCE in the donor, then use the same NPZ for both captures. Mesh
buffers are restored verbatim: matching constructor arguments is insufficient.
No reference is overwritten, no source is patched, and no RK formula is copied.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.field import Field
from vfs.ocean.column import OceanColumn
from vfs.ocean.domain import OceanDomain
from vfs.ocean.state import OceanState
from vfs.ocean.dynamics import MultilayerOceanDynamics, HydrostaticBoussinesqOceanDynamics
from vfs.ocean.physics import (OceanSurfaceForcing, OceanSurfaceFluxPhysics,
    OceanEkmanSurfacePhysics, OceanBulkMixedLayerPhysics, CompositeOceanPhysics)
from vfs.core.integrators import forward_euler, ssprk3, rk4

FIELDS = ('u_e', 'h', 'temp_h', 'salt_h', 'tracer_h')


def tensor(value):
    return value.values if isinstance(value, Field) else value


def create_inputs(path):
    if path.exists():
        raise FileExistsError(path)
    mesh = VoronoiMesh(n_cells=32, lloyd_iters=1, verbose=False)
    gen = torch.Generator().manual_seed(731)
    rand = lambda *shape: torch.randn(*shape, generator=gen)
    x = mesh.cell_xyz[:, :1]
    h = 1000.0 * (1 + .15 * x * torch.tensor([-1., 0., 1.])) + .1*x/3
    temp = 10 + x + torch.tensor([2., 0., -2.])
    salt = 35 + .1*x + torch.tensor([-.1, 0., .1])
    arrays = {'mesh.'+k: v.cpu().numpy() for k,v in mesh.named_buffers()}
    arrays.update(u_e=(.05*rand(mesh.nEdges,3)).numpy(), h=h.numpy(),
                  temp_h=(h*temp).numpy(), salt_h=(h*salt).numpy(),
                  tracer_h=(h[...,None]*(.5+.1*rand(32,3,2))).numpy())
    arrays['mesh_metadata'] = np.array(json.dumps({k:getattr(mesh,k) for k in
        ('nCells','nEdges','nVertices','maxEdges','radius')}))
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_inputs(path):
    with np.load(path, allow_pickle=False) as z:
        mesh = VoronoiMesh.__new__(VoronoiMesh)
        torch.nn.Module.__init__(mesh)
        for k,v in json.loads(str(z['mesh_metadata'])).items():
            setattr(mesh,k,v)
        for k in z.files:
            if k.startswith('mesh.'):
                mesh.register_buffer(k[5:],torch.from_numpy(z[k].copy()))
        fields = {k:torch.from_numpy(z[k].copy()) for k in FIELDS}
    return mesh, fields


def capture(path):
    mesh, fields = load_inputs(path)
    out = {}
    def record(key,value):
        if isinstance(value, OceanState):
            for f in FIELDS: record(key+'/'+f,getattr(value,f))
        elif isinstance(value,dict):
            for k,v in value.items(): record(key+'/'+k,v)
        elif isinstance(value,(tuple,list)):
            for i,v in enumerate(value): record(key+'/'+str(i),v)
        elif value is not None:
            value=tensor(value)
            out[key]=value.detach().cpu().numpy().copy() if isinstance(value,torch.Tensor) else np.asarray(value)

    for masked in (False,True):
        domain = OceanDomain(mesh,mesh.cell_xyz[:,2]>-.4) if masked else None
        depth = 3000*(1+.1*mesh.cell_xyz[:,:1]) if masked else 3000.
        column = OceanColumn(3,total_depth=depth,rest_thickness=torch.full((3,),1000.))
        state = OceanState(**fields,mesh=mesh,column=column)
        if domain: state=domain.project_state(state)
        prefix='coast' if masked else 'global'
        record(prefix+'/initial',state)
        record(prefix+'/domain_projection',state if domain is None else domain.project_state(state))
        for eos in ('linear','jmd95'):
            for pg in ('montgomery','sigma_depth'):
                dyn=MultilayerOceanDynamics(domain=domain,eos=eos,pressure_gradient_mode=pg)
                key=f'{prefix}/{eos}/{pg}'
                record(key+'/rhs',dyn.tendency(state));record(key+'/terms',dyn.tendency_terms(state))
                record(key+'/pressure_work',dyn.pressure_work_terms(state))
                record(key+'/rotation_work',dyn.coriolis_work_terms(state))
                record(key+'/budgets',dyn.conservation_terms(state))
                record(key+'/energy',dyn.total_energy(state))
                for name,step in [('euler',forward_euler),('rk3',ssprk3),('rk4',rk4)]:
                    stages=[]
                    def rhs(s):
                        stages.append(s)
                        return dyn.tendency(s)
                    record(key+'/'+name+'/final',step(rhs,state,30.))
                    record(key+'/'+name+'/stages',stages)
                leaves=[tensor(getattr(state,k)).detach().clone().requires_grad_() for k in FIELDS]
                active=OceanState(*leaves,mesh=mesh,column=column)
                tendency=dyn.tendency(active)
                objective=sum(tensor(getattr(tendency,k)).square().sum() for k in FIELDS)
                record(key+'/rhs_gradient',torch.autograd.grad(objective,leaves))
                record(key+'/no_tracer',dyn.tendency(state.with_tensors(*[getattr(state,k) for k in FIELDS[:4]],None)))
        for scheme,options in [
            ('vertical',dict(scalar_vertical_diffusivity=.01,momentum_vertical_viscosity=.02)),
            ('constant',dict(horizontal_momentum_viscosity=2e4)),
            ('scaled',dict(horizontal_momentum_viscosity_scheme='mesh_cfl_scaled',horizontal_momentum_viscosity_cfl=.01,horizontal_momentum_viscosity_dt=30.)),
            ('biharmonic',dict(horizontal_momentum_viscosity_scheme='biharmonic',horizontal_momentum_biharmonic_viscosity=1e14)),
            ('biharmonic_scaled',dict(horizontal_momentum_viscosity_scheme='biharmonic',horizontal_momentum_biharmonic_cfl=.001,horizontal_momentum_viscosity_dt=30.)),
            ('remap',dict(vertical_remap_mode='target_thickness_relaxation',vertical_remap_timescale=86400.)),
        ]:
            dyn=HydrostaticBoussinesqOceanDynamics(domain=domain,**options)
            key=f'{prefix}/process/{scheme}'
            record(key+'/rhs',dyn.tendency(state));record(key+'/terms',dyn.tendency_terms(state))
            record(key+'/w',dyn.interface_vertical_velocity(state))
            record(key+'/step',ssprk3(dyn,state,30.))
            record(key+'/remap',dyn.conservative_vertical_transport_tendency(state,torch.full((32,2),1e-4)))
        flux=OceanSurfaceFluxPhysics(domain=domain,mixed_layer_depth=2000.,mixed_layer_scalar_diffusivity=.01,mixed_layer_momentum_viscosity=.02,freshwater_flux_amplitude=1e-8)
        components={'restoring':OceanSurfaceForcing(domain=domain),'flux':flux,
            'ekman':OceanEkmanSurfacePhysics(domain=domain,vertical_momentum_viscosity=.02),
            'prescribed':OceanEkmanSurfacePhysics(domain=domain,stress_mode='prescribed',prescribed_stress_edge=torch.full((mesh.nEdges,1),.1)),
            'bulk':OceanBulkMixedLayerPhysics(domain=domain,surface_flux_component=flux),
            'bulk_jmd95':OceanBulkMixedLayerPhysics(domain=domain,surface_flux_component=flux,density_mode='jmd95')}
        components['composite']=CompositeOceanPhysics(flux,components['bulk'])
        dyn=MultilayerOceanDynamics(domain=domain)
        for name,source in components.items():
            record(prefix+'/source/'+name,source.tendency(state))
            current=state
            for _ in range(3):
                current=ssprk3(lambda s:dyn.tendency(s)+source.tendency(s),current,30.)
            record(prefix+'/forced_steps/'+name,current)
        unstable=OceanState.from_temperature(tensor(state.u_e),tensor(state.h),torch.full((32,3),10.)+torch.tensor([-2.,0.,2.]),torch.full((32,3),35.),mesh=mesh,column=column)
        record(prefix+'/unstable_bulk',components['bulk'].tendency(unstable))
    return out


def compare(actual,reference):
    if actual.keys()!=reference.keys():
        raise AssertionError('capture keys differ')
    differences=[]
    for key,a in actual.items():
        b=reference[key]
        if not np.array_equal(a,b):
            differences.append({'key':key,'max_abs':float(np.max(np.abs(a-b)))})
    return {'arrays':len(actual),'exact':len(actual)-len(differences),'differences':differences}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--create-inputs',action='store_true')
    p.add_argument('--save',type=Path)
    p.add_argument('--compare',type=Path)
    p.add_argument('--report',type=Path)
    args=p.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    if args.create_inputs: create_inputs(args.inputs)
    result=capture(args.inputs)
    report={'torch':torch.__version__,'input_sha256':hashlib.sha256(args.inputs.read_bytes()).hexdigest(),'arrays':len(result)}
    if args.save:
        if args.save.exists(): raise FileExistsError(args.save)
        args.save.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.save,**result)
    if args.compare:
        with np.load(args.compare,allow_pickle=False) as z: report.update(compare(result,dict(z)))
    if args.report: args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    if report.get('differences'): raise SystemExit(1)
