"""Small JMD95 Hamiltonian campaign; measurements, no scientific pass cutoffs.

Independent pressure references integrate the archived EOS in physical height.
The internal-wave reference uses a continuum two-layer hydrostatic linearization.
Run against an installed wheel outside the checkout. No binary fields are saved.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import vfs
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import Ocean, OceanColumn
from vfs.ocean.eos import jmd95_seawater_density
from vfs.ocean.jmd95 import JMD95Dynamics, checked_ssprk3
from vfs.ocean.jmd95_energy import JMD95Energy


def gauss(order=24):
    x, w = np.polynomial.legendre.leggauss(order)
    return torch.tensor((x+1)/2), torch.tensor(w/2)


def rms(value, weights):
    return float(((weights*value.square()).sum()/weights.expand_as(value).sum()).sqrt())


def make_case(n=32, radius=6371220., lloyd=2):
    mesh = VoronoiMesh(n_cells=n, radius=radius, lloyd_iters=lloyd, verbose=False)
    column = OceanColumn(3, total_depth=3000.)
    model = Ocean(mesh,column,dynamics=JMD95Dynamics(),stepper=checked_ssprk3)
    x,y = mesh.cell_xyz[:,:1],mesh.cell_xyz[:,1:2]
    h = torch.tensor([500.,1000.,1500.])[None,:].expand(n,3).clone()
    t = torch.tensor([17.,10.,4.])[None,:]+.4*x
    s = torch.tensor([34.,35.,35.5])[None,:]+.05*y
    state = model.state(u_e=torch.zeros(mesh.nEdges,3),h=h,temperature=t,salinity=s)
    return model,state


def independent_energy(state, order=24):
    """Integrate the existing EOS twice; no candidate energy helper/derivative."""
    node, weight = gauss(order)
    h,t,s = state.h,state.temperature(),state.salinity()
    bottom = -state.column.total_depth_like(h) + h.flip(-1).cumsum(-1).flip(-1)-h
    z = bottom[...,None]+h[...,None]*node
    path = z[...,None]*node
    g,rho0 = state.column.gravity,state.column.rho0
    p = -rho0*g*path/1e4
    rho = jmd95_seawater_density(s[...,None,None],t[...,None,None],p)
    ref = jmd95_seawater_density(torch.full_like(s[...,None,None],35.),torch.full_like(t[...,None,None],10.),p)
    parcel = g/rho0*z*((rho-ref)*weight).sum(-1)
    return (rho0*state.mesh.cell_area[:,None]*h*(parcel*weight).sum(-1)).sum()


def derivative_work_probe():
    model,state = make_case()
    h = state.h*(1+.001*state.mesh.cell_xyz[:,2:3])
    u = .03*torch.sin(torch.arange(state.u_e.numel()).reshape_as(state.u_e)*.37)
    state = model.state(u_e=u,h=h,temperature=state.temperature(),salinity=state.salinity())
    names = ('u_e','h','temp_h','salt_h')
    leaves = [getattr(state,key).clone().requires_grad_() for key in names]
    state = state.with_tensors(*leaves,None)
    gradients = torch.autograd.grad(model.dynamics.total_energy(state),leaves)
    analytic = model.dynamics.energy.gradients(state)
    rhs = model.tendency(state)
    local = [g*getattr(rhs,k) for g,k in zip(gradients,names)]
    result = {'gradient_relative_errors':{k:float((analytic[k]-g).abs().max()/g.abs().max()) for k,g in zip(names,gradients)},
        'energy_rate_W':float(sum(x.sum() for x in local)),
        'absolute_local_work_W':float(sum(x.abs().sum() for x in local))}
    result['normalized_work'] = result['energy_rate_W']/result['absolute_local_work_W']
    result['finite_differences'] = []
    for name in names:
        direction = torch.sin(torch.arange(getattr(state,name).numel()).reshape_as(getattr(state,name))*.3+.1)
        predicted = float((analytic[name]*direction).sum())
        for eps in (.1,.01,.001):
            energies=[]
            for sign in (-1,1):
                fields=[getattr(state,k)+(sign*eps*direction if k==name else 0) for k in names]
                changed=state.with_tensors(*fields,None)
                energies.append(float(model.dynamics.total_energy(changed)))
            fd=(energies[1]-energies[0])/(2*eps)
            result['finite_differences'].append(dict(field=name,epsilon=eps,relative_error=abs(fd-predicted)/abs(predicted)))
    ref = float(independent_energy(state))
    result['quadrature'] = [{'order':n,'relative_energy_error_against_independent_24':abs(float(JMD95Energy(quadrature_order=n).stratification(state)[0])-ref)/abs(ref)} for n in (2,4,8,16)]
    result['zero_velocity'] = []
    for eps in (1e-3,1e-6,1e-9,1e-12):
        forces=[];total=[]
        for sign in (-1,1):
            changed=state.with_tensors(torch.full_like(u,sign*eps),state.h,state.temp_h,state.salt_h,None)
            terms=model.dynamics.tendency_terms(changed)
            forces.append(terms['u_e']['hamiltonian_pressure'])
            total.append(sum(terms['u_e'].values()))
        weights=(state.mesh.dc_edge*state.mesh.dv_edge)[:,None]
        result['zero_velocity'].append(dict(epsilon_m_s=eps,pressure_jump_rms_m_s2=rms(forces[1]-forces[0],weights),rhs_jump_rms_m_s2=rms(total[1]-total[0],weights)))
    return result


def independent_pressure(state):
    """Layer-mean -grad int_z^0 f dz at flat interfaces and nonuniform T/S.

    Edge T and S come from the analytic spherical fields, and horizontal
    derivatives from their analytic surface gradient. EOS partials here use AD
    of the pre-existing EOS, independently of the candidate polynomial partials.
    """
    mesh=state.mesh;g,rho0=state.column.gravity,state.column.rho0
    t=(torch.tensor([17.,10.,4.])[None,:]+.4*mesh.edge_xyz[:,:1]).requires_grad_()
    s=(torch.tensor([34.,35.,35.5])[None,:]+.05*mesh.edge_xyz[:,1:2]).requires_grad_()
    h=state.h[0];ztop=-torch.cat([torch.zeros(1),h.cumsum(-1)[:-1]])
    bottom=ztop-h;node,weight=gauss()
    z=bottom[:,None]+h[:,None]*node
    # Independent AD one quadrature node at a time avoids summing the needed
    # vertical derivative profile before applying physical pressure weights.
    profiles=[]
    for q in range(len(node)):
        rho=jmd95_seawater_density(s,t,-rho0*g*z[:,q][None,:]/1e4)
        rt,rs=torch.autograd.grad(rho.sum(),(t,s))
        profiles.append(g/rho0*(rt*.4*mesh.edge_normal[:,:1]/mesh.radius+rs*.05*mesh.edge_normal[:,1:2]/mesh.radius))
    profile=torch.stack(profiles,-1)
    full=h*(profile*weight).sum(-1)
    above=full.cumsum(-1)-full
    within=h*(profile*weight*node).sum(-1)
    return -(above+within)


def pressure_convergence():
    rows=[]
    for n in (64,256,1024):
        model,state=make_case(n,lloyd=20)
        predicted=model.tendency(state).u_e
        reference=independent_pressure(state)
        weights=(state.mesh.dc_edge*state.mesh.dv_edge)[:,None]
        rows.append(dict(cells=n,rms_error_m_s2=rms(predicted-reference,weights),reference_rms_m_s2=rms(reference,weights),max_error_m_s2=float((predicted-reference).abs().max())))
    return rows


def rest_probe():
    model,initial=make_case()
    mesh=initial.mesh;rows=[]
    for topography in (False,True):
        bottom=3000+200*mesh.cell_xyz[:,:1] if topography else torch.full((mesh.nCells,1),3000.)
        column=OceanColumn(3,total_depth=bottom,rest_thickness=torch.tensor([500.,1000.,1500.]))
        h=torch.cat([torch.full_like(bottom,500.),torch.full_like(bottom,1000.),bottom-1500.],-1)
        model=Ocean(mesh,column,dynamics=JMD95Dynamics(),stepper=checked_ssprk3)
        state=model.state(u_e=torch.zeros(mesh.nEdges,3),h=h,temperature=torch.tensor([17.,10.,4.]).expand_as(h),salinity=torch.tensor([34.,35.,35.5]).expand_as(h))
        force=model.tendency(state).u_e
        rows.append(dict(topography=topography,rms_acceleration_m_s2=rms(force,(mesh.dc_edge*mesh.dv_edge)[:,None]),max_acceleration_m_s2=float(force.abs().max())))
    return rows


def internal_wave(n, fraction=.05):
    mesh=VoronoiMesh(n_cells=n,radius=1e5,lloyd_iters=10,verbose=False)
    g,rho0=9.80665,1025.
    h0=torch.tensor([500.,2500.]);t=torch.tensor([17.,4.]);s=torch.tensor([34.,35.5])
    column=OceanColumn(2,total_depth=3000.,rest_thickness=h0)
    model=Ocean(mesh,column,dynamics=JMD95Dynamics(omega=0.),stepper=checked_ssprk3)
    def f(layer,z):
        p=torch.tensor(-rho0*g*z/1e4)
        return g/rho0*(jmd95_seawater_density(s[layer],t[layer],p)-jmd95_seawater_density(torch.tensor(35.),torch.tensor(10.),p))
    # Continuum pressure Hessian for material layers, evaluated at rest.
    d1=g+f(0,0.)
    d2=d1-f(0,-float(h0[0]))+f(1,-float(h0[0]))
    C=torch.stack([torch.stack([d1,d1]),torch.stack([d1,d2])])
    eigenvalues,eigenvectors=torch.linalg.eig(torch.diag(h0)@C)
    index=eigenvalues.real.argmin();speed2=float(eigenvalues[index].real)
    vector=eigenvectors[:,index].real;vector=vector/vector.abs().max()
    area=mesh.cell_area[:,None];mode=mesh.cell_xyz[:,2:3];mode=mode-(area*mode).sum()/area.sum()
    amplitude=.1
    initial=model.state(u_e=torch.zeros(mesh.nEdges,2),h=h0[None,:]+amplitude*mode*vector[None,:],temperature=t.expand(n,2),salinity=s.expand(n,2))
    frequency=math.sqrt(2*speed2)/mesh.radius
    end=fraction*2*math.pi/frequency
    # Resolve the fast external mode as well as the internal mode.
    dt=min(30.,float(mesh.dc_edge.min())/math.sqrt(float(eigenvalues.real.max()))*.15)
    steps=math.ceil(end/dt);dt=end/steps
    state=initial
    with torch.no_grad():
        for _ in range(steps):state=model.step(state,dt)
    projection=(area*mode*(state.h-h0[None,:])).sum(0)/(area*mode.square()).sum()
    expected=amplitude*vector*math.cos(frequency*end)
    E0=float(model.dynamics.total_energy(initial));E1=float(model.dynamics.total_energy(state))
    # Reference offset dominates H. Also report drift against perturbation energy.
    rest=model.state(u_e=torch.zeros_like(initial.u_e),h=h0.expand(n,2),temperature=t.expand(n,2),salinity=s.expand(n,2))
    perturbation=abs(E0-float(model.dynamics.total_energy(rest)))
    inventory0=model.budgets(initial);inventory1=model.budgets(state)
    return dict(cells=n,internal_speed_m_s=math.sqrt(speed2),period_s=2*math.pi/frequency,fraction_of_period=fraction,initial_interface_amplitude_m=amplitude,steps=steps,dt_s=dt,
        projection_error_over_amplitude=float((projection-expected).norm()/amplitude),
        energy_change_J=E1-E0,relative_total_energy_change=(E1-E0)/abs(E0),energy_change_over_initial_perturbation=(E1-E0)/perturbation,
        inventory_relative_changes={key:float((inventory1[key]-inventory0[key])/inventory0[key]) for key in ('volume','temperature_content','salinity_content')},
        min_thickness_m=float(state.h.min()),salinity_range=[float(state.salinity().min()),float(state.salinity().max())])


def temporal_probe():
    model,initial=make_case(32,radius=1e5)
    model=Ocean(initial.mesh,initial.column,dynamics=JMD95Dynamics(omega=0.),stepper=checked_ssprk3)
    h=initial.h + .2*initial.mesh.cell_xyz[:,2:3]*torch.tensor([1.,-.4,-.6])[None,:]
    initial=model.state(u_e=initial.u_e,h=h,temperature=initial.temperature(),salinity=initial.salinity())
    def run(dt):
        state=initial
        with torch.no_grad():
            for _ in range(round(400/dt)):state=model.step(state,dt)
        return state
    reference=run(1.25);finer=run(.625)
    def error(a,b):
        values=[(a.u_e-b.u_e)/math.sqrt(9.80665*3000),(a.h-b.h)/3000,
            (a.temp_h-b.temp_h)/(3000*10),(a.salt_h-b.salt_h)/(3000*35)]
        return float(torch.cat([v.flatten() for v in values]).square().mean().sqrt())
    e0=float(model.dynamics.total_energy(initial));rows=[]
    for dt in (40.,20.,10.):
        state=run(dt)
        rows.append(dict(dt_s=dt,rms_nondimensional_error=error(state,reference),energy_change_J=float(model.dynamics.total_energy(state))-e0))
    return dict(duration_s=400.,reference_dt_s=1.25,reference_refinement_error=error(reference,finer),rows=rows,
        pairwise_orders=[math.log(rows[i]['rms_nondimensional_error']/rows[i+1]['rms_nondimensional_error'],2) for i in range(2)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quick',action='store_true',help='omit wave/temporal runs and pressure mesh refinement')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    package=Path(vfs.__file__).parent
    result=dict(device='cpu',dtype='float64',threads=1,torch=str(torch.__version__),numpy=np.__version__,
        work=derivative_work_probe(),rest=rest_probe())
    if not args.quick:
        result['pressure']=pressure_convergence()
        print('pressure',result['pressure'],flush=True)
        result['internal_wave']=[]
        for n in (32,64,128):
            row=internal_wave(n);result['internal_wave'].append(row);print('wave',row,flush=True)
        result['temporal']=temporal_probe();print('temporal',result['temporal'],flush=True)
    result['runtime_sha256']={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.rglob('*.py'))}
    result['driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['scope']='experimental adiabatic centered JMD95; no long-run, limiter/mixing, GPU or full Poisson qualification'
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('work',result['work']['normalized_work'],'rest',result['rest'],flush=True)


if __name__=='__main__':
    main()
