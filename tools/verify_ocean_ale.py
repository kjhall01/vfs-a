"""ALE measurements with analytic references; no scientific acceptance gates.

Run with an installed wheel. --quick omits complete-period wave integrations.
The wave observable is column temperature content, which is invariant under
remapping; layer thickness itself is not a meaningful cross-grid observable.
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
from vfs.ocean.ale import ALEStepper, ReferenceGrid, VerticalRemapper, remap_contents
from vfs.ocean.eos import jmd95_seawater_density
from vfs.ocean.jmd95 import JMD95Dynamics, checked_ssprk3


def smooth_means(edges):
    # Exact layer averages of q(z)=2+cos(pi*z), 0<=z<=1.
    return 2 + torch.sin(math.pi*edges).diff()/(math.pi*edges.diff())


def reconstruction_measurements():
    rows = []
    for method in ('pcm','plm'):
        for n in (16,32,64,128):
            old = torch.linspace(0,1,n+1)
            new = old + .08*torch.sin(2*math.pi*old)
            h,target = old.diff(),new.diff()
            content = h*smooth_means(old)
            mapped = remap_contents(h,content,target,method=method)
            error = mapped/target-smooth_means(new)
            cycled = content
            for _ in range(50):
                cycled = remap_contents(target,remap_contents(h,cycled,target,method=method),h,method=method)
            rows.append(dict(method=method,layers=n,
                weighted_l2_error=float((target*error.square()).sum().sqrt()),
                inventory_residual=float(mapped.sum()-content.sum()),
                cycles=50,cycle_l2_error=float((h*(cycled/h-content/h).square()).sum().sqrt()),
                cycle_inventory_residual=float(cycled.sum()-content.sum()),
                cycle_range=[float((cycled/h).min()),float((cycled/h).max())]))
    return rows


def rest_measurements():
    mesh = VoronoiMesh(n_cells=32,lloyd_iters=2,verbose=False)
    rows = []
    for n in (8,16,32):
        column = OceanColumn(n,total_depth=3000.)
        sigma = torch.linspace(0,1,n+1)
        depth = 3000*(sigma[None,:]+.08*mesh.cell_xyz[:,:1]*torch.sin(math.pi*sigma)[None,:])
        h = depth.diff(dim=-1)
        center = .5*(depth[:,1:]+depth[:,:-1])
        model = Ocean(mesh,column,dynamics=JMD95Dynamics(omega=0.))
        state = model.state(u_e=torch.zeros(mesh.nEdges,n),h=h,
            temperature=20-16*center/3000,salinity=34.5+center/3000)
        after,report = VerticalRemapper().remap_with_report(model,state,ReferenceGrid()(state))
        edge_area = (mesh.dc_edge*mesh.dv_edge)[:,None]
        def acceleration(s):
            force = model.tendency(s).u_e
            return float((edge_area*force.square()).sum().div(edge_area.expand_as(force).sum()).sqrt())
        rows.append(dict(layers=n,initial_acceleration_rms_m_s2=acceleration(state),
            remapped_acceleration_rms_m_s2=acceleration(after),
            max_temperature_error_C=float((after.temperature()-(20-16*(after.h.cumsum(-1)-.5*after.h)/3000)).abs().max()),
            remap_energy_J={k:float(v) for k,v in report.energy_change.items()}))
    return rows


def wave_case(n_layers):
    """Two water masses, each subdivided into n_layers/2 material layers.

    Same independent continuum two-layer reference as the original JMD95
    investigation. Subdivision permits assessment of remapping resolution.
    """
    mesh = VoronoiMesh(n_cells=32,radius=1e5,lloyd_iters=10,verbose=False)
    g,rho0 = 9.80665,1025.
    masses = torch.tensor([500.,2500.])
    temperatures,salinities = torch.tensor([17.,4.]),torch.tensor([34.,35.5])
    def f(layer,z):
        p = torch.tensor(-rho0*g*z/1e4)
        return g/rho0*(jmd95_seawater_density(salinities[layer],temperatures[layer],p)
            -jmd95_seawater_density(torch.tensor(35.),torch.tensor(10.),p))
    d1 = g+f(0,0.)
    d2 = d1-f(0,-500.)+f(1,-500.)
    C = torch.stack((torch.stack((d1,d1)),torch.stack((d1,d2))))
    eigenvalues,eigenvectors = torch.linalg.eig(torch.diag(masses)@C)
    index = eigenvalues.real.argmin()
    vector = eigenvectors[:,index].real
    vector = vector/vector.abs().max()
    count = n_layers//2
    h0 = masses.repeat_interleave(count)/count
    temperature = temperatures.repeat_interleave(count)
    salinity = salinities.repeat_interleave(count)
    vector = vector.repeat_interleave(count)/count
    column = OceanColumn(n_layers,total_depth=3000.,rest_thickness=h0)
    model = Ocean(mesh,column,dynamics=JMD95Dynamics(omega=0.),stepper=checked_ssprk3)
    area = mesh.cell_area[:,None]
    mode = mesh.cell_xyz[:,2:3]
    mode = mode-(area*mode).sum()/area.sum()
    state = model.state(u_e=torch.zeros(mesh.nEdges,n_layers),h=h0[None,:]+.1*mode*vector[None,:],
        temperature=temperature.expand(mesh.nCells,n_layers),salinity=salinity.expand(mesh.nCells,n_layers))
    rest = model.state(u_e=torch.zeros_like(state.u_e),h=h0.expand_as(state.h),
        temperature=state.temperature(),salinity=state.salinity())
    frequency = math.sqrt(2*float(eigenvalues[index].real))/mesh.radius
    period = 2*math.pi/frequency
    max_dt = min(30.,float(mesh.dc_edge.min())/math.sqrt(float(eigenvalues.real.max()))*.15)
    return model,state,rest,mode,period,max_dt


def internal_wave(n_layers,remap_every,periods=1.):
    model,state,rest,mode,period,max_dt = wave_case(n_layers)
    initial = state
    steps = math.ceil(periods*period/max_dt)
    dt = periods*period/steps
    area = model.mesh.cell_area[:,None]
    denominator = (area*mode.square()).sum()
    def projection(s):
        column_anomaly = (s.temp_h-rest.temp_h).sum(-1,keepdim=True)
        return float((area*mode*column_anomaly).sum()/denominator)
    amplitude = projection(initial)
    energy0 = float(model.dynamics.total_energy(initial))
    perturbation = abs(energy0-float(model.dynamics.total_energy(rest)))
    energy_remap = {k:0. for k in model.dynamics.energy.terms(state)}
    remapper,target = VerticalRemapper(),ReferenceGrid()
    samples = [dict(time_over_period=0.,projection_over_initial=1.,reference=1.)]
    inventory0 = model.budgets(initial)
    sample_stride = max(1,steps//32)
    for step in range(1,steps+1):
        state = checked_ssprk3(model,state,dt)
        if remap_every and step%remap_every == 0:
            state,report = remapper.remap_with_report(model,state,target(state))
            for key,value in report.energy_change.items():
                energy_remap[key] += float(value)
        if step%sample_stride == 0 or step == steps:
            time = step*dt
            samples.append(dict(time_over_period=time/period,
                projection_over_initial=projection(state)/amplitude,reference=math.cos(2*math.pi*time/period)))
        if step%1000 == 0:
            print(f'wave layers={n_layers} remap_every={remap_every}: {step}/{steps}',flush=True)
    inventory1 = model.budgets(state)
    total_change = float(model.dynamics.total_energy(state))-energy0
    x = np.array([row['time_over_period'] for row in samples])
    y = np.array([row['projection_over_initial'] for row in samples])
    design = np.stack((np.cos(2*np.pi*x),np.sin(2*np.pi*x),np.ones_like(x)),-1)
    fit = np.linalg.lstsq(design,y,rcond=None)[0]
    return dict(layers=n_layers,cells=model.mesh.nCells,remap_every_steps=remap_every,
        steps=steps,dt_s=dt,period_s=period,periods=periods,
        projection_rms_error=float(np.sqrt(np.mean((y-np.cos(2*np.pi*x))**2))),
        fitted_reference_frequency_amplitude=float(np.hypot(fit[0],fit[1])),
        fitted_phase_rad=float(np.arctan2(-fit[1],fit[0])),
        initial_perturbation_energy_J=perturbation,total_energy_change_J=total_change,
        remap_energy_change_J=energy_remap,remaining_dynamics_change_J=total_change-energy_remap['total'],
        remap_energy_over_initial_perturbation=energy_remap['total']/perturbation,
        total_energy_over_initial_perturbation=total_change/perturbation,
        inventory_relative_changes={key:float((inventory1[key]-inventory0[key])/inventory0[key])
            for key in ('volume','temperature_content','salinity_content')},
        min_thickness_m=float(state.h.min()),temperature_range_C=[float(state.temperature().min()),float(state.temperature().max())],
        salinity_range=[float(state.salinity().min()),float(state.salinity().max())],samples=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quick',action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    result = dict(device='cpu',dtype='float64',threads=1,torch=str(torch.__version__),
        scope='Conservative bounded vertical remap; horizontal stages remain centered and checked. No energy conservation, GPU or long-backward claim.')
    with torch.no_grad():
        result['reconstruction'] = reconstruction_measurements()
        result['rest'] = rest_measurements()
        result['waves'] = []
        if not args.quick:
            for layers,interval in ((4,0),(4,1),(8,1),(8,10)):
                row = internal_wave(layers,interval)
                result['waves'].append(row)
                print({key:value for key,value in row.items() if key != 'samples'},flush=True)
                (args.output/'partial.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    package = Path(vfs.__file__).parent
    result['runtime_sha256'] = {str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob('*.py'))}
    result['driver_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Wrote',args.output/'results.json',flush=True)


if __name__ == '__main__':
    main()
