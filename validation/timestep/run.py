"""Finite-duration timestep qualification for the explicit Roquet ocean.

Each trial runs in an isolated CPU/float64 process. No timestep is called safe
solely because it completed. Local endpoint tensors are never upload artifacts.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def write(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def substeps(dt, mixing_dt):
    return max(1, math.ceil(dt / mixing_dt))


def signature(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def compare(a, b):
    import torch
    result = {}
    for name in ('u', 'CT', 'SA', 'h', 'k'):
        w = b['edge_weight'] if name == 'u' else b['weight']
        result[name] = float(torch.sqrt((w * (a[name] - b[name]).square()).sum() / w.sum()))
    return result


def qualifies(row, errors, limits):
    if row.get('status') != 'completed' or errors is None:
        return False
    return (all(errors[k] <= limits[k] for k in errors)
            and row['max_energy_error_J_kg'] <= limits['energy']
            and row['max_mass_error_relative'] <= limits['inventory']
            and row['max_salt_error_relative'] <= limits['inventory']
            and row['max_tracer_error_relative'] <= limits['inventory'])


def run_logged(command, log, timeout, verbose):
    """Tee worker output without blocking the timeout on a silent worker."""
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        def relay():
            for line in process.stdout:
                log.write(line); log.flush()
                if verbose:print(line,end='',flush=True)
        thread=threading.Thread(target=relay,daemon=True);thread.start()
        try:
            returncode=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill();process.wait();thread.join()
            raise
        thread.join()
        return returncode


def worker(config, output):
    import numpy as np
    import torch
    sys.path[:0] = [str(ROOT/'validation/temperature_coordinate'), str(ROOT/'validation/ocean_column')]
    from roquet_campaign import model
    from vfs.geometry.mesh import VoronoiMesh
    from vfs.ocean import OceanColumn
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(config['threads'])
    torch.set_grad_enabled(False)
    started = time.perf_counter()
    row = dict(config=config, status='initializing', completed_steps=0, time_s=0.)
    write(output, row)
    def progress(message):
        if config.get('verbose'):print(message,flush=True)
    progress('Initializing mesh and model...')
    try:
        xyz = np.load(config['mesh'], allow_pickle=False) if config['mesh'] else None
        mesh = VoronoiMesh(n_cells=config['cells'], generator_xyz=xyz,
                           lloyd_iters=config['lloyd'], rotation_scheme='compatible', verbose=False)
        row['phase']='model construction'; write(output,row)
        # Fixed exponential depth map: surface resolution varies explicitly with nk.
        r = torch.linspace(0, 1, config['layers']+1)
        interfaces = config['depth']*torch.expm1(4*r)/math.expm1(4)
        rest = interfaces.diff()
        column = OceanColumn(config['layers'], total_depth=config['depth'], rest_thickness=rest)
        full = config['case'] == 'forced'
        m = model(mesh, column, substeps(config['dt'], config['mixing_dt']), physics=full)
        if config.get('compile'):
            progress('Enabling torch.compile kernels; first calls may take several minutes...')
            m.compile_thermodynamics()
        row['phase']='initial state'; write(output,row)
        h = rest[None].expand(config['cells'], -1).clone()
        depth = (interfaces[1:]+interfaces[:-1])[None]/2
        ct = (2+18*torch.exp(-depth/500)).expand_as(h).clone()
        sa = (35-.5*torch.exp(-depth/200)).expand_as(h).clone()
        if config['case'] in ('wave', 'forced'):
            # Localized, resolved free-surface pulse excites more than degree one.
            pulse = torch.exp((mesh.cell_xyz[:, 0:1]-1)/.015)
            pulse -= (mesh.cell_area[:, None]*pulse).sum()/mesh.cell_area.sum()
            h += config['wave_amplitude']*pulse*rest[None]/config['depth']
        if config['case'] == 'front':
            ct += 2*torch.tanh(mesh.cell_xyz[:, 1:2]/.03)*torch.exp(-depth/500)
        east = torch.stack((-mesh.edge_xyz[:, 1], mesh.edge_xyz[:, 0], torch.zeros(mesh.nEdges)), -1)
        speed = .5 if config['case'] in ('front', 'forced') else 0.
        u = (speed*(east*mesh.edge_normal).sum(-1))[:, None]*torch.exp(-depth/1000)
        state = m.state(u_e=u, h=h, conservative_temperature=ct, absolute_salinity=sa,
                        tracer=(1+.2*mesh.cell_xyz[:, 2:3]).expand_as(h)[..., None], tke=1e-4 if full else 0.)
        initial = state.ocean
        area = m.energy.mass_area(initial)
        mass = (area*h).sum()
        salt0 = (area*initial.salt_h).sum()
        tracer0 = (area[..., None]*initial.tracer_h).sum()
        row['phase']='initial energy'; write(output,row)
        progress('Evaluating initial energy (may compile kernels)...')
        energy0 = m.total_energy(state)
        forcing = {}
        if full:
            stress = .1*(east*mesh.edge_normal).sum(-1)
            forcing = dict(surface_stress=stress, shortwave=torch.full_like(h[:, 0], 200.),
                           surface_heat=torch.full_like(h[:, 0], -100.))
        row.update(status='running', phase='timestepping', min_dc_m=float(mesh.dc_edge.min()),
                   surface_layer_m=float(rest[0]), bottom_layer_m=float(rest[-1]),
                   external_wave_speed_m_s=math.sqrt(column.gravity*config['depth']),
                   mesh_sha256=hashlib.sha256(mesh.cell_xyz.numpy().tobytes()).hexdigest(),
                   max_energy_error_J_kg=0., max_mass_error_relative=0.,
                   max_salt_error_relative=0., max_tracer_error_relative=0.,
                   max_speed_m_s=0., min_h_m=float(h.min()), min_tke=0.,
                   max_local_budget_ratio=0., max_wave_courant=0.,
                   setup_wall_s=time.perf_counter()-started, samples=[])
        elapsed = 0.; external = 0.; run_started = time.perf_counter()
        total_steps=math.ceil(config['duration']/config['dt'])
        progress(f'Starting {total_steps} steps; first step may trigger additional compilation.')
        with torch.no_grad():
            while elapsed < config['duration']-1e-9:
                dt = min(config['dt'], config['duration']-elapsed)
                trial, report = m.step_with_report(state, dt, **forcing)
                added = 0.
                if full:
                    added = report.boundary.surface_work+report.heating.external_heat
                    trial, _ = m.remap_with_report(trial, trial.ocean.h.sum(-1, keepdim=True)*rest[None]/config['depth'])
                    row['max_local_budget_ratio'] = max(row['max_local_budget_ratio'], float(report.turbulence.budget_error))
                m.check(trial)
                state = trial; external += added; elapsed += dt
                s = state.ocean
                error = float((m.total_energy(state)-energy0-external)/mass)
                metrics = dict(energy_error_J_kg=error,
                    mass_error_relative=float((area*(s.h-initial.h)).sum()/mass),
                    salt_error_relative=float((area*(s.salt_h-initial.salt_h)).sum()/salt0),
                    tracer_error_relative=float((area[..., None]*(s.tracer_h-initial.tracer_h)).sum()/tracer0))
                for key, value in metrics.items():
                    row['max_'+key] = max(row['max_'+key], abs(value))
                row['max_speed_m_s'] = max(row['max_speed_m_s'], float(s.u_e.abs().max()))
                row['min_h_m'] = min(row['min_h_m'], float(s.h.min()))
                row['min_tke'] = min(row['min_tke'], float((state.tke_h/s.h).min()))
                row['max_wave_courant'] = max(row['max_wave_courant'], dt*(math.sqrt(column.gravity*float(s.h.sum(-1).max()))+float(s.u_e.abs().max()))/row['min_dc_m'])
                row.update(time_s=elapsed, completed_steps=row['completed_steps']+1,
                           run_wall_s=time.perf_counter()-run_started)
                if len(row['samples']) < 65 and (elapsed >= (len(row['samples'])+1)*config['duration']/64 or elapsed >= config['duration']):
                    row['samples'].append(dict(time_s=elapsed, **metrics))
                write(output, row)
                average=row['run_wall_s']/row['completed_steps']
                eta=average*(total_steps-row['completed_steps'])
                progress(f"step {row['completed_steps']}/{total_steps} | t={elapsed:g}s | wall={row['run_wall_s']:.1f}s | mean={average:.2f}s/step | ETA={eta:.1f}s | energy error={error:.3e} J/kg")
            row.update(status='completed', sypd=elapsed/(365*row['run_wall_s']))
            endpoint = dict(u=s.u_e, CT=s.temperature(), SA=s.salinity(), h=s.h,
                            k=state.tke_h/s.h, weight=area*s.h,
                            edge_weight=mesh.dc_edge[:, None]*mesh.dv_edge[:, None]*mesh.cell_to_edge(s.h))
            torch.save(endpoint, output.with_suffix('.pt'))
    except Exception as exc:
        row.update(status='rejected' if isinstance(exc,(ValueError,FloatingPointError)) else 'worker_failed', error_type=type(exc).__name__, error=str(exc))
    row['total_wall_s'] = time.perf_counter()-started
    write(output, row)
    return int(row['status'] != 'completed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--compare', type=Path, nargs=2, help=argparse.SUPPRESS)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cells', type=int, nargs='+', default=[16384])
    p.add_argument('--layers', type=int, default=40)
    p.add_argument('--depth', type=float, default=4000.)
    p.add_argument('--dt', type=float, nargs='+', default=[120,240,360,480,600,900,1200])
    p.add_argument('--reference-dt', type=float, default=60.)
    p.add_argument('--hours', type=float, default=6.)
    p.add_argument('--cases', nargs='+', choices=['rest','wave','front','forced'], default=['wave','front','forced'])
    p.add_argument('--mixing-dt', type=float, default=60.)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--lloyd', type=int, default=0)
    p.add_argument('--mesh', type=Path, help='Saved (N,3) .npy generators; no implicit extra relaxation')
    p.add_argument('--wave-amplitude', type=float, default=10.)
    p.add_argument('--timeout', type=float, default=43200., help='Wall seconds per trial')
    p.add_argument('--compile', action='store_true', help='Compile selected Roquet tensor kernels with Inductor; guards remain eager')
    p.add_argument('--verbose', action='store_true', help='Stream initialization and every accepted step with wall time and ETA')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--screen', action='store_true', help='Skip references: completion/budgets only; never qualifies a timestep')
    p.add_argument('--velocity-tolerance', type=float, default=.001)
    p.add_argument('--temperature-tolerance', type=float, default=.01)
    p.add_argument('--salinity-tolerance', type=float, default=.001)
    p.add_argument('--thickness-tolerance', type=float, default=.1)
    p.add_argument('--tke-tolerance', type=float, default=1e-5)
    p.add_argument('--energy-tolerance', type=float, default=1e-3)
    p.add_argument('--inventory-tolerance', type=float, default=1e-10)
    a = p.parse_args()
    if a.compare:
        import torch
        print(json.dumps(compare(*(torch.load(f, weights_only=True) for f in a.compare))))
        return 0
    if a.worker:
        return worker(json.loads(a.worker.read_text()), a.output)
    if min(a.cells)<16 or min(a.layers,a.depth,a.reference_dt,a.hours,a.mixing_dt,a.threads,a.timeout,*a.dt)<=0 or a.lloyd<0:
        p.error('positive sizes, timesteps, duration and timeout required; cells >=16')
    if a.mesh and len(a.cells)!=1:p.error('--mesh requires one --cells value')
    limits = dict(u=a.velocity_tolerance, CT=a.temperature_tolerance, SA=a.salinity_tolerance,
                  h=a.thickness_tolerance, k=a.tke_tolerance, energy=a.energy_tolerance, inventory=a.inventory_tolerance)
    if any(not math.isfinite(x) or x<=0 for x in [*limits.values(),a.hours,a.depth,a.reference_dt,a.mixing_dt,*a.dt]):p.error('finite positive values required')
    from importlib.util import find_spec
    from importlib.metadata import version
    source = hashlib.sha256()
    for folder in (HERE, ROOT/'validation/temperature_coordinate', ROOT/'validation/ocean_column', Path(find_spec('vfs').origin).parent):
        for f in sorted(folder.rglob('*.py')):source.update(str(f.relative_to(folder)).encode()+f.read_bytes())
    provenance = dict(source_sha256=source.hexdigest(), torch=version('torch'), python=sys.version,
                      mesh_file_sha256=hashlib.sha256(a.mesh.read_bytes()).hexdigest() if a.mesh else None)
    summary = dict(provenance=provenance, limits=limits, scope='finite-duration Roquet CPU float64; no climate guarantee', results=[])
    def trial(cells, case, dt):
        cfg = dict(cells=cells,layers=a.layers,depth=a.depth,dt=dt,duration=a.hours*3600,
                   case=case,compile=a.compile,verbose=a.verbose,mixing_dt=a.mixing_dt,threads=a.threads,lloyd=a.lloyd,
                   mesh=str(a.mesh.resolve()) if a.mesh else None,wave_amplitude=a.wave_amplitude,provenance=provenance)
        name=f'{cells}x{a.layers}-{case}-dt{dt:g}'
        path=a.output/(name+'.json');request=a.output/(name+'.config.json')
        if a.resume and path.exists():
            row=json.loads(path.read_text())
            if row.get('config')==cfg and row.get('status')=='completed' and path.with_suffix('.pt').exists():return row,path
        write(request,cfg)
        print('Running',name,flush=True)
        with path.with_suffix('.log').open('w') as log:
            try:
                returncode=run_logged([sys.executable,'-I',str(Path(__file__).resolve()),'--worker',str(request.resolve()),'--output',str(path.resolve())],log,a.timeout,a.verbose)
                row=json.loads(path.read_text()) if path.exists() else dict(config=cfg,status='worker_failed')
                if returncode and row.get('status') in ('initializing','running','completed'):row.update(status='worker_failed',returncode=returncode)
            except subprocess.TimeoutExpired:
                row=json.loads(path.read_text()) if path.exists() else dict(config=cfg)
                row['status']='timeout'
        write(path,row)
        print(name,row['status'],row.get('error',''),flush=True)
        return row,path
    def compare_files(first, second):
        # Keep large tensor libraries/endpoints out of the coordinator process.
        result=subprocess.run([sys.executable,'-I',str(Path(__file__).resolve()),
            '--compare',str(first),str(second),'--output',str(a.output)],
            capture_output=True,text=True,check=True,timeout=a.timeout)
        return json.loads(result.stdout)
    for cells in a.cells:
        case_qualified=[]
        for case in a.cases:
            reference=None;ref_ok=False
            if not a.screen:
                coarse,cp=trial(cells,case,a.reference_dt)
                fine,fp=trial(cells,case,a.reference_dt/2)
                if coarse['status']==fine['status']=='completed':
                    reference=fp.with_suffix('.pt').resolve()
                    ref_errors=compare_files(cp.with_suffix('.pt').resolve(),reference)
                    tighter={k:v/4 for k,v in limits.items()}
                    ref_ok=qualifies(coarse,ref_errors,tighter) and qualifies(fine,ref_errors,tighter)
                else:ref_errors=None
                summary['results'].append(dict(cells=cells,case=case,reference_check=True,reference_qualified=ref_ok,errors=ref_errors))
            accepted=[]
            for dt in sorted(set(a.dt)):
                row,path=trial(cells,case,dt)
                errors=compare_files(path.with_suffix('.pt').resolve(),reference) if row['status']=='completed' and reference is not None else None
                passed=ref_ok and qualifies(row,errors,limits)
                if passed:accepted.append(dt)
                summary['results'].append(dict(cells=cells,case=case,dt=dt,status=row['status'],qualified=passed,errors=errors,result=path.name))
                write(a.output/'summary.json',summary)
            case_qualified.append(set(accepted))
        common=set.intersection(*case_qualified)
        summary.setdefault('largest_tested_qualified_dt',{})[str(cells)]=max(common) if common else None
        write(a.output/'summary.json',summary)
    print(json.dumps(summary.get('largest_tested_qualified_dt',{})),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
