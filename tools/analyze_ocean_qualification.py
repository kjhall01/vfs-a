"""Reduce completed campaign measurements; no empirical acceptance thresholds."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch


def load_driver():
    spec = importlib.util.spec_from_file_location('qualification_driver', Path(__file__).with_name('qualify_ocean_entropy.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def phase_metrics(times, phase, envelope, period):
    """Unwrap a resolved modal phase; envelope includes mode redistribution."""
    times = np.asarray(times)
    unwrapped = np.unwrap(phase)
    return dict(frequency_over_reference=float(np.polyfit(times, unwrapped, 1)[0]*period/(2*math.pi)),
                final_phase_error_rad=float(unwrapped[-1]-2*math.pi*times[-1]/period),
                envelope_min=float(np.min(envelope)), envelope_max=float(np.max(envelope)))


def geometry_reference(driver, cells, times):
    """Independent linear -div grad spectrum, not a nonlinear solver oracle.

    An initial continuum Y10 is not an eigenvector of the mesh Laplacian.
    Project its linear superposition onto itself to expose geometric phase
    bias and beating without any timestep or transport dissipation.
    """
    model, start, energy, h0, mode, vector, omega, fast = driver.wave_setup(cells)
    mesh = model.mesh
    root_area = mesh.cell_area.sqrt()
    laplacian = -mesh.div(mesh.grad(torch.eye(cells)))
    symmetric = root_area[:, None]*laplacian/root_area[None, :]
    symmetry_error = float((symmetric-symmetric.T).abs().max())
    eigenvalues, eigenvectors = torch.linalg.eigh((symmetric+symmetric.T)/2)
    initial = root_area*mode[:, 0]
    weights = (eigenvectors.T@initial).square()/initial.square().sum()
    speedsq = omega**2*mesh.radius**2/2
    frequencies = (speedsq*eigenvalues.clamp_min(0)).sqrt()
    t = torch.tensor(times)
    projection = .1*(torch.cos(t[:, None]*frequencies)*weights).sum(-1)
    rate = -.1*(torch.sin(t[:, None]*frequencies)*frequencies*weights).sum(-1)
    phase = torch.atan2(-rate/omega, projection)
    envelope = ((projection.square()+(rate/omega).square()).sqrt()/.1)
    return dict(scope='linear continuum vertical mode with mesh scalar Laplacian; no nonlinear dynamics or time integrator',
                weighted_laplacian_symmetry_error=symmetry_error,
                minimum_eigenvalue=float(eigenvalues.min()),
                projected_amplitude_m=projection.tolist(),
                **phase_metrics(times, phase.numpy(), envelope.numpy(), 2*math.pi/omega))



def jet_reference_scales(driver, cells):
    w = driver.p.load('qualification_scale_williamson', 'verify_ocean_williamson.py')
    mesh = driver.VoronoiMesh(n_cells=cells, lloyd_iters=20, verbose=False)
    model = w.make_model(mesh)
    state = w.initial_state(model)
    edge = (mesh.dc_edge*mesh.dv_edge)[:, None]
    return dict(initial_velocity_rms_m_s=driver.rms(state.u_e, edge),
                initial_kinetic_energy_J_m2=float(model.dynamics.energy.terms(state)['kinetic']/mesh.cell_area.sum()))


def summarize(directory):
    reports = []
    initial_reports = {}
    for path in sorted(directory.glob('*.json')):
        data = json.loads(path.read_text())
        if data.get('scope') == 'initial tendencies only; no integration':
            if data['status'] != 'completed':
                raise ValueError(f'{path.name}: initial case is {data["status"]}')
            initial_reports[path.name] = {k:v for k,v in data.items() if k not in ('source_sha256', 'lloyd_diagnostics')}
            initial_reports[path.name]['lloyd_final'] = data['lloyd_diagnostics']['final']
        if 'kind' not in data:
            continue
        if data['status'] != 'completed':
            raise ValueError(f'{path.name}: campaign case is {data["status"]}')
        reports.append((path.name, data))
    driver = load_driver()
    result = {'candidate_commit': '705720c79e38c2f07de39d45a33c9ede50707b87', 'cases': {}, 'initial_balance_cases': initial_reports}
    for name, data in reports:
        row = dict(kind=data['kind'], cells=data['cells'], completed_steps=data['completed_steps'])
        if data['kind'] == 'transport':
            row.update(profile=data['profile'], fields=data['fields'])
        else:
            row.update(final=data['samples'][-1], dt_s=data['dt_s'])
            if data['kind'] == 'wave':
                model, start, energy, h0, mode, vector, omega, _ = driver.wave_setup(data['cells'])
                times = [s['time_s'] for s in data['samples']]
                row['modal_metrics'] = phase_metrics(times, [s['phase_rad'] for s in data['samples']], [s['envelope_over_initial'] for s in data['samples']], data['period_s'])
                row['geometry_reference'] = geometry_reference(driver, data['cells'], times)
                row['projection_vs_geometry_rms_m'] = float(np.sqrt(np.mean((np.array([s['modal_amplitude_m'] for s in data['samples']])-row['geometry_reference']['projected_amplitude_m'])**2)))
                area = start.mesh.cell_area[:, None]
                speed_squared = omega**2*model.mesh.radius**2/2
                row['initial_linear_wave_energy_J_m2'] = float(.5*energy.base.rho0*speed_squared*(vector.square()/h0).sum()*(area*(.1*mode).square()).sum()/area.sum())
                mass = energy.mass_area(start)
                row['entropy_change_over_initial_absolute_entropy'] = float(row['final']['entropy_change_J_K_m2']*start.mesh.cell_area.sum()/(mass*start.entropy_h.abs()).sum())
            else:
                row.update(profile=data['profile'], initial_momentum_rms_m_s2=data['initial_momentum_rms_m_s2'])
                if data['profile'] == 'jet':
                    row.update(jet_reference_scales(driver, data['cells']))
                    row['final_velocity_error_over_initial_rms'] = row['final']['u_error_rms_m_s']/row['initial_velocity_rms_m_s']
                    row['energy_change_over_initial_kinetic_energy'] = row['final']['energy_change_J_m2']/row['initial_kinetic_energy_J_m2']
        result['cases'][name] = row
    result['temporal_comparisons'] = {}
    for kind, field in [('wave', 'modal_amplitude_m'), ('balance', 'u_error_rms_m_s')]:
        rows = sorted([r for r in result['cases'].values() if r['kind']==kind and r['cells']==(32 if kind=='wave' else 64) and (kind=='wave' or r.get('profile')=='jet')], key=lambda r:r['dt_s'])
        if len(rows)==2:
            fine, coarse = rows
            result['temporal_comparisons'][kind] = dict(scope='differences of endpoint diagnostics, not a norm of full-state differences',
                coarse_dt_s=coarse['dt_s'], fine_dt_s=fine['dt_s'], diagnostic=field,
                fine_minus_coarse=fine['final'][field]-coarse['final'][field],
                coarse_over_fine_energy_change=coarse['final']['energy_change_J_m2']/fine['final']['energy_change_J_m2'])
    for field in ('salinity', 'passive', 'entropy'):
        rows = sorted([d for _, d in reports if d['kind']=='transport' and d['profile']=='smooth'], key=lambda d:d['cells'])
        result[f'smooth_{field}_orders'] = [math.log(a['fields'][field]['rms']/b['fields'][field]['rms'])/math.log(b['cells']/a['cells']) for a,b in zip(rows, rows[1:])]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    with torch.no_grad(): result=summarize(args.directory)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__': main()
