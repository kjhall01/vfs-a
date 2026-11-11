"""Measure the ocean Gibbs/entropy candidate; no scientific accuracy gates."""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import torch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


faces = load('ocean_gibbs_faces', 'ocean_gibbs_faces.py')
candidate = load('ocean_entropy_candidate', 'ocean_entropy_research.py')
previous = load('ocean_seawater_parent', 'probe_ocean_seawater.py')
wave_reference = load('ocean_wave_reference', 'verify_ocean_jmd95.py')


def case(kind='front', coast=False):
    model, physical = previous.previous.make_case('front' if kind == 'near_fresh' else 'tilted' if kind == 'upper_salt' else kind, coast)
    x = model.mesh.cell_xyz[:, :1]
    if kind in ('near_fresh', 'upper_salt'):
        salt = torch.where(x > 0, 35., 1e-10) if kind == 'near_fresh' else torch.where(x < 0, 42.-1e-10, 41.99)
        physical = physical.with_tensors(physical.u_e, physical.h, physical.temp_h, salt*physical.h, physical.tracer_h)
        if model.domain is not None:
            physical = model.domain.project_state(physical)
    ref = previous.reference()
    state = candidate.EntropyState.from_temperature(physical, ref)
    return model, state, candidate.EntropyEnergy(model.dynamics.energy, ref)


def energy_change(energy, before, after):
    old, a = energy.fields(before)
    new, b = energy.fields(after)
    dynamic = energy.base.terms(new)['total']-energy.base.terms(old)['total']
    thermal = (energy.mass_area(before)*(after.h*b['h0']-before.h*a['h0'])).sum()
    return dynamic+thermal


def audit(kind, coast=False):
    model, state, energy = case(kind, coast)
    dt = 10.
    rhs, work = candidate.evaluate(model, state, dt, energy)
    rate, scale = candidate.contraction(energy.gradients(state), rhs)
    m = energy.mass_area(state)
    entropy_rate = (m*rhs.entropy_h).sum()
    production = (m*work.entropy_source).sum()
    entropy_scale = (m*rhs.entropy_h).abs().sum()
    end, stages = candidate.step(model, state, dt, energy)
    old, new = model.budgets(state.physical_state(energy.reference)), model.budgets(end.physical_state(energy.reference))
    entropy_change = (m*(end.entropy_h-state.entropy_h)).sum()
    predicted = dt*sum(w*(m*l.entropy_source).sum() for w, l in zip((1/6, 1/6, 2/3), stages))
    return dict(kind=kind, coast=coast, dt_s=dt, energy_rate_W=float(rate),
        normalized_energy_rate=float(rate/scale), absolute_local_work_W=float(scale),
        entropy_rate_W_K=float(entropy_rate), nonnegative_entropy_production_W_K=float(production),
        entropy_flux_roundoff_normalized=float((entropy_rate-production)/entropy_scale),
        heat_W=float(work.heat.sum()), mechanically_paired_work_W=float(work.face_work.clamp_min(0).sum()),
        limited_active_faces=int((work.transport.beta[:, 1] < 1).sum()),
        limited_passive_faces=int((work.transport.beta[:, 2:] < 1).sum()),
        maximum_force_correction_m_s2=float(work.force_correction.abs().max()),
        maximum_gibbs_identity_residual_J_kg=float(work.gibbs_identity.abs().max()),
        finite_entropy_change_J_K=float(entropy_change), predicted_entropy_change_J_K=float(predicted),
        finite_entropy_budget_error_over_initial_absolute_entropy=float((entropy_change-predicted)/(m*state.entropy_h.abs()).sum()),
        energy_change_J_m2=float(energy_change(energy, state, end)/(m.sum()/energy.base.rho0)),
        stage_salinity_range=[min(float((l.transport.amounts[..., 1]/l.transport.volume).min()) for l in stages),
                              max(float((l.transport.amounts[..., 1]/l.transport.volume).max()) for l in stages)],
        minimum_stage_passive=min(float((l.transport.amounts[..., 2:]/l.transport.volume[..., None]).min()) for l in stages),
        relative_inventory_errors={k: float(((new[k]-old[k])/old[k].abs()).abs().max())
                                   for k in ('volume', 'salinity_content', 'tracer_content')})


def zero_speed():
    rows = []
    for kind in ('front', 'near_fresh', 'upper_salt'):
        model, state, energy = case(kind)
        for factor in (1., 1e-3, 1e-6, 1e-9, 1e-12, 0.):
            for sign in (-1., 1.):
                _, work = candidate.evaluate(model, replace(state, u_e=sign*factor*state.u_e), 10., energy)
                rows.append(dict(kind=kind, velocity_factor=sign*factor,
                    maximum_force_correction_m_s2=float(work.force_correction.abs().max()),
                    entropy_production_W_K=float((energy.mass_area(state)*work.entropy_source).sum()),
                    limited_active_faces=int((work.transport.beta[:, 1] < 1).sum())))
    return rows


def single_edge_zero():
    """Hold every other edge fixed, including their active limiting."""
    model, state, energy = case('upper_salt')
    _, initial = candidate.evaluate(model, state, 10., energy)
    index = int(initial.force_correction.abs().flatten().argmax())
    rows = []
    for speed in (1e-2, 1e-3, 1e-6, 1e-9, 0.):
        for sign in (-1., 1.):
            velocity = state.u_e.clone()
            velocity.flatten()[index] = sign*speed
            _, work = candidate.evaluate(model, replace(state, u_e=velocity), 10., energy)
            rows.append(dict(edge=index//state.h.shape[-1], layer=index%state.h.shape[-1],
                velocity_m_s=sign*speed, force_correction_m_s2=float(work.force_correction.flatten()[index]),
                limited_active_faces=int((work.transport.beta[:, 1] < 1).sum())))
    return rows


def pressure_refinement():
    rows = []
    for n in (64, 256, 1024):
        model, physical = wave_reference.make_case(n, lloyd=20)
        ref = previous.reference()
        state = candidate.EntropyState.from_temperature(physical, ref)
        energy = candidate.EntropyEnergy(model.dynamics.energy, ref)
        prediction, work = candidate.evaluate(model, state, 10., energy)
        exact = wave_reference.independent_pressure(physical)
        weights = (model.mesh.dc_edge*model.mesh.dv_edge)[:, None]
        rows.append(dict(cells=n, rms_error_m_s2=wave_reference.rms(prediction.u_e-exact, weights),
            reference_rms_m_s2=wave_reference.rms(exact, weights),
            maximum_error_m_s2=float((prediction.u_e-exact).abs().max())))
    return rows


def temporal_refinement():
    rows = []
    for dt in (1000., 500., 250.):
        model, start, energy = case()
        end = start
        for _ in range(round(1000/dt)):
            end, _ = candidate.step(model, end, dt, energy)
        rows.append(dict(duration_s=1000., dt_s=dt,
            energy_change_J_m2=float(energy_change(energy, start, end)/(energy.mass_area(start).sum()/energy.base.rho0))))
    return rows


def internal_wave(n):
    """Repeat #31's continuum two-layer reference with this candidate step."""
    from vfs.geometry.mesh import VoronoiMesh
    from vfs.ocean import Ocean, OceanColumn
    from vfs.ocean.jmd95 import JMD95Dynamics
    from vfs.ocean.eos import jmd95_seawater_density
    mesh = VoronoiMesh(n_cells=n, radius=1e5, lloyd_iters=10, verbose=False)
    h0, theta, salt = torch.tensor([500., 2500.]), torch.tensor([17., 4.]), torch.tensor([34., 35.5])
    model = Ocean(mesh, OceanColumn(2, total_depth=3000., rest_thickness=h0), dynamics=JMD95Dynamics(omega=0.))
    g, rho = model.dynamics.gravity, model.dynamics.rho0
    def buoyancy(layer, z):
        p = torch.tensor(-rho*g*z/1e4)
        return g/rho*(jmd95_seawater_density(salt[layer], theta[layer], p)-jmd95_seawater_density(torch.tensor(35.), torch.tensor(10.), p))
    a = g+buoyancy(0, 0.)
    b = a-buoyancy(0, -500.)+buoyancy(1, -500.)
    c = torch.stack([torch.stack([a, a]), torch.stack([a, b])])
    eigenvalues, vectors = torch.linalg.eig(torch.diag(h0)@c)
    index = eigenvalues.real.argmin()
    speed2 = float(eigenvalues[index].real)
    vector = vectors[:, index].real
    vector = vector/vector.abs().max()
    area = mesh.cell_area[:, None]
    mode = mesh.cell_xyz[:, 2:3]
    mode = mode-(area*mode).sum()/area.sum()
    amplitude = .1
    physical = model.state(u_e=torch.zeros(mesh.nEdges, 2), h=h0[None, :]+amplitude*mode*vector[None, :],
                           temperature=theta.expand(n, 2), salinity=salt.expand(n, 2))
    ref = previous.reference()
    start = candidate.EntropyState.from_temperature(physical, ref)
    energy = candidate.EntropyEnergy(model.dynamics.energy, ref)
    frequency = math.sqrt(2*speed2)/mesh.radius
    duration = .05*2*math.pi/frequency
    dt = min(30., float(mesh.dc_edge.min())/math.sqrt(float(eigenvalues.real.max()))*.15)
    steps = math.ceil(duration/dt)
    dt = duration/steps
    end = start
    for _ in range(steps):
        end, _ = candidate.step(model, end, dt, energy)
    projection = (area*mode*(end.h-h0[None, :])).sum(0)/(area*mode.square()).sum()
    expected = amplitude*vector*math.cos(frequency*duration)
    return dict(cells=n, period_s=2*math.pi/frequency, fraction_of_period=.05, steps=steps, dt_s=dt,
        projection_error_over_amplitude=float((projection-expected).norm()/amplitude),
        energy_change_J_m2=float(energy_change(energy, start, end)/area.sum()),
        minimum_thickness_m=float(end.h.min()), salinity_range=[float((end.salt_h/end.h).min()), float((end.salt_h/end.h).max())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--waves', action='store_true', help='also run 0.05-period internal waves at 64 and 256 cells')
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    with torch.no_grad():
        cases = [audit(kind, coast) for kind in ('front', 'salt_only', 'tilted', 'near_fresh', 'upper_salt') for coast in (False, True)]
        speeds = zero_speed()
        single_edge = single_edge_zero()
        temporal = temporal_refinement()
    # The independent pressure reference intentionally uses AD of the old EOS.
    pressure = pressure_refinement()
    waves = []
    if args.waves:
        with torch.no_grad():
            for n in (64, 256):
                row = internal_wave(n)
                waves.append(row)
                print(json.dumps({'internal_wave': row}), flush=True)
    result = dict(status='research candidate; strictly positive active salt, no local-extremum guarantee',
        cases=cases, zero_velocity=speeds, single_edge_zero_velocity=single_edge,
        pressure_refinement=pressure, temporal_refinement=temporal, internal_wave=waves,
        files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), Path(faces.__file__), Path(candidate.__file__), previous.DATA]},
        environment=dict(torch=torch.__version__, dtype='float64', device='cpu', threads=torch.get_num_threads()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'max_normalized_work': max(abs(c['normalized_energy_rate']) for c in cases),
                      'entropy_production_W_K': {c['kind']+('_coast' if c['coast'] else ''): c['nonnegative_entropy_production_W_K'] for c in cases},
                      'pressure_refinement': pressure, 'temporal_refinement': temporal}), flush=True)


if __name__ == '__main__':
    main()
