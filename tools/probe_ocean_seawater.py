"""Seawater-reference and enthalpy-coordinate measurements, without accuracy gates.

Requires gsw==3.6.23 only for independent reference evaluations. Forward
research code uses torch, and installed VFS runtime remains unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


surface = load('ocean_surface_reference', 'ocean_surface_gibbs.py')
enthalpy = load('ocean_enthalpy_reference', 'ocean_enthalpy_research.py')
previous = load('ocean_thermal_reference', 'probe_ocean_thermal_work.py')
DATA = Path(__file__).with_name('data')/'surface-gibbs-reference.json'


def reference():
    return surface.SurfaceGibbs.from_file(DATA, temperature_scale='IPTS-68')


def case(kind='front', coast=False):
    model, physical = previous.make_case(kind, coast)
    ref = reference()
    state = enthalpy.EnthalpyState.from_temperature(physical, ref)
    return model, state, enthalpy.EnthalpyEnergy(model.dynamics.energy, ref)


def errors(value, target):
    delta = torch.as_tensor(value)-torch.as_tensor(target)
    return dict(maximum_absolute=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()))


def surface_accuracy():
    import gsw
    import numpy as np
    if gsw.__version__ != '3.6.23':
        raise RuntimeError('reference measurement requires gsw==3.6.23')
    ref = reference()
    rng = np.random.default_rng(20261106)
    s, t = np.meshgrid(np.linspace(0, 42, 19), np.linspace(-2.5, 40, 23))
    s = np.concatenate((s.ravel(), rng.uniform(0, 42, 4096), [1e-10, 1e-6, 1e-3]))
    t = np.concatenate((t.ravel(), rng.uniform(-2.5, 40, 4096), [10., 10., 10.]))
    sp, theta = torch.tensor(s), torch.tensor(t/ref.to_90)
    sa, kelvin = s*ref.data['ups'], t+273.15
    v = ref.evaluate(sp, theta)
    target = dict(gibbs=gsw.gibbs(0, 0, 0, sa, t, 0), h0=gsw.enthalpy_t_exact(sa, t, 0),
        entropy=gsw.entropy_from_t(sa, t, 0), cp=gsw.cp_t_exact(sa, t, 0),
        h_theta=-kelvin*gsw.gibbs(0, 2, 0, sa, t, 0)*ref.to_90,
        entropy_theta=-gsw.gibbs(0, 2, 0, sa, t, 0)*ref.to_90)
    result = {k: errors(v[k], expected) for k, expected in target.items()}
    positive = s > 0
    vpos = ref.evaluate(sp[positive], theta[positive], entropy_derivatives=True)
    gs = gsw.gibbs(1, 0, 0, sa[positive], t[positive], 0)
    gst = gsw.gibbs(1, 1, 0, sa[positive], t[positive], 0)
    result['h_s'] = errors(vpos['h_s'], (gs-kelvin[positive]*gst)*ref.data['ups'])
    result['entropy_s'] = errors(vpos['entropy_s'], -gst*ref.data['ups'])
    recovered = ref.temperature_from_enthalpy(sp, v['h0'])
    result['temperature_inverse'] = errors(recovered, theta)
    result.update(samples=len(s), gsw_version=gsw.__version__, minimum_cp_J_kg_K=float(v['cp'].min()),
                  training=ref.data['training'])
    return result


def hybrid_accuracy():
    """Difference from full GSW at depth; NOT a claim of full TEOS-10 density."""
    import gsw
    model, state, energy = case()
    theta90, sp, z = torch.meshgrid(torch.tensor([0., 10., 25.]), torch.tensor([5., 35., 40.]),
                                  torch.tensor([0., -500., -2000., -4000.]), indexing='ij')
    theta = theta90/energy.reference.to_90
    ref = energy.reference.evaluate(sp, theta, entropy_derivatives=True)
    _, pt, ps, _ = energy.base.parcel_potential(theta, sp, z)
    temperature = ref['kelvin']*(1+pt/ref['h_theta'])
    chemical = ref['h_s']+ps-temperature*ref['entropy_s']
    sa = (sp*energy.reference.data['ups']).numpy()
    pressure = (-model.dynamics.rho0*model.dynamics.gravity*z/1e4).numpy()
    # A parcel at surface theta has this temperature when referenced to p.
    in_situ = gsw.pt_from_t(sa, theta90.numpy(), 0, pressure)
    mu = gsw.gibbs(1, 0, 0, sa, in_situ, pressure)*energy.reference.data['ups']
    return [dict(depth_m=float(-z[0, 0, k]),
                 temperature_K=errors(temperature[..., k], in_situ[..., k]+273.15),
                 chemical_J_kg_PSU=errors(chemical[..., k], mu[..., k])) for k in range(z.shape[-1])]


def audit(kind, coast, scheme):
    model, state, energy = case(kind, coast)
    dt = 10.
    rhs, ledger = enthalpy.evaluate(model, state, dt, energy, scheme)
    rate, scale = enthalpy.contraction(energy.gradients(state), rhs)
    end, stages = enthalpy.step(model, state, dt, energy, scheme)
    area = state.mesh.cell_area[:, None]
    physical = state.physical_state(energy.reference)
    final = end.physical_state(energy.reference)
    dynamic = energy.base.terms(final)['total']-energy.base.terms(physical)['total']
    thermal = (energy.mass_area(state)*(end.enthalpy_h-state.enthalpy_h)).sum()
    prediction = dt*sum(w*(area*l.enthalpy_content_source).sum() for w, l in zip((1/6, 1/6, 2/3), stages))
    actual = (area*(end.enthalpy_h-state.enthalpy_h)).sum()
    before, after = model.budgets(physical), model.budgets(final)
    entropy = {}
    if kind != 'zero_salt':
        a, b = enthalpy.entropy_terms(energy, state, ledger, dt)
        nodal, magnitude = enthalpy.contraction(energy.entropy_gradients(state), rhs)
        entropy = dict(advective_W_K=float(a.sum()), conversion_W_K=float(b.sum()),
            total_W_K=float(a.sum()+b.sum()), nodal_W_K=float(nodal),
            face_nodal_normalized_difference=float((a.sum()+b.sum()-nodal)/magnitude))
    return dict(kind=kind, coast=coast, scheme=scheme, dt_s=dt,
        energy_rate_W=float(rate), absolute_local_work_W=float(scale), normalized_energy_rate=float(rate/scale),
        limiter_work_W=float(ledger.face_work.sum()), thermal_power_W=float(ledger.thermal_power.sum()),
        entropy=entropy, potential_enthalpy_content_change=float(actual), predicted_content_change=float(prediction),
        content_budget_residual=float(actual-prediction),
        total_energy_change_J_m2=float((dynamic+thermal)/(energy.mass_area(state).sum()/energy.base.rho0)),
        minimum_stage_salinity=min(float((l.transport.amounts[..., 1]/l.transport.volume).min()) for l in stages),
        minimum_stage_passive=min(float((l.transport.amounts[..., 2:]/l.transport.volume[..., None]).min()) for l in stages),
        relative_inventory_errors={k: float(((after[k]-before[k])/before[k].abs()).abs().max())
                                   for k in ('volume', 'salinity_content', 'tracer_content')})


def convergence():
    rows = []
    for dt in (1000., 500., 250.):
        model, start, energy = case()
        end = start
        for _ in range(round(1000/dt)):
            end, _ = enthalpy.step(model, end, dt, energy)
        old = energy.base.terms(start.physical_state(energy.reference))['total']
        new = energy.base.terms(end.physical_state(energy.reference))['total']
        thermal = (energy.mass_area(start)*(end.enthalpy_h-start.enthalpy_h)).sum()
        rows.append(dict(dt_s=dt, duration_s=1000., energy_change_J_m2=float((new-old+thermal)/(energy.mass_area(start).sum()/energy.base.rho0))))
    return rows


def exchange():
    rows = []
    for mobility in (1e14, 1e24):
        model, state, energy = case('salt_only')
        rhs, info = enthalpy.thermodynamic_exchange(model, state, 10., energy,
            heat_conductance=1e9, salt_mobility=mobility)
        rate, scale = enthalpy.contraction(energy.gradients(state), rhs)
        entropy_rate, magnitude = enthalpy.contraction(energy.entropy_gradients(state), rhs)
        end = state+10*rhs
        finite = {}
        try:
            enthalpy.validate(model, end, energy)
            finite['entropy_change_J_K'] = float(energy.entropy(end)-energy.entropy(state))
        except (ValueError, FloatingPointError) as error:
            finite['rejected'] = str(error)
        rows.append(dict(salt_mobility=mobility, normalized_energy_rate=float(rate/scale),
            entropy_rate_W_K=float(entropy_rate), nonnegative_face_production_W_K=float(info['entropy_production'].sum()),
            entropy_identity_normalized_difference=float((entropy_rate-info['entropy_production'].sum())/magnitude),
            minimum_salt_limiter=float(info['salt_limiter'].min()), minimum_euler_salinity=float((end.salt_h/end.h).min()),
            maximum_euler_salinity=float((end.salt_h/end.h).max()), finite_euler=finite))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    cases = [audit(kind, coast, scheme) for kind in ('front', 'salt_only', 'tilted', 'zero_salt')
             for coast in (False, True) for scheme in ('donor', 'fct')]
    result = dict(status='research only; bounded advection still fails the tilted-column entropy audit',
        surface_reference=surface_accuracy(), hybrid_at_depth=hybrid_accuracy(), cases=cases,
        short_temporal_refinement=convergence(), separate_exchange=exchange(),
        files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), DATA, Path(surface.__file__), Path(enthalpy.__file__)]},
        environment=dict(torch=torch.__version__, dtype='float64', device='cpu', threads=torch.get_num_threads()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'surface_reference': result['surface_reference'],
                      'tilted_entropy_W_K': [x['entropy']['total_W_K'] for x in cases if x['kind'] == 'tilted'],
                      'max_normalized_work': max(abs(x['normalized_energy_rate']) for x in cases)}))


if __name__ == '__main__':
    main()
