"""Explicit portable fixture policy; numerical package code is never modified.

The manifest enumerates every leaf. Bounds are field/component-scaled absolute
errors, not pointwise relative tolerances. See docs/PORTABLE_REGRESSION.md.
"""
import json
import math
from pathlib import Path

import torch

EPS128 = 128 * torch.finfo(torch.float64).eps
TOLERANCES = {
    'state': 5e-13,
    'dry_helper': 5e-13,
    'tendency': 2e-12,
    'gradient': 2e-13,
    'inventory': EPS128,
    'ledger': EPS128,
    'energy': 2e-13,
    'jw_thermal': 1e-11,
    'jw_tendency': 2e-10,
    'limiter_force': 1e-9,
    'limiter_heat': 2e-11,
    'cfl': EPS128,
    'energy_rate': EPS128,
    'rotation_work': EPS128,
    'rotation_relative': EPS128,
}
FIELDS = ('u_e', 'ps', 'entropy_content', 'water_content')
MANIFEST = json.loads(Path(__file__).with_name('fixture_quantities.json').read_text())


def require(condition, label):
    if not bool(condition):
        raise AssertionError(label)


def flatten_pair(actual, reference, manifest, path=''):
    """Reject unknown containers as well as unknown leaves; never broadcast."""
    require(type(actual) is type(reference), f'{path}: type mismatch')
    if isinstance(reference, (dict, list, tuple)):
        require(not path or any(p.startswith(path + '/') for p in manifest),
                f'{path}: unmapped container')
        if isinstance(reference, dict):
            require(actual.keys() == reference.keys(), f'{path}: dictionary keys mismatch')
            keys = reference.keys()
        else:
            require(len(actual) == len(reference), f'{path}: sequence length mismatch')
            keys = range(len(reference))
        for key in keys:
            child = f'{path}/{key}' if path else str(key)
            yield from flatten_pair(actual[key], reference[key], manifest, child)
    else:
        require(path in manifest, f'{path}: unmapped quantity')
        if isinstance(reference, torch.Tensor):
            require(actual.shape == reference.shape, f'{path}: shape mismatch')
            require(actual.dtype == reference.dtype == torch.float64, f'{path}: dtype mismatch')
            require(actual.device.type == reference.device.type == 'cpu', f'{path}: CPU fixture required')
            require(torch.isfinite(actual).all() and torch.isfinite(reference).all(),
                    f'{path}: nonfinite tensor')
        elif type(reference) is float:
            require(math.isfinite(actual) and math.isfinite(reference), f'{path}: nonfinite scalar')
        yield path, actual, reference


def scaled_error(actual, reference, quantity, components=0, scale=None):
    """Return maximum fraction of budget; zero reference scale permits no error."""
    require(quantity in TOLERANCES, f'unknown quantity category: {quantity}')
    a, b = torch.as_tensor(actual, dtype=torch.float64), torch.as_tensor(reference, dtype=torch.float64)
    require(a.shape == b.shape, 'scaled error: shape mismatch')
    require(torch.isfinite(a).all() and torch.isfinite(b).all(), 'scaled error: nonfinite input')
    if components:
        require(b.ndim > 0 and b.shape[-1] == components, 'component axis mismatch')
        a, b = a.reshape(-1, components), b.reshape(-1, components)
        error, natural_scale = (a - b).abs().amax(0), b.abs().amax(0)
    else:
        error, natural_scale = (a - b).abs().amax(), b.abs().amax()
    scale = natural_scale if scale is None else torch.as_tensor(scale, dtype=torch.float64)
    require(torch.isfinite(scale).all() and (scale >= 0).all(), 'invalid reference scale')
    bound = TOLERANCES[quantity] * scale
    require((error <= bound).all(),
            f'{quantity} tolerance: error={error.tolist()}, bound={bound.tolist()}')
    # Reporting only: avoid division by zero without introducing a tolerance floor.
    ratio = torch.zeros_like(error)
    positive = bound > 0
    ratio[positive] = error[positive] / bound[positive]
    return float(ratio.max())


def work_scale(case):
    scale = sum((case['gradients'][k] * case['rhs'][k]).abs().sum() for k in FIELDS)
    require(torch.isfinite(scale), 'energy work scale: nonfinite')
    return scale


def conserved(final, initial, label, scale=None):
    scale = initial.abs() if scale is None else scale
    require(torch.isfinite(final).all() and torch.isfinite(initial).all()
            and torch.isfinite(scale).all(), f'{label}: nonfinite conservation data')
    delta = final - initial
    require((delta.abs() <= EPS128 * scale).all(), f'{label}: conservation')
    return delta.tolist()


def positive(value, label, strict=True):
    require(torch.isfinite(value).all(), f'{label}: nonfinite')
    require((value > 0).all() if strict else (value >= 0).all(), f'{label}: positivity')


def geometry(n_cells, levels, sigma_power):
    from vfs import VoronoiMesh, DryHydrostaticDynamics, make_hybrid_coefficients
    mesh = VoronoiMesh(n_cells=n_cells, lloyd_iters=0, verbose=False)
    A, B = make_hybrid_coefficients(levels, p_top=5000., sigma_power=sigma_power)
    return mesh, torch.as_tensor(A), torch.as_tensor(B), DryHydrostaticDynamics().gravity


def check_checkpoint_invariants(actual, reference):
    from vfs.atmosphere.moist_cp import interface_pressure_mass
    mesh, A, B, gravity = geometry(64, 6, 1.)
    area = mesh.cell_area[:, None]
    report = {}
    for name in ('smooth', 'cloud', 'cloud_reverse'):
        case, ref = actual[name], reference[name]
        states, totals = {}, {}
        for stage in ('initial', 'euler', 'final'):
            s = case[stage]
            p = A + B * s['ps']
            dp = p.diff()
            positive(s['ps'], f'{name}/{stage}/surface pressure')
            positive(p, f'{name}/{stage}/pressure')
            positive(dp, f'{name}/{stage}/layer mass')
            dm = interface_pressure_mass(dp)
            water = s['water_content']
            positive(water[..., 0], f'{name}/{stage}/vapor')
            positive(water[..., 1:], f'{name}/{stage}/condensate', strict=False)
            positive(dm - water.sum(-1), f'{name}/{stage}/dry fraction')
            states[stage] = (dm, water)
            totals[stage] = ((area * dm).sum() / gravity,
                             (area[..., None] * water).sum((0, 1)) / gravity)
        item = {}
        for stage in ('euler', 'final'):
            item[stage + '_mass_delta'] = conserved(totals[stage][0], totals['initial'][0], f'{name}/{stage}/mass')
            item[stage + '_species_delta'] = conserved(totals[stage][1], totals['initial'][1], f'{name}/{stage}/species')
        for stage in ('initial', 'final'):
            for key, total in zip(('mass', 'species'), totals[stage]):
                conserved(case[stage + '_budgets'][key], total, f'{name}/{stage}/reported {key}')
        info = case['limiter']
        positive(info['amounts'], f'{name}/ledger amounts', strict=False)
        positive(info['heat'], f'{name}/heat', strict=False)
        beta = info['beta']
        require(((beta >= 0) & (beta <= 1)).all(), f'{name}/beta bounds')
        require(torch.equal(beta < 1, ref['limiter']['beta'] < 1), f'{name}/limiter active mask')
        require(type(info['limited_faces']) is int and
                info['limited_faces'] == ref['limiter']['limited_faces'] == int((beta < 1).sum()),
                f'{name}/limited face count')
        require(case['stages'] == ref['stages'], f'{name}/limiter stage counts')
        water_from_ledger = info['amounts'][:, 1:].reshape_as(case['euler']['water_content']) / area[..., None]
        require(torch.equal(water_from_ledger, case['euler']['water_content']), f'{name}/exact Euler ledger')
        dm0, water0 = states['initial']
        initial_amounts = area[..., None] * torch.cat([(dm0 - water0.sum(-1))[..., None], water0], -1)
        item['ledger_delta'] = conserved(info['amounts'].sum(0), initial_amounts.sum((0, 1)),
                                        f'{name}/ledger species', initial_amounts.abs().sum((0, 1)))
        conserved(info['amounts'].sum(-1), (area * states['euler'][0]).flatten(), f'{name}/ledger carrier')
        scale = work_scale(case)
        rate = sum((case['gradients'][k] * case['rhs'][k]).sum() for k in FIELDS)
        require(torch.isfinite(rate) and rate.abs() <= EPS128 * scale, f'{name}/energy closure')
        require(case['energy_rate'].abs() <= EPS128 * scale, f'{name}/reported energy closure')
        require(torch.equal(rate, case['energy_rate']), f'{name}/energy rate identity')
        item['energy_rate'] = float(rate)
        item['energy_work_scale'] = float(scale)
        item['energy_drift'] = float(case['final_budgets']['energy']['total'] - case['initial_budgets']['energy']['total'])
        report[name] = item
    dry = actual['dry']
    dry_mass = {}
    for stage in ('initial', 'final'):
        s = dry[stage]
        positive(A + B * s['ps'], f'dry/{stage}/pressure')
        dp = (A + B * s['ps']).diff()
        positive(dp, f'dry/{stage}/layer mass')
        positive(s['mtheta_i'], f'dry/{stage}/thermal contents')
        dry_mass[stage] = (area * dp).sum() / gravity
        conserved(dry[stage + '_mass'], dry_mass[stage], f'dry/{stage}/reported mass')
    report['dry_mass_delta'] = conserved(dry_mass['final'], dry_mass['initial'], 'dry/mass')
    return report


def check_dry_invariants(actual):
    mesh, A, B, gravity = geometry(32, 4, 1.2)
    area = mesh.cell_area[:, None]
    report = {}
    for name in ('canonical', 'diagonal'):
        case = actual[name]
        for key, value in case['pressure_fields'].items():
            positive(value, f'{name}/pressure_fields/{key}')
        for key in ('dp', 'dp_e', 'dm', 'theta'):
            positive(case['primitives'][key], f'{name}/primitives/{key}')
        for key in ('theta', 'theta_full', 'temperature', 'exner_interface', 'exner_full'):
            positive(case['thermo_fields'][key], f'{name}/thermo_fields/{key}')
        step = case['step']
        p = A + B * step['ps']
        positive(p, f'{name}/step/pressure')
        positive(p.diff(), f'{name}/step/layer mass')
        positive(step['mtheta_i'], f'{name}/step/thermal contents')
        initial_mass = (area * case['pressure_fields']['dp']).sum() / gravity
        final_mass = (area * p.diff()).sum() / gravity
        conserved(case['total_mass'], initial_mass, f'{name}/reported mass')
        report[name + '_mass_delta'] = conserved(final_mass, initial_mass, f'{name}/mass')
        work = case['rotation_work_terms']
        positive(work['work_abs_scale'], f'{name}/rotation work scale')
        require(work['work'].abs() <= EPS128 * work['work_abs_scale'], f'{name}/rotation closure')
        require(0 <= work['work_relative'] <= EPS128, f'{name}/normalized rotation closure')
        require(torch.equal(work['work_relative'], work['work'].abs() / work['work_abs_scale']),
                f'{name}/rotation work identity')
    tracer_rate = area[..., None] * actual['canonical']['tracer_rhs']['tracer_m'] / gravity
    conserved(tracer_rate.sum((0, 1)), torch.zeros(2, dtype=torch.float64),
              'canonical/tracer tendency', tracer_rate.abs().sum((0, 1)))
    return report


def compare_portable(actual, reference, fixture):
    require(fixture in MANIFEST, f'unknown fixture: {fixture}')
    manifest = MANIFEST[fixture]
    leaves = list(flatten_pair(actual, reference, manifest))
    require({p for p, _, _ in leaves} == manifest.keys(), 'manifest coverage mismatch')
    maxima, checked = {}, 0
    for path, a, b in leaves:
        spec = manifest[path]
        q, components = spec['quantity'], spec['components']
        if q == 'exact':
            require(a == b, f'{path}: discrete value mismatch')
            continue
        scale = None
        if q == 'energy_rate':
            scale = work_scale(reference[path.split('/')[0]])
        elif q == 'rotation_work':
            scale = reference[path.split('/')[0]]['rotation_work_terms']['work_abs_scale']
        elif q == 'rotation_relative':
            scale = 1.  # Already dimensionless and normalized by absolute work.
        try:
            ratio = scaled_error(a, b, q, components, scale)
        except AssertionError as exc:
            raise AssertionError(f'{path}: {exc}') from exc
        maxima[q] = max(maxima.get(q, 0.), ratio)
        checked += isinstance(b, torch.Tensor)
    invariants = (check_checkpoint_invariants(actual, reference) if fixture == 'checkpoint54.pt'
                  else check_dry_invariants(actual))
    return dict(tensor_comparisons=checked, maximum_budget_fraction=maxima, invariants=invariants)


def compare_selected(actual, reference, fixture, mode, exact):
    if mode == 'exact':
        return dict(tensor_comparisons=exact(actual, reference))
    if mode == 'portable':
        return compare_portable(actual, reference, fixture)
    raise ValueError(f'unknown comparison mode: {mode}')
