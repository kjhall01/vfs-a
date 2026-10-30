"""Adversarial checks of the portable policy, independent of numerical kernels."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = load('fixture_regression')
TOOLS = {'checkpoint54.pt': load('check_baseline'), 'dry_helpers_pass2.pt': load('check_dry_helpers')}


def at(tree, path):
    for key in path.split('/'):
        tree = tree[int(key)] if isinstance(tree, (tuple, list)) else tree[key]
    return tree


@pytest.fixture(scope='module')
def samples():
    result = {}
    for name, tool in TOOLS.items():
        actual = tool.capture(tool.imports()) if name == 'checkpoint54.pt' else tool.capture()
        reference = policy.retained_reference(torch.load(ROOT / 'tests/fixtures' / name, weights_only=True), name)
        result[name] = actual, reference
    return result


def test_manifest_covers_all_235_retained_tensors_and_all_categories(samples):
    tensors = 0
    categories = set()
    for name, (actual, reference) in samples.items():
        leaves = list(policy.flatten_pair(actual, reference, policy.MANIFEST[name]))
        assert len(leaves) == len(policy.MANIFEST[name])
        tensors += sum(isinstance(b, torch.Tensor) for _, _, b in leaves)
        categories.update(s['quantity'] for s in policy.MANIFEST[name].values())
        assert policy.compare_portable(actual, reference, name)['tensor_comparisons'] == (142 if name == 'checkpoint54.pt' else 93)
    assert tensors == 235
    assert categories == set(policy.TOLERANCES) | {'exact'}


@pytest.mark.parametrize('quantity', sorted(policy.TOLERANCES))
def test_each_numerical_category_rejects_above_budget_and_accepts_below(quantity):
    b = torch.tensor([1., -2.], dtype=torch.float64)
    tau = policy.TOLERANCES[quantity]
    good, bad = b.clone(), b.clone()
    good[0] += .25 * tau * 2
    bad[0] += 4 * tau * 2
    policy.scaled_error(good, b, quantity)
    with pytest.raises(AssertionError, match=quantity + ' tolerance'):
        policy.scaled_error(bad, b, quantity)


TENSOR_PATHS = [(name, path) for name, entries in policy.MANIFEST.items()
                for path, spec in entries.items() if spec['quantity'] not in ('exact', 'cfl')]


@pytest.mark.parametrize('name,path', TENSOR_PATHS)
def test_every_tensor_mapping_rejects_a_deliberate_error(samples, name, path):
    source, reference = samples[name]
    actual = copy.deepcopy(source)
    a, b = at(actual, path), at(reference, path)
    spec = policy.MANIFEST[name][path]
    q = spec['quantity']
    scale = float(b.abs().max())
    if q == 'energy_rate':
        scale = float(policy.work_scale(reference[path.split('/')[0]]))
    elif q == 'rotation_work':
        scale = float(reference[path.split('/')[0]]['rotation_work_terms']['work_abs_scale'])
    elif q == 'rotation_relative':
        scale = 1.
    # This intentionally exceeds even a whole-tensor bound. Separate tests
    # below ensure a dominant species cannot hide errors in a dilute one.
    index = int(b.abs().argmax())
    a.flatten()[index] += 10 * policy.TOLERANCES[q] * scale if scale else 1e-100
    with pytest.raises(AssertionError, match=path):
        policy.compare_portable(actual, reference, name)


@pytest.mark.parametrize('components', [2, 3, 4])
@pytest.mark.parametrize('q', ['state', 'tendency', 'gradient', 'inventory', 'ledger', 'dry_helper'])
def test_component_scales_cannot_be_borrowed_from_dominant_species(components, q):
    b = torch.full((2, components), 1e-12, dtype=torch.float64)
    b[:, 0] = 1e12
    a = b.clone()
    a[0, -1] += 1e-10  # Invisible to a whole-tensor scale, huge for this species.
    policy.scaled_error(a, b, q)  # Demonstrate why independent scales matter.
    with pytest.raises(AssertionError, match='tolerance'):
        policy.scaled_error(a, b, q, components=components)


def test_zero_scale_is_exact_and_has_no_floor():
    b = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(AssertionError, match='tolerance'):
        policy.scaled_error(b + 1e-300, b, 'state')


@pytest.mark.parametrize('side', ['actual', 'reference'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_values_never_pass_even_when_matching(samples, side, value):
    actual, reference = map(copy.deepcopy, samples['checkpoint54.pt'])
    target = actual if side == 'actual' else reference
    target['smooth']['initial']['u_e'][0, 0] = value
    with pytest.raises(AssertionError, match='nonfinite tensor'):
        policy.compare_portable(actual, reference, 'checkpoint54.pt')
    actual['smooth']['initial']['u_e'][0, 0] = value
    reference['smooth']['initial']['u_e'][0, 0] = value
    with pytest.raises(AssertionError, match='nonfinite tensor'):
        policy.compare_portable(actual, reference, 'checkpoint54.pt')


@pytest.mark.parametrize('change,message', [
    ('shape', 'shape mismatch'), ('dtype', 'dtype mismatch'),
    ('container', 'type mismatch'), ('keys', 'dictionary keys mismatch'),
    ('length', 'sequence length mismatch'), ('count', 'discrete value mismatch'),
    ('cfl_nonfinite', 'nonfinite scalar'), ('cfl_error', 'cfl tolerance'),
])
def test_structure_and_scalar_checks(samples, change, message):
    actual, reference = samples['checkpoint54.pt']
    actual = copy.deepcopy(actual)
    case = actual['smooth']
    if change == 'shape': case['initial']['u_e'] = case['initial']['u_e'][:1]
    if change == 'dtype': case['initial']['u_e'] = case['initial']['u_e'].float()
    if change == 'container': case['stages'] = tuple(case['stages'])
    if change == 'keys': del case['initial']['ps']
    if change == 'length': case['stages'][0].pop()
    if change == 'count': case['stages'][0][0] += 1
    if change == 'cfl_nonfinite': case['cfl'] = float('nan')
    if change == 'cfl_error': case['cfl'] *= 1.01
    with pytest.raises(AssertionError, match=message):
        policy.compare_portable(actual, reference, 'checkpoint54.pt')


@pytest.mark.parametrize('value', [torch.tensor(1.), {}])
def test_unknown_paths_rejected_even_when_both_inputs_agree(samples, value):
    actual, reference = map(copy.deepcopy, samples['checkpoint54.pt'])
    actual['new_quantity'] = copy.deepcopy(value)
    reference['new_quantity'] = copy.deepcopy(value)
    with pytest.raises(AssertionError, match='unmapped'):
        policy.compare_portable(actual, reference, 'checkpoint54.pt')


def test_unknown_category_and_fixture_rejected():
    with pytest.raises(AssertionError, match='unknown quantity'):
        policy.scaled_error(torch.ones(1), torch.ones(1), 'invented')
    with pytest.raises(AssertionError, match='unknown fixture'):
        policy.compare_portable({}, {}, 'invented.pt')


MOIST_MUTATIONS = [
    'pressure', 'layer mass', 'zero vapor', 'negative vapor', 'condensate',
    'dry fraction', 'ledger amounts', 'heat', 'beta lower', 'beta upper',
    'active mask', 'face count', 'stage counts', 'Euler ledger', 'mass',
    'species', 'absent species', 'reported mass', 'ledger species', 'ledger carrier',
    'energy closure', 'reported energy closure', 'energy rate identity',
]


def mutate_moist(actual, change):
    c = actual['cloud']
    if change == 'pressure': c['initial']['ps'][0, 0] = -1
    elif change == 'layer mass': c['initial']['ps'][0, 0] = 4000
    elif change == 'zero vapor': c['initial']['water_content'][0, 0, 0] = 0
    elif change == 'negative vapor': c['initial']['water_content'][0, 0, 0] = -1e-300
    elif change == 'condensate':
        liquid = c['initial']['water_content'][..., 1]
        i, j = (liquid == 0).nonzero()[0].tolist()
        liquid[i, j] = -1e-300
    elif change == 'dry fraction': c['initial']['water_content'][0, 0, 0] = 1e6
    elif change == 'ledger amounts': c['limiter']['amounts'][0, 3] = -1e-300
    elif change == 'heat': c['limiter']['heat'][0] = -1e-300
    elif change == 'beta lower': c['limiter']['beta'][0] = -1e-20
    elif change == 'beta upper': c['limiter']['beta'][0] = 1 + torch.finfo(torch.float64).eps
    elif change == 'active mask':
        beta = c['limiter']['beta']; i = int((beta == 1).nonzero()[0])
        beta[i] = torch.nextafter(beta[i], torch.zeros_like(beta[i]))
    elif change == 'face count': c['limiter']['limited_faces'] += 1
    elif change == 'stage counts': c['stages'][0][0] += 1
    elif change == 'Euler ledger':
        water = c['euler']['water_content']
        water[0, 0, 0] = torch.nextafter(water[0, 0, 0], torch.tensor(float('inf')))
    elif change == 'mass': c['final']['ps'] *= 1 + 1e-10
    elif change == 'species': c['final']['water_content'][..., 1] *= 1 + 1e-10
    elif change == 'absent species': c['final']['water_content'][0, 0, 2] = 1e-300
    elif change == 'reported mass': c['final_budgets']['mass'] *= 1 + 1e-10
    elif change == 'ledger species': c['limiter']['amounts'][:, 0] += 1e6
    elif change == 'ledger carrier':
        c['limiter']['amounts'][0, 0] += 1e6
        c['limiter']['amounts'][1, 0] -= 1e6
    elif change == 'energy closure': c['rhs']['u_e'] += 1e-7
    elif change == 'reported energy closure': c['energy_rate'] += 1e8
    elif change == 'energy rate identity': c['energy_rate'] += 1
    else: raise AssertionError(change)


@pytest.mark.parametrize('change', MOIST_MUTATIONS)
def test_each_moist_invariant_rejects_incorrect_results_independently(samples, change):
    source, reference = samples['checkpoint54.pt']
    actual = copy.deepcopy(source)
    mutate_moist(actual, change)
    labels = {
        'pressure': 'surface pressure: positivity', 'layer mass': 'layer mass: positivity',
        'zero vapor': 'vapor: positivity', 'negative vapor': 'vapor: positivity',
        'condensate': 'condensate: positivity', 'dry fraction': 'dry fraction: positivity',
        'ledger amounts': 'ledger amounts: positivity', 'heat': 'heat: positivity',
        'beta lower': 'beta bounds', 'beta upper': 'beta bounds', 'active mask': 'active mask',
        'face count': 'limited face count', 'stage counts': 'limiter stage counts',
        'Euler ledger': 'exact Euler ledger', 'mass': 'final/mass: conservation',
        'species': 'final/species: conservation', 'absent species': 'final/species: conservation',
        'reported mass': 'reported mass: conservation', 'ledger species': 'ledger species: conservation',
        'ledger carrier': 'ledger carrier: conservation', 'energy closure': 'energy closure',
        'reported energy closure': 'reported energy closure', 'energy rate identity': 'energy rate identity',
    }
    with pytest.raises(AssertionError, match=labels[change]):
        policy.check_checkpoint_invariants(actual, reference)


@pytest.mark.parametrize('change,message', [
    ('active mask', 'active mask'), ('Euler ledger', 'exact Euler ledger'),
    ('condensate', 'condensate: positivity'), ('heat', 'heat: positivity'),
    ('energy rate identity', 'energy rate identity'),
])
def test_full_portable_check_keeps_invariants_below_numerical_budgets(samples, change, message):
    source, reference = samples['checkpoint54.pt']
    actual = copy.deepcopy(source)
    mutate_moist(actual, change)
    with pytest.raises(AssertionError, match=message):
        policy.compare_portable(actual, reference, 'checkpoint54.pt')


@pytest.mark.parametrize('change', ['pressure', 'layer mass', 'temperature', 'theta',
                                   'mass', 'reported mass', 'rotation scale', 'rotation closure',
                                   'normalized rotation closure', 'rotation identity'])
def test_dry_invariants_reject_incorrect_results_independently(samples, change):
    actual = copy.deepcopy(samples['dry_helpers_pass2.pt'][0]); c = actual['diagonal']
    if change == 'pressure': c['pressure_fields']['ps'][0, 0] = 0
    elif change == 'layer mass': c['step']['ps'][0, 0] = 4000
    elif change == 'temperature': c['thermo_fields']['temperature'][0, 0] = 0
    elif change == 'theta': c['step']['mtheta_i'][0, 0] = 0
    elif change == 'mass': c['step']['ps'] *= 1 + 1e-10
    elif change == 'reported mass': c['total_mass'] *= 1 + 1e-10
    elif change == 'rotation scale': c['rotation_work_terms']['work_abs_scale'].zero_()
    elif change == 'rotation closure': c['rotation_work_terms']['work'] = c['rotation_work_terms']['work_abs_scale'].clone()
    elif change == 'normalized rotation closure': c['rotation_work_terms']['work_relative'].fill_(1.)
    elif change == 'rotation identity': c['rotation_work_terms']['work_relative'] += 1e-16
    labels = {
        'pressure': 'pressure_fields/ps: positivity', 'layer mass': 'step/layer mass: positivity',
        'temperature': 'temperature: positivity', 'theta': 'thermal contents: positivity',
        'mass': 'mass: conservation', 'reported mass': 'reported mass: conservation',
        'rotation scale': 'rotation work scale: positivity', 'rotation closure': 'rotation closure',
        'normalized rotation closure': 'normalized rotation closure',
        'rotation identity': 'rotation work identity', 'tracer': 'tracer tendency: conservation',
    }
    with pytest.raises(AssertionError, match=labels[change]):
        policy.check_dry_invariants(actual)


def test_conservation_uses_each_species_and_rejects_new_absent_species():
    initial = torch.tensor([1e20, 1., 0.], dtype=torch.float64)
    for final in [torch.tensor([1e20, 1.001, 0.], dtype=torch.float64),
                  torch.tensor([1e20, 1., 1e-300], dtype=torch.float64)]:
        with pytest.raises(AssertionError, match='conservation'):
            policy.conserved(final, initial, 'species')


@pytest.mark.parametrize('name', TOOLS)
def test_exact_comparator_and_default_still_reject_one_ulp(samples, name):
    tool = TOOLS[name]
    source = samples[name][0]
    actual = copy.deepcopy(source)
    path = 'smooth/initial/u_e' if name == 'checkpoint54.pt' else 'diagonal/wind_fields/u_e'
    a = at(actual, path)
    a[0, 0] = torch.nextafter(a[0, 0], torch.tensor(float('inf')))
    tool.check(actual, source, 'portable')
    for kwargs in ({}, {'mode': 'exact'}):
        with pytest.raises(AssertionError):
            tool.check(actual, source, **kwargs)
    with pytest.raises(ValueError, match='unknown comparison mode'):
        tool.check(source, source, 'automatic')
    assert tool.check(source, source, 'exact')['tensor_comparisons'] == (142 if name == 'checkpoint54.pt' else 93)


@pytest.mark.parametrize('tool', ['check_baseline', 'check_dry_helpers'])
@pytest.mark.parametrize('mode', ['exact', 'portable'])
def test_cli_records_mode_and_preserves_failure_exit_code(tmp_path, tool, mode):
    report = tmp_path / 'result.json'
    proc = subprocess.run([sys.executable, '-I', str(ROOT / 'tools' / (tool + '.py')),
                           '--mode', mode, '--report', str(report)], cwd=tmp_path,
                          capture_output=True, text=True)
    data = json.loads(report.read_text())
    assert data['mode'] == mode + '_comparison'
    assert proc.returncode == (0 if data['status'] == 'pass' else 1)
    if mode == 'portable':
        assert data['status'] == 'pass', proc.stdout + proc.stderr
    elif data['status'] == 'fail':
        assert data['error']
