"""Report every fixture difference without changing exact regression gates.

Run against an installed wheel outside the checkout. This is a diagnostic,
not a portable pass/fail mode. It never writes fixtures or sets tolerances.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
from unittest.mock import patch

import torch
import vfs

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(a, b, path):
    row = dict(path=path, shape=list(a.shape), reference_shape=list(b.shape),
               dtype=str(a.dtype), reference_dtype=str(b.dtype),
               actual_nonfinite=int((~torch.isfinite(a)).sum()),
               reference_nonfinite=int((~torch.isfinite(b)).sum()))
    if a.shape != b.shape:
        return row
    finite = torch.isfinite(a) & torch.isfinite(b)
    row['exact'] = torch.equal(a, b)
    row['unequal'] = int((a != b).sum())
    row['elements'] = a.numel()
    if not finite.any():
        return row
    x, y = a[finite].double(), b[finite].double()
    error = (x - y).abs()
    scale = float(y.abs().max())
    index = int(error.argmax())
    nonzero = y != 0
    zeros = ~nonzero
    row.update(max_abs=float(error.max()), rms_abs=float(error.square().mean().sqrt()),
               reference_max_abs=scale,
               max_abs_over_reference_max=float(error.max()) / scale if scale else None,
               max_relative_nonzero=float((error[nonzero] / y[nonzero].abs()).max()) if nonzero.any() else None,
               reference_zero_count=int(zeros.sum()),
               reference_zero_changed=int((x[zeros] != 0).sum()),
               max_abs_at_reference_zero=float(error[zeros].max()) if zeros.any() else None,
               worst_actual=float(x[index]), worst_reference=float(y[index]),
               actual_min=float(x.min()), reference_min=float(y.min()),
               sign_changes=int(((x * y) < 0).sum()))
    return row


def walk(a, b, path, rows, scalars, structures, components):
    if isinstance(b, torch.Tensor):
        if not isinstance(a, torch.Tensor):
            structures.append(dict(path=path, error='tensor type mismatch'))
            return
        rows.append(metrics(a, b, path))
        if a.shape == b.shape and a.ndim and a.shape[-1] in (3, 4) and path.split('/')[-1] in (
            'water_content', 'species', 'amounts', 'integrated_flux', 'tracer_m', 'tracer_inventory'
        ):
            for i in range(a.shape[-1]):
                components.append(metrics(a[..., i], b[..., i], f'{path}[{i}]'))
    elif isinstance(b, dict):
        if not isinstance(a, dict) or a.keys() != b.keys():
            structures.append(dict(path=path, error='dictionary keys/type mismatch'))
        if isinstance(a, dict):
            for key in b.keys() & a.keys():
                walk(a[key], b[key], path + '/' + key, rows, scalars, structures, components)
    elif isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            structures.append(dict(path=path, error='sequence length/type mismatch'))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f'{path}/{i}', rows, scalars, structures, components)
    else:
        scalars.append(dict(path=path, actual=a, reference=b, exact=type(a) is type(b) and a == b))


def relative_change(a, b):
    delta = a - b
    return dict(delta=delta.tolist(), reference=b.tolist(),
                relative=[float(x / y) if y != 0 else None
                          for x, y in zip(delta.flatten(), b.flatten())])


def moist_invariants(actual, expected):
    from vfs.atmosphere.moist_cp import interface_pressure_mass
    from vfs.geometry.column import make_hybrid_coefficients
    mesh = vfs.VoronoiMesh(n_cells=64, lloyd_iters=0, verbose=False)
    A, B = make_hybrid_coefficients(6, p_top=5000., sigma_power=1.)
    A, B = torch.as_tensor(A), torch.as_tensor(B)
    results = {}
    for name in ('smooth', 'cloud', 'cloud_reverse'):
        case, ref = actual[name], expected[name]
        info, ref_info = case['limiter'], ref['limiter']
        item = dict(limited_faces=info['limited_faces'],
                    reference_limited_faces=ref_info['limited_faces'],
                    stages=case['stages'], reference_stages=ref['stages'],
                    beta_in_unit_interval=bool(((info['beta'] >= 0) & (info['beta'] <= 1)).all()),
                    active_mask_equal=torch.equal(info['beta'] < 1, ref_info['beta'] < 1),
                    active_mask_changed=int(((info['beta'] < 1) != (ref_info['beta'] < 1)).sum()),
                    ledger_min_by_species=info['amounts'].amin(0).tolist(),
                    heat_min=float(info['heat'].min()))
        for stage in ('initial', 'euler', 'final'):
            s = case[stage]
            p = A + B * s['ps']
            dp = p.diff()
            dm = interface_pressure_mass(dp)
            q = s['water_content'] / dm[..., None]
            item[stage] = dict(pressure_min=float(p.min()), layer_dp_min=float(dp.min()),
                               dry_fraction_min=float((1 - q.sum(-1)).min()),
                               water_fraction_min=q.amin((0, 1)).tolist())
        recovered = info['amounts'][:, 1:].reshape_as(case['euler']['water_content']) / mesh.cell_area[:, None, None]
        item['euler_water_is_ledger_exact'] = torch.equal(recovered, case['euler']['water_content'])
        for key in ('mass', 'species'):
            item[key + '_conservation'] = relative_change(case['final_budgets'][key], case['initial_budgets'][key])
        q0 = case['initial']['water_content']
        p0 = A + B * case['initial']['ps']
        dm0 = interface_pressure_mass(p0.diff())
        Q0 = torch.cat([(dm0-q0.sum(-1))[..., None], q0], -1) * mesh.cell_area[:, None, None]
        item['ledger_conservation'] = relative_change(info['amounts'].sum(0), Q0.sum((0, 1)))
        # Report energy drift; SSPRK3 does not imply exact finite-step energy conservation.
        item['energy_budget_changes'] = {k: relative_change(case['final_budgets']['energy'][k], v)
                                         for k, v in case['initial_budgets']['energy'].items()}
        results[name] = item
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    environment = dict(platform=platform.platform(), python=platform.python_version(),
                       torch=str(torch.__version__), source=vfs.__file__,
                       cpu_capability=torch.backends.cpu.get_cpu_capability(),
                       torch_config=torch.__config__.show(),
                       env={k: os.environ.get(k) for k in ('ATEN_CPU_CAPABILITY', 'MKL_CBWR', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})
    result = dict(environment=environment, fixtures={})
    for name, filename in [('check_baseline', 'checkpoint54.pt'), ('check_dry_helpers', 'dry_helpers_pass2.pt')]:
        module = load(name)
        fixture = ROOT / 'tests/fixtures' / filename
        before = hashlib.sha256(fixture.read_bytes()).hexdigest()
        with patch('torch.autograd.grad', side_effect=AssertionError('forward autodiff')):
            actual = module.capture(module.imports()) if name == 'check_baseline' else module.capture()
        expected = torch.load(fixture, map_location='cpu', weights_only=True)
        rows, scalars, structures, components = [], [], [], []
        walk(actual, expected, name, rows, scalars, structures, components)
        rows.sort(key=lambda x: x['path'])
        try:
            module.compare(actual, expected)
            exact_result = 'pass'
        except AssertionError as exc:
            exact_result = str(exc)
        item = dict(fixture_sha256=before, tensors=rows, components=components,
                    scalars=scalars, structural_errors=structures, exact_comparator=exact_result,
                    tensor_count=len(rows), exact_tensor_count=sum(x.get('exact', False) for x in rows))
        if name == 'check_baseline':
            item['moist_invariants'] = moist_invariants(actual, expected)
        assert before == hashlib.sha256(fixture.read_bytes()).hexdigest()
        result['fixtures'][filename] = item
    environment['torch_threads'] = torch.get_num_threads()
    environment['torch_interop_threads'] = torch.get_num_interop_threads()
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: {n: v[n] for n in ('tensor_count', 'exact_tensor_count')}
                      for k, v in result['fixtures'].items()}, indent=2))


if __name__ == '__main__':
    main()
