"""Compare reviewed versus integrated ocean equations in the same environment.

Uses frozen input/mesh bytes and production model calls for both equation sets,
with and without a coast. Gradients are evaluated before serialization detaches.
This never regenerates the archived extraction fixtures.
"""
import argparse
import importlib.util
from pathlib import Path

import torch
from vfs.ocean import Ocean, OceanColumn, OceanDomain, OceanState
from vfs.ocean.conservative import ConstantDensityDynamics
from vfs.ocean.dynamics import MultilayerOceanDynamics


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    ocean = load('capture_ocean')
    clean = load('capture_readability').clean
    result = {}
    for coast in (False, True):
        mesh, fields = ocean.load_inputs(ROOT / 'tests/fixtures/ocean/inputs.npz')
        column = OceanColumn(3, total_depth=3000.)
        domain = OceanDomain(mesh, mesh.cell_xyz[:, 2] > -.4) if coast else None
        initial = OceanState(**fields, mesh=mesh, column=column)
        if coast:
            initial = domain.project_state(initial)
        for equations in (MultilayerOceanDynamics, ConstantDensityDynamics):
            model = Ocean(mesh, column, domain=domain, dynamics=equations(domain=domain))
            leaves = [getattr(initial, key).clone().requires_grad_() for key in ocean.FIELDS]
            state = initial.with_tensors(*leaves)
            rhs, step = model.tendency(state), model.step(state, 30.)
            # Probe every stored variable, including passive scalar transport.
            objective = sum(getattr(step, key).square().mean() for key in ocean.FIELDS)
            gradients = torch.autograd.grad(objective, leaves)
            values = {
                'initial': dict(zip(ocean.FIELDS, leaves)),
                'rhs': {key: getattr(rhs, key) for key in ocean.FIELDS},
                'step': {key: getattr(step, key) for key in ocean.FIELDS},
                'terms': model.tendency_terms(state),
                'physical_fields': model.physical_fields(state),
                'budgets': model.budgets(state),
                'step_gradients': gradients,
            }
            result[f'{equations.__name__}/{coast}'] = clean(values)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--save', type=Path)
    group.add_argument('--compare', type=Path)
    args = parser.parse_args()
    actual = capture()
    if args.save:
        if args.save.exists():
            raise FileExistsError(args.save)
        torch.save(actual, args.save)
    else:
        count = load('capture_readability').compare(actual, torch.load(args.compare, weights_only=True))
        print('Exactly equal ocean integration tensors:', count)
