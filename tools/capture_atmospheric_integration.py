"""Existing atmospheric migration cases plus complete-step gradient captures.

No expanded benchmark: reuse the retained smooth/active-limiter gradient cases.
Serialization copies are detached only after derivatives are evaluated.
"""
import argparse
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
from test_bounded_gradients import smooth_case, objective, KEYS, load
from vfs import MoistAtmosphere


def capture():
    result = load(ROOT/'tools/capture_readability.py').capture()
    for name in ('smooth', 'active'):
        if name == 'smooth':
            model, state = smooth_case()
            dt = 10.
        else:
            _, state = load(ROOT/'examples/bounded_moist.py').initialize()
            model = MoistAtmosphere(state.mesh, state.column)
            dt = 30.
        leaves = [getattr(state,k).clone().requires_grad_() for k in KEYS]
        connected = replace(state, **dict(zip(KEYS,leaves)))
        output, ledgers = model.transport.step_with_info(connected,dt)
        gradients = torch.autograd.grad(objective(output),leaves)
        clean = load(ROOT/'tools/capture_readability.py').clean
        result['bounded_gradient_'+name] = clean(dict(output=output,ledgers=ledgers,gradients=gradients))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--save',type=Path)
    group.add_argument('--compare',type=Path)
    args=parser.parse_args()
    result=capture()
    if args.save:
        if args.save.exists():raise FileExistsError(args.save)
        torch.save(result,args.save)
    else:
        count=load(ROOT/'tools/capture_readability.py').compare(result,torch.load(args.compare,weights_only=True))
        print('Exactly equal atmospheric integration tensors:',count)
