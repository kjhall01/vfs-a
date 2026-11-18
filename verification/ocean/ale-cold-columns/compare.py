"""Reproduce the cold-column acceptance discrepancy against either source tree.

Run with an installed VFS wheel; --research-root selects only research modules.
The deterministic input builder is shared with the regression test.
"""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--research-root', type=Path, default=ROOT)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--salinity-scale', type=float, default=1.7105959511064084)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path[:0] = [str(args.research_root / 'validation' / name)
               for name in ('temperature_coordinate', 'ocean_column')]
import ale
import roquet
sys.path.insert(0, str(ROOT / 'tests/ocean'))
from test_temperature_ale_roundoff import cold_columns
from vfs.ocean.stratified.energy_columns import column_terms

model, state, target = cold_columns(seed=args.seed, salinity_scale=args.salinity_scale)
try:
    end, report = model.remap_with_report(state, target)
    old, _ = column_terms(model.energy, state.ocean)
    mass = model.energy.mass_area(state.ocean)
    eps = torch.finfo(target.dtype).eps
    et = 256 * eps * sum(v.abs() for v in old.values()).clamp_min(1)
    st = 512 * eps * (mass * ale.entropy_content(model.energy, state.ocean).abs()).sum(-1).clamp_min(1)
    result = dict(status='pass', iterations=report.iterations,
                  maximum_energy_residual_over_tolerance=float((report.column_energy_residual.abs()/et).max()),
                  minimum_entropy_change_over_tolerance=float((report.column_entropy_change/st).min()))
except (ValueError, FloatingPointError) as error:
    tb = error.__traceback__
    local = None
    while tb:
        if tb.tb_frame.f_code.co_filename.endswith('/temperature_coordinate/ale.py'):
            local = tb.tb_frame.f_locals
        tb = tb.tb_next
    if local is None or 'entropy_tolerance' not in local:
        raise
    ratio = local['entropy_change'] / local['entropy_tolerance']
    index = int(ratio.argmin())
    # ref is the temporary-coordinate evaluation used in the old loop.
    temporary = (local['mass'] * (local['after'].h * local['ref']['entropy']
                 - ale.entropy_content(local['energy'], local['state']))).sum(-1)
    result = dict(status='failed', error=str(error), iterations=local['iteration']+1,
                  column=index, temporary_entropy_change_over_tolerance=float(temporary[index]/local['entropy_tolerance'][index]),
                  stored_entropy_change_over_tolerance=float(ratio[index]),
                  maximum_energy_residual_over_tolerance=float((local['residual'].abs()/local['energy_tolerance']).max()))
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
