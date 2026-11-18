"""Compare complete eager/Inductor coupled intervals on the same configuration.

Records startup/first calls separately from later intervals, state differences,
independent inventory ledgers and a bitwise compiled restart replay. No optical
tables are inferred: omit --radiation-data for an explicit radiation-OFF trial.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics
import sys
import time
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workflows.coupled import campaign as c


def tensors(value, path=''):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensors(item, f'{path}/{key}')


def differences(left, right):
    result = {}
    for (name, a), (other, b) in zip(tensors(left), tensors(right), strict=True):
        assert name == other and a.shape == b.shape and a.dtype == b.dtype
        if a.dtype == torch.bool:
            result[name] = dict(unequal=int((a != b).sum()))
        else:
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError(f'nonfinite comparison tensor {name}')
            error = float((a-b).abs().max()) if a.numel() else 0.
            scale = float(a.abs().max()) if a.numel() else 0.
            result[name] = dict(max_absolute=error, max_scaled=error/max(1., scale))
    return result


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a == b


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'workflows/coupled/coupled_aquaplanet_1yr.yaml')
    parser.add_argument('--mesh', type=Path)
    parser.add_argument('--cells', type=int)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--radiation-data', type=Path)
    parser.add_argument('--steps', type=int, default=5, help='total intervals per mode, including warmup')
    parser.add_argument('--warmup', type=int, default=2, help='initial intervals excluded from steady timing')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.steps <= args.warmup:
        parser.error('need steps > warmup >= 1')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be new/empty')
    args.output.mkdir(parents=True, exist_ok=True)
    config = c.CampaignConfig(**yaml.safe_load(args.config.read_text()))
    changes = {name: getattr(args, name) for name in ('cells', 'threads') if getattr(args, name) is not None}
    config = replace(config, **changes, compile=False).validate()
    config, xyz, mesh_info = c.load_mesh(config, args.mesh)
    result = dict(status='running', torch=str(torch.__version__), configuration=asdict(config),
                  mesh=mesh_info, radiation_enabled=args.radiation_data is not None,
                  steps=args.steps, warmup=args.warmup, source_sha256=c.source_hashes(), modes={})
    c.atomic_json(args.output/'comparison.json', result)
    models = {}
    try:
        for label, compiled in (('eager', False), ('compiled', True)):
            print(f'Building {label}; initial calls may compile', flush=True)
            cfg = replace(config, compile=compiled)
            start = time.perf_counter()
            a, air, r, sea = c.build(cfg, args.source, args.radiation_data,
                                    smoke_no_radiation=args.radiation_data is None, generator_xyz=xyz)
            initial = c.inventories(a, air, r, sea)
            models[label] = [a, air, r, sea, cfg]
            result['modes'][label] = dict(setup_and_initial_diagnostics_seconds=time.perf_counter()-start,
                compilation=a.compilation_manifest, initial=initial, intervals=[], external=0., upstream=0.)
        for step in range(args.steps):
            # Alternate order after warmup to reduce order/thermal bias.
            order = ('eager', 'compiled') if step % 2 == 0 else ('compiled', 'eager')
            for label in order:
                a, air, r, sea, cfg = models[label]
                if label == 'compiled' and step == args.steps-1:
                    saved = c.pack(a, air, r, sea, 'benchmark', {})
                start = time.perf_counter()
                air, sea, report = c.advance(a, air, r, sea, cfg.physics_dt, cfg)
                elapsed = time.perf_counter()-start
                models[label][1], models[label][3] = air, sea
                row = result['modes'][label]
                row['external'] += report['external']; row['upstream'] += report['upstream']
                current = c.inventories(a, air, r, sea)
                row['intervals'].append(dict(index=step, wall_seconds=elapsed, report=report,
                    numerical_energy_residual_J_m2=current['energy']-row['initial']['energy']-row['external']-row['upstream'],
                    water_residual_kg_m2=current['water']-row['initial']['water'],
                    salt_relative=(current['salt']-row['initial']['salt'])/row['initial']['salt']))
                print(f'{label} interval {step+1}/{args.steps}: {elapsed:.6f} s', flush=True)
            c.atomic_json(args.output/'comparison.json', result)
        payloads = {label: c.pack(m[0],m[1],m[2],m[3],'compare',{}) for label,m in models.items()}
        result['state_differences'] = differences(payloads['eager'], payloads['compiled'])
        a, air, r, sea, cfg = models['compiled']
        ra, rs, _ = c.unpack(saved, a, r, sea, 'benchmark')
        ra, rs, rr = c.advance(a, ra, r, rs, cfg.physics_dt, cfg)
        result['compiled_restart_bitwise'] = equal(c.pack(a,ra,r,rs,'compare',{}), payloads['compiled']) and equal(rr,result['modes']['compiled']['intervals'][-1]['report'])
        if not result['compiled_restart_bitwise']:
            raise AssertionError('compiled restart replay differs')
        for row in result['modes'].values():
            median = statistics.median(x['wall_seconds'] for x in row['intervals'][args.warmup:])
            row['steady_median_seconds'] = median
            row['steady_sypd'] = config.physics_dt/(365*median)
        result['steady_speedup'] = result['modes']['eager']['steady_median_seconds']/result['modes']['compiled']['steady_median_seconds']
        result['status'] = 'passed'
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        c.atomic_json(args.output/'comparison.json', result)
        raise
    c.atomic_json(args.output/'comparison.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'steady_speedup', 'compiled_restart_bitwise')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
