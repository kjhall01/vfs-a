"""Diagnose one saved coupled interval using its original cached radiation.

This explicitly reconstructs state for cross-version diagnosis, not production
restart migration. No output checkpoint is offered as a resumable campaign.
A fresh optics call fails; real radiation tables are never substituted.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, default=ROOT)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--restart', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be new/empty')
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.research_root))
    from workflows.coupled import campaign as c
    from workflows.pyam21.state import PhysicsState, MoistState
    from workflows.pyam21.radiation import HostRadiationDriver

    config = c.CampaignConfig(**json.loads(args.config.read_text()))
    if args.compile is not None:
        config = replace(config, compile=args.compile)
    config, xyz, mesh = c.load_mesh(config.validate(), args.mesh)
    payload = torch.load(args.restart, map_location='cpu', weights_only=True)
    pa = payload['air']
    if 'vfs_radiation_pflux' not in pa['radiation'].get('saved', {}):
        raise ValueError('diagnostic expects a checkpoint with native radiation pressure interfaces')
    calls = 0

    def unavailable(*a, **kw):
        nonlocal calls
        calls += 1
        raise RuntimeError('fresh radiation optics required; this diagnostic only reuses saved radiation')

    def cached(nml, model, data_dir, **kwargs):
        return SimpleNamespace(driver=HostRadiationDriver.from_nml(nml, unavailable),
                               identity='diagnostic-original-cache', manifest={})

    result = dict(status='running', diagnostic_only=True,
                  start_seconds=pa['time_seconds'], configuration=asdict(config), mesh=mesh,
                  checkpoint_sha256=hashlib.sha256(args.restart.read_bytes()).hexdigest(),
                  source_sha256=c.source_hashes(), torch=str(torch.__version__))
    c.atomic_json(args.output/'result.json', result)
    start = time.perf_counter()
    try:
        with patch('workflows.pyam21.radiation.build_radiation', cached):
            adapter, _, receiver, template = c.build(config, args.source, 'diagnostic-cache', generator_xyz=xyz)
        air = PhysicsState(MoistState(**pa['atmosphere'], mesh=adapter.model.mesh, column=adapter.model.column),
                           pa['cloud_fraction'], pa['memory'], pa['surface'], pa['radiation'], pa['time_seconds'])
        ocean = replace(template.coupled.ocean, ocean=template.coupled.ocean.ocean.with_tensors(**payload['ocean']),
                        tke_h=payload['tke_h'], eddy_h=payload['eddy_h'])
        sea = c.ThermalIceState(c.RidgingState(ocean, c.CategoryIce(**payload['ice'])),
                               payload['snow_mass'], payload['cold_content'])
        adapter.check(air)
        receiver.snow.check(sea)
        result['compilation'] = adapter.compilation_manifest
        initial = c.inventories(adapter, air, receiver, sea)
        original = c.ale_step

        def capture(receiver, water, cfg):
            state = water.coupled.ocean
            fractions = c.rest_thickness(cfg).to(state.ocean.h)
            target = state.ocean.h.sum(-1, keepdim=True)*(fractions/fractions.sum())[None]
            c.checkpoint(args.output/'pre_ale.pt', dict(
                ocean={k:getattr(state.ocean, k) for k in payload['ocean']},
                tke_h=state.tke_h, eddy_h=state.eddy_h, target=target))
            return original(receiver, water, cfg)

        with patch.object(c, 'ale_step', capture):
            air, sea, report = c.advance(adapter, air, receiver, sea, config.physics_dt, config)
        final = c.inventories(adapter, air, receiver, sea)
        result.update(status='passed', end_seconds=air.time_seconds, report=report,
                      numerical_energy_residual_J_m2=final['energy']-initial['energy']-report['external']-report['upstream'],
                      water_residual_kg_m2=final['water']-initial['water'],
                      salt_relative=(final['salt']-initial['salt'])/initial['salt'])
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result.update(fresh_optics_calls=calls, wall_seconds=time.perf_counter()-start)
        c.atomic_json(args.output/'result.json', result)
    print(json.dumps({k:result[k] for k in ('status', 'start_seconds', 'end_seconds', 'fresh_optics_calls')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
