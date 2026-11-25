"""Capture or compare the maintained CPU paths across VFS checkouts.

Run this driver twice with the same interpreter, once per --checkout.
The small geometries, physics configurations and restart exercises come from
the selected checkout's tests. PyAM21 and the test dependencies are required.
"""

import argparse
from dataclasses import fields, is_dataclass
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch


def snapshot(value):
    """Save prognostic fields and reports without pickling model/geometry objects."""
    from vfs.ocean.state import OceanState

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, OceanState):
        return {k: snapshot(getattr(value, k))
                for k in ("u_e", "h", "temp_h", "salt_h", "tracer_h", "attrs")}
    if hasattr(value, "resolved") and hasattr(value, "tke_h"):
        # Compare stored layer fields across the change in state ownership.
        result = dict(ocean=snapshot(value.resolved), tke_h=snapshot(value.tke_h))
        if hasattr(value, "eddy_h"):
            result["eddy_h"] = snapshot(value.eddy_h)
        return result
    if is_dataclass(value):
        return {f.name: snapshot(getattr(value, f.name)) for f in fields(value)
                if f.name not in ("mesh", "column", "domain")}
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [snapshot(v) for v in value]
    return value


def compare(actual, expected, path):
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        a, b = (x.contiguous().reshape(-1).view(torch.uint8) for x in (actual, expected))
        assert torch.equal(a, b), path
        return 1
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        return sum(compare(actual[k], expected[k], f"{path}/{k}") for k in expected)
    if isinstance(expected, list):
        assert len(actual) == len(expected), path
        return sum(compare(a, b, f"{path}/{i}")
                   for i, (a, b) in enumerate(zip(actual, expected)))
    assert actual == expected, path
    return 0


def capture():
    import test_coastal_campaign as coastal
    import test_coupled_campaign as coupled
    import test_coupled_shelf_redi as shelf
    import test_pyam21_workflow as atmosphere
    import test_roquet_configuration as ocean
    from tools.capture_readability import capture as capture_atmospheres
    from workflows.coupled import campaign

    result = capture_atmospheres()
    for case in result.values():
        for key in list(case):
            if key.startswith("ark4_"):
                del case[key]

    with torch.no_grad(), TemporaryDirectory() as tmp:
        adapter, state = atmosphere.initial(atmosphere.model.__wrapped__())
        end, report = adapter.apply(state, 60)
        result["forced_atmosphere"] = snapshot([adapter.pack(end), report])
        model, sea, forcing = ocean.setup.__wrapped__()
        result["forced_ocean"] = snapshot(model.step_with_report(sea, 1., **forcing))

        directory = Path(tmp)
        source = coupled.source.__wrapped__(directory)
        config = coupled.config.__wrapped__()
        advance = campaign.advance

        def recorded(adapter, air, receiver, sea, dt, config, *args, **kwargs):
            dynamics = receiver.H.model.dynamics
            tendency = dynamics.tendency_with_report
            stages = []

            def record_stage(state, stage_dt):
                rhs, work = tendency(state, stage_dt)
                stages.append(snapshot([rhs, work.transport]))
                return rhs, work

            with patch.object(dynamics, "tendency_with_report", record_stage):
                air, sea, report = advance(adapter, air, receiver, sea, dt, config, *args, **kwargs)
            steps.append(stages)
            steps.append([campaign.pack(adapter, air, receiver, sea, "comparison", {}), report])
            return air, sea, report

        with patch.object(campaign, "advance", recorded):
            steps = []
            coupled.test_explicit_legacy_cadences_reproduce_default_coupled_step(source, config)
            result["aquaplanet"] = steps
            steps = []
            coastal.test_coastal_runoff_ice_and_restart(config, source, directory)
            result["coastal_earth"] = steps
            steps = []
            shelf.test_prognostic_partial_columns_through_coupling_ale_and_disk_restart(
                config, source, directory
            )
            result["active_partial_columns"] = steps
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--save", type=Path)
    mode.add_argument("--compare", type=Path)
    args = parser.parse_args()
    root = args.checkout.resolve()
    sys.path[:0] = [str(root), str(root / "tests"), str(root / "tests/ocean")]
    torch.set_num_threads(1)
    actual = capture()
    if args.save:
        with args.save.open("xb") as dest:
            torch.save(actual, dest)
    else:
        expected = torch.load(args.compare, weights_only=True)
        assert actual.keys() == expected.keys()
        for case in actual:
            count = compare(actual[case], expected[case], case)
            print(f"{case}: {count} bitwise-identical tensors")
