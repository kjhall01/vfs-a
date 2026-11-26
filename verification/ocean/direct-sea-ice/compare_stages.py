"""Compare both ice integrators to a separate checkout on the same CPU runtime."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys
from unittest.mock import patch

import torch


def fields(state):
    """Select the same stored quantities from either representation."""
    mechanical = getattr(state, "coupled", state)
    thermal = state if hasattr(state, "coupled") else state.ice
    result = {k: getattr(mechanical.ocean, k) for k in
              ("u_e", "h", "temp_h", "salt_h", "tracer_h", "tke_h", "eddy_h")}
    result.update({k: getattr(mechanical.ice, k) for k in ("area", "volume", "velocity")})
    result.update(snow_mass=thermal.snow_mass, cold_content=thermal.cold_content)
    return result


def capture():
    from test_pyam21_workflow import model
    from test_ice_temperature_workflow import thermal_case

    receiver, state = thermal_case(model.__wrapped__())
    owner = receiver.ocean_ice if hasattr(receiver, "ocean_ice") else receiver.snow
    mechanics = owner.mechanics if hasattr(owner, "mechanics") else owner.coupled
    ice = getattr(state, "coupled", state).ice
    snow = 0.15 * owner.hamiltonian.slab.parameters.density * ice.volume
    if hasattr(state, "coupled"):
        state = replace(state, snow_mass=snow)
    else:
        state = replace(state, ice=replace(ice, snow_mass=snow))
    results = {}
    for integrator in ("ssprk3", "ssprk43"):
        record = {}
        rhs = owner.tendency
        drag = mechanics.drag_step
        stages, kicks = [], []

        def stage(s, dt, **options):
            r, work = rhs(s, dt, **options)
            stages.append((dt, fields(s), fields(r), asdict(work)))
            return r, work

        def kick(s, dt):
            end, report = drag(s, dt)
            kicks.append(asdict(report))
            return end, report

        def split(name, s, external):
            record[name] = (fields(s), external)

        with patch.object(owner, "tendency", stage), patch.object(mechanics, "drag_step", kick):
            end, report = owner.step(state, 1., dynamics_integrator=integrator, stage_callback=split)
        assert len(stages) == (3 if integrator == "ssprk3" else 4)
        assert len(kicks) == 2 and kicks[0]["tke_gain"].sum() > 0
        record.update(end=fields(end), stages=stages, drag=kicks, report=asdict(report),
                      total_energy=owner.total_energy(end), ice_mass=owner.hamiltonian.mass(
                          getattr(end, "coupled", end)))
        results[integrator] = record
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--save", type=Path)
    mode.add_argument("--compare", type=Path)
    args = parser.parse_args()
    root = args.checkout.resolve()
    sys.path[:0] = [str(root), str(root / "tests"), str(root / "verification/coupled/cleanup")]
    from compare import compare, snapshot

    with torch.no_grad():
        actual = snapshot(capture())
    if args.save:
        with args.save.open("xb") as f:
            torch.save(actual, f)
    else:
        expected = torch.load(args.compare, weights_only=True)
        for method in actual:
            print(f"{method}: {compare(actual[method], expected[method], method)} identical tensors")
