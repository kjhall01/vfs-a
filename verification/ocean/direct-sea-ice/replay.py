"""Replay only the joint ocean–ice step of a shelf NetCDF restart on either checkout."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys
from time import perf_counter

import torch
import xarray as xr
import yaml


def replay(args):
    root = args.checkout.resolve()
    sys.path[:0] = [str(root), str(root / "verification/coupled/cleanup")]
    from workflows.coupled import campaign
    from workflows.coupled.state_io import read_checkpoint
    from compare import compare

    preset = root / "workflows/coupled/earth_shelf_4096_benchmark.yaml"
    config = replace(campaign.CampaignConfig(**yaml.safe_load(preset.read_text())),
                     compile=args.compile, threads=args.threads)
    config, xyz, mesh = campaign.load_mesh(config, args.mesh)
    with xr.open_dataset(args.restart, decode_times=False) as ds:
        if mesh["sha256"] != ds.mesh_sha256:
            raise ValueError("mesh differs from the restart")
        geography = dict(land=ds.land_mask.values.astype(bool),
                         height_m=ds.surface_height_m.values, depth_m=ds.ocean_depth_m.values)
        reference = torch.from_numpy(ds.ocean_reference_thickness.values.copy())
    _, _, receiver, template = campaign.build(config, args.source, None, smoke_no_radiation=True,
                                              generator_xyz=xyz, geography=geography)
    data = read_checkpoint(args.restart)
    direct = hasattr(receiver, "ocean_ice")
    initial = template.ocean if direct else template.coupled.ocean
    wet = torch.from_numpy(~geography["land"])
    current = initial.column.layer_reference.cpu()[wet]
    stored = reference[wet]
    if not torch.equal(current > 0, stored > 0):
        raise ValueError("wet-layer topology differs from the restart")
    # The native file masks land. Stored wet depths may have been generated
    # with a different CPU math library than this same-host paired comparison.
    torch.testing.assert_close(current, stored, atol=2e-12, rtol=2e-14)
    ocean = initial.with_layers(initial.ocean.with_tensors(**data["ocean"]),
                                tke_h=data["tke_h"], eddy_h=data["eddy_h"])
    if direct:
        ice = campaign.SeaIceState(**data["ice"], snow_mass=data["snow_mass"],
                                   cold_content=data["cold_content"])
        state = campaign.OceanIceState(ocean, ice)
        model = receiver.ocean_ice
    else:
        categories = campaign.CategoryIce(**data["ice"])
        state = campaign.ThermalIceState(campaign.RidgingState(ocean, categories),
                                         data["snow_mass"], data["cold_content"])
        model = receiver.snow
    state = replace(state, land=campaign.LandState(**data["land"]))
    model.check(state)

    def step():
        return model.step(state, args.dt, dynamics_integrator=args.integrator)

    end, report = step()
    for _ in range(args.repeats):
        start = perf_counter()
        end, report = step()
        print(f"Joint step {perf_counter() - start:.6f} s", flush=True)
    mechanical = end if direct else end.coupled
    thermal = end.ice if direct else end
    payload = dict(ocean={k: getattr(mechanical.ocean, k) for k in
                          ("u_e", "h", "temp_h", "salt_h", "tracer_h", "tke_h", "eddy_h")},
                   ice={k: getattr(mechanical.ice, k) for k in ("area", "volume", "velocity")},
                   snow_mass=thermal.snow_mass, cold_content=thermal.cold_content,
                   land=asdict(end.land), report=asdict(report), energy=model.total_energy(end))
    if args.save:
        with args.save.open("xb") as dest:
            torch.save(payload, dest)
    if args.compare:
        expected = torch.load(args.compare, weights_only=True)
        print(f"{compare(payload, expected, 'shelf')} bitwise-identical tensors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "mesh", "source", "restart"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dt", type=int, default=600)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=0)
    parser.add_argument("--integrator", choices=("ssprk3", "ssprk43"), default="ssprk3")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--save", type=Path)
    mode.add_argument("--compare", type=Path)
    with torch.no_grad():
        replay(parser.parse_args())
