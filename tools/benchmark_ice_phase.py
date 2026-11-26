"""Compare both Hamiltonian evaluations on identical thermal ice phase calls."""
import argparse
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from statistics import median
import sys
import time
from unittest.mock import patch

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.coupled import campaign
from workflows.coupled.device import execution_context, resolve_device, synchronize
from vfs.ice.thermal_phase import ThermalPhase
from vfs.ice.category_phase import CategoryThermodynamics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="PyAM21 source checkout")
    parser.add_argument("--config", type=Path, default=ROOT / "workflows/coupled/aquaplanet_multirate_4096_40.yaml")
    parser.add_argument("--mesh", type=Path, help="Saved Voronoi generators")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--repeats", type=int, default=16, help="Measured calls per method after two warmup pairs")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    config = replace(campaign.CampaignConfig(**yaml.safe_load(args.config.read_text())),
                     device=args.device, dtype="float64", threads=args.threads, compile=args.compile)
    config, xyz, _ = campaign.load_mesh(config, args.mesh)
    device = resolve_device(config.device)
    with execution_context(device), torch.no_grad():
        _, _, receiver, sea = campaign.build(config, args.source, None, smoke_no_radiation=True, generator_xyz=xyz)
        owner = receiver.ocean_ice
        flux = 150 * sea.ocean.mesh.cell_xyz[:, 2:3]
        times = {"reference": [], "reduced": []}
        results = {}
        for i in range(args.repeats + 2):
            order = ("reference", "reduced") if i % 2 == 0 else ("reduced", "reference")
            for name in order:
                method = (patch.object(ThermalPhase, "_solve_mass", CategoryThermodynamics._solve_mass)
                          if name == "reference" else nullcontext())
                with method:
                    synchronize(device)
                    start = time.perf_counter()
                    state, _, _ = owner.phase_step(sea, 600, flux)
                    synchronize(device)
                    elapsed = time.perf_counter() - start
                results[name] = state
                if i >= 2:
                    times[name].append(elapsed)
        a, b = results["reference"], results["reduced"]
        ice_error = (owner.hamiltonian.mass(a) - owner.hamiltonian.mass(b)).abs().max().item()
        temp_error = (a.ocean.resolved.temperature() - b.ocean.resolved.temperature()).abs().max().item()
        reference, reduced = median(times["reference"]), median(times["reduced"])
        print(f"{config.cells} cells, {config.ocean_levels} ocean levels, {device}, float64, "
              f"compile={config.compile}, {args.repeats} calls per method")
        print(f"Median phase call: reference {1000*reference:.2f} ms, reduced {1000*reduced:.2f} ms, "
              f"{reference/reduced:.3f}x")
        print(f"Maximum endpoint differences: ice {ice_error:.4g} kg/m², ocean CT {temp_error:.4g} K")


if __name__ == "__main__":
    main()
