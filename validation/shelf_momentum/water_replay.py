"""Paired ice-drag replay using the campaign's water scheduler and checkpoint clock.

Includes ocean–ice–snow physics and ALE. No new atmosphere/land fluxes. Uses
the saved manifest settings for the 4096-cell pilot.
"""

import argparse
import json
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from netCDF4 import Dataset

from vfs.ocean.temperature.backscatter import EddyBackscatter
from vfs.ocean.temperature.roquet import RoquetMixing
from workflows.coupled.campaign import CampaignConfig, _advance_water, ale_step
from workflows.coupled.ice_temperature import (
    ThermalIceReceiver,
    ThermalIceState,
    ThermalSnowOceanIce,
)

from .operators import edge_drag, use_openings
from .replay import load_restart


class StageWork:
    """Global Hamiltonian changes from actual prognostic stores at each water stage."""

    ocean_stages = True
    substeps = True

    def __init__(self, mode, snow, sea):
        self.mode = mode
        self.area = sea.coupled.ocean.ocean.mesh.cell_area
        self.snow = snow
        self.previous = self.terms(sea)

    def terms(self, sea):
        terms = self.snow.hamiltonian.terms(sea.coupled)
        out = {key: value.sum().item() for key, value in terms.items()}
        out["cold_content"] = (self.area[:, None] * sea.cold_content).sum().item()
        return out

    def observe(
        self, stage, adapter, air, receiver, sea, *, external=0.0, details=None
    ):
        current = self.terms(sea)
        delta = (
            sum(current[n] - self.previous[n] for n in current) / self.area.sum().item()
        )
        self.previous = current
        print(
            json.dumps(
                {"mode": self.mode, "stage": stage, "residual_J_m2": delta - external}
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("restart")
    parser.add_argument("manifest")
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["baseline", "edge", "opening", "both"],
        default=["baseline", "edge", "opening", "both"],
    )
    parser.add_argument("--trace-first-interval", action="store_true")
    args = parser.parse_args()
    if args.hours < 1:
        parser.error("choose at least one hour")
    with open(args.manifest) as stream:
        config = CampaignConfig(**json.load(stream)["configuration"])
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(3)
    with torch.no_grad(), Dataset(args.restart) as ds:
        start = int(np.asarray(ds["time"][:]).item())

        def field(name):
            return torch.tensor(np.asarray(ds[name][:]))

        mesh = None
        for mode in args.variants:
            cp, st = load_restart(args.restart, mesh)
            mesh = st.ocean.ocean.mesh
            cp.ocean.closure = RoquetMixing(
                config.mixing_length_m,
                substeps=config.mixing_substeps,
                budget_rtol=1e-9,
            )
            cp.ocean.backscatter = EddyBackscatter(
                config.backscatter_viscosity_m2_s,
                config.backscatter_days * 86400,
                config.backscatter_fraction,
                config.biharmonic_viscosity_m4_s,
            )
            if mode in ("opening", "both"):
                use_openings(cp.ocean)
            if mode in ("edge", "both"):
                cp.drag_step = MethodType(edge_drag, cp)
            snow = ThermalSnowOceanIce(cp)
            receiver = ThermalIceReceiver(snow)
            adapter = SimpleNamespace(model=SimpleNamespace(mesh=mesh))
            sea = ThermalIceState(st, field("snow_mass"), field("cold_content"))
            snow.check(sea)
            i = int(st.ice.velocity.abs().argmax())
            area = mesh.cell_area.sum()
            wet = cp.ocean.ocean.domain.wet_cell
            A = mesh.cell_area[wet, None]
            energy0 = snow.total_energy(sea)
            trace = StageWork(mode, snow, sea) if args.trace_first_interval else None
            e, k = divmod(
                st.ocean.ocean.u_e.abs().argmax().item(), st.ocean.ocean.h.shape[1]
            )
            heat = 0.0

            def stores(state, cp=cp, A=A, wet=wet):
                oc = state.coupled.ocean.ocean
                m = cp.hamiltonian.mass(state.coupled)
                return torch.stack(
                    (
                        (
                            A
                            * (oc.column.rho0 * oc.h.sum(1, keepdim=True)[wet] + m[wet])
                        ).sum(),
                        (A * oc.salt_h[wet]).sum(),
                    )
                )

            initial = stores(sea)
            counts = {"gmredi_steps": 0, "ale_steps": 0}
            for sec in range(0, args.hours * 3600 + 1, config.physics_dt):
                if sec % 3600 == 0:
                    oc = sea.coupled.ocean.ocean
                    ice = sea.coupled.ice
                    slip = (ice.velocity - oc.u_e[:, :1]).abs()
                    pack = ice.area.sum(-1)[mesh.cellsOnEdge].amin(-1) >= 0.15
                    print(
                        json.dumps(
                            dict(
                                mode=mode,
                                hours=sec / 3600,
                                edge=i,
                                slip=slip[i].item(),
                                pack_slip_p95=torch.quantile(slip[pack], 0.95).item(),
                                peak_ice=ice.velocity.abs().max().item(),
                                ice_u=ice.velocity[i].item(),
                                ocean_u=oc.u_e[e, k].item(),
                                relative_stores=(
                                    (stores(sea) - initial) / initial
                                ).tolist(),
                                energy_change_J_m2=(
                                    (snow.total_energy(sea) - energy0) / area
                                ).item()
                                - heat,
                                **counts,
                            )
                        ),
                        flush=True,
                    )
                if sec == args.hours * 3600:
                    break
                sea, boundary, r = _advance_water(
                    adapter,
                    None,
                    receiver,
                    sea,
                    config.physics_dt,
                    config,
                    trace if sec == 0 else None,
                )
                counts["gmredi_steps"] += r["gmredi_steps"]
                heat += boundary
                if (start + sec + config.physics_dt) % config.ale_dt == 0:
                    sea, r = ale_step(receiver, sea, config)
                    counts["ale_steps"] += r["ale_steps"]


if __name__ == "__main__":
    main()
