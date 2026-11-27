"""Replay a 4096-cell Earth-shelf checkpoint without external column forcing.

Run from repo root: python -m validation.shelf_momentum.replay restart.nc --hours 6
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset

from vfs import VoronoiMesh
from vfs.ice.ocean_ice import OceanIce
from vfs.ice.ridging import CategoryIce, RidgingState
from vfs.ocean.column import OceanColumn
from vfs.ocean.domain import OceanDomain
from vfs.ocean.state import OceanState
from vfs.ocean.temperature.eddy_energy import EddyState
from vfs.ocean.temperature.roquet import RoquetOcean
from vfs.ocean.temperature.shelf_transport import ShelfTemperatureOcean

from .operators import edge_drag, use_openings


def load_restart(path, mesh=None):
    with Dataset(path) as ds:
        ds.set_auto_mask(False)

        def array(key):
            return torch.from_numpy(np.asarray(ds[key][:]).copy())

        if mesh is None:
            mesh = VoronoiMesh(
                n_cells=ds["cell_xyz"].shape[0],
                generator_xyz=array("cell_xyz").numpy(),
                lloyd_iters=0,
                rotation_scheme="compatible",
                verbose=False,
            )
        assert torch.equal(mesh.cellsOnEdge, array("cells_on_edge").long())
        wet = ~array("land_mask").bool()
        nominal = array("ocean_nominal_thickness")
        depth = torch.where(wet, array("ocean_depth_m"), nominal.sum())
        column = OceanColumn(
            len(nominal),
            gravity=9.8,
            total_depth=depth,
            rest_thickness=nominal,
            partial_cells=True,
        )
        torch.testing.assert_close(
            column.layer_reference[wet],
            array("ocean_reference_thickness")[wet],
            atol=1e-10,
            rtol=1e-14,
        )
        domain = OceanDomain(mesh, wet, depth=depth)
        model = RoquetOcean(mesh, column, domain=domain)
        model.dynamics = ShelfTemperatureOcean(model.ocean)
        s = OceanState(
            *(
                array(k)
                for k in (
                    "ocean_u_e",
                    "ocean_h",
                    "ocean_temp_h",
                    "ocean_salt_h",
                    "ocean_tracer_h",
                )
            ),
            mesh=mesh,
            column=column,
        )
        ice = CategoryIce(
            *(array(k) for k in ("ice_area", "ice_volume", "ice_velocity"))
        )
        state = RidgingState(EddyState(s, array("tke_h"), array("eddy_h")), ice)
    owner = OceanIce(model)
    owner.check(state)
    return owner, state


def inventories(owner, state):
    s = state.ocean.ocean
    mech = owner.ocean.energy.base.mechanical
    terms = owner.hamiltonian.terms(state)
    # h*dK/dh does not partition K when fixed bathymetry enters the opening.
    terms["kinetic"] = 0.5 * mech.metric(s) * s.u_e.square()
    energy = {name: value.sum().item() for name, value in terms.items()}
    wet = owner.ocean.ocean.domain.wet_cell
    area = s.mesh.cell_area[wet, None]
    water = (area * s.h[wet]).sum().item()
    salt = (area * s.salt_h[wet]).sum().item()
    return energy, water, salt


def trajectory(path, variant, hours, mesh=None, *, dt=1200.0):
    owner, state = load_restart(path, mesh)
    s = state.ocean.ocean
    mesh = s.mesh
    if variant in ("opening", "both"):
        use_openings(owner.ocean)

    def drag(current):
        if variant in ("drag", "both"):
            return edge_drag(owner, current, dt / 2)[0]
        return owner.drag_step(current, dt / 2)[0]

    ocean_edge, level = np.unravel_index(s.u_e.abs().argmax().item(), s.u_e.shape)
    ice_edge = int(state.ice.velocity.abs().argmax())
    points = mesh.verticesOnEdge[ocean_edge]
    initial, water0, salt0 = inventories(owner, state)
    start = time.perf_counter()
    nsteps = round(hours * 3600 / dt)
    for step in range(nsteps + 1):
        if step == 0 or step == nsteps or step % max(1, round(3600 / dt)) == 0:
            physical = state.ocean.ocean
            slip = (state.ice.velocity - physical.u_e[:, :1]).abs()
            energy, water, salt = inventories(owner, state)
            cover = state.ice.area.sum(-1)[mesh.cellsOnEdge].amin(-1) >= 0.15
            ediff = (
                sum(energy[n] - initial[n] for n in energy)
                / mesh.cell_area.sum().item()
            )
            record = {
                "variant": variant,
                "hours": step * dt / 3600,
                "ocean_edge": int(ocean_edge),
                "ice_edge": ice_edge,
                "ocean_u": physical.u_e[ocean_edge, level].item(),
                "peak_ocean": physical.u_e.abs().max().item(),
                "zeta": mesh.curl(physical.u_e)[points, level].tolist(),
                "ice_u": state.ice.velocity[ice_edge].item(),
                "slip": slip[ice_edge].item(),
                "pack_slip_p95": torch.quantile(slip[cover], 0.95).item(),
                "max_ice": state.ice.velocity.abs().max().item(),
                "energy_change_J_m2": ediff,
                "relative_water": (water - water0) / water0,
                "relative_salt": (salt - salt0) / salt0,
                "wall_seconds": time.perf_counter() - start,
            }
            print(json.dumps(record), flush=True)
        if step < nsteps:
            try:
                state = drag(state)
                state, _ = owner.dynamics_step(
                    state, dt, integrator="ssprk43", drag=False
                )
                state = drag(state)
            except (ValueError, FloatingPointError) as exc:
                print(
                    json.dumps(
                        {
                            "variant": variant,
                            "status": "rejected",
                            "completed_hours": step * dt / 3600,
                            "requested_hours": hours,
                            "message": str(exc),
                        }
                    ),
                    flush=True,
                )
                return mesh
    return mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("restart", type=Path)
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument("--dt", type=int, default=1200)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["baseline", "drag", "opening", "both"],
        default=["baseline", "drag", "opening", "both"],
    )
    args = parser.parse_args()
    if args.dt < 1 or 3600 % args.dt:
        parser.error("dt must divide one hour into integer steps")
    if args.hours < 1:
        parser.error("choose at least one hour")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    with torch.no_grad():
        mesh = None
        for variant in args.variants:
            mesh = trajectory(args.restart, variant, args.hours, mesh, dt=args.dt)


if __name__ == "__main__":
    main()
