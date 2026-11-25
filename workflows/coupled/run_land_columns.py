"""Atmospheric surface-exchange experiment with grassland and a finite slab ocean."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from workflows.coupled.device import execution_context, resolve_device
from workflows.coupled.driver import exchange_step
from workflows.coupled.grid_io import read_mesh
from workflows.coupled.land import LandSlabReceiver
from workflows.pyam21.setup import example
from workflows.pyam21.dynamics import advance_dynamics, pressure_velocity


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path, help="pinned PyAM21 checkout")
    p.add_argument("--mesh", type=Path, help="saved Voronoi generators [cell,3]")
    p.add_argument("--land-mask", type=Path, help="boolean .npy in mesh-cell order")
    p.add_argument("--surface-height", type=Path, help="surface elevation in metres, .npy in mesh-cell order")
    p.add_argument("--cells", type=int, default=32)
    p.add_argument("--levels", type=int, default=12)
    p.add_argument("--dt", type=int, default=600)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--atmosphere-dt", type=int, help="enable bounded atmospheric dynamics at this timestep (s)")
    p.add_argument("--soil-water", type=float, default=50., help="initial soil water in kg/m² of land")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda", "cuda:0"))
    p.add_argument("--radiation-data", type=Path, help="SeaEsfRad tables; omit for radiation-off")
    p.add_argument("--radiation-step", type=int, default=7200)
    args = p.parse_args(argv)
    if args.dt < 1 or args.steps < 1 or args.cells < 16 or args.levels < 4:
        p.error("positive dt and steps, at least 16 cells and four levels required")
    if args.atmosphere_dt is not None and (args.atmosphere_dt <= 0 or args.dt % args.atmosphere_dt):
        p.error("atmosphere timestep must divide the physics interval")
    if args.radiation_data and (args.radiation_step <= 0 or args.radiation_step % args.dt):
        p.error("radiation step must be a positive multiple of dt")
    geometry = None
    cells = args.cells
    if args.mesh:
        xyz, _ = read_mesh(args.mesh)
        cells = len(xyz)
        geometry = dict(generator_xyz=xyz, lloyd_iters=0, verbose=False)
    height = None
    if args.surface_height:
        height = np.load(args.surface_height, allow_pickle=False)
        if height.shape != (cells,) or height.dtype.kind not in "fiu" or not np.isfinite(height).all():
            p.error("--surface-height requires one finite elevation per mesh cell")
    device = resolve_device(args.device)
    with execution_context(device), torch.no_grad():
        adapter, air = example(args.source, cells=cells, levels=args.levels, aquaplanet=True,
                               mesh_options=geometry, surface_height_m=height, device=device, radiation_data=args.radiation_data,
                               namelist_overrides={"surface_flux_nml": {"gust_min": 0.1}},
                               radiation_step=args.radiation_step)
        mesh = adapter.model.mesh
        if args.land_mask:
            mask = torch.as_tensor(np.load(args.land_mask, allow_pickle=False), device=device)
        else:
            # A meridional sector, one quarter of the longitude range.
            mask = mesh.cell_xyz[:, 0] > mesh.cell_xyz[:, 1].abs()
        receiver = LandSlabReceiver(adapter, mask)
        surface = receiver.initialize(air.surface["t_surf"], soil_water=args.soil_water)
        area = mesh.cell_area
        land_area = area[mask]
        land_water0 = surface.land.water()

        def inventories():
            water, energy = receiver.inventories(surface)
            air_diag = adapter.model.diagnostics(air.atmosphere)
            return (air_diag["species"].sum() + (water * area).sum(),
                    air_diag["energy"] + (energy * area).sum())
        w0, e0 = inventories()
        upstream = toa = routed = gravity_heat = rain_snow = vapor = dynamics_energy = 0.
        max_heat = max_water = 0.
        for _ in range(args.steps):
            omega = None
            if args.atmosphere_dt is not None:
                e_start = adapter.model.diagnostics(air.atmosphere)["energy"]
                for _substep in range(args.dt // args.atmosphere_dt):
                    air = advance_dynamics(adapter, air, args.atmosphere_dt)
                dynamics_energy += float((adapter.model.diagnostics(air.atmosphere)["energy"] - e_start) / area.sum())
                omega = pressure_velocity(adapter, air)
            air, surface, exchange, ar, sr = exchange_step(adapter, air, receiver, surface, args.dt, omega=omega)
            upstream += float((area * ar.energy_changes["am2_energy_residual"]).sum() / area.sum())
            toa += float((area * exchange.toa).sum() / area.sum())
            rain_snow += float((land_area * (exchange.rain + exchange.snow)[mask]).sum() / land_area.sum())
            vapor += float((land_area * exchange.vapor[mask, 0]).sum() / land_area.sum())
            routed += float((area * sr.runoff_received).sum() / area.sum())
            gravity_heat += float((area * sr.runoff_gravity_heat).sum() / area.sum())
            max_heat = max(max_heat, float(sr.energy_residual.abs().max()))
            max_water = max(max_water, float(sr.water_residual.abs().max()))
        w1, e1 = inventories()
        result = dict(cells=cells, land_cells=int(mask.sum()), seconds=air.time_seconds,
                      radiation=bool(args.radiation_data), runoff_delivered_kg_m2_global=routed,
                      runoff_gravity_heat_J_m2_global=gravity_heat,
                      precipitation_kg_m2_land=rain_snow, vapor_to_air_kg_m2_land=vapor,
                      storage_change_kg_m2_land=float((land_area * (surface.land.water()-land_water0)).sum() / land_area.sum()),
                      runoff_kg_m2_land=routed * float(area.sum() / land_area.sum()),
                      water_residual_kg_m2=float((w1-w0)/area.sum()),
                      numerical_energy_residual_J_m2=float((e1-e0)/area.sum()) - toa - upstream,
                      upstream_energy_residual_J_m2=upstream, toa_J_m2=toa,
                      dynamics_energy_change_J_m2=dynamics_energy, atmosphere_dt=args.atmosphere_dt,
                      max_surface_energy_residual_J_m2=max_heat,
                      max_surface_water_residual_kg_m2=max_water)
        print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
