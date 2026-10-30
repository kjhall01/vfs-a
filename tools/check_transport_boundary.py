"""Single-Euler entry probe; not a time-integrated prescribed transport test."""
import argparse
import importlib.util
import math
from pathlib import Path

import torch
from vfs import (
    MoistAtmosphere, VoronoiMesh, CharneyPhillipsColumn, make_hybrid_coefficients,
)

spec = importlib.util.spec_from_file_location(
    "manifest", Path(__file__).with_name("run_verification.py")
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    period = 12 * 86400.
    configuration = dict(
        cells=[16, 64, 256], layers=1, dt_s=1., period_s=period, time_s=0.,
        dtype="float64", device="cpu", threads=1,
        purpose="single Euler boundary probe, NOT a transport return experiment",
        velocity="direct analytical midpoint sample; no projection",
    )
    with common.recorded_run(
        args.repo, args.output, "C5-BOUNDARY-PROBE", configuration, __file__
    ):
        rows = []
        for cells in configuration["cells"]:
            mesh = VoronoiMesh(
                n_cells=cells, lloyd_iters=0, radius=6371220., verbose=False
            )
            xyz = mesh.edge_xyz
            lon = torch.atan2(xyz[:, 1], xyz[:, 0])
            lat = torch.asin(xyz[:, 2])
            east = torch.stack((-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)), -1)
            north = torch.stack((
                -torch.sin(lat)*torch.cos(lon), -torch.sin(lat)*torch.sin(lon),
                torch.cos(lat),
            ), -1)
            for flow in ("rotation_equatorial", "rotation_tilted", "NL2010_case4"):
                # These are t=0 samples only. See the original equations and
                # dimensionalization in INCREMENT_1C_BOUNDARY_PROPOSAL.md.
                if flow.startswith("rotation"):
                    alpha = 0. if flow == "rotation_equatorial" else math.pi/4
                    axis = xyz.new_tensor([-math.sin(alpha), 0., math.cos(alpha)])
                    velocity = 2*math.pi*mesh.radius/period * torch.linalg.cross(
                        axis.expand_as(xyz), xyz
                    )
                else:
                    # Original unit-sphere Case 4: k=2, T*=5. Time unit=T/5.
                    u = (2*torch.sin(lon)**2*torch.sin(2*lat)
                         + 2*math.pi/5*torch.cos(lat))*mesh.radius*5/period
                    v = 2*torch.sin(2*lon)*torch.cos(lat)*mesh.radius*5/period
                    velocity = u[:, None]*east + v[:, None]*north
                wind = (velocity*mesh.edge_normal).sum(-1)[:, None]
                a, b = make_hybrid_coefficients(1, p_top=5000.)
                model = MoistAtmosphere(mesh, CharneyPhillipsColumn(
                    1, a, b, phi_full_mode="direct_u_gl2_adjoint"
                ))
                state = model.state_from_fields(
                    temperature=280., surface_pressure=100000., normal_wind=wind,
                    vapor=.01, liquid=.001, ice=.0001,
                )
                # Exercise the ORIGINAL coupled Euler map. Its wind output is
                # not used to claim an isolated prescribed-flow integration.
                updated, info = model.stepper.transport.euler_with_info(state, 1.)
                dp = updated.column.interface_pressure(updated.ps).diff()
                dm = torch.cat((.5*dp, .5*dp), -1)
                fractions = updated.water_content/dm[..., None]
                expected_water = (
                    info["amounts"][:, 1:].reshape_as(updated.water_content)
                    / mesh.cell_area[:, None, None]
                )
                rows.append(dict(
                    cells=cells, flow=flow,
                    max_sampled_velocity_divergence_per_s=float(mesh.div(wind).abs().max()),
                    max_ps_change_Pa=float((updated.ps-state.ps).abs().max()),
                    constant_composition_max_error=float(
                        (fractions-fractions.new_tensor([.01, .001, .0001])).abs().max()
                    ),
                    exact_water_ledger=torch.equal(updated.water_content, expected_water),
                    limited_faces=info["limited_faces"], substeps=info["substeps"],
                ))
                torch.save(dict(
                    mesh=mesh.state_dict(), column=model.column.state_dict(),
                    wind=wind, initial_ps=state.ps, final_ps=updated.ps,
                    initial_water=state.water_content, final_water=updated.water_content,
                    entropy_content=state.entropy_content, info=info,
                ), args.output / f"{flow}-{cells}.pt")
        common.write_json(args.output / "metrics.json", rows)


if __name__ == "__main__":
    main()
