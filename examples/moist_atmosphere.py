"""Run from an installed package: python examples/moist_atmosphere.py.

Small illustrative cloudy flow, not a balanced atmospheric benchmark.
The ARK4 variant uses smooth, positive composition: it cannot bound the sharp
cloud edge used by the default explicit example.
"""

import argparse
import torch
from vfs import (
    MoistAtmosphere,
    CenteredIMEX,
    VoronoiMesh,
    CharneyPhillipsColumn,
    make_hybrid_coefficients,
)


def initialize(*, stepper=None, smooth_composition=False):
    mesh = VoronoiMesh(n_cells=64, lloyd_iters=0, verbose=False)
    A, B = make_hybrid_coefficients(6, p_top=5000.0, sigma_power=1.0)
    column = CharneyPhillipsColumn(
        6, A, B, phi_s=0.0, phi_full_mode="direct_u_gl2_adjoint"
    )
    model = MoistAtmosphere(mesh, column, stepper=stepper)
    pressure = model.interface_pressure(100_000.0)
    temperature = 280 * (pressure / 100_000.0) ** 0.08
    # Solid-body zonal flow sampled at edges, then projected to edge normals.
    cos_lat = torch.linalg.vector_norm(mesh.edge_xyz[:, :2], dim=-1)[:, None]
    wind = model.normal_wind(eastward=10 * cos_lat, northward=0.0)
    liquid = 0.002 * (mesh.cell_xyz[:, :1] > 0) * (pressure / 100_000.0) ** 2
    if smooth_composition:
        liquid = torch.full_like(pressure, .001)
    state = model.state_from_fields(
        temperature=temperature,
        surface_pressure=100_000.0,
        normal_wind=wind,
        vapor=0.01,
        liquid=liquid,
        ice=.0001 if smooth_composition else 0.,
    )
    return model, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imex", action="store_true", help="centered ARK4; no species bounds")
    args = parser.parse_args()
    steps, dt = (1, 1.) if args.imex else (10, 30.)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    # Opt out of gradient recording only for this forward example.
    with torch.no_grad():
        model, state = initialize(
            stepper=CenteredIMEX() if args.imex else None,
            smooth_composition=args.imex,
        )
        before = model.diagnostics(state)
        for _ in range(steps):
            state = model.step(state, dt=dt)
        after = model.diagnostics(state)
        if args.imex:
            print("Solver:", model.stepper.solver.last_step_info)
        fields = model.physical_fields(state)
        print("Stepper:", "centered ARK4 (not bounded)" if args.imex else "bounded SSPRK3")
        print("Simulated seconds:", steps * dt)
        print(
            "Relative mass change:",
            float((after["mass"] - before["mass"]) / before["mass"]),
        )
        print(
            "Species mass changes kg:", (after["species"] - before["species"]).tolist()
        )
        print(
            "Energy change J/m²:",
            float((after["energy"] - before["energy"]) / model.mesh.cell_area.sum()),
        )
        print(
            "Minimum vapor/liquid/ice:",
            [float(fields[k].min()) for k in ("vapor", "liquid", "ice")],
        )


if __name__ == "__main__":
    main()
