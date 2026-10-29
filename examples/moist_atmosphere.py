"""Run from an installed package: python examples/moist_atmosphere.py.

Small illustrative cloudy flow, not a balanced atmospheric benchmark.
"""

import torch
from vfs import (
    MoistAtmosphere,
    VoronoiMesh,
    CharneyPhillipsColumn,
    make_hybrid_coefficients,
)


def initialize():
    mesh = VoronoiMesh(n_cells=64, lloyd_iters=0, verbose=False)
    A, B = make_hybrid_coefficients(6, p_top=5000.0, sigma_power=1.0)
    column = CharneyPhillipsColumn(
        6, A, B, phi_s=0.0, phi_full_mode="direct_u_gl2_adjoint"
    )
    model = MoistAtmosphere(mesh, column)
    pressure = model.interface_pressure(100_000.0)
    temperature = 280 * (pressure / 100_000.0) ** 0.08
    # Solid-body zonal flow sampled at edges, then projected to edge normals.
    cos_lat = torch.linalg.vector_norm(mesh.edge_xyz[:, :2], dim=-1)[:, None]
    wind = model.normal_wind(eastward=10 * cos_lat, northward=0.0)
    liquid = 0.002 * (mesh.cell_xyz[:, :1] > 0) * (pressure / 100_000.0) ** 2
    state = model.state_from_fields(
        temperature=temperature,
        surface_pressure=100_000.0,
        normal_wind=wind,
        vapor=0.01,
        liquid=liquid,
    )
    return model, state


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    # Opt out of gradient recording only for this forward example.
    with torch.no_grad():
        model, state = initialize()
        before = model.diagnostics(state)
        for _ in range(10):
            state = model.step(state, dt=30.0)
        after = model.diagnostics(state)
        fields = model.physical_fields(state)
        print("Simulated seconds:", 300)
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
