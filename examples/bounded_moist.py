"""Small cloudy forward run using only the installed package (CPU/float64).

Initialization is spelled out until the physical-fields constructor is added.
The cloud edge activates the limiter. This is an example, not a benchmark.
"""

import torch
from vfs.atmosphere import MoistConservativeDynamics, MoistState, BoundedGibbs
from vfs.atmosphere.moist_cp import interface_pressure_mass
from vfs.atmosphere.moist_budgets import inventories
from vfs.geometry.mesh import VoronoiMesh
from vfs.geometry.column import CharneyPhillipsColumn, make_hybrid_coefficients


def initialize():
    torch.set_default_dtype(torch.float64)
    mesh = VoronoiMesh(n_cells=64, lloyd_iters=0, verbose=False)
    A, B = make_hybrid_coefficients(6, p_top=5000.0, sigma_power=1.0)
    column = CharneyPhillipsColumn(
        6, A, B, phi_s=0.0, phi_full_mode="direct_u_gl2_adjoint"
    )
    base = MoistConservativeDynamics()
    ps = torch.full((mesh.nCells, 1), 100_000.0)
    pressure = column.interface_pressure(ps)
    mass = interface_pressure_mass(pressure.diff())  # pressure thickness, Pa
    temperature = 280 * (pressure / 100_000.0) ** 0.08
    water = torch.zeros(mesh.nCells, 7, 3)  # vapor, liquid, ice mass fractions
    water[..., 0] = 0.01
    water[..., 1] = 0.002 * (mesh.cell_xyz[:, :1] > 0) * (pressure / 100_000.0) ** 2
    fractions = torch.cat([1 - water.sum(-1, keepdim=True), water], dim=-1)
    axis = torch.tensor([0.0, 0.0, 1.0]).expand_as(mesh.edge_xyz)
    wind = (
        10
        * (torch.linalg.cross(axis, mesh.edge_xyz) * mesh.edge_normal)
        .sum(-1, keepdim=True)
        .expand(-1, 6)
        .clone()
    )
    state = MoistState(
        wind,
        ps,
        mass * base.thermodynamics.entropy(temperature, pressure, fractions),
        mass[..., None] * water,
        mesh,
        column,
    )
    return base, state


def main():
    torch.set_num_threads(1)
    with torch.no_grad():
        base, state = initialize()
        model = BoundedGibbs(base, release_as_heat=True)
        before = inventories(base, state)
        for _ in range(10):
            state = model.step(state, dt=30.0)
        after = inventories(base, state)
        print("Simulated seconds:", 300)
        print(
            "Relative mass change:",
            float((after["mass"] - before["mass"]) / before["mass"]),
        )
        print(
            "Energy change J/m²:",
            float((after["energy"] - before["energy"]) / state.mesh.cell_area.sum()),
        )


if __name__ == "__main__":
    main()
