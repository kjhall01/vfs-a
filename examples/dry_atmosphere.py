"""Conservative dry SSPRK3 or ARK4 on a small sphere; not a balanced benchmark."""
import argparse
import torch
from vfs import DryAtmosphere, DryIMEX, State, VoronoiMesh, CharneyPhillipsColumn, make_hybrid_coefficients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imex", action="store_true", help="use the retained analytic ARK4 split")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    mesh = VoronoiMesh(n_cells=16, lloyd_iters=0, verbose=False)
    a, b = make_hybrid_coefficients(3, p_top=5000.)
    column = CharneyPhillipsColumn(3, a, b, phi_full_mode="direct_u_gl2_adjoint")
    model = DryAtmosphere(mesh, column, stepper=DryIMEX() if args.imex else None).to(dtype=torch.float64)
    ref = model.mesh.cell_area
    state = State.from_theta(
        ref.new_full((mesh.nEdges, 3), 5.),
        ref.new_full((mesh.nCells, 4), 300.),
        ref.new_full((mesh.nCells, 1), 100000.), mesh=mesh, column=column,
    )
    before = model.diagnostics(state)
    state = model(state, 1.)
    after = model.diagnostics(state)
    print("Stepper:", "ARK4" if args.imex else "SSPRK3")
    print("Simulated seconds:", 1.)
    print("Relative mass change:", float(after['mass']/before['mass']-1))
    print("Energy change J/m²:", float((after['boundary_total_energy']-before['boundary_total_energy'])/mesh.cell_area.sum()))
    if args.imex:
        print("Solver:", model.stepper.solver.last_step_info)


if __name__ == '__main__':
    main()
