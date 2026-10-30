"""One explicitly forced conservative dry SSPRK3 step; no climate qualification."""
import torch
from vfs import DryAtmosphere, State, VoronoiMesh, CharneyPhillipsColumn, make_hybrid_coefficients
from vfs.atmosphere.held_suarez import HeldSuarezForcing
from vfs.core.integrators import ssprk3


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    mesh = VoronoiMesh(n_cells=16, lloyd_iters=0, verbose=False)
    a, b = make_hybrid_coefficients(3, p_top=5000.)
    column = CharneyPhillipsColumn(3, a, b, phi_full_mode="direct_u_gl2_adjoint")
    model = DryAtmosphere(mesh, column).to(dtype=torch.float64)
    forcing = HeldSuarezForcing().to(dtype=torch.float64)
    ref = mesh.cell_area
    state = State.from_theta(
        ref.new_full((mesh.nEdges, 3), 5.),
        ref.new_full((mesh.nCells, 4), 300.),
        ref.new_full((mesh.nCells, 1), 100000.), mesh=mesh, column=column,
    )

    # Evaluate BOTH sources at each existing SSPRK3 stage. model(state, dt)
    # alone remains unforced; this is not post-step relaxation or a new solver.
    def rhs(stage):
        return model.tendency(stage) + forcing.tendency(stage)

    terms = {"conservative": model.tendency(state), **forcing.tendency_terms(state)}
    for name, term in terms.items():
        power = model.dynamics.boundary_energy_rate_closed(state, term)
        print(name, "instantaneous power W:", float(power))
    next_state = ssprk3(rhs, state, 1.)
    print("Relative mass change:", float(next_state.total_mass() / state.total_mass() - 1))
    print("Finite-step energy change J:", float(
        model.dynamics.boundary_total_energy(next_state)
        - model.dynamics.boundary_total_energy(state)))


if __name__ == "__main__":
    main()
