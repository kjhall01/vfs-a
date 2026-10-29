"""Tensor-valued global inventories for the mixture state (no detach/reporting).

Energy includes the fixed-pressure top-boundary reservoir. Existing energy
components/rates remain methods of MoistConservativeDynamics.
"""

from .moist_cp import interface_pressure_mass
from .conservative import v


def inventories(model, state):
    area = state.mesh.cell_area[:, None]
    mass = interface_pressure_mass(v(state.column.interface_pressure(state.ps)).diff())
    gravity = model.kernel.gravity
    return {
        "mass": (area * mass).sum() / gravity,
        "species": (area[..., None] * state.water_content).sum((0, 1)) / gravity,
        "entropy": (area * state.entropy_content).sum() / gravity,
        "energy": model.boundary_total_energy(state),
    }
