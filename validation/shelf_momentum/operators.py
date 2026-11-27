"""Candidate operators for a controlled ocean–ice replay, outside the campaign."""

from vfs.ice.ice_drag import edge_drag
from vfs.ocean.energy import ConstantDensityEnergy
from vfs.ocean.temperature.shelf_momentum import OpeningEnergy, install_opening_momentum


def use_openings(model, *, match_ale=True):
    energy = install_opening_momentum(model)
    if not match_ale:
        # Deliberately inconsistent control for validation/shelf_momentum/ale_check.py.
        energy.edge_thickness = ConstantDensityEnergy.edge_thickness.__get__(energy)
        energy.column_kinetic_energy = ConstantDensityEnergy.column_kinetic_energy.__get__(energy)
        model.dynamics.matched_openings = False
    return energy
