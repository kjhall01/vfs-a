"""Experimental ocean core package for VFS-O.

This package exports ocean-owned state, column/domain geometry, EOS, dynamics,
physics, and initial-condition symbols. Named ocean recipe models live in
``vfs.ocean_model``.
"""

from __future__ import annotations

from .column import OceanColumn
from .constants import (
    OCEAN_ALPHA_T,
    OCEAN_BETA_S,
    OCEAN_CP,
    OCEAN_EOS_MODES,
    OCEAN_GRAVITY,
    OCEAN_PRESSURE_GRADIENT_MODES,
    OCEAN_RHO0,
    OCEAN_ROTATION_RATE,
    OCEAN_S0,
    OCEAN_T0,
)
from .domain import OceanDomain
from .dynamics import HydrostaticBoussinesqOceanDynamics, MultilayerOceanDynamics
from .eos import jmd95_bulk_modulus, jmd95_seawater_density, linear_seawater_density
from .initial_conditions import ocean_planet_initial_state
from .physics import (
    CompositeOceanPhysics,
    OceanBulkMixedLayerPhysics,
    OceanEkmanSurfacePhysics,
    OceanSurfaceFluxPhysics,
    OceanSurfaceForcing,
)
from .state import OceanState, OceanStateRequirement


__all__ = [
    "HydrostaticBoussinesqOceanDynamics",
    "MultilayerOceanDynamics",
    "OCEAN_ALPHA_T",
    "OCEAN_BETA_S",
    "OCEAN_CP",
    "OCEAN_EOS_MODES",
    "OCEAN_GRAVITY",
    "OCEAN_PRESSURE_GRADIENT_MODES",
    "OCEAN_RHO0",
    "OCEAN_ROTATION_RATE",
    "OCEAN_S0",
    "OCEAN_T0",
    "OceanColumn",
    "OceanBulkMixedLayerPhysics",
    "CompositeOceanPhysics",
    "OceanDomain",
    "OceanEkmanSurfacePhysics",
    "OceanState",
    "OceanStateRequirement",
    "OceanSurfaceFluxPhysics",
    "OceanSurfaceForcing",
    "jmd95_bulk_modulus",
    "jmd95_seawater_density",
    "linear_seawater_density",
    "ocean_planet_initial_state",
]
