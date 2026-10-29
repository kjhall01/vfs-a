"""Compatibility exports for sea-ice components.

Sea-ice implementation lives in :mod:`vfs.ice`; this module keeps the existing
``vfs.surface.sea_ice`` import path stable.
"""

from ..ice import (
    SeaIceBoundaryProperties,
    SeaIceEVP,
    SeaIceEVPStepResult,
    SeaIceParameters,
    SeaIceState,
    SeaIceStepResult,
    SeaIceThermo,
    SeaIceTransport,
    SeaIceTransportStepResult,
    sea_ice_boundary_properties,
    sea_ice_snow_cover_fraction,
    sea_ice_snow_insulation_factor,
)

__all__ = [
    "SeaIceBoundaryProperties",
    "SeaIceEVP",
    "SeaIceEVPStepResult",
    "SeaIceParameters",
    "SeaIceState",
    "SeaIceStepResult",
    "SeaIceThermo",
    "SeaIceTransport",
    "SeaIceTransportStepResult",
    "sea_ice_boundary_properties",
    "sea_ice_snow_cover_fraction",
    "sea_ice_snow_insulation_factor",
]
