"""Compatibility imports; implementation owned by vfs.atmosphere.boundary."""
from vfs.atmosphere.boundary import (
    SURFACE_GEOPOTENTIAL_ALIASES,
    SURFACE_TEMPERATURE_ALIASES,
    synthetic_earth_boundary,
    earth_boundary_requirement,
    validate_boundary_data,
    regrid_boundary_data,
    write_boundary_netcdf,
    load_native_boundary_field,
    _resolve_key,
    _payload,
    _attrs,
    _as_tensor,
)

__all__ = ['SURFACE_GEOPOTENTIAL_ALIASES', 'SURFACE_TEMPERATURE_ALIASES', 'earth_boundary_requirement', 'load_native_boundary_field', 'regrid_boundary_data', 'synthetic_earth_boundary', 'validate_boundary_data', 'write_boundary_netcdf']
