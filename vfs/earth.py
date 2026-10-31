"""Compatibility imports; implementation owned by vfs.atmosphere.earth."""
from vfs.atmosphere.earth import (
    prescribed_earth_temperature_interfaces,
    earth_like_dry_initial_state,
    _as_tensor,
)

__all__ = ['earth_like_dry_initial_state', 'prescribed_earth_temperature_interfaces']
