"""Compatibility imports for the land model.

The land model now lives in :mod:`vfs.land`. This module keeps older
``vfs.surface.land_surface`` imports working while surface-system code moves
to the dedicated land package.
"""

from ..land import LandSurface, LandSurfaceState, LandSurfaceStepResult

__all__ = ["LandSurface", "LandSurfaceState", "LandSurfaceStepResult"]
