"""Reusable atmospheric operator implementations.

Named dry recipes select compatible operators from this package; the canonical
dry RHS itself remains in ``vfs.atmosphere.dry_dynamics``.
"""

from .fluxes import HorizontalScalarFluxPackage

__all__ = ["HorizontalScalarFluxPackage"]
