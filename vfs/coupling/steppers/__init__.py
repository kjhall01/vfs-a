"""Coupled-domain stepping adapters.

Numerical ARK/Krylov machinery lives in ``vfs.integrators.imex``.  This
package contains only the adapter that applies it to ``CoupledState`` and the
atmosphere--ocean tendency contract.
"""

from .imex_ark2 import CoupledAtmosphereOceanARK2, CoupledIMEXStepInfo

__all__ = ["CoupledAtmosphereOceanARK2", "CoupledIMEXStepInfo"]
