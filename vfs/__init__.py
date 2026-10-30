"""Stable, lightweight root API for VFS.

The root namespace is intentionally small. It is for beginner-friendly dry-core
usage plus a few documented provisional integration points. Optional or
experimental systems such as ETOPO/ERA5 ingest, SCM adapters, IMEX, split-explicit steppers, and ESMF/xESMF regridders stay
behind explicit owner-module imports. That keeps ``import vfs`` reliable on
machines without the heavier geoscience workflow stack installed.
"""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version as _package_version
except ImportError:  # pragma: no cover - Python <3.8 fallback kept for safety.
    PackageNotFoundError = Exception  # type: ignore[assignment]
    _package_version = None  # type: ignore[assignment]

if _package_version is None:
    __version__ = "1.0.0+extraction4"
else:
    try:
        __version__ = _package_version("vfs")
    except PackageNotFoundError:
        # Source-tree imports before editable installation still need a version.
        __version__ = "1.0.0+extraction4"

from .core.state import State
from .geometry.column import CharneyPhillipsColumn
from .geometry.column import make_hybrid_coefficients
from .geometry.mesh import VoronoiMesh
from .core.integrators import ssprk3
from .core.integrators import forward_euler
from .core.integrators import rk4
from .atmosphere import BoundedGibbs
from .atmosphere import MoistState
from .atmosphere import MoistConservativeDynamics
from .atmosphere import DryHydrostaticConservativeDynamics

from .atmosphere.model import MoistAtmosphere

from .field import Field
from .core.state import PrognosticFields, StateSpec, TracerSpec
from .model import DryAtmosphere
from .atmosphere.steppers import BoundedSSPRK3, CenteredIMEX, DrySSPRK3, DryIMEX

__all__ = [
    "State", "MoistState", "PrognosticFields", "StateSpec", "TracerSpec", "Field",
    "DryAtmosphere", "MoistAtmosphere", "DryHydrostaticConservativeDynamics",
    "MoistConservativeDynamics", "BoundedGibbs", "BoundedSSPRK3", "CenteredIMEX",
    "DrySSPRK3", "DryIMEX", "VoronoiMesh", "CharneyPhillipsColumn",
    "make_hybrid_coefficients", "ssprk3", "forward_euler", "rk4", "__version__",
]
