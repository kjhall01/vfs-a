"""Public atmospheric models, tensor states, geometry and explicit steppers.

Optional ETOPO/ERA5 ingestion and ESMF/xESMF regridding remain behind their
owner-module imports. Model configuration uses constructed component instances;
there is no recipe registry or alternate legacy model package.
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

from vfs.atmosphere.state import State
from vfs.atmosphere.column import CharneyPhillipsColumn
from vfs.atmosphere.column import make_hybrid_coefficients
from .geometry.mesh import VoronoiMesh
from vfs.integrators.explicit import ssprk3
from vfs.integrators.explicit import forward_euler
from vfs.integrators.explicit import rk4
from .atmosphere import BoundedGibbs
from .atmosphere import MoistState
from .atmosphere import MoistConservativeDynamics
from .atmosphere import DryHydrostaticConservativeDynamics

from .atmosphere.model import MoistAtmosphere

from .field import Field
from vfs.atmosphere.state import PrognosticFields, StateSpec, TracerSpec
from vfs.atmosphere.dry_model import DryAtmosphere
from .atmosphere.steppers import BoundedSSPRK3, CenteredIMEX, DrySSPRK3, DryIMEX

__all__ = [
    "State", "MoistState", "PrognosticFields", "StateSpec", "TracerSpec", "Field",
    "DryAtmosphere", "MoistAtmosphere", "DryHydrostaticConservativeDynamics",
    "MoistConservativeDynamics", "BoundedGibbs", "BoundedSSPRK3", "CenteredIMEX",
    "DrySSPRK3", "DryIMEX", "VoronoiMesh", "CharneyPhillipsColumn",
    "make_hybrid_coefficients", "ssprk3", "forward_euler", "rk4", "__version__",
]
