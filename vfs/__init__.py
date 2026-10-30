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

from .atmosphere import DryHydrostaticDynamics
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

# Retained atmospheric compatibility imports resolve only on request.
_COMPAT_EXPORTS = {
    "ComponentConfig": (".core.config", "ComponentConfig"),
    "DataFieldRequirement": (".core.config", "DataFieldRequirement"),
    "DataRequirement": (".core.config", "DataRequirement"),
    "LatLonToMeshConfig": (".core.config", "LatLonToMeshConfig"),
    "MeshToLatLonConfig": (".core.config", "MeshToLatLonConfig"),
    "ProcessStepConfig": (".core.config", "ProcessStepConfig"),
    "RKStepConfig": (".core.config", "RKStepConfig"),
    "TensorFieldRequirement": (".core.config", "TensorFieldRequirement"),
    "VFSModelConfig": (".core.config", "VFSModelConfig"),
    "split_process_plan": (".core.config", "split_process_plan"),
    "compile_tendency_provider": (".core.integrators", "compile_tendency_provider"),
    "custom_schedule": (".core.io", "custom_schedule"),
    "daily_schedule": (".core.io", "daily_schedule"),
    "schedule_steps": (".core.io", "schedule_steps"),
    "snapshot_schedule": (".core.io", "snapshot_schedule"),
    "RecipeMetadata": (".core.registry", "RecipeMetadata"),
    "get_component": (".core.registry", "get_component"),
    "get_integrator": (".core.registry", "get_integrator"),
    "get_model": (".core.registry", "get_model"),
    "get_model_metadata": (".core.registry", "get_model_metadata"),
    "get_regrid_operator": (".core.registry", "get_regrid_operator"),
    "registered_model_metadata": (".core.registry", "registered_model_metadata"),
    "registered_models": (".core.registry", "registered_models"),
    "registered_regrid_operators": (".core.registry", "registered_regrid_operators"),
    "PrognosticFields": (".core.state", "PrognosticFields"),
    "StateRequirement": (".core.state", "StateRequirement"),
    "StateSpec": (".core.state", "StateSpec"),
    "TracerRequirement": (".core.state", "TracerRequirement"),
    "TracerSpec": (".core.state", "TracerSpec"),
    "DryHydrostaticDiagnostics": (".diagnostics", "DryHydrostaticDiagnostics"),
    "Field": (".field", "Field"),
    "load_mpas_mesh": (".geometry.mesh", "load_mpas_mesh"),
    "LatLonGrid": (".geometry.regrid", "LatLonGrid"),
    "LatLonToMeshBilinear": (".geometry.regrid", "LatLonToMeshBilinear"),
    "MeshToLatLonKNN": (".geometry.regrid", "MeshToLatLonKNN"),
    "DryHydrostaticModel": (".model", "DryHydrostaticModel"),
    "HeldSuarezModel": (".model", "HeldSuarezModel"),
    "VFSModel": (".model", "VFSModel"),
    "HeldSuarezConstants": (".physics", "HeldSuarezConstants"),
    "HeldSuarezPhysics": (".physics", "HeldSuarezPhysics"),
    "held_suarez_equilibrium_temperature": (
        ".physics",
        "held_suarez_equilibrium_temperature",
    ),
    "PhysicsState": (".process", "PhysicsState"),
    "PhysicsStepResult": (".process", "PhysicsStepResult"),
    "ProcessPhysicsDriver": (".process", "ProcessPhysicsDriver"),
}

from .atmosphere.steppers import BoundedSSPRK3, CenteredSSPRK3, CenteredIMEX

__all__ = [
    "BoundedSSPRK3", "CenteredSSPRK3", "CenteredIMEX",
    "MoistAtmosphere",
    "DryHydrostaticDynamics",
    "State",
    "CharneyPhillipsColumn",
    "make_hybrid_coefficients",
    "VoronoiMesh",
    "ssprk3",
    "forward_euler",
    "rk4",
    "BoundedGibbs",
    "MoistState",
    "MoistConservativeDynamics",
    "DryHydrostaticConservativeDynamics",
    "__version__",
]


def __getattr__(name):
    if name not in _COMPAT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module, attribute = _COMPAT_EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_COMPAT_EXPORTS))
