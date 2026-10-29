"""Core exports with optional compatibility names loaded only on request."""

from .state import State
from .state import PrognosticFields
from .integrators import ssprk3
from .integrators import forward_euler
from .integrators import rk4

# Historical imports remain available explicitly, but are not eagerly loaded.
_COMPAT_EXPORTS = {
    "ComponentConfig": (".config", "ComponentConfig"),
    "DataFieldRequirement": (".config", "DataFieldRequirement"),
    "DataRequirement": (".config", "DataRequirement"),
    "LatLonToMeshConfig": (".config", "LatLonToMeshConfig"),
    "MeshToLatLonConfig": (".config", "MeshToLatLonConfig"),
    "ProcessStepConfig": (".config", "ProcessStepConfig"),
    "RKStepConfig": (".config", "RKStepConfig"),
    "TensorFieldRequirement": (".config", "TensorFieldRequirement"),
    "VFSModelConfig": (".config", "VFSModelConfig"),
    "split_process_plan": (".config", "split_process_plan"),
    "DiagnosticsProvider": (".diagnostic_contracts", "DiagnosticsProvider"),
    "TendencyProvider": (".diagnostic_contracts", "TendencyProvider"),
    "TendencyTermsProvider": (".diagnostic_contracts", "TendencyTermsProvider"),
    "TermDict": (".diagnostic_contracts", "TermDict"),
    "compile_tendency_provider": (".integrators", "compile_tendency_provider"),
    "custom_schedule": (".io", "custom_schedule"),
    "daily_schedule": (".io", "daily_schedule"),
    "schedule_steps": (".io", "schedule_steps"),
    "snapshot_schedule": (".io", "snapshot_schedule"),
    "get_component": (".registry", "get_component"),
    "get_integrator": (".registry", "get_integrator"),
    "get_model": (".registry", "get_model"),
    "get_regrid_operator": (".registry", "get_regrid_operator"),
    "register_component": (".registry", "register_component"),
    "register_integrator": (".registry", "register_integrator"),
    "register_model": (".registry", "register_model"),
    "register_regrid_operator": (".registry", "register_regrid_operator"),
    "registered_models": (".registry", "registered_models"),
    "registered_regrid_operators": (".registry", "registered_regrid_operators"),
    "StateRequirement": (".state", "StateRequirement"),
    "StateSpec": (".state", "StateSpec"),
    "TracerRequirement": (".state", "TracerRequirement"),
    "TracerSpec": (".state", "TracerSpec"),
}

__all__ = ["State", "PrognosticFields", "ssprk3", "forward_euler", "rk4"]


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
