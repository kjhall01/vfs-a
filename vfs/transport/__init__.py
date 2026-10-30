"""Transport benchmark helpers.

This owner module contains mesh-native, benchmark-facing transport utilities.
It is intentionally not part of the root ``vfs`` import surface.
"""

from .synthetic_tracers import (
    gaussian_blob,
    make_tracer_field,
    multi_blob,
    red_noise_field,
)
from .shadow_diagnostics import DEFAULT_SHADOW_DIAGNOSTIC_NAMES, compute_shadow_diagnostics
from .shadow_tracers import (
    ShadowTracerConfig,
    initialize_shadow_mass,
    shadow_concentration,
    shadow_mass_from_diagnostics,
    shadow_memory_diagnostics,
    shadow_tendency,
)

__all__ = [
    "gaussian_blob", "make_tracer_field", "multi_blob", "red_noise_field",
    "DEFAULT_SHADOW_DIAGNOSTIC_NAMES", "compute_shadow_diagnostics",
    "ShadowTracerConfig", "initialize_shadow_mass", "shadow_concentration",
    "shadow_mass_from_diagnostics", "shadow_memory_diagnostics", "shadow_tendency",
]
