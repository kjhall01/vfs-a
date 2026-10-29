"""Cross-domain coupling helpers for VFS atmosphere/ocean experiments.

The lightweight coupling contracts are imported eagerly. SCM/process adapters
are resolved lazily so ``import vfs`` does not pull in the surface-system
package just because the root API exposes the basic coupled-state contracts.
"""

from .model import CoupledAtmosphereOceanModel
from .mass_correction import DryAirMassCorrectionProcess
from .momentum import AirSeaMomentumCoupler
from .saturation_adjustment import SaturationAdjustmentProcess
from .state import CoupledState, CoupledStepResult
from .surface_flux import AirSeaSurfaceFluxCoupler

__all__ = [
    "AirSeaMomentumCoupler",
    "AirSeaSurfaceFluxCoupler",
    "CoupledARKStageSource",
    "CoupledARKStageSourceResult",
    "CoupledAtmosphereOceanARK2",
    "CoupledAtmosphereStepperProcess",
    "CoupledIMEXStepInfo",
    "CoupledAtmosphereOceanModel",
    "CoupledSCMProcess",
    "CoupledSurfaceSystemStageSource",
    "CoupledSurfaceStrangARK2",
    "CoupledSurfaceStageMapARK2",
    "CoupledState",
    "CoupledStepResult",
    "DryAirMassCorrectionProcess",
    "SaturationAdjustmentProcess",
]


def __getattr__(name: str):
    """Load optional coupling adapters only when users ask for them."""

    if name == "CoupledAtmosphereStepperProcess":
        from .atmosphere_stepper import CoupledAtmosphereStepperProcess

        return CoupledAtmosphereStepperProcess
    if name == "CoupledSCMProcess":
        from .scm_process import CoupledSCMProcess

        return CoupledSCMProcess
    if name == "CoupledSurfaceSystemStageSource":
        from .surface.finite_map import CoupledSurfaceSystemStageSource

        return CoupledSurfaceSystemStageSource
    if name == "CoupledSurfaceStrangARK2":
        from .surface.strang import CoupledSurfaceStrangARK2

        return CoupledSurfaceStrangARK2
    if name == "CoupledSurfaceStageMapARK2":
        from .surface.stage_map import CoupledSurfaceStageMapARK2

        return CoupledSurfaceStageMapARK2
    if name in {"CoupledAtmosphereOceanARK2", "CoupledIMEXStepInfo"}:
        from .steppers.imex_ark2 import CoupledAtmosphereOceanARK2, CoupledIMEXStepInfo

        return {
            "CoupledAtmosphereOceanARK2": CoupledAtmosphereOceanARK2,
            "CoupledIMEXStepInfo": CoupledIMEXStepInfo,
        }[name]
    if name in {"CoupledARKStageSource", "CoupledARKStageSourceResult"}:
        from .ark_stage_sources import CoupledARKStageSource, CoupledARKStageSourceResult

        return {
            "CoupledARKStageSource": CoupledARKStageSource,
            "CoupledARKStageSourceResult": CoupledARKStageSourceResult,
        }[name]
    raise AttributeError(f"module 'vfs.coupling' has no attribute {name!r}")
