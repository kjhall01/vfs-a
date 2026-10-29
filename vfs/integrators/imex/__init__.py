"""Domain-aware IMEX integrators and retained pressure-prototype recipes.

``ark2`` is the active Field-native ARK2/GMRES/Schur implementation.  The
older pressure-correction recipes are isolated in ``legacy_pressure`` so they
remain available for controlled comparisons without obscuring the supported
ARK2 path.
"""

from .ark2 import (
    ARK2_TABLEAU,
    DryPEIMEXARK2,
    DryPELinearOperator,
    IMEXConvergenceError,
    IMEXSolverConfig,
    IMEXStageInfo,
    IMEXStepInfo,
)
from .legacy_pressure import (
    BlockBiCGSTAB,
    DryCorePressureIMEXEuler,
    DryCorePressureProjectionSIConservative,
    DryCorePressureProjectionSIMidpointFlux,
    DryCorePressureProjectionSIPicard1,
    DryCorePressureProjectionSIThetaFlux,
    DryCorePressureScalarSIPressureOnlyCentered,
    DryCorePressureScalarSIPressureOnlyLaggedThermo,
    DryCorePressureScalarSIProjectionProbe,
    DryCorePressureScalarSIThermoMidpoint,
    DryCorePressureSIMidpoint,
    DryCorePressureSIThermoMidpoint,
    PressureCorrectionCG,
)

__all__ = [
    "ARK2_TABLEAU",
    "BlockBiCGSTAB",
    "DryCorePressureIMEXEuler",
    "DryCorePressureProjectionSIConservative",
    "DryCorePressureProjectionSIMidpointFlux",
    "DryCorePressureProjectionSIPicard1",
    "DryCorePressureProjectionSIThetaFlux",
    "DryCorePressureScalarSIPressureOnlyCentered",
    "DryCorePressureScalarSIPressureOnlyLaggedThermo",
    "DryCorePressureScalarSIProjectionProbe",
    "DryCorePressureScalarSIThermoMidpoint",
    "DryCorePressureSIMidpoint",
    "DryCorePressureSIThermoMidpoint",
    "DryPEIMEXARK2",
    "DryPELinearOperator",
    "IMEXConvergenceError",
    "IMEXSolverConfig",
    "IMEXStageInfo",
    "IMEXStepInfo",
    "PressureCorrectionCG",
]

from .higher_order import ARK3_TABLEAU, ARK4_TABLEAU, DryPEIMEXARK
__all__ += ["ARK3_TABLEAU", "ARK4_TABLEAU", "DryPEIMEXARK"]

from .moist_ark import MoistPEIMEXARK
__all__ += ["MoistPEIMEXARK"]
