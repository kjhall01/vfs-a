"""Core exports with optional compatibility names loaded only on request."""

from .constants import GRAVITY
from .constants import OMEGA
from .dry_dynamics import DryHydrostaticDynamics
from .conservative import DryHydrostaticConservativeDynamics
from .moist_conservative import MoistConservativeDynamics
from .moist_conservative import MoistState
from .bounded_moist import BoundedGibbs

from .model import MoistAtmosphere

# Historical imports remain available explicitly, but are not eagerly loaded.
_COMPAT_EXPORTS = {
    "AdaptiveDivergenceDamping": (".dry_strategies", "AdaptiveDivergenceDamping"),
    "AdaptiveVelocityHyperdiffusion": (
        ".dry_strategies",
        "AdaptiveVelocityHyperdiffusion",
    ),
    "AdaptiveVerticalModeDamping": (".dry_strategies", "AdaptiveVerticalModeDamping"),
    "BlendedMildUpwindVerticalMomentumTransport": (
        ".dry_strategies",
        "BlendedMildUpwindVerticalMomentumTransport",
    ),
    "BiharmonicDivergenceDamping": (".dry_strategies", "BiharmonicDivergenceDamping"),
    "CompositeMomentumDiffusion": (".dry_strategies", "CompositeMomentumDiffusion"),
    "DryThermodynamics": (".dry_strategies", "DryThermodynamics"),
    "EdgeUpwindVerticalMomentumTransport": (
        ".dry_strategies",
        "EdgeUpwindVerticalMomentumTransport",
    ),
    "ExnerSecantSigmaPressureGradient": (
        ".dry_strategies",
        "ExnerSecantSigmaPressureGradient",
    ),
    "FCTTracerTransport": (".dry_strategies", "FCTTracerTransport"),
    "LogPressureSigmaPressureGradient": (
        ".dry_strategies",
        "LogPressureSigmaPressureGradient",
    ),
    "MildCellSupportedVerticalMomentumTransport": (
        ".dry_strategies",
        "MildCellSupportedVerticalMomentumTransport",
    ),
    "ScaleNormalizedCoordinateWork": (
        ".dry_strategies",
        "ScaleNormalizedCoordinateWork",
    ),
    "TerrainAwarePressureGradient": (".dry_strategies", "TerrainAwarePressureGradient"),
    "TopRayleighSponge": (".dry_strategies", "TopRayleighSponge"),
    "UnscaledReferenceCoordinateWork": (
        ".dry_strategies",
        "UnscaledReferenceCoordinateWork",
    ),
    "UpwindTracerTransport": (".dry_strategies", "UpwindTracerTransport"),
    "VelocityHyperdiffusion": (".dry_strategies", "VelocityHyperdiffusion"),
    "VirtualTemperatureThermodynamics": (
        ".dry_strategies",
        "VirtualTemperatureThermodynamics",
    ),
    "DryOperatorSet": (".recipes", "DryOperatorSet"),
    "DryHydrostaticCenteredMthetaTransportDynamics": (
        ".recipes",
        "DryHydrostaticCenteredMthetaTransportDynamics",
    ),
    "DryHydrostaticDivergenceDampingDynamics": (
        ".recipes",
        "DryHydrostaticDivergenceDampingDynamics",
    ),
    "DryHydrostaticExnerSecantSigmaDynamics": (
        ".recipes",
        "DryHydrostaticExnerSecantSigmaDynamics",
    ),
    "DryHydrostaticFCTMthetaTransportDynamics": (
        ".recipes",
        "DryHydrostaticFCTMthetaTransportDynamics",
    ),
    "DryHydrostaticFluxPackageMUSCLDynamics": (
        ".recipes",
        "DryHydrostaticFluxPackageMUSCLDynamics",
    ),
    "DryHydrostaticKiteCenteredMthetaTransportDynamics": (
        ".recipes",
        "DryHydrostaticKiteCenteredMthetaTransportDynamics",
    ),
    "DryHydrostaticLogPressureSigmaDynamics": (
        ".recipes",
        "DryHydrostaticLogPressureSigmaDynamics",
    ),
    "DryHydrostaticMassKiteCenteredMthetaTransportDynamics": (
        ".recipes",
        "DryHydrostaticMassKiteCenteredMthetaTransportDynamics",
    ),
    "DryHydrostaticMildVerticalMomentumDynamics": (
        ".recipes",
        "DryHydrostaticMildVerticalMomentumDynamics",
    ),
    "DryHydrostaticMUSCLMthetaTransportDynamics": (
        ".recipes",
        "DryHydrostaticMUSCLMthetaTransportDynamics",
    ),
    "DryHydrostaticSoftMUSCLMthetaTransportDynamics": (
        ".recipes",
        "DryHydrostaticSoftMUSCLMthetaTransportDynamics",
    ),
    "DryHydrostaticTopSpongeDynamics": (".recipes", "DryHydrostaticTopSpongeDynamics"),
    "DryHydrostaticUnscaledCoordWorkDynamics": (
        ".recipes",
        "DryHydrostaticUnscaledCoordWorkDynamics",
    ),
    "DryHydrostaticVelocityHyperdiffusionDynamics": (
        ".recipes",
        "DryHydrostaticVelocityHyperdiffusionDynamics",
    ),
    "DryHydrostaticVirtualTemperatureDynamics": (
        ".recipes",
        "DryHydrostaticVirtualTemperatureDynamics",
    ),
    "build_dry_dynamics": (".recipes", "build_dry_dynamics"),
    "dry_operator_sets": (".recipes", "dry_operator_sets"),
    "get_dry_operator_set": (".recipes", "get_dry_operator_set"),
    "HorizontalScalarFluxPackage": (".operators", "HorizontalScalarFluxPackage"),
}

__all__ = [
    "MoistAtmosphere",
    "GRAVITY",
    "OMEGA",
    "DryHydrostaticDynamics",
    "DryHydrostaticConservativeDynamics",
    "MoistConservativeDynamics",
    "MoistState",
    "BoundedGibbs",
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
