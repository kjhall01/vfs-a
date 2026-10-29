"""Surface boundary, exchange, and surface-system coupling components.

Land owns modules that depend on lightweight surface packets.  Keep the
surface-system assembly and land re-exports lazy so importing ``vfs.land``
does not recurse through this package initializer.
"""

from .conservation import (
    ConservativeTransfer,
    TransferLedger,
    ledger_from,
    relative_residual,
    storage_budget_summary,
)
from .exchange import SurfaceBoundary, SurfaceExchange
from .fractions import SurfaceFractions
from .glacier_land_ice import GlacierLandIce, GlacierState, GlacierStepResult
from .icebergs import IcebergState, IcebergStepResult, IcebergTracer
from .packets import AtmosSurfaceFluxPacket, OceanBoundaryFluxPacket, SurfaceBoundaryPacket
from .runoff_router import RunoffRouteResult, RunoffRouter, RunoffState

__all__ = [
    "AtmosSurfaceFluxPacket",
    "ConservativeTransfer",
    "GlacierLandIce",
    "GlacierState",
    "GlacierStepResult",
    "IcebergState",
    "IcebergStepResult",
    "IcebergTracer",
    "LandSurface",
    "LandSurfaceState",
    "LandSurfaceStepResult",
    "LandUseFields",
    "LandUseParameterTable",
    "OceanBoundaryFluxPacket",
    "RunoffRouteResult",
    "RunoffRouter",
    "RunoffState",
    "SeaIceBoundaryProperties",
    "SeaIceEVP",
    "SeaIceEVPStepResult",
    "SeaIceParameters",
    "SeaIceState",
    "SeaIceStepResult",
    "SeaIceThermo",
    "SeaIceTransport",
    "SeaIceTransportStepResult",
    "SurfaceBoundary",
    "SurfaceBoundaryPacket",
    "SurfaceExchange",
    "SurfaceFractions",
    "SurfaceState",
    "SurfaceStepResult",
    "SurfaceSystem",
    "TransferLedger",
    "ledger_from",
    "load_surface_system_config",
    "relative_residual",
    "storage_budget_summary",
    "surface_system_from_config",
]


def __getattr__(name: str):
    """Load land-dependent surface assembly symbols only when requested."""

    if name in {"load_surface_system_config", "surface_system_from_config"}:
        from .config import load_surface_system_config, surface_system_from_config

        return {"load_surface_system_config": load_surface_system_config, "surface_system_from_config": surface_system_from_config}[name]
    if name in {"LandSurface", "LandSurfaceState", "LandSurfaceStepResult", "LandUseFields", "LandUseParameterTable"}:
        from ..land import LandSurface, LandSurfaceState, LandSurfaceStepResult, LandUseFields, LandUseParameterTable

        return {
            "LandSurface": LandSurface,
            "LandSurfaceState": LandSurfaceState,
            "LandSurfaceStepResult": LandSurfaceStepResult,
            "LandUseFields": LandUseFields,
            "LandUseParameterTable": LandUseParameterTable,
        }[name]
    if name in {
        "SeaIceBoundaryProperties",
        "SeaIceEVP",
        "SeaIceEVPStepResult",
        "SeaIceParameters",
        "SeaIceState",
        "SeaIceStepResult",
        "SeaIceThermo",
        "SeaIceTransport",
        "SeaIceTransportStepResult",
    }:
        from .sea_ice import (
            SeaIceBoundaryProperties,
            SeaIceEVP,
            SeaIceEVPStepResult,
            SeaIceParameters,
            SeaIceState,
            SeaIceStepResult,
            SeaIceThermo,
            SeaIceTransport,
            SeaIceTransportStepResult,
        )

        return {
            "SeaIceBoundaryProperties": SeaIceBoundaryProperties,
            "SeaIceEVP": SeaIceEVP,
            "SeaIceEVPStepResult": SeaIceEVPStepResult,
            "SeaIceParameters": SeaIceParameters,
            "SeaIceState": SeaIceState,
            "SeaIceStepResult": SeaIceStepResult,
            "SeaIceThermo": SeaIceThermo,
            "SeaIceTransport": SeaIceTransport,
            "SeaIceTransportStepResult": SeaIceTransportStepResult,
        }[name]
    if name == "SurfaceState":
        from .state import SurfaceState

        return SurfaceState
    if name in {"SurfaceStepResult", "SurfaceSystem"}:
        from .system import SurfaceStepResult, SurfaceSystem

        return {"SurfaceStepResult": SurfaceStepResult, "SurfaceSystem": SurfaceSystem}[name]
    raise AttributeError(f"module 'vfs.surface' has no attribute {name!r}")
