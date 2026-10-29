"""Component-graph configuration subsystem.

``contracts`` defines the small runtime interfaces, ``registry`` describes
the YAML-visible module catalog, and ``model`` owns graph construction,
validation, schedules, and snapshots. The established ``vfs.core.graph``
import path remains the public entry point.
"""

from .contracts import GraphComponentAPI, GraphDiagnosticAPI, GraphInitializerAPI, GraphObserverAPI, GraphState
from .model import (
    AirSeaHeatExchangeForcing,
    ComponentGraphModel,
    DryAtmosphereHealthDiagnostics,
    DryHydrostaticAtmosphere,
    ExternalFields,
    HeldSuarezForcing,
    HeldSuarezInitializer,
    JablonskiWilliamsonInitializer,
    LandSurfaceComponent,
    LandSurfaceInitializer,
    MeshToLatLonObserver,
    NativeExportObserver,
    OceanRestInitializer,
    OceanStateComponent,
    SCMForcing,
    WeatherBench2ERA5DryInitializer,
)
from .registry import GraphComponentMetadata, graph_component_catalog, graph_component_metadata

__all__ = [
    "AirSeaHeatExchangeForcing",
    "ComponentGraphModel",
    "DryAtmosphereHealthDiagnostics",
    "DryHydrostaticAtmosphere",
    "ExternalFields",
    "GraphComponentAPI",
    "GraphComponentMetadata",
    "GraphDiagnosticAPI",
    "GraphInitializerAPI",
    "GraphObserverAPI",
    "GraphState",
    "HeldSuarezForcing",
    "HeldSuarezInitializer",
    "JablonskiWilliamsonInitializer",
    "LandSurfaceComponent",
    "LandSurfaceInitializer",
    "MeshToLatLonObserver",
    "NativeExportObserver",
    "OceanRestInitializer",
    "OceanStateComponent",
    "SCMForcing",
    "WeatherBench2ERA5DryInitializer",
    "graph_component_catalog",
    "graph_component_metadata",
]
