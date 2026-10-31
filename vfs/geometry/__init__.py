"""Core exports with optional compatibility names loaded only on request."""

from .mesh import VoronoiMesh
from vfs.atmosphere.column import CharneyPhillipsColumn
from vfs.atmosphere.column import make_hybrid_coefficients

# Historical imports remain available explicitly, but are not eagerly loaded.
_COMPAT_EXPORTS = {
    "CellTransfer": (".mesh", "CellTransfer"),
    "VoronoiMeshHierarchy": (".mesh", "VoronoiMeshHierarchy"),
    "load_mpas_mesh": (".mesh", "load_mpas_mesh"),
    "ESMFGridToMesh": (".regrid", "ESMFGridToMesh"),
    "ESMFMeshToGrid": (".regrid", "ESMFMeshToGrid"),
    "LatLonGrid": (".regrid", "LatLonGrid"),
    "LatLonToMeshBilinear": (".regrid", "LatLonToMeshBilinear"),
    "MeshToLatLonAdjoint": (".regrid", "MeshToLatLonAdjoint"),
    "MeshToLatLonKNN": (".regrid", "MeshToLatLonKNN"),
}

__all__ = ["VoronoiMesh", "CharneyPhillipsColumn", "make_hybrid_coefficients"]


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
