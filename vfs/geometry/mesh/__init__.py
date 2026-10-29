"""Core exports with optional compatibility names loaded only on request."""

from .grid import VoronoiMesh
from .grid import EARTH_RADIUS_M
from .grid import MESH_OPERATOR_VERSION
from .grid import EXACT_THUBURN_VERSION

# Historical imports remain available explicitly, but are not eagerly loaded.
_COMPAT_EXPORTS = {
    "THUBURN_CORNER_CONVENTION": (".grid", "THUBURN_CORNER_CONVENTION"),
    "THUBURN_IMPLEMENTATION": (".grid", "THUBURN_IMPLEMENTATION"),
    "_FLOAT_BUFFERS": (".grid", "_FLOAT_BUFFERS"),
    "_INT_BUFFERS": (".grid", "_INT_BUFFERS"),
    "_load_mesh": (".grid", "_load_mesh"),
    "_project_padded_edge_operator_m_skew_numpy": (
        ".grid",
        "_project_padded_edge_operator_m_skew_numpy",
    ),
    "_rebuild_derived_buffers": (".grid", "_rebuild_derived_buffers"),
    "_save_mesh": (".grid", "_save_mesh"),
    "load_mpas_mesh": (".io", "load_mpas_mesh"),
    "load_vfs_imex_mesh": (".io", "load_vfs_imex_mesh"),
    "CellMeshKernelConv": (".kernels", "CellMeshKernelConv"),
    "CellMeshKernelNetwork": (".kernels", "CellMeshKernelNetwork"),
    "EdgeMeshKernelConv": (".kernels", "EdgeMeshKernelConv"),
    "EdgeMeshKernelNetwork": (".kernels", "EdgeMeshKernelNetwork"),
    "MeshKernelConfig": (".kernels", "MeshKernelConfig"),
    "CellTransfer": (".multilevel", "CellTransfer"),
    "VoronoiMeshHierarchy": (".multilevel", "VoronoiMeshHierarchy"),
}

__all__ = [
    "VoronoiMesh",
    "EARTH_RADIUS_M",
    "MESH_OPERATOR_VERSION",
    "EXACT_THUBURN_VERSION",
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
