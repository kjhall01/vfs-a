"""Compatibility imports; implementation owned by vfs.geometry.remap."""
from vfs.geometry.remap import (
    _as_float_tensor,
    _xyz_from_lat_lon,
    _cell_lat_lon_rad,
    _local_east_north_from_xyz,
    _squeeze_last_singleton,
    _native_field_to_source_layout,
    _latlon_field_from_source_layout,
    _latlon_field_to_source_layout,
    _native_field_from_latlon_layout,
    _cell_field,
    LatLonGrid,
    MeshToLatLonAdjoint,
    LatLonToMeshBilinear,
    ESMFGridToMesh,
    ESMFMeshToGrid,
    _regular_edges_from_centers,
    _regular_bilinear_weights,
    _import_xesmf,
)
from .remap import MeshToLatLonKNN as _FieldRemapper


class MeshToLatLonKNN(_FieldRemapper):
    """Legacy convenience surface; new code uses explicit component adapters.

    Only this compatibility boundary infers a component. The shared field
    implementation in geometry.remap never inspects or imports a state class.
    """

    def state_to_latlon(self, state, *, fields="default"):
        if hasattr(state, "h") and hasattr(state, "temp_h") and not hasattr(state, "ps"):
            from vfs.ocean.regrid import state_to_latlon
        else:
            from vfs.atmosphere.regrid import state_to_latlon
        return state_to_latlon(self, state, fields=fields)

__all__ = ['ESMFGridToMesh', 'ESMFMeshToGrid', 'LatLonGrid', 'LatLonToMeshBilinear', 'MeshToLatLonAdjoint', 'MeshToLatLonKNN']
