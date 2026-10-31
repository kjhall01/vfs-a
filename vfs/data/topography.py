"""Compatibility imports; implementation owned by vfs.data.elevation."""
from vfs.data.elevation import (
    ETOPO_2022_60S_BEDROCK_OPENDAP_URL,
    ETOPO_2022_60S_BEDROCK_DOWNLOAD_URL,
    ETOPO_2022_60S_BEDROCK_NETCDF_URL,
    ETOPO_2022_METADATA_URL,
    ETOPO_ELEVATION_ALIASES,
    download_etopo2022,
    open_etopo2022,
    etopo_grid_from_xarray,
    etopo_elevation_from_xarray,
    etopo_native_elevation_from_xarray,
    _as_tensor,
    _etopo_lat_lon_names,
    _etopo_elevation_dataarray,
    _etopo_coarsen_cache_path,
    _etopo_xarray_source_id,
    _etopo_cache_token,
    _read_etopo_coarsen_cache,
    _write_etopo_coarsen_cache,
    _is_remote_url,
    _etopo_direct_download_url,
)
"""Compatibility imports; implementation owned by vfs.atmosphere.topography."""
from vfs.atmosphere.topography import (
    GRAVITY,
    EARTH_GRAVITY,
    R_D,
    DRY_AIR_GAS_CONSTANT,
    DEFAULT_SCALE_HEIGHT,
    build_atmospheric_topographic_column,
    build_topographic_column,
    atmospheric_surface_pressure_from_phi_s,
    earth_surface_pressure_from_phi_s,
    atmospheric_surface_height_from_elevation,
    atmospheric_surface_geopotential_from_elevation,
)

__all__ = ['ETOPO_2022_60S_BEDROCK_DOWNLOAD_URL', 'ETOPO_2022_60S_BEDROCK_NETCDF_URL', 'ETOPO_2022_60S_BEDROCK_OPENDAP_URL', 'ETOPO_2022_METADATA_URL', 'ETOPO_ELEVATION_ALIASES', 'DEFAULT_SCALE_HEIGHT', 'DRY_AIR_GAS_CONSTANT', 'EARTH_GRAVITY', 'GRAVITY', 'R_D', 'atmospheric_surface_pressure_from_phi_s', 'atmospheric_surface_geopotential_from_elevation', 'atmospheric_surface_height_from_elevation', 'build_atmospheric_topographic_column', 'build_topographic_column', 'download_etopo2022', 'earth_surface_pressure_from_phi_s', 'etopo_native_elevation_from_xarray', 'etopo_elevation_from_xarray', 'etopo_grid_from_xarray', 'open_etopo2022']
