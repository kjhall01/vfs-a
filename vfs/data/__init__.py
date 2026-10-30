"""External data and atmospheric terrain helpers."""

from .topography import (
    ETOPO_2022_60S_BEDROCK_DOWNLOAD_URL,
    ETOPO_2022_60S_BEDROCK_NETCDF_URL,
    ETOPO_2022_60S_BEDROCK_OPENDAP_URL,
    ETOPO_2022_METADATA_URL,
    atmospheric_surface_geopotential_from_elevation,
    atmospheric_surface_height_from_elevation,
    atmospheric_surface_pressure_from_phi_s,
    build_atmospheric_topographic_column,
    build_topographic_column,
    download_etopo2022,
    earth_surface_pressure_from_phi_s,
    etopo_elevation_from_xarray,
    etopo_grid_from_xarray,
    etopo_native_elevation_from_xarray,
    open_etopo2022,
)

__all__ = [
    "ETOPO_2022_60S_BEDROCK_DOWNLOAD_URL",
    "ETOPO_2022_60S_BEDROCK_NETCDF_URL",
    "ETOPO_2022_60S_BEDROCK_OPENDAP_URL",
    "ETOPO_2022_METADATA_URL",
    "atmospheric_surface_geopotential_from_elevation",
    "atmospheric_surface_height_from_elevation",
    "atmospheric_surface_pressure_from_phi_s",
    "build_atmospheric_topographic_column",
    "build_topographic_column",
    "download_etopo2022",
    "earth_surface_pressure_from_phi_s",
    "etopo_elevation_from_xarray",
    "etopo_grid_from_xarray",
    "etopo_native_elevation_from_xarray",
    "open_etopo2022",
]
