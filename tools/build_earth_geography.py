#!/usr/bin/env python3
"""Build Earth shelf geography in the order of a VFS mesh's saved generators."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlretrieve

import numpy as np
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.geometry import shape


RELIEF = (
    "https://oceania.generic-mapping-tools.org/server/earth/earth_relief/earth_relief_10m_p.grd",
    "bf61348f617ae6c3d6df4f71192362f6eae6861eccabaa0c0334bf3686de8ec9",
)
COASTLINE = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson",
    "e874b27a51d146452be360cafb3cc50c86001074a67d534113e6534682f9826b",
)


def source_file(local, cache, source):
    url, digest = source
    path = local if local is not None else cache / url.rsplit("/", 1)[-1]

    def verify(candidate):
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Unexpected SHA-256 for {path}. Expected {digest}.")

    if local is not None or path.exists():
        verify(path)
    else:
        cache.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=cache, delete=False) as tmp:
            partial = Path(tmp.name)
        try:
            urlretrieve(url, partial)
            verify(partial)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    return path


def land_pixels(lon, lat, coastline):
    """Rasterize the coastline at the relief pixel centres, retaining polygon holes."""
    land = np.zeros((lat.size, lon.size), dtype=bool)
    for feature in json.loads(coastline.read_text())["features"]:
        polygon = shape(feature["geometry"])
        west, south, east, north = polygon.bounds
        x = slice(lon.searchsorted(west), lon.searchsorted(east, side="right"))
        y = slice(lat.searchsorted(south), lat.searchsorted(north, side="right"))
        land[y, x] |= contains_xy(polygon, lon[None, x], lat[y, None])
    return land


def build(mesh, relief, coastline, min_depth):
    if not np.isfinite(min_depth) or min_depth <= 0:
        raise ValueError("Minimum wet depth must be positive and finite.")
    raw = mesh.read_bytes()
    xyz = np.load(io.BytesIO(raw), allow_pickle=False)
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or len(xyz) < 16
        or xyz.dtype.kind != "f"
        or not np.isfinite(xyz).all()
        or not np.allclose(np.linalg.norm(xyz, axis=1), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError("Expected a mesh .npy file with N >= 16 unit XYZ generators.")
    with Dataset(relief) as grid:
        lon, lat = grid["lon"][:].data, grid["lat"][:].data
        z = grid["z"][:]
        if np.ma.getmaskarray(z).any():
            raise ValueError("Missing relief pixels in the global source grid.")
        z = np.asarray(z, dtype=np.float64)
        title = grid.title

    polygon_land = land_pixels(lon, lat, coastline)
    wet = ~polygon_land & (z < 0)
    rad_lon, rad_lat = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    cos_lat = np.cos(rad_lat)
    points = np.column_stack(
        (
            (cos_lat * np.cos(rad_lon)).ravel(),
            (cos_lat * np.sin(rad_lon)).ravel(),
            np.sin(rad_lat).ravel(),
        )
    )
    cell = cKDTree(xyz).query(points)[1]

    def area_sum(field):
        return np.bincount(cell, weights=(cos_lat * field).ravel(), minlength=len(xyz))

    area = area_sum(1.0)
    if (area == 0).any():
        raise ValueError("A mesh cell contains no 10-arc-minute relief pixel centres.")
    land_area, wet_area = area_sum(~wet), area_sum(wet)
    land_fraction = land_area / area
    land = land_fraction >= 0.5
    height = np.where(
        land, area_sum(np.maximum(z, 0.0) * ~wet) / np.where(land, land_area, 1.0), 0.0
    )
    raw_depth = np.where(
        land, 0.0, area_sum(np.maximum(-z, 0.0) * wet) / np.where(land, 1.0, wet_area)
    )
    metadata = dict(
        mesh_sha256=hashlib.sha256(raw).hexdigest(),
        relief=title,
        coastline="Natural Earth 1:50m land polygons",
        method="Area-weighted pixel-centre Voronoi binning; majority land; retain disconnected wet basins; land and ocean conditional means with relief clipped at sea level within each class.",
        source_sha256=[RELIEF[1], COASTLINE[1]],
        min_wet_depth_m=min_depth,
        wet_pixel_rule="Outside Natural Earth land polygons and GMT elevation below sea level",
        source_urls=[RELIEF[0], COASTLINE[0]],
    )
    return dict(
        land=land,
        height_m=height,
        depth_m=np.where(land, 0.0, np.maximum(raw_depth, min_depth)),
        raw_depth_m=raw_depth,
        land_fraction=land_fraction,
        metadata=json.dumps(metadata),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh",
        type=Path,
        required=True,
        help="Saved VFS unit-sphere generators (.npy)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination geography archive (.npz)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Raw download directory (default: earth_sources beside output)",
    )
    parser.add_argument(
        "--relief",
        type=Path,
        help="Use an existing GMT 10 arc-minute pixel-registered grid",
    )
    parser.add_argument(
        "--coastline",
        type=Path,
        help="Use an existing Natural Earth 1:50m land GeoJSON",
    )
    parser.add_argument(
        "--min-wet-depth-m",
        type=float,
        default=600,
        help="Bathymetry floor in wet cells (default: 600 m)",
    )
    args = parser.parse_args()
    cache = args.cache_dir or args.output.parent / "earth_sources"
    result = build(
        args.mesh,
        source_file(args.relief, cache, RELIEF),
        source_file(args.coastline, cache, COASTLINE),
        args.min_wet_depth_m,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.output.parent, delete=False) as tmp:
        partial = Path(tmp.name)
    try:
        with partial.open("wb") as f:
            np.savez_compressed(f, **result)
        partial.replace(args.output)
    finally:
        partial.unlink(missing_ok=True)
    land = result["land"]
    print(f"Wrote {args.output} ({land.sum()} land, {(~land).sum()} wet cells)")


if __name__ == "__main__":
    main()
