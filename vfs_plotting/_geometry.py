"""Display geometry for native VFS C-grid samples."""
from dataclasses import dataclass

import numpy as np
import xarray as xr


def as_array(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asanyarray(value)


def _indices(values, limit):
    # xarray decodes the -1 padding in the NetCDF connectivity as NaN.
    a = as_array(values)
    a = np.where(np.isnan(a), -1, a)
    ids = a.astype(np.intp)
    if not np.array_equal(ids, a) or np.any((ids < -1) | (ids >= limit)):
        raise ValueError("invalid mesh connectivity index")
    return ids


def _unit(xyz):
    return xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)


@dataclass
class Geometry:
    cells: np.ndarray
    vertices: np.ndarray
    face_nodes: np.ndarray
    edge_nodes: np.ndarray | None
    edge_cells: np.ndarray | None
    ordered_vertices: bool = True

    @classmethod
    def read(cls, mesh):
        if not isinstance(mesh, (xr.Dataset, np.ndarray)):
            return cls(as_array(mesh.cell_xyz), as_array(mesh.vertex_xyz),
                       as_array(mesh.verticesOnCell), as_array(mesh.verticesOnEdge),
                       as_array(mesh.cellsOnEdge))

        if isinstance(mesh, xr.Dataset):
            centers = mesh.cell_xyz
            if "time" in centers.dims:  # historical instantaneous snapshots
                centers = centers.isel(time=0)
            cells = centers.values
            if "vertices_on_cell" in mesh:
                lon = np.deg2rad(mesh.vertex_longitude.values)
                lat = np.deg2rad(mesh.vertex_latitude.values)
                vertices = np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))
                return cls(cells, vertices, _indices(mesh.vertices_on_cell, len(vertices)),
                           _indices(mesh.vertices_on_edge, len(vertices)), _indices(mesh.cells_on_edge, len(cells)))
            if "verticesOnCell" in mesh:  # VoronoiMesh.save() operator cache
                return cls(cells, mesh.vertex_xyz.values, _indices(mesh.verticesOnCell, mesh.sizes["vertex"]),
                           _indices(mesh.verticesOnEdge, mesh.sizes["vertex"]), _indices(mesh.cellsOnEdge, len(cells)))
        else:
            cells = mesh
        # Generator-only geography and older snapshots have no stored polygons.
        # Rebuild the spherical Voronoi regions in the original generator order.
        from scipy.spatial import SphericalVoronoi
        if cells.ndim != 2 or cells.shape[1] != 3:
            raise ValueError("expected Cartesian mesh centers with shape (cell, 3)")
        vor = SphericalVoronoi(_unit(cells.astype(float)))
        vor.sort_vertices_of_regions()
        nodes = np.full((len(cells), max(map(len, vor.regions))), -1)
        for i, ring in enumerate(vor.regions):
            nodes[i, :len(ring)] = ring
        edge_nodes = edge_cells = None
        if isinstance(mesh, xr.Dataset) and "cells_on_edge" in mesh:
            pair = mesh.cells_on_edge
            if "time" in pair.dims:
                pair = pair.isel(time=0)
            edge_cells = _indices(pair, len(cells))
            # Keep the edge order from the file. Each cell pair has two
            # shared Voronoi vertices, even when Qhull numbered them differently.
            regions = [set(ring) for ring in vor.regions]
            endpoints = [sorted(regions[a] & regions[b]) for a, b in edge_cells]
            if any(len(ends) != 2 for ends in endpoints):
                raise ValueError("saved edge adjacency disagrees with the Voronoi geometry")
            edge_nodes = np.asarray(endpoints)
        return cls(cells, vor.vertices, nodes, edge_nodes, edge_cells, ordered_vertices=False)

    def rings(self, location):
        if location == "cell":
            if (np.sum(self.face_nodes >= 0, axis=1) < 3).any() or (self.face_nodes[:, 0] < 0).any():
                raise ValueError("each Voronoi cell needs at least three ordered vertices")
            ids = np.where(self.face_nodes < 0, self.face_nodes[:, :1], self.face_nodes)
            return self.vertices[ids], np.arange(len(ids))
        if self.edge_nodes is None:
            raise ValueError(f"{location} fields need stored mesh connectivity to establish their native ordering")
        if location == "edge":
            ends = self.vertices[self.edge_nodes]
            # Each side of the native edge joins its two endpoints to its cell
            # center. Both triangles carry the same signed edge sample.
            n = len(ends)
            rings = np.empty((n, 2, 3, 3))
            rings[:, :, 0] = self.cells[self.edge_cells]
            rings[:, :, 1:] = ends[:, None]
            return rings.reshape(-1, 3, 3), np.repeat(np.arange(n), 2)
        if location == "vertex":
            cells, slots = np.nonzero(self.face_nodes >= 0)
            vertex_ids = self.face_nodes[cells, slots]
            if not np.all(np.bincount(vertex_ids, minlength=len(self.vertices)) == 3):
                raise ValueError("a native dual triangle needs three incident cells per vertex")
            triangles = cells[np.argsort(vertex_ids)].reshape(-1, 3)
            return self.cells[triangles], np.arange(len(triangles))
        raise ValueError("expected a scalar at cell, edge or vertex locations")


def robinson_polygons(rings, owners, target):
    """Project spherical outlines, retaining the owner of both pieces at the map cut."""
    import cartopy.crs as ccrs
    from shapely.geometry import Polygon

    # Geodesic edges become curves on a map. Sample the geometry along each
    # minor great-circle arc; the field value is constant over the whole ring.
    rings = _unit(rings)
    following = np.roll(rings, -1, axis=1)
    max_arc = np.rad2deg(np.arccos(np.clip((rings * following).sum(-1).min(), -1., 1.)))
    steps = max(1, int(np.ceil(max_arc / 2.)))
    t = (np.arange(steps) / steps)[:, None]
    points = (rings[:, :, None] * (1 - t) + following[:, :, None] * t).reshape(len(rings), -1, 3)
    x, y, z = points.transpose(2, 0, 1)
    central_longitude = target.proj4_params.get("lon_0", 0.)
    lon = (np.rad2deg(np.arctan2(y, x)) - central_longitude + 180) % 360 - 180
    lat = np.rad2deg(np.arctan2(z, np.hypot(x, y)))
    # Rotating longitudes first puts the cut at +/-180 for every view.
    proj = ccrs.Robinson(globe=target.globe,
                         false_easting=target.proj4_params.get("x_0", 0.),
                         false_northing=target.proj4_params.get("y_0", 0.))
    ordinary = np.ptp(lon, axis=1) < 180
    visible = proj.transform_points(ccrs.PlateCarree(), lon[ordinary], lat[ordinary])[..., :2]
    polygons, ids = list(visible), owners[ordinary].tolist()

    # Cartopy clips great-circle polygons at its curved outline and closes
    # polar caps. A geodetic exterior must wind counterclockwise on the sphere.
    for i in np.flatnonzero(~ordinary):
        ring = np.column_stack((lon[i], lat[i]))
        normal = np.cross(rings[i], following[i]).sum(axis=0)
        if normal @ rings[i].mean(axis=0) < 0:
            ring = ring[::-1]
        for part in proj.project_geometry(Polygon(ring), ccrs.Geodetic()).geoms:
            polygons.append(np.asarray(part.exterior.coords))
            ids.append(owners[i])
    return polygons, np.asarray(ids, dtype=np.intp)
