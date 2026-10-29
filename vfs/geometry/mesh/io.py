"""Interoperability helpers for externally generated horizontal meshes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .grid import (
    EARTH_RADIUS_M,
    _FLOAT_BUFFERS,
    _INT_BUFFERS,
    _rebuild_derived_buffers,
    VoronoiMesh,
)


def load_vfs_imex_mesh(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> VoronoiMesh:
    """Load an immutable geometry/operator cache written by the final IMEX donor.

    This is deliberately separate from :meth:`VoronoiMesh.load`: the donor
    cache uses the older ``UnstructuredGrid`` NetCDF schema and stores its
    already-qualified Thuburn/rotation buffers verbatim.  It exists only for
    dry-path reproduction controls; ordinary main-VFS caches retain their
    current schema/version checks.
    """

    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - packaging dependent.
        raise ImportError("vfs-imex mesh loading requires the VFS NetCDF dependencies") from exc

    source_path = Path(path)
    with xr.open_dataset(source_path) as ds:
        attrs = dict(ds.attrs)
        if str(attrs.get("grid_class", "")) != "UnstructuredGrid":
            raise ValueError(
                f"{source_path} is not a vfs-imex UnstructuredGrid cache; "
                f"grid_class={attrs.get('grid_class')!r}"
            )
        required_attrs = ("n_cells", "n_edges", "n_vertices", "radius")
        missing_attrs = [name for name in required_attrs if name not in attrs]
        if missing_attrs:
            raise ValueError(f"{source_path} is missing vfs-imex mesh attributes: {missing_attrs}")

        source_names = {
            "cell_xyz": "primal_centers",
            "vertex_xyz": "primal_vertices",
        }
        required_buffers = {
            source_names.get(name, name)
            for name in (*_FLOAT_BUFFERS, *_INT_BUFFERS)
        }
        missing_buffers = sorted(required_buffers - set(ds.data_vars))
        if missing_buffers:
            raise ValueError(
                f"{source_path} is missing buffers required for donor reproduction: {missing_buffers}"
            )

        target_device = torch.device("cpu") if device is None else torch.device(device)
        mesh = VoronoiMesh.__new__(VoronoiMesh)
        nn.Module.__init__(mesh)
        mesh.nCells = int(attrs["n_cells"])
        mesh.nEdges = int(attrs["n_edges"])
        mesh.nVertices = int(attrs["n_vertices"])
        mesh.maxEdges = int(ds["edgesOnCell"].shape[1])
        mesh.radius = float(attrs["radius"])

        for name in _FLOAT_BUFFERS:
            source_name = source_names.get(name, name)
            mesh.register_buffer(
                name,
                torch.as_tensor(ds[source_name].values, dtype=torch.float64, device=target_device),
            )
        for name in _INT_BUFFERS:
            mesh.register_buffer(
                name,
                torch.as_tensor(ds[name].values, dtype=torch.int64, device=target_device),
            )

    if int(mesh.cellsOnEdge.shape[0]) != mesh.nEdges or int(mesh.cell_xyz.shape[0]) != mesh.nCells:
        raise ValueError(f"{source_path} geometry dimensions disagree with vfs-imex mesh metadata")
    if int(mesh.vertex_xyz.shape[0]) != mesh.nVertices:
        raise ValueError(f"{source_path} vertex geometry disagrees with vfs-imex mesh metadata")

    _rebuild_derived_buffers(mesh)
    mesh.source_mesh_format = "vfs-imex"
    mesh.source_mesh_path = str(source_path)
    mesh.source_mesh_lloyd_iters = int(attrs.get("lloyd_iters", 0))
    mesh.source_mesh_kind = str(attrs.get("mesh_kind", "fibonacci_lloyd"))
    return mesh


def load_mpas_mesh(
    path: str | Path,
    *,
    radius: float | None = None,
    device: torch.device | str | None = None,
    verify_connectivity: bool = True,
) -> VoronoiMesh:
    """Rebuild a :class:`VoronoiMesh` from an MPAS mesh NetCDF file.

    MPAS cell generators are normalized onto the sphere, then VFS rebuilds
    its own Delaunay/Voronoi metric geometry.  ``cellsOnEdge`` is compared as
    an undirected edge set by default.  A mismatch is an error rather than a
    silent topology substitution, because constrained or degenerate MPAS
    meshes cannot necessarily be represented by VFS's reconstructed topology.

    This lightweight reader needs only xarray/netCDF support.  It neither
    vendors nor requires MPAS-Tools or JIGSAW.
    """

    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - packaging dependent.
        raise ImportError("MPAS mesh loading requires the VFS NetCDF dependencies") from exc

    source_path = Path(path)
    with xr.open_dataset(source_path) as ds:
        coordinate_names = ("xCell", "yCell", "zCell")
        missing = [name for name in coordinate_names if name not in ds]
        if missing:
            raise ValueError(f"{source_path} is missing MPAS cell coordinates: {missing}")
        xyz = np.stack([np.asarray(ds[name].values) for name in coordinate_names], axis=-1)
        norms = np.linalg.norm(xyz, axis=-1)
        if not np.isfinite(xyz).all() or bool((norms <= 0.0).any()):
            raise ValueError(f"{source_path} contains non-finite or zero-length MPAS cell coordinates")
        if radius is None:
            attr_radius = float(ds.attrs.get("sphere_radius", 0.0) or 0.0)
            coordinate_radius = float(np.mean(norms))
            radius = coordinate_radius if coordinate_radius > 1_000.0 else attr_radius
            if radius <= 1_000.0:
                radius = EARTH_RADIUS_M
        source_edges: set[tuple[int, int]] | None = None
        if verify_connectivity:
            if "cellsOnEdge" not in ds:
                raise ValueError(f"{source_path} has no cellsOnEdge for connectivity verification")
            cells_on_edge = np.asarray(ds["cellsOnEdge"].values, dtype=np.int64) - 1
            if cells_on_edge.ndim != 2 or cells_on_edge.shape[1] != 2:
                raise ValueError(
                    f"{source_path} cellsOnEdge must have shape [nEdges, 2], "
                    f"got {tuple(cells_on_edge.shape)}"
                )
            valid = np.all(cells_on_edge >= 0, axis=1)
            source_edges = {tuple(sorted((int(a), int(b)))) for a, b in cells_on_edge[valid]}

    mesh = VoronoiMesh(
        n_cells=int(xyz.shape[0]),
        lloyd_iters=0,
        radius=float(radius),
        generator_xyz=torch.as_tensor(xyz, dtype=torch.float64),
    )
    if device is not None:
        mesh = mesh.to(device=device)
    mesh.source_mesh_format = "MPAS"
    mesh.source_mesh_path = str(source_path)
    if source_edges is not None:
        rebuilt_edges = {
            tuple(sorted((int(a), int(b))))
            for a, b in mesh.cellsOnEdge.detach().cpu().tolist()
        }
        if source_edges != rebuilt_edges:
            raise ValueError(
                "Rebuilt VFS Delaunay topology differs from MPAS cellsOnEdge: "
                f"missing={len(source_edges - rebuilt_edges)}, "
                f"added={len(rebuilt_edges - source_edges)}. "
                "The source may use constrained or degenerate topology."
            )
    return mesh


__all__ = ["load_mpas_mesh", "load_vfs_imex_mesh"]
