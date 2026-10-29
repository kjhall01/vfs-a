"""
Spherical Voronoi C-grid mesh.

Numerical grid implementation containing:
  - Fibonacci sphere initialization
  - Spherical Lloyd relaxation (correct spherical centroids)
  - Topology construction from convex hull / Delaunay triangulation
  - Geometry computation (areas, lengths, normals)
  - TRiSK-consistent discrete operators: grad, div, curl
  - cell_to_edge / edge_to_cell (adjoint pair under cell_edge_area weights)
  - TRiSK tangential velocity reconstruction weights (Thuburn 2009)
  - Helpers: integrate_cell, integrate_dual, cell_wind_vector, regrid

Design principles:
  - All operators are consistent with a single primal/dual pairing.
  - No subtriangle operator family. No mixed-family terms.
  - grad and div are discrete adjoints: sum(f * div(g)) = -sum(g * grad(f))
    under cell_area and dvEdge weights respectively (up to boundary).
  - curl and grad satisfy curl(grad(f)) = 0 exactly.
  - div and curl satisfy div(curl(u)) = 0 exactly (on closed sphere).
  - KE accumulation uses the same cell_edge_area weights as cell_to_edge,
    so Bernoulli and PGF both use mesh.grad — no mixed gradient operators.

Variables and their locations:
  - Scalars (ps, theta, phi, dp): cell centers  [nCells]
  - Edge-normal velocity (u):     edges          [nEdges]
  - Vorticity / PV:               vertices       [nVertices]

References:
  Thuburn et al. 2009, J. Comput. Phys. 228, 8321-8335
  Ringler et al. 2010, J. Comput. Phys. 229, 3065-3090
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from scipy.spatial import ConvexHull

from vfs.field import Field, require_field


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARTH_RADIUS_M = 6.371e6
MESH_OPERATOR_VERSION = 2
EXACT_THUBURN_VERSION = 2
THUBURN_CORNER_CONVENTION = "current_next_exit_vertex_v2"
THUBURN_IMPLEMENTATION = "vfs.geometry.mesh._build_exact_thuburn_edge_operator_numpy"


# ---------------------------------------------------------------------------
# Mesh generation: Fibonacci sphere + spherical Lloyd relaxation
# ---------------------------------------------------------------------------

def _fibonacci_xyz(n: int) -> np.ndarray:
    """Return n points on the unit sphere via the Fibonacci / golden-angle method."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    golden_angle = 2.0 * math.pi * (1.0 - 1.0 / phi)
    i = np.arange(n, dtype=np.float64)
    z = 1.0 - 2.0 * (i + 0.5) / n
    z = np.clip(z, -1.0, 1.0)
    r = np.sqrt(1.0 - z * z)
    theta = golden_angle * i
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)


def _normalize_rows(xyz: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    return xyz / np.clip(norms, 1e-15, None)


def _spherical_triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Spherical excess area of a unit-sphere triangle (van Oosterom & Strackee)."""
    triple = float(np.dot(a, np.cross(b, c)))
    denom = 1.0 + float(np.dot(a, b)) + float(np.dot(b, c)) + float(np.dot(c, a))
    return 2.0 * math.atan2(abs(triple), denom)


def _spherical_polygon_moment(vertices: np.ndarray) -> np.ndarray:
    """Exact integral of x dA for a convex minor-arc unit-sphere polygon.

    Each oriented edge contributes half its arc length times its unit normal.
    Accepts either winding. The polygon must lie in an open hemisphere.
    This is a vector area moment, not a geodesic-distance Frechet mean.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError('Expected at least three spherical polygon vertices')
    if not np.isfinite(vertices).all() or not np.allclose(
        np.linalg.norm(vertices, axis=1), 1.0, rtol=0, atol=1e-10
    ):
        raise ValueError('Polygon vertices must be finite unit vectors')
    following = np.roll(vertices, -1, axis=0)
    cross = np.cross(vertices, following)
    sine = np.linalg.norm(cross, axis=1)
    cosine = np.sum(vertices * following, axis=1)
    if np.any((sine < 1e-15) & (cosine < 0)):
        raise ValueError('Antipodal polygon edge is ambiguous')
    factors = np.divide(np.arctan2(sine, cosine), sine,
                        out=np.ones_like(sine), where=sine > 0)
    moment = 0.5 * np.sum(cross * factors[:, None], axis=0)
    if np.dot(moment, vertices.mean(axis=0)) < 0:
        moment = -moment
    if np.linalg.norm(moment) <= 1e-15:
        raise ValueError('Degenerate spherical polygon')
    return moment


def _spherical_polygon_centroid(vertices: np.ndarray) -> np.ndarray:
    """Direction of the exact area first moment (uniform-density chordal CVT)."""
    moment = _spherical_polygon_moment(vertices)
    return moment / np.linalg.norm(moment)


def _spherical_circumcenter(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Circumcenter of a spherical triangle.

    A point p on the unit sphere is equidistant from a, b, c (in spherical
    distance) iff a·p = b·p = c·p. The set {p : a·p = b·p} is the plane
    (a-b)·p = 0. The intersection of (a-b)·p = 0 and (b-c)·p = 0 is the
    line spanned by (a-b)×(b-c) = a×b + b×c + c×a.
    """
    cc = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
    norm = np.linalg.norm(cc)
    if norm < 1e-15:
        cc = a + b + c          # fallback for degenerate triangle
        norm = np.linalg.norm(cc)
    cc = cc / norm
    # Ensure we pick the hemisphere containing the triangle
    if np.dot(cc, a + b + c) < 0.0:
        cc = -cc
    return cc


def lloyd_relaxation(
    xyz: np.ndarray,
    n_iters: int = 200,
    tol: float = 1e-6,
    patience: int = 100,
    min_rel_improvement: float = 1e-5,
    verbose: bool = False,
    *,
    relative_tol: float = 1e-4,
    max_relative_tol: float = 5e-4,
    return_info: bool = False,
):
    """Uniform-density spherical Lloyd iteration using exact area moments.

    Returns the coordinates as before; return_info=True returns (xyz, info).
    Convergence requires mean angular centroid offset <= tol AND RMS/max
    offsets divided by sqrt(cell area) <= relative_tol/max_relative_tol.
    All residuals describe the RETURNED mesh. Budget exhaustion and objective
    plateau are reported separately and never imply convergence.

    The reported objective is integral sum_i |x-generator_i|^2 dA on the
    unit sphere (chordal distance). It is not squared generator displacement.
    Set min_rel_improvement=0 to disable the optional objective plateau stop.
    Existing explicit generator arrays with lloyd_iters=0 are unchanged.
    """
    from scipy.spatial import SphericalVoronoi

    xyz = np.asarray(xyz, dtype=np.float64)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 4
            or not np.isfinite(xyz).all()
            or np.any(np.linalg.norm(xyz, axis=1) == 0)):
        raise ValueError('xyz must contain at least four finite nonzero 3-vectors')
    if not isinstance(n_iters, (int, np.integer)) or n_iters < 0:
        raise ValueError('n_iters must be a nonnegative integer')
    if any(not np.isfinite(x) or x < 0 for x in
           (tol, relative_tol, max_relative_tol, min_rel_improvement)):
        raise ValueError('Tolerances must be finite and nonnegative')
    xyz = _normalize_rows(xyz.copy())
    significant_best = math.inf
    best_iter = 0
    history = []
    for iteration in range(n_iters + 1):
        vor = SphericalVoronoi(xyz, radius=1.0, center=np.zeros(3))
        vor.sort_vertices_of_regions()
        # Vectorized boundary integrals, with independent winding per cell.
        starts = np.concatenate([vor.vertices[r] for r in vor.regions])
        ends = np.concatenate([np.roll(vor.vertices[r], -1, axis=0) for r in vor.regions])
        owners = np.repeat(np.arange(len(xyz)), [len(r) for r in vor.regions])
        cross = np.cross(starts, ends)
        sine = np.linalg.norm(cross, axis=1)
        cosine = np.sum(starts * ends, axis=1)
        if np.any((sine < 1e-15) & (cosine < 0)):
            raise ValueError('Antipodal Voronoi edge')
        factors = np.divide(np.arctan2(sine, cosine), sine,
                            out=np.ones_like(sine), where=sine > 0)
        moments = np.zeros_like(xyz)
        np.add.at(moments, owners, 0.5 * cross * factors[:, None])
        moments *= np.where(np.sum(moments * xyz, axis=1) < 0, -1., 1.)[:, None]
        if np.any(np.linalg.norm(moments, axis=1) <= 1e-15):
            raise ValueError('Degenerate Voronoi cell')
        centers = _normalize_rows(moments)
        areas = vor.calculate_areas()
        offsets = 2 * np.arcsin(np.clip(np.linalg.norm(centers-xyz, axis=1)/2, 0, 1))
        relative = offsets / np.sqrt(areas)
        objective = float(2 * areas.sum() - 2 * np.sum(xyz * moments))
        row = dict(iteration=iteration, objective=objective,
                   mean_offset_rad=float(offsets.mean()), max_offset_rad=float(offsets.max()),
                   rms_relative_offset=float(np.sqrt(np.mean(relative**2))),
                   max_relative_offset=float(relative.max()))
        history.append(row)
        converged = (row['mean_offset_rad'] <= tol
                     and row['rms_relative_offset'] <= relative_tol
                     and row['max_relative_offset'] <= max_relative_tol)
        if verbose:
            print(f"Lloyd {iteration}: objective={objective:.10g}, "
                  f"relative RMS={row['rms_relative_offset']:.3e}, "
                  f"max={row['max_relative_offset']:.3e}")
        if converged:
            reason = 'converged'
            break
        if iteration == n_iters:
            reason = 'iteration_budget'
            break
        if objective < significant_best * (1 - min_rel_improvement):
            significant_best, best_iter = objective, iteration
        if min_rel_improvement > 0 and patience > 0 and iteration-best_iter >= patience:
            reason = 'objective_plateau'
            break
        xyz = centers
    info = dict(converged=bool(converged), stop_reason=reason, iterations=iteration,
                tolerances=dict(mean_offset_rad=tol, rms_relative_offset=relative_tol,
                                max_relative_offset=max_relative_tol),
                final=history[-1], history=history)
    if verbose:
        print(f'Lloyd stopped: {reason}; converged={converged}')
    return (xyz, info) if return_info else xyz


def _csr_from_padded_edge_operator_numpy(
    *,
    n_edges: int,
    stencil: np.ndarray,
    weights: np.ndarray,
) -> csr_matrix:
    """Build a CSR edge operator from padded row stencils."""
    stencil = np.asarray(stencil, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if stencil.shape != weights.shape or stencil.ndim != 2 or stencil.shape[0] != n_edges:
        raise ValueError("stencil and weights must have matching [n_edges, width] shapes")

    valid = stencil >= 0
    if np.any(stencil[valid] >= n_edges):
        raise ValueError("stencil contains an edge index outside [0, n_edges)")

    rows = np.broadcast_to(np.arange(n_edges, dtype=np.int64)[:, None], stencil.shape)[valid]
    operator = csr_matrix(
        (weights[valid], (rows, stencil[valid])),
        shape=(n_edges, n_edges),
        dtype=np.float64,
    )
    operator.sum_duplicates()
    operator.eliminate_zeros()
    operator.sort_indices()
    return operator


def _padded_from_csr_edge_operator_numpy(
    operator: csr_matrix,
    *,
    atol: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a CSR edge operator back to padded row stencils."""
    operator = operator.tocsr(copy=True)
    if operator.shape[0] != operator.shape[1]:
        raise ValueError("edge operator must be square")
    if atol > 0.0:
        operator.data[np.abs(operator.data) <= atol] = 0.0
    operator.eliminate_zeros()
    operator.sort_indices()

    n_edges = operator.shape[0]
    counts = np.diff(operator.indptr).astype(np.int64, copy=False)
    max_nnz = int(counts.max(initial=0))
    stencil = -np.ones((n_edges, max_nnz), dtype=np.int64)
    weights = np.zeros((n_edges, max_nnz), dtype=np.float64)

    for edge, count in enumerate(counts):
        if count == 0:
            continue
        start = operator.indptr[edge]
        stop = operator.indptr[edge + 1]
        stencil[edge, :count] = operator.indices[start:stop]
        weights[edge, :count] = operator.data[start:stop]

    return stencil, weights, counts


def _project_padded_edge_operator_m_skew_numpy(
    *,
    n_edges: int,
    stencil: np.ndarray,
    weights: np.ndarray,
    metric: np.ndarray,
    atol: float = 0.0,
) -> dict[str, np.ndarray]:
    """Project a padded edge operator onto the metric-skew subspace in CSR."""
    metric = np.asarray(metric, dtype=np.float64).reshape(-1)
    if metric.size != n_edges:
        raise ValueError(f"metric size {metric.size} does not match n_edges {n_edges}")
    if np.any(metric <= 0.0) or not np.all(np.isfinite(metric)):
        raise ValueError("metric must be finite and strictly positive")

    operator = _csr_from_padded_edge_operator_numpy(
        n_edges=n_edges,
        stencil=stencil,
        weights=weights,
    )
    metric_adjoint = operator.transpose().multiply(metric[None, :])
    metric_adjoint = metric_adjoint.multiply((1.0 / metric)[:, None]).tocsr()
    projected = (0.5 * (operator - metric_adjoint)).tocsr()
    projected.setdiag(0.0)
    projected.eliminate_zeros()
    projected.sort_indices()

    proj_stencil, proj_weights, proj_count = _padded_from_csr_edge_operator_numpy(
        projected,
        atol=atol,
    )

    metric_projected = projected.multiply(metric[:, None]).tocsr()
    defect = (metric_projected + metric_projected.transpose()).tocsr()
    defect.eliminate_zeros()
    defect_max = 0.0 if defect.nnz == 0 else float(np.abs(defect.data).max())

    return {
        "stencil": proj_stencil,
        "weights": proj_weights,
        "count": proj_count,
        "skew_defect_max": np.array(defect_max, dtype=np.float64),
    }

# ---------------------------------------------------------------------------
# Topology construction from convex hull
# ---------------------------------------------------------------------------

def _triangle_neighbors(triangles: np.ndarray) -> np.ndarray:
    """Return the three face neighbors for a closed triangular sphere mesh."""

    triangles = np.asarray(triangles, dtype=np.int64)
    neighbors = -np.ones(triangles.shape, dtype=np.int64)
    owners: dict[tuple[int, int], tuple[int, int]] = {}
    for face_id, (a, b, c) in enumerate(triangles):
        for local, edge in enumerate(((b, c), (a, c), (a, b))):
            key = tuple(sorted((int(edge[0]), int(edge[1]))))
            previous = owners.get(key)
            if previous is None:
                owners[key] = (face_id, local)
                continue
            previous_face, previous_local = previous
            neighbors[face_id, local] = previous_face
            neighbors[previous_face, previous_local] = face_id
    if bool((neighbors < 0).any()):
        raise ValueError("supplied Delaunay triangles must form a closed two-face edge mesh")
    return neighbors


def _build_topology(
    xyz: np.ndarray,
    *,
    delaunay_triangles: np.ndarray | None = None,
) -> dict:
    """
    Build Voronoi C-grid topology from unit-sphere generator positions.

    Returns a dict with:
      cell_xyz        [nCells, 3]     unit-sphere cell centers (= generators)
      vertex_xyz      [nVertices, 3]  unit-sphere dual vertices (circumcenters)
      cellsOnEdge     [nEdges, 2]     primal cell indices per edge
      verticesOnEdge  [nEdges, 2]     dual vertex indices per edge
      cellsOnVertex   [nVertices, 3]  primal cell indices per dual triangle
      edgesOnVertex   [nVertices, 3]  edge indices per dual vertex
      verticesOnCell  [nCells, maxE]  dual vertex indices per cell (CCW, padded -1)
      edgesOnCell     [nCells, maxE]  edge indices per cell (CCW, padded -1)
      edgeSignOnCell  [nCells, maxE]  +1 if edge outward, -1 if inward
      edgeSignOnVertex[nVertices, 3]  +1/-1 for CCW circulation sense
      nEdgesOnCell    [nCells]        actual valency of each cell
    """
    xyz = _normalize_rows(np.asarray(xyz, dtype=np.float64))
    n_cells = xyz.shape[0]

    if delaunay_triangles is None:
        hull = ConvexHull(xyz, qhull_options="QJ")
        triangles = hull.simplices.astype(np.int64)    # [nVertices, 3] Delaunay triangles
        tri_neighbors = hull.neighbors.astype(np.int64) # [nVertices, 3]
    else:
        triangles = np.asarray(delaunay_triangles, dtype=np.int64)
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("delaunay_triangles must have shape [n_faces, 3]")
        if int(triangles.min()) < 0 or int(triangles.max()) >= n_cells:
            raise ValueError("delaunay_triangles contains an invalid cell index")
        tri_neighbors = _triangle_neighbors(triangles)
    n_vertices = triangles.shape[0]

    # --- Dual vertex positions: spherical circumcenters of each Delaunay triangle ---
    vertex_xyz = np.empty((n_vertices, 3), dtype=np.float64)
    for v, (a, b, c) in enumerate(triangles):
        vertex_xyz[v] = _spherical_circumcenter(xyz[a], xyz[b], xyz[c])

    # --- Build edge list ---
    # Each edge is shared by exactly two Delaunay triangles (dual vertices).
    # We visit each pair once (tri_idx < neighbor_idx).
    cells_on_edge_list = []
    vertices_on_edge_list = []
    edge_counter = 0
    edges_on_vertex = -np.ones((n_vertices, 3), dtype=np.int64)
    for tri in range(n_vertices):
        a, b, c = triangles[tri]
        # The three edges of triangle `tri` are opposite to vertices a, b, c:
        #   opposite a: edge (b, c)  shared with neighbor tri_neighbors[tri][0]
        #   opposite b: edge (a, c)  shared with neighbor tri_neighbors[tri][1]
        #   opposite c: edge (a, b)  shared with neighbor tri_neighbors[tri][2]
        opposite = [(b, c), (a, c), (a, b)]
        for local, nbr in enumerate(tri_neighbors[tri]):
            if tri < nbr:
                ci, cj = opposite[local]
                if ci > cj:
                    ci, cj = cj, ci
                cells_on_edge_list.append((ci, cj))
                vertices_on_edge_list.append((tri, nbr))
                eidx = edge_counter
                edge_counter += 1
                edges_on_vertex[tri, local] = eidx
                # Find the back-pointer from nbr to tri
                back = np.where(tri_neighbors[nbr] == tri)[0]
                edges_on_vertex[nbr, back[0]] = eidx

    cells_on_edge = np.asarray(cells_on_edge_list, dtype=np.int64)
    vertices_on_edge = np.asarray(vertices_on_edge_list, dtype=np.int64)
    n_edges = cells_on_edge.shape[0]

    # --- Build cell-edge adjacency by graph traversal ---
    #
    # For each primal cell ci, collect all incident edges (edges where ci
    # appears in cellsOnEdge).  Then walk the edge ring in CCW order by
    # following the dual-vertex graph: each edge has two dual-vertex
    # endpoints; consecutive edges in the ring share exactly one dual vertex.
    #
    # This approach never sorts by angle and never does a set-intersection
    # lookup, so it is immune to near-degenerate circumcenter positions.

    # cell -> list of incident edge indices
    cell_incident_edges: list[list[int]] = [[] for _ in range(n_cells)]
    for eidx in range(n_edges):
        c0, c1 = cells_on_edge[eidx]
        cell_incident_edges[int(c0)].append(eidx)
        cell_incident_edges[int(c1)].append(eidx)

    n_edges_on_cell = np.array([len(e) for e in cell_incident_edges], dtype=np.int64)
    max_edges = int(n_edges_on_cell.max())

    edges_on_cell    = -np.ones((n_cells, max_edges), dtype=np.int64)
    edge_sign_on_cell = np.zeros((n_cells, max_edges), dtype=np.float64)
    vertices_on_cell = -np.ones((n_cells, max_edges), dtype=np.int64)

    for ci in range(n_cells):
        inc = cell_incident_edges[ci]
        ne = len(inc)

        # Build a map: dual_vertex -> [edge_indices incident on ci at that vertex]
        # Each dual vertex in the ring appears in exactly 2 of ci's edges.
        vtx_to_edges: dict[int, list[int]] = {}
        for eidx in inc:
            for v in vertices_on_edge[eidx]:
                v = int(v)
                vtx_to_edges.setdefault(v, []).append(eidx)

        # Walk the ring: start from the first edge, pick a direction.
        # At each step: current edge has two dual vertices; one was the
        # "entry" vertex (shared with previous edge), the other is the
        # "exit" vertex (shared with next edge).
        ordered_edges = []
        ordered_verts = []  # the "exit" dual vertex after each edge

        start_edge = inc[0]
        v0, v1 = int(vertices_on_edge[start_edge, 0]), int(vertices_on_edge[start_edge, 1])

        # Choose orientation: pick whichever vertex comes first CCW.
        # Use a single cross-product check against the cell center.
        cx = xyz[ci]
        vp0, vp1 = vertex_xyz[v0], vertex_xyz[v1]
        # If (vp0 -> vp1) is CCW around cx, start with v0 as entry.
        ccw = np.dot(cx, np.cross(vp0, vp1))
        if ccw >= 0.0:
            entry_v, exit_v = v0, v1
        else:
            entry_v, exit_v = v1, v0

        ordered_edges.append(start_edge)
        ordered_verts.append(exit_v)

        prev_edge = start_edge
        cur_exit  = exit_v

        for _ in range(ne - 1):
            # The next edge shares cur_exit as one of its dual vertices,
            # and is not the edge we just came from.
            candidates = [e for e in vtx_to_edges[cur_exit] if e != prev_edge]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Cell {ci}: expected 1 next edge at vertex {cur_exit}, "
                    f"got {candidates}. Ring walk failed."
                )
            next_edge = candidates[0]
            ordered_edges.append(next_edge)
            # The exit vertex of next_edge is the one that is NOT cur_exit
            nv0, nv1 = int(vertices_on_edge[next_edge, 0]), int(vertices_on_edge[next_edge, 1])
            cur_exit = nv1 if nv0 == cur_exit else nv0
            ordered_verts.append(cur_exit)
            prev_edge = next_edge

        for k, eidx in enumerate(ordered_edges):
            edges_on_cell[ci, k] = eidx
            c0e = int(cells_on_edge[eidx, 0])
            edge_sign_on_cell[ci, k] = 1.0 if ci == c0e else -1.0
            vertices_on_cell[ci, k] = ordered_verts[k]

    # --- edgeSignOnVertex: sign for CCW circulation at each dual vertex ---
    edge_sign_on_vertex = np.zeros((n_vertices, 3), dtype=np.float64)
    for v in range(n_vertices):
        a, b, c = triangles[v]
        # Determine orientation (should be CCW viewed from outside sphere)
        orient = np.dot(xyz[a], np.cross(xyz[b], xyz[c]))
        if orient >= 0.0:
            ordered_cells = (a, b, c)
        else:
            ordered_cells = (a, c, b)
        # CCW boundary pairs for this triangle
        oc = ordered_cells
        boundary = {(oc[0], oc[1]), (oc[1], oc[2]), (oc[2], oc[0])}
        for loc in range(3):
            eidx = edges_on_vertex[v, loc]
            c0, c1 = cells_on_edge[eidx]
            if (c0, c1) in boundary:
                edge_sign_on_vertex[v, loc] = 1.0
            else:
                edge_sign_on_vertex[v, loc] = -1.0

    return {
        "cell_xyz": xyz,
        "vertex_xyz": vertex_xyz,
        "cellsOnEdge": cells_on_edge,
        "verticesOnEdge": vertices_on_edge,
        "cellsOnVertex": triangles.copy(),
        "edgesOnVertex": edges_on_vertex,
        "verticesOnCell": vertices_on_cell,
        "edgesOnCell": edges_on_cell,
        "edgeSignOnCell": edge_sign_on_cell,
        "edgeSignOnVertex": edge_sign_on_vertex,
        "nEdgesOnCell": n_edges_on_cell,
        "nCells": n_cells,
        "nEdges": n_edges,
        "nVertices": n_vertices,
        "maxEdges": max_edges,
    }


# ---------------------------------------------------------------------------
# Geometry: areas, lengths, normals, TRiSK weights
# ---------------------------------------------------------------------------

def _arc_length(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Great-circle arc length between unit vectors (vectorized, numerically stable)."""
    chord = np.linalg.norm(a - b, axis=-1)
    return 2.0 * np.arcsin(np.clip(0.5 * chord, 0.0, 1.0))


def _great_circle_tangent(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Unit tangent at the midpoint of the great-circle arc from a to b.
    Points in the direction of increasing arc from a toward b.
    """
    mid = a + b
    norm_mid = np.linalg.norm(mid, axis=-1, keepdims=True)
    mid = mid / np.clip(norm_mid, 1e-15, None)
    gc_normal = np.cross(a, b)
    tangent = np.cross(gc_normal, mid)
    norm_t = np.linalg.norm(tangent, axis=-1, keepdims=True)
    return tangent / np.clip(norm_t, 1e-15, None)


def _spherical_triangle_area_batch(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Spherical excess area for batches of unit-sphere triangles."""
    triple = np.sum(a * np.cross(b, c), axis=-1)
    denom = 1.0 + np.sum(a * b, axis=-1) + np.sum(b * c, axis=-1) + np.sum(c * a, axis=-1)
    return 2.0 * np.arctan2(np.abs(triple), denom)


def _unit_midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit-sphere midpoint between matching point pairs."""
    m = a + b
    return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-15, None)


def _build_exact_thuburn_edge_operator_numpy(
    *,
    cell_xyz: np.ndarray,
    vertex_xyz: np.ndarray,
    edge_xyz: np.ndarray,
    edge_normal: np.ndarray,
    edge_tangent: np.ndarray,
    cells_on_edge: np.ndarray,
    vertices_on_edge: np.ndarray,
    edges_on_vertex: np.ndarray,
    edge_sign_on_vertex: np.ndarray,
    vertices_on_cell: np.ndarray,
    edges_on_cell: np.ndarray,
    edge_sign_on_cell: np.ndarray,
    n_edges_on_cell: np.ndarray,
    cell_area_unit: np.ndarray,
    dc_edge: np.ndarray,
    dv_edge: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Build the exact-Thuburn tangential and raw-rotation edge operators.

    This ports the retained survivor-path construction into the standalone
    VoronoiMesh geometry builder.

    Outputs
    -------
    tan_stencil / tan_weights:
        Tangential velocity reconstruction u_t = T u_n.

    rotation_stencil / rotation_weights:
        Raw rotated transport R u_n, intended for the PV-flux / Coriolis
        rotation term. This must be kept separate from tan_weights.

    The important metric scaling is dv_edge[source] / dc_edge[target].
    """

    n_cells = cell_xyz.shape[0]
    n_edges = cells_on_edge.shape[0]
    max_edges = edges_on_cell.shape[1]

    # ------------------------------------------------------------
    # 1. Exact geometric corner fractions R_{i,v}
    # ------------------------------------------------------------
    corner_area = np.zeros((n_cells, max_edges), dtype=np.float64)
    corner_fraction = np.zeros((n_cells, max_edges), dtype=np.float64)
    corner_fraction_sum = np.zeros(n_cells, dtype=np.float64)

    for ci in range(n_cells):
        ne = int(n_edges_on_cell[ci])
        if ne <= 0:
            continue

        ring_vertices = vertices_on_cell[ci, :ne]
        ring_edges = edges_on_cell[ci, :ne]

        center = np.repeat(cell_xyz[ci:ci + 1], ne, axis=0)
        vtx = vertex_xyz[ring_vertices]
        # vertices_on_cell[ci, k] is the exit vertex of edges_on_cell[ci, k].
        # Its adjacent boundary-edge midpoints are therefore the current edge
        # and the next edge, not the previous/current pair.  Using the latter
        # over-counts the corner partition by ~3/2 on hex-like meshes.
        prev_edge_xyz = edge_xyz[ring_edges]
        next_edge_xyz = edge_xyz[np.roll(ring_edges, -1)]

        local_corner_area = (
            _spherical_triangle_area_batch(center, prev_edge_xyz, vtx)
            + _spherical_triangle_area_batch(center, vtx, next_edge_xyz)
        )

        corner_area[ci, :ne] = local_corner_area
        corner_fraction[ci, :ne] = local_corner_area / max(float(cell_area_unit[ci]), 1.0e-30)
        corner_fraction_sum[ci] = corner_fraction[ci, :ne].sum()

    # ------------------------------------------------------------
    # 2. Endpoint circulation sign t_e^v
    # ------------------------------------------------------------
    endpoint_sign = np.zeros((n_edges, 2), dtype=np.float64)

    for vi in range(edges_on_vertex.shape[0]):
        for local_slot, e in enumerate(edges_on_vertex[vi]):
            e = int(e)
            if e < 0:
                continue

            ev0, ev1 = vertices_on_edge[e]
            if int(ev0) == vi:
                endpoint_sign[e, 0] = edge_sign_on_vertex[vi, local_slot]
            elif int(ev1) == vi:
                endpoint_sign[e, 1] = edge_sign_on_vertex[vi, local_slot]
            else:
                raise RuntimeError(
                    f"Vertex {vi} is not an endpoint of edge {e}."
                )

    # ------------------------------------------------------------
    # 3. Edge handedness used to convert perpendicular rotation
    #    into tangential velocity reconstruction.
    # ------------------------------------------------------------
    rotation_hand = np.cross(edge_xyz, edge_tangent)
    edge_sign = np.sign(np.sum(rotation_hand * edge_normal, axis=1))
    edge_sign[edge_sign == 0.0] = 1.0

    # ------------------------------------------------------------
    # 4. Exact path-sum Thuburn weights.
    # ------------------------------------------------------------
    row_entries: list[dict[int, float]] = []
    row_nnz = np.zeros(n_edges, dtype=np.int64)
    row_sum = np.zeros(n_edges, dtype=np.float64)
    row_abs_sum = np.zeros(n_edges, dtype=np.float64)

    for target_edge in range(n_edges):
        local_weights: dict[int, float] = {}

        for ci in cells_on_edge[target_edge]:
            ci = int(ci)
            if ci < 0:
                continue

            ne = int(n_edges_on_cell[ci])
            ring_edges = [int(x) for x in edges_on_cell[ci, :ne]]
            ring_vertices = [int(x) for x in vertices_on_cell[ci, :ne]]
            ring_signs = [float(x) for x in edge_sign_on_cell[ci, :ne]]

            try:
                target_local = ring_edges.index(int(target_edge))
            except ValueError as exc:
                raise RuntimeError(
                    f"Cell {ci} does not contain target edge {target_edge}."
                ) from exc

            target_vertex = ring_vertices[target_local]
            ev0, ev1 = vertices_on_edge[target_edge]
            if int(ev0) == int(target_vertex):
                target_endpoint_slot = 0
            elif int(ev1) == int(target_vertex):
                target_endpoint_slot = 1
            else:
                raise RuntimeError(
                    f"Target edge {target_edge} does not contain ring vertex "
                    f"{target_vertex} on cell {ci}."
                )

            target_circulation_sign = float(endpoint_sign[target_edge, target_endpoint_slot])

            for source_local, source_edge in enumerate(ring_edges):
                source_edge = int(source_edge)
                if source_edge == int(target_edge):
                    continue

                source_sign = float(ring_signs[source_local])

                # Sum corner fractions along the ordered cell boundary from
                # source_edge toward target_edge.
                path_sum = 0.0
                vertex_local = (source_local + 1) % ne
                while True:
                    path_sum += float(corner_fraction[ci, vertex_local])
                    if vertex_local == target_local:
                        break
                    vertex_local = (vertex_local + 1) % ne

                exact_weight = target_circulation_sign * source_sign * (path_sum - 0.5)

                # This is the crucial Thuburn metric scaling.
                perpendicular_weight = exact_weight * float(dv_edge[source_edge]) / max(
                    float(dc_edge[target_edge]),
                    1.0e-30,
                )

                local_weights[source_edge] = local_weights.get(source_edge, 0.0) + perpendicular_weight

        row_entries.append(local_weights)
        row_nnz[target_edge] = len(local_weights)
        if local_weights:
            vals = np.asarray(list(local_weights.values()), dtype=np.float64)
            row_sum[target_edge] = vals.sum()
            row_abs_sum[target_edge] = np.abs(vals).sum()

    max_nnz = max((len(entries) for entries in row_entries), default=0)
    stencil = -np.ones((n_edges, max_nnz), dtype=np.int64)
    valid = np.zeros((n_edges, max_nnz), dtype=bool)
    perp_weights = np.zeros((n_edges, max_nnz), dtype=np.float64)

    for e, entries in enumerate(row_entries):
        if not entries:
            continue
        source_edges = list(entries.keys())
        vals = list(entries.values())
        count = len(source_edges)

        stencil[e, :count] = np.asarray(source_edges, dtype=np.int64)
        valid[e, :count] = True
        perp_weights[e, :count] = np.asarray(vals, dtype=np.float64)

    # Match survivor-path convention:
    #   tan_weights      = edge_sign * perp_weights
    #   rotation_weights = -perp_weights
    tan_weights = np.where(valid, edge_sign[:, None] * perp_weights, 0.0)
    rotation_weights = np.where(valid, -perp_weights, 0.0)

    return {
        "corner_area": corner_area,
        "corner_fraction": corner_fraction,
        "corner_fraction_sum": corner_fraction_sum,
        "endpoint_sign": endpoint_sign,
        "edge_sign": edge_sign,
        "tan_stencil": stencil,
        "tan_weights": tan_weights,
        "rotation_stencil": stencil.copy(),
        "rotation_weights": rotation_weights,
        "rotation_stencil_count": row_nnz,
        "rotation_stencil_valid": valid,
        "rotation_row_sum": row_sum,
        "rotation_row_abs_sum": row_abs_sum,
    }

def _compute_geometry(topo: dict, radius: float) -> dict:
    """
    Compute all metric geometry from topology + radius.

    Returns a dict of numpy arrays ready to register as buffers.
    """
    cell_xyz = topo["cell_xyz"]       # unit sphere
    vertex_xyz = topo["vertex_xyz"]   # unit sphere
    cells_on_edge = topo["cellsOnEdge"]
    vertices_on_edge = topo["verticesOnEdge"]
    cells_on_vertex = topo["cellsOnVertex"]
    vertices_on_cell = topo["verticesOnCell"]
    edges_on_cell = topo["edgesOnCell"]
    n_edges_on_cell = topo["nEdgesOnCell"]
    n_cells = topo["nCells"]
    n_edges = topo["nEdges"]
    n_vertices = topo["nVertices"]
    max_edges = topo["maxEdges"]

    # --- Edge geometry ---
    c0 = cell_xyz[cells_on_edge[:, 0]]
    c1 = cell_xyz[cells_on_edge[:, 1]]
    v0 = vertex_xyz[vertices_on_edge[:, 0]]
    v1 = vertex_xyz[vertices_on_edge[:, 1]]

    dc_edge = radius * _arc_length(c0, c1)      # primal edge length [nEdges]
    dv_edge = radius * _arc_length(v0, v1)      # dual edge length [nEdges]
    edge_normal = _great_circle_tangent(c0, c1) # unit normal (c0→c1) [nEdges, 3]
    edge_tangent = _great_circle_tangent(v0, v1)# unit tangent (v0→v1) [nEdges, 3]

    # orient signs
    edge_mid = c0 + c1
    edge_mid = edge_mid / np.clip(np.linalg.norm(edge_mid, axis=-1, keepdims=True), 1e-15, None)

    edge_normal = _great_circle_tangent(c0, c1)

    # Convention matching momentum RHS rot = -q * u_t:
    # t = n × r, where r is outward radial direction.
    edge_tangent = _great_circle_tangent(v0, v1)
    target_tangent = np.cross(edge_normal, edge_mid)
    target_tangent = target_tangent / np.clip(
        np.linalg.norm(target_tangent, axis=-1, keepdims=True), 1e-15, None
    )

    flip = np.sum(edge_tangent * target_tangent, axis=-1) < 0.0
    edge_tangent[flip] *= -1.0
    # end orient signs

    #
    # --- Primal cell areas: sum of spherical triangles (center, vi, vi+1) ---
    cell_area = np.zeros(n_cells, dtype=np.float64)
    # Also compute cell_edge_area[cell, local_edge] = area of the sub-triangle
    # (cell_center, edge_midpoint_v0, edge_midpoint_v1) for TRiSK weights.
    # We approximate this as the spherical triangle (cell, v_k, v_{k+1}).
    cell_edge_area = np.zeros((n_cells, max_edges), dtype=np.float64)
    for ci in range(n_cells):
        ne = n_edges_on_cell[ci]
        ring = vertices_on_cell[ci, :ne]
        ctr = cell_xyz[ci:ci+1].repeat(ne, axis=0)   # [ne, 3]
        va = vertex_xyz[np.roll(ring, 1)]              # ring[(k-1)%ne]: entry vertex of edge k
        vb = vertex_xyz[ring]                          # ring[k]:       exit  vertex of edge k
        areas = _spherical_triangle_area_batch(ctr, va, vb) * (radius ** 2)
        cell_area[ci] = areas.sum()
        cell_edge_area[ci, :ne] = areas

    # --- Dual cell (triangle) areas ---
    a = cell_xyz[cells_on_vertex[:, 0]]
    b = cell_xyz[cells_on_vertex[:, 1]]
    c = cell_xyz[cells_on_vertex[:, 2]]
    dual_area = _spherical_triangle_area_batch(a, b, c) * (radius ** 2)

    # --- Exact-Thuburn tangential and raw-rotation edge operators ---
    #
    # This replaces the previous local dot-product tangential reconstruction.
    # The survivor path keeps two separate operators:
    #   tan_weights       : tangential velocity reconstruction for KE diagnostics
    #   rotation_weights  : raw rotated transport for PV-flux / Coriolis rotation
    #
    # Do not collapse these into one buffer.
    edge_xyz = _unit_midpoint(v0, v1)  # unit-sphere midpoint of the Voronoi edge

    cell_area_unit = cell_area / (radius ** 2)

    thuburn = _build_exact_thuburn_edge_operator_numpy(
        cell_xyz=cell_xyz,
        vertex_xyz=vertex_xyz,
        edge_xyz=edge_xyz,
        edge_normal=edge_normal,
        edge_tangent=edge_tangent,
        cells_on_edge=cells_on_edge,
        vertices_on_edge=vertices_on_edge,
        edges_on_vertex=topo["edgesOnVertex"],
        edge_sign_on_vertex=topo["edgeSignOnVertex"],
        vertices_on_cell=vertices_on_cell,
        edges_on_cell=edges_on_cell,
        edge_sign_on_cell=topo["edgeSignOnCell"],
        n_edges_on_cell=n_edges_on_cell,
        cell_area_unit=cell_area_unit,
        dc_edge=dc_edge,
        dv_edge=dv_edge,
    )

    tan_stencil = thuburn["tan_stencil"]
    tan_weights = thuburn["tan_weights"]
    rotation_stencil = thuburn["rotation_stencil"]
    rotation_weights = thuburn["rotation_weights"]

    # --- M-skew projection of the raw rotation operator ---
    #
    # The exact-Thuburn port can still have small metric-skew defects because
    # this standalone mesh geometry/order is not identical to the survivor path.
    # Project the raw rotation operator onto:
    #
    #     M R + R^T M = 0
    #
    # Use the same edge metric as the grad/div adjointness test.
    rotation_metric = dc_edge * dv_edge

    projected_rotation = _project_padded_edge_operator_m_skew_numpy(
        n_edges=n_edges,
        stencil=rotation_stencil,
        weights=rotation_weights,
        metric=rotation_metric,
        atol=0.0,
    )

    rotation_stencil = projected_rotation["stencil"]
    rotation_weights = projected_rotation["weights"]
    rotation_stencil_count = projected_rotation["count"]

    return {
        "dc_edge": dc_edge.astype(np.float64),
        "dv_edge": dv_edge.astype(np.float64),
        "edge_normal": edge_normal.astype(np.float64),
        "edge_tangent": edge_tangent.astype(np.float64),
        "edge_xyz": edge_xyz.astype(np.float64),

        "cell_area": cell_area.astype(np.float64),
        "dual_area": dual_area.astype(np.float64),
        "cell_edge_area": cell_edge_area.astype(np.float64),

        # Tangential reconstruction
        "tan_stencil": tan_stencil.astype(np.int64),
        "tan_weights": tan_weights.astype(np.float64),

        # Separate exact-Thuburn raw rotation operator
        "rotation_stencil": rotation_stencil.astype(np.int64),
        "rotation_weights": rotation_weights.astype(np.float64),
        "rotation_stencil_count": rotation_stencil_count.astype(np.int64),
        "rotation_row_sum": rotation_weights.sum(axis=1).astype(np.float64),
        "rotation_row_abs_sum": np.abs(rotation_weights).sum(axis=1).astype(np.float64),
        "rotation_skew_defect_max": projected_rotation["skew_defect_max"].astype(np.float64),

        # Useful diagnostics
        "thuburn_corner_area": thuburn["corner_area"].astype(np.float64),
        "thuburn_corner_fraction": thuburn["corner_fraction"].astype(np.float64),
        "thuburn_corner_fraction_sum": thuburn["corner_fraction_sum"].astype(np.float64),
        "thuburn_endpoint_sign": thuburn["endpoint_sign"].astype(np.float64),
        "thuburn_edge_sign": thuburn["edge_sign"].astype(np.float64),
    }


# ---------------------------------------------------------------------------
# VoronoiMesh
# ---------------------------------------------------------------------------

class VoronoiMesh(nn.Module):
    """
    Spherical Voronoi C-grid mesh.

    Construction
    ------------
    mesh = VoronoiMesh(n_cells=1024, lloyd_iters=100, radius=EARTH_RADIUS_M)

    Key integer attributes
    ----------------------
    nCells, nEdges, nVertices, maxEdges

    Buffers (unit sphere, float64)
    --------------------------------
    cell_xyz        [nCells,    3]
    vertex_xyz      [nVertices, 3]
    edge_normal     [nEdges,    3]  unit outward normal (cell0 → cell1)
    edge_tangent    [nEdges,    3]  unit tangent (vertex0 → vertex1)

    Buffers (metric, float64)
    --------------------------
    dc_edge         [nEdges]        primal edge length (m)
    dv_edge         [nEdges]        dual edge length (m)
    cell_area       [nCells]        primal cell area (m²)
    dual_area       [nVertices]     dual triangle area (m²)
    cell_edge_area  [nCells, maxE]  sub-triangle areas for TRiSK weights (m²)

    Connectivity (int64)
    --------------------
    cellsOnEdge     [nEdges,    2]
    verticesOnEdge  [nEdges,    2]
    cellsOnVertex   [nVertices, 3]
    edgesOnVertex   [nVertices, 3]
    verticesOnCell  [nCells, maxE]  padded with -1
    edgesOnCell     [nCells, maxE]  padded with -1
    nEdgesOnCell    [nCells]

    Sign arrays (float64)
    ---------------------
    edgeSignOnCell  [nCells, maxE]  +1 outward, -1 inward
    edgeSignOnVertex[nVertices, 3]  +1/-1 for CCW circulation

    TRiSK tangential reconstruction (float64)
    -----------------------------------------
    tan_stencil     [nEdges, maxS]  indices of contributing edges (-1 = pad)
    tan_weights     [nEdges, maxS]  reconstruction weights

    Operators (all accept [..., nX] tensors)
    ----------------------------------------
    grad(phi_c)           cell   → edge   (m⁻¹)
    div(flux_e)           edge   → cell   (m⁻¹)
    curl(u_e)             edge   → vertex (m⁻¹, i.e. s⁻¹ if u in m/s)
    cell_to_edge(phi_c)   cell   → edge   (area-weighted average)
    edge_to_cell(phi_e)   edge   → cell   (area-weighted average, adjoint of above)
    tangential_velocity(u_e)  edge-normal → edge-tangential via TRiSK weights
    ke_cell(u_e)          edge   → cell   (kinetic energy per unit mass, m²/s²)
    integrate_cell(q_c)         → scalar  (sum q * cell_area)
    integrate_dual(q_v)         → scalar  (sum q * dual_area)
    cell_wind_vector(u_e) edge   → (nCells, 3) Cartesian cell-center wind vector
    """

    def __init__(
        self,
        n_cells: int = 1024,
        lloyd_iters: int = 100,
        radius: float = EARTH_RADIUS_M,
        verbose: bool = False,
        generator_xyz: torch.Tensor | np.ndarray | None = None,
        delaunay_triangles: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        super().__init__()

        # --- Generate and relax generator positions ---
        if generator_xyz is None:
            if verbose:
                print(f"Generating {n_cells} Fibonacci points...")
            xyz = _fibonacci_xyz(n_cells)
        else:
            source_xyz = torch.as_tensor(generator_xyz, dtype=torch.float64).detach().cpu().numpy()
            if source_xyz.ndim != 2 or tuple(source_xyz.shape[1:]) != (3,):
                raise ValueError(
                    "generator_xyz must have shape [n_cells, 3]; "
                    f"got {tuple(source_xyz.shape)}"
                )
            if int(source_xyz.shape[0]) != int(n_cells):
                raise ValueError(
                    f"generator_xyz contains {int(source_xyz.shape[0])} cells, "
                    f"but n_cells={int(n_cells)}"
                )
            if not np.isfinite(source_xyz).all() or np.any(np.linalg.norm(source_xyz, axis=1) <= 0.0):
                raise ValueError("generator_xyz must contain finite, non-zero coordinates")
            xyz = _normalize_rows(source_xyz)

        self.lloyd_diagnostics = None
        if lloyd_iters > 0:
            if verbose:
                print(f"Running up to {lloyd_iters} Lloyd iterations...")
            # Disable the plateau stop in the mesh constructor: callers set
            # `lloyd_iters` expecting a real iteration budget, and the old
            # plateau logic could stop on noisy CVT-energy wiggles long before
            # the mesh quality was where we wanted it.
            xyz, self.lloyd_diagnostics = lloyd_relaxation(
                xyz,
                n_iters=lloyd_iters,
                min_rel_improvement=0.0,
                return_info=True,
                verbose=verbose,
            )

        # --- Build topology ---
        if verbose:
            print("Building topology...")
        triangles = None
        if delaunay_triangles is not None:
            triangles = torch.as_tensor(delaunay_triangles, dtype=torch.int64).detach().cpu().numpy()
        topo = _build_topology(xyz, delaunay_triangles=triangles)

        # --- Compute geometry ---
        if verbose:
            print("Computing geometry...")
        geom = _compute_geometry(topo, radius)

        # --- Store scalar attributes ---
        self.nCells: int = topo["nCells"]
        self.nEdges: int = topo["nEdges"]
        self.nVertices: int = topo["nVertices"]
        self.maxEdges: int = topo["maxEdges"]
        self.radius: float = float(radius)

        # --- Register unit-sphere position buffers ---
        self._reg_f("cell_xyz",    topo["cell_xyz"])
        self._reg_f("vertex_xyz",  topo["vertex_xyz"])
        self._reg_f("edge_normal", geom["edge_normal"])
        self._reg_f("edge_tangent",geom["edge_tangent"])

        # --- Register metric geometry buffers ---
        self._reg_f("dc_edge",        geom["dc_edge"])
        self._reg_f("dv_edge",        geom["dv_edge"])
        self._reg_f("cell_area",      geom["cell_area"])
        self._reg_f("dual_area",      geom["dual_area"])
        self._reg_f("cell_edge_area", geom["cell_edge_area"])
        self._reg_f(
            "cell_edge_area_masked",
            np.where(topo["edgesOnCell"] >= 0, geom["cell_edge_area"], 0.0),
        )

        # --- Register connectivity buffers ---
        self._reg_i("cellsOnEdge",      topo["cellsOnEdge"])
        self._reg_i("verticesOnEdge",   topo["verticesOnEdge"])
        self._reg_i("cellsOnVertex",    topo["cellsOnVertex"])
        self._reg_i("edgesOnVertex",    topo["edgesOnVertex"])
        self._reg_i("verticesOnCell",   topo["verticesOnCell"])
        self._reg_i("edgesOnCell",      topo["edgesOnCell"])
        self._reg_i("nEdgesOnCell",     topo["nEdgesOnCell"])

        # --- Register sign arrays ---
        self._reg_f("edgeSignOnCell",   topo["edgeSignOnCell"])
        self._reg_f("edgeSignOnVertex", topo["edgeSignOnVertex"])

        # --- Register TRiSK tangential reconstruction ---
        self._reg_i("tan_stencil", geom["tan_stencil"])
        self._reg_f("tan_weights", geom["tan_weights"])
        self._reg_f(
            "tan_weights_masked",
            np.where(geom["tan_stencil"] >= 0, geom["tan_weights"], 0.0),
        )

        # --- Register exact-Thuburn raw rotation operator ---
        self._reg_i("rotation_stencil", geom["rotation_stencil"])
        self._reg_f("rotation_weights", geom["rotation_weights"])
        self._reg_i("rotation_stencil_count", geom["rotation_stencil_count"])
        self._reg_f("rotation_row_sum", geom["rotation_row_sum"])
        self._reg_f("rotation_row_abs_sum", geom["rotation_row_abs_sum"])
        self._reg_f(
            "rotation_weights_masked",
            np.where(geom["rotation_stencil"] >= 0, geom["rotation_weights"], 0.0),
        )
        self._reg_f(
            "coriolis_unit_vertex",
            2.0 * np.sin(np.arcsin(np.clip(topo["vertex_xyz"][:, 2], -1.0, 1.0))),
        )

        # --- Optional exact-Thuburn diagnostics ---
        self._reg_f("edge_xyz", geom["edge_xyz"])
        self._reg_f("thuburn_corner_area", geom["thuburn_corner_area"])
        self._reg_f("thuburn_corner_fraction", geom["thuburn_corner_fraction"])
        self._reg_f("thuburn_corner_fraction_sum", geom["thuburn_corner_fraction_sum"])
        self._reg_f("thuburn_endpoint_sign", geom["thuburn_endpoint_sign"])
        self._reg_f("thuburn_edge_sign", geom["thuburn_edge_sign"])

        # --- Precompute safe (no -1) index arrays for gather ops ---
        self._reg_i("safe_edges_on_cell",
                    np.where(topo["edgesOnCell"] >= 0, topo["edgesOnCell"], 0))
        self._reg_i("safe_cells_on_edge",
                    np.where(topo["cellsOnEdge"] >= 0, topo["cellsOnEdge"], 0))
        self.register_buffer("valid_edges_on_cell",
                    torch.as_tensor(topo["edgesOnCell"] >= 0))
        self._reg_i("safe_tan_stencil",
                    np.where(geom["tan_stencil"] >= 0, geom["tan_stencil"], 0))
        self.register_buffer("valid_tan_stencil",
                    torch.as_tensor(geom["tan_stencil"] >= 0))

        self._reg_i(
            "safe_rotation_stencil",
            np.where(geom["rotation_stencil"] >= 0, geom["rotation_stencil"], 0),
        )
        self.register_buffer(
            "valid_rotation_stencil",
            torch.as_tensor(geom["rotation_stencil"] >= 0),
        )
        self._reg_f("rotation_skew_defect_max", geom["rotation_skew_defect_max"])

        if verbose:
            self._print_quality_summary()

    @classmethod
    def refined_icosahedron(
        cls,
        frequency: int,
        *,
        radius: float = EARTH_RADIUS_M,
        device: torch.device | str | None = None,
        verbose: bool = False,
    ) -> "VoronoiMesh":
        """Construct the donor-equivalent class-I refined-icosahedral mesh.

        This is used for static multilevel preconditioner levels. Main VFS
        simulations may instead bind arbitrary Voronoi or MPAS fine meshes.
        """

        from ..geodesic import icosahedral_cell_count, refined_icosahedron

        frequency = int(frequency)
        xyz, triangles = refined_icosahedron(frequency)
        mesh = cls(
            n_cells=icosahedral_cell_count(frequency),
            lloyd_iters=0,
            radius=radius,
            verbose=verbose,
            generator_xyz=xyz,
            delaunay_triangles=triangles,
        )
        mesh.mesh_kind = "icosahedral"
        mesh.icosahedral_frequency = frequency
        return mesh if device is None else mesh.to(device=device)

    @classmethod
    def from_mpas_netcdf(
        cls,
        path,
        *,
        radius: float | None = None,
        device: torch.device | str | None = None,
        verify_connectivity: bool = True,
    ) -> "VoronoiMesh":
        """Build a VFS mesh from MPAS cell generators after topology checks."""

        from .io import load_mpas_mesh

        return load_mpas_mesh(
            path,
            radius=radius,
            device=device,
            verify_connectivity=verify_connectivity,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reg_f(self, name: str, arr: np.ndarray) -> None:
        self.register_buffer(name, torch.as_tensor(arr, dtype=torch.float64))

    def _reg_i(self, name: str, arr: np.ndarray) -> None:
        self.register_buffer(name, torch.as_tensor(arr, dtype=torch.int64))

    # ------------------------------------------------------------------
    # Discrete operators
    #
    # State tensor convention:  [..., nSpatial, nLayers]
    #   - Cell fields:    [..., nCells,    nLayers]
    #   - Edge fields:    [..., nEdges,    nLayers]
    #   - Vertex fields:  [..., nVertices, nLayers]
    #
    # The spatial dimension is always second-to-last (-2).
    # nLayers is the last dimension (-1) and is carried unchanged.
    # All geometry buffers are 1-D [nSpatial]; they broadcast over
    # leading batch dims and over the layers dim via [..., None].
    # Stencil reductions (over gathered neighbours) use dim=-2 so
    # that the layers dim is preserved.
    # ------------------------------------------------------------------

    def _field_operator_values(self, field: Field) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]:
        """Flatten a Field payload for the historical tensor kernels.

        Geometry kernels predate Field and operate on ``[..., horizontal,
        payload]`` tensors.  This adapter keeps those numerically audited
        kernels intact while making the public operator contract Field-native.
        """

        axis = field.horizontal_axis % field.values.ndim
        values = field.values
        batch_shape = tuple(values.shape[:axis])
        payload_shape = tuple(values.shape[axis + 1 :])
        # ``Tensor.flatten`` keeps this adapter visible to ``torch.compile``.
        # Computing a Python ``np.prod`` of symbolic shapes forces Dynamo to
        # materialize a non-tensor intermediate and breaks full-graph capture.
        if payload_shape:
            flattened = values.flatten(start_dim=axis + 1)
        else:
            flattened = values.unsqueeze(-1)
        return flattened, batch_shape, payload_shape

    @staticmethod
    def _field_from_operator_values(
        template: Field,
        values: torch.Tensor,
        *,
        location: str,
        name: str | None,
        batch_shape: tuple[int, ...],
        payload_shape: tuple[int, ...],
    ) -> Field:
        shape = (*batch_shape, values.shape[-2], *payload_shape)
        return Field(
            values.reshape(shape),
            location=location,
            vertical_location=template.vertical_location,
            name=name,
            units=template.units,
            component_axes=template.component_axes,
        )

    def grad(self, phi_c: Field | torch.Tensor) -> Field | torch.Tensor:
        """Compute an edge-normal gradient of a cell Field.

        Field inputs are the VFS 1.0 public contract.  The tensor dispatch is
        retained temporarily for internal migration of the large owner-module
        surface and is deliberately not documented as public API.
        """

        if not isinstance(phi_c, Field):
            return self._grad_values(phi_c)
        field = require_field(phi_c, location="cell", name="phi_c")
        values, batch_shape, payload_shape = self._field_operator_values(field)
        result = self._grad_values(values)
        return self._field_from_operator_values(
            field,
            result,
            location="edge",
            name=None if field.name is None else f"grad({field.name})",
            batch_shape=batch_shape,
            payload_shape=payload_shape,
        )

    def _grad_values(self, phi_c: torch.Tensor) -> torch.Tensor:
        """
        Cell-scalar → edge-normal gradient.

        Input:  [..., nCells,    nLayers]
        Output: [..., nEdges,    nLayers]

        grad(phi)[..., e, k] = (phi[..., cell1, k] - phi[..., cell0, k])
                                / dc_edge[e]

        Adjoint of -div under weights (cell_area, dv_edge * dc_edge):
            sum_{e,k} dv*dc[e] * g[...,e,k] * grad(f)[...,e,k]
          = -sum_{c,k} cell_area[c] * f[...,c,k] * div(g)[...,c,k]
        """
        c0 = phi_c[..., self.cellsOnEdge[:, 0], :]   # [..., nEdges, nLayers]
        c1 = phi_c[..., self.cellsOnEdge[:, 1], :]
        return (c1 - c0) / self.dc_edge[:, None]      # dc_edge: [nEdges] -> [nEdges,1]

    def div(self, flux_e: Field | torch.Tensor) -> Field | torch.Tensor:
        """Compute a cell divergence of an edge Field."""

        if not isinstance(flux_e, Field):
            return self._div_values(flux_e)
        field = require_field(flux_e, location="edge", name="flux_e")
        values, batch_shape, payload_shape = self._field_operator_values(field)
        result = self._div_values(values)
        return self._field_from_operator_values(
            field,
            result,
            location="cell",
            name=None if field.name is None else f"div({field.name})",
            batch_shape=batch_shape,
            payload_shape=payload_shape,
        )

    def _div_values(self, flux_e: torch.Tensor) -> torch.Tensor:
        """
        Edge-normal flux → cell divergence.

        Input:  [..., nEdges, nLayers]
        Output: [..., nCells, nLayers]

        div(F)[..., i, k] = (1/cell_area[i])
                            * sum_{e in i} dv_edge[e] * sign[i,e] * F[...,e,k]
        """
        # gather: [..., nCells, maxEdges, nLayers]
        gathered = flux_e[..., self.safe_edges_on_cell, :]
        # dv_sign: [nCells, maxEdges] -> [nCells, maxEdges, 1] for broadcast
        dv_sign = (self.dv_edge[self.safe_edges_on_cell] * self.edgeSignOnCell)[..., None]
        weighted = gathered * dv_sign
        valid = self.valid_edges_on_cell[..., None]   # [nCells, maxEdges, 1]
        weighted = torch.where(valid, weighted, torch.zeros_like(weighted))
        return weighted.sum(dim=-2) / self.cell_area[:, None]

    def curl(self, u_e: Field | torch.Tensor) -> Field | torch.Tensor:
        """Compute vertex vorticity from an edge Field."""

        if not isinstance(u_e, Field):
            return self._curl_values(u_e)
        field = require_field(u_e, location="edge", name="u_e")
        values, batch_shape, payload_shape = self._field_operator_values(field)
        result = self._curl_values(values)
        return self._field_from_operator_values(
            field,
            result,
            location="vertex",
            name=None if field.name is None else f"curl({field.name})",
            batch_shape=batch_shape,
            payload_shape=payload_shape,
        )

    def _curl_values(self, u_e: torch.Tensor) -> torch.Tensor:
        """
        Edge-normal velocity → vertex vorticity.

        Input:  [..., nEdges,    nLayers]
        Output: [..., nVertices, nLayers]

        curl(u)[..., v, k] = (1/dual_area[v])
                             * sum_{e in v} dc_edge[e] * sign[v,e] * u[...,e,k]

        Satisfies curl(grad(phi)) = 0 exactly.
        """
        # gather: [..., nVertices, 3, nLayers]
        gathered = u_e[..., self.edgesOnVertex, :]
        dc_sign = (self.dc_edge[self.edgesOnVertex] * self.edgeSignOnVertex)[..., None]
        circulation = (gathered * dc_sign).sum(dim=-2)   # [..., nVertices, nLayers]
        return circulation / self.dual_area[:, None]

    def curl_adjoint(self, zeta_v: Field | torch.Tensor) -> Field | torch.Tensor:
        """Return the edge Field adjoint to :meth:`curl`."""

        if not isinstance(zeta_v, Field):
            return self._curl_adjoint_values(zeta_v)
        field = require_field(zeta_v, location="vertex", name="zeta_v")
        values, batch_shape, payload_shape = self._field_operator_values(field)
        result = self._curl_adjoint_values(values)
        return self._field_from_operator_values(
            field,
            result,
            location="edge",
            name=None if field.name is None else f"curl_adjoint({field.name})",
            batch_shape=batch_shape,
            payload_shape=payload_shape,
        )

    def _curl_adjoint_values(self, zeta_v: torch.Tensor) -> torch.Tensor:
        """
        Vertex scalar → edge-normal field, adjoint of ``curl``.

        Input:  [..., nVertices, nLayers]
        Output: [..., nEdges,    nLayers]

        Under the edge metric ``dc_edge * dv_edge`` and vertex metric
        ``dual_area``,

            <curl(u), zeta>_dual = <u, curl_adjoint(zeta)>_edge
        """

        vertices = self.verticesOnEdge.to(device=zeta_v.device)
        endpoint_edges = self.edgesOnVertex[vertices].to(device=zeta_v.device)
        endpoint_sign = self.edgeSignOnVertex[vertices].to(dtype=zeta_v.dtype, device=zeta_v.device)
        edge_index = torch.arange(int(self.nEdges), device=zeta_v.device).reshape(-1, 1, 1)
        sign = (endpoint_sign * (endpoint_edges == edge_index).to(dtype=zeta_v.dtype)).sum(dim=-1)
        zeta_end = zeta_v[..., vertices, :]
        circulation = (zeta_end * sign[..., None]).sum(dim=-2)
        return circulation / self.dv_edge.to(dtype=zeta_v.dtype, device=zeta_v.device)[:, None]

    def vector_laplacian_edge(self, u_e: Field | torch.Tensor, *, mode: str = "div_curl") -> Field | torch.Tensor:
        """
        Compatible edge-normal vector Laplacian.

        Input:  [..., nEdges, nLayers]
        Output: [..., nEdges, nLayers]

        The default ``div_curl`` form is the negative-semidefinite Hodge-style
        operator

            L(u) = grad(div(u)) - curl_adjoint(curl(u)).

        It satisfies the discrete energy identity

            <u, L(u)>_edge = -||div(u)||^2_cell - ||curl(u)||^2_dual

        under the VFS mesh metrics. Positive viscosity multiplying this
        operator therefore damps kinetic energy.
        """

        if not isinstance(u_e, Field):
            return self._vector_laplacian_edge_values(u_e, mode=mode)
        field = require_field(u_e, location="edge", name="u_e")
        values, batch_shape, payload_shape = self._field_operator_values(field)
        result = self._vector_laplacian_edge_values(values, mode=mode)
        return self._field_from_operator_values(
            field,
            result,
            location="edge",
            name=None if field.name is None else f"laplacian({field.name})",
            batch_shape=batch_shape,
            payload_shape=payload_shape,
        )

    def _vector_laplacian_edge_values(self, u_e: torch.Tensor, *, mode: str = "div_curl") -> torch.Tensor:
        if mode != "div_curl":
            raise ValueError(f"vector_laplacian_edge mode must be 'div_curl', got {mode!r}")
        return self._grad_values(self._div_values(u_e)) - self._curl_adjoint_values(self._curl_values(u_e))

    def trisk_pv_flux_rotation(self, u_e: Field | torch.Tensor, rotation_rate: float) -> Field | torch.Tensor:
        """
        Exact-Thuburn PV-flux rotation shared by atmosphere and ocean cores.

        Invalid padded rotation entries have zero precomputed weights, so the
        runtime path does not allocate a mask/zero tensor for every call.
        """

        if not isinstance(u_e, Field):
            return self._trisk_pv_flux_rotation_values(u_e, rotation_rate)
        field = require_field(u_e, location="edge", name="u_e")
        values, batch_shape, payload_shape = self._field_operator_values(field)
        result = self._trisk_pv_flux_rotation_values(values, rotation_rate)
        return self._field_from_operator_values(
            field,
            result,
            location="edge",
            name=None if field.name is None else f"pv_rotation({field.name})",
            batch_shape=batch_shape,
            payload_shape=payload_shape,
        )

    def _trisk_pv_flux_rotation_values(self, u_e: torch.Tensor, rotation_rate: float) -> torch.Tensor:
        zeta_v = self._curl_values(u_e)
        coriolis = (float(rotation_rate) * self.coriolis_unit_vertex).to(
            dtype=u_e.dtype,
            device=u_e.device,
        )
        abs_v = zeta_v + coriolis[:, None]

        v0 = self.verticesOnEdge[:, 0]
        v1 = self.verticesOnEdge[:, 1]
        q_e = 0.5 * (abs_v[..., v0, :] + abs_v[..., v1, :])

        s = self.safe_rotation_stencil
        R = self.rotation_weights_masked
        u_s = u_e[..., s, :]
        q_s = q_e[..., s, :]
        q_t = q_e[..., :, None, :]
        term = 0.5 * (q_t + q_s) * R[..., None] * u_s
        return -term.sum(dim=-2)

    def cell_to_edge(self, phi_c: Field | torch.Tensor) -> Field | torch.Tensor:
        """Interpolate a cell Field to edges."""

        if not isinstance(phi_c, Field):
            return self._cell_to_edge_values(phi_c)
        field = require_field(phi_c, location="cell", name="phi_c")
        values, batch_shape, payload_shape = self._field_operator_values(field)
        result = self._cell_to_edge_values(values)
        return self._field_from_operator_values(
            field,
            result,
            location="edge",
            name=None if field.name is None else f"edge({field.name})",
            batch_shape=batch_shape,
            payload_shape=payload_shape,
        )

    def _cell_to_edge_values(self, phi_c: torch.Tensor) -> torch.Tensor:
        """
        Cell scalar → edge scalar via area-weighted average.

        Input:  [..., nCells, nLayers]
        Output: [..., nEdges, nLayers]

        Discrete adjoint of edge_to_cell:
            sum_{e,k} (a0+a1)[e] * cell_to_edge(f)[...,e,k] * g[...,e,k]
          = sum_{c,k} cell_area[c] * f[...,c,k] * edge_to_cell(g)[...,c,k]
        """
        phi0 = phi_c[..., self.cellsOnEdge[:, 0], :]   # [..., nEdges, nLayers]
        phi1 = phi_c[..., self.cellsOnEdge[:, 1], :]
        a0 = self.edge_side_area[:, 0, None]            # [nEdges, 1]
        a1 = self.edge_side_area[:, 1, None]
        return (phi0 * a0 + phi1 * a1) / (a0 + a1).clamp_min(1e-30)

    def edge_to_cell(self, phi_e: Field | torch.Tensor) -> Field | torch.Tensor:
        """Interpolate an edge Field to cells."""

        if not isinstance(phi_e, Field):
            return self._edge_to_cell_values(phi_e)
        field = require_field(phi_e, location="edge", name="phi_e")
        values, batch_shape, payload_shape = self._field_operator_values(field)
        result = self._edge_to_cell_values(values)
        return self._field_from_operator_values(
            field,
            result,
            location="cell",
            name=None if field.name is None else f"cell({field.name})",
            batch_shape=batch_shape,
            payload_shape=payload_shape,
        )

    def _edge_to_cell_values(self, phi_e: torch.Tensor) -> torch.Tensor:
        """
        Edge scalar → cell scalar via area-weighted average.

        Input:  [..., nEdges, nLayers]
        Output: [..., nCells, nLayers]

        Discrete adjoint of cell_to_edge (see above).
        """
        # gather: [..., nCells, maxEdges, nLayers]
        gathered = phi_e[..., self.safe_edges_on_cell, :]
        weights = self.cell_edge_area[..., None]          # [nCells, maxEdges, 1]
        valid   = self.valid_edges_on_cell[..., None]
        weighted = torch.where(valid, gathered * weights, torch.zeros_like(gathered))
        return weighted.sum(dim=-2) / self.cell_area[:, None].clamp_min(1e-30)

    def tangential_velocity(self, u_e: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct edge-tangential velocity from edge-normal velocities.

        Input:  [..., nEdges, nLayers]
        Output: [..., nEdges, nLayers]

        u_t[..., e, k] = sum_{e'} tan_weights[e, e'] * u_n[..., e', k]

        Uses TRiSK weights (Thuburn 2009 eq 20).  Must use the same weights
        in ke_cell so that the rotation term does zero net work.
        """
        # gather: [..., nEdges, maxStencil, nLayers]
        gathered = u_e[..., self.safe_tan_stencil, :]
        weights = self.tan_weights_masked[..., None]      # [nEdges, maxStencil, 1]
        return (gathered * weights).sum(dim=-2)           # [..., nEdges, nLayers]

    def tangential_velocity_adjoint(self, y_e: torch.Tensor) -> torch.Tensor:
        """
        Euclidean adjoint of :meth:`tangential_velocity`.

        If ``u_t = T u``, this returns ``T^T y``. It is used by exact
        pressure-work diagnostics because ``ke_cell`` includes reconstructed
        tangential kinetic energy, not only the diagonal normal-wind part.
        Invalid padded stencil weights are already zero.
        """
        if y_e.ndim < 2:
            raise ValueError("tangential_velocity_adjoint expects [..., nEdges, nLayers]")
        leading = y_e.shape[:-2]
        n_layers = y_e.shape[-1]
        y_flat = y_e.reshape(-1, self.nEdges, n_layers)
        weights = self.tan_weights_masked.to(dtype=y_e.dtype, device=y_e.device)
        contrib = y_flat[:, :, None, :] * weights[None, :, :, None]
        idx = self.safe_tan_stencil.to(device=y_e.device).reshape(-1)
        scatter_idx = idx[None, :, None].expand(y_flat.shape[0], -1, n_layers)
        out = torch.zeros_like(y_flat)
        out.scatter_add_(1, scatter_idx, contrib.reshape(y_flat.shape[0], -1, n_layers))
        return out.reshape(*leading, self.nEdges, n_layers)

    def kinetic_energy_gradient_edge(
        self,
        u_e: torch.Tensor,
        dp_c: torch.Tensor,
        *,
        gravity: float = 1.0,
    ) -> torch.Tensor:
        """
        Closed-form gradient of integrated C-grid kinetic energy with respect
        to edge-normal wind.

        ``ke_cell`` implies

            KE = 0.5 * sum_e,k W[e,k] * (u[e,k]^2 + (T u)[e,k]^2)

        where ``T`` is the tangential reconstruction and

            W[e,k] = (A_left[e] * dp_left[k] + A_right[e] * dp_right[k]) / g.

        Thus

            dKE/du = W * u + T^T(W * T u).
        """
        if u_e.ndim < 2 or dp_c.ndim < 2:
            raise ValueError(
                "kinetic_energy_gradient_edge expects u [..., nEdges, nLayers] "
                "and dp [..., nCells, nLayers]"
            )
        c0 = self.cellsOnEdge[:, 0]
        c1 = self.cellsOnEdge[:, 1]
        area = self.edge_side_area.to(dtype=u_e.dtype, device=u_e.device)
        dp = dp_c.to(dtype=u_e.dtype, device=u_e.device)
        weight = (area[:, 0, None] * dp[..., c0, :] + area[:, 1, None] * dp[..., c1, :]) / float(gravity)
        u_t = self.tangential_velocity(u_e)
        return weight * u_e + self.tangential_velocity_adjoint(weight * u_t)

    def ke_cell(self, u_e: torch.Tensor) -> torch.Tensor:
        """
        Edge-normal velocity → cell-center kinetic energy per unit mass.

        Input:  [..., nEdges, nLayers]
        Output: [..., nCells, nLayers]   (units: m²/s²)

        KE[..., i, k] = (1/cell_area[i])
                        * sum_{e in i} cell_edge_area[i,e]
                        * 0.5 * (u_n[...,e,k]^2 + u_t[...,e,k]^2)

        Using the same TRiSK weights for u_t here and in tangential_velocity
        guarantees that grad(KE) and the rotation term are energy-conjugate.
        """
        u_t  = self.tangential_velocity(u_e)
        ke_e = 0.5 * (u_e ** 2 + u_t ** 2)               # [..., nEdges, nLayers]
        gathered = ke_e[..., self.safe_edges_on_cell, :]  # [..., nCells, maxEdges, nLayers]
        weights = self.cell_edge_area_masked[..., None]
        weighted = gathered * weights
        return weighted.sum(dim=-2) / self.cell_area[:, None].clamp_min(1e-30)

    def integrate_cell(self, q_c: torch.Tensor) -> torch.Tensor:
        """
        Integrate a cell field over the sphere.

        Input:  [..., nCells, nLayers]
        Output: [..., nLayers]   (weighted sum over cells)
        """
        return (q_c * self.cell_area[:, None]).sum(dim=-2)

    def integrate_dual(self, q_v: torch.Tensor) -> torch.Tensor:
        """
        Integrate a vertex field over the sphere.

        Input:  [..., nVertices, nLayers]
        Output: [..., nLayers]
        """
        return (q_v * self.dual_area[:, None]).sum(dim=-2)

    def cell_wind_vector(self, u_e: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct cell-center Cartesian wind vector from edge-normal velocities.

        Input:  [..., nEdges,         nLayers]
        Output: [..., nCells, nLayers, 3]

        Uses the least-squares projection precomputed at init.
        """
        gathered = u_e[..., self.safe_edges_on_cell, :]   # [..., nCells, maxEdges, nLayers]
        valid    = self.valid_edges_on_cell[..., None]
        # The reconstruction weights are built from cell-local outward normals
        # sign(cell, edge) * edge_normal[edge].  Convert the globally oriented
        # edge-normal velocity to that same local convention before applying
        # the weights.
        gathered = gathered * self.edgeSignOnCell[..., None]
        gathered = torch.where(valid, gathered, torch.zeros_like(gathered))
        # cell_rec_w1/w2: [nCells, maxEdges] -> [nCells, maxEdges, 1]
        coeff1 = (gathered * self.cell_rec_w1[..., None]).sum(dim=-2)  # [..., nCells, nLayers]
        coeff2 = (gathered * self.cell_rec_w2[..., None]).sum(dim=-2)
        # cell_rec_basis1: [nCells, 3]
        # coeff1[..., None]: [..., nCells, nLayers, 1]
        # basis[:, None, :]: [nCells, 1, 3]
        # product: [..., nCells, nLayers, 3]
        b1 = self.cell_rec_basis1[:, None, :]   # [nCells, 1, 3]
        b2 = self.cell_rec_basis2[:, None, :]
        return coeff1[..., None] * b1 + coeff2[..., None] * b2

    # ------------------------------------------------------------------
    # Simple diagnostic
    # ------------------------------------------------------------------

    def _print_quality_summary(self) -> None:
        dv = self.dv_edge.numpy()
        dc = self.dc_edge.numpy()
        ca = self.cell_area.numpy()
        R = self.radius
        print(f"  nCells={self.nCells}  nEdges={self.nEdges}  nVertices={self.nVertices}")
        print(f"  dcEdge  min={dc.min():.3e}  max={dc.max():.3e}  CV={dc.std()/dc.mean():.3f}  (m)")
        dv_min, dv_mean, dv_max = dv.min(), dv.mean(), dv.max()
        ratio_str = f"{dv_max/dv_min:.1f}" if dv_min > 0 else "inf (DEGENERATE)"
        print(f"  dvEdge  min={dv_min:.3e}  max={dv_max:.3e}  CV={dv.std()/dv_mean:.3f}  max/min={ratio_str}  (m)")
        n_zero_dv = int((dv < 1.0).sum())
        if n_zero_dv > 0:
            print(f"  WARNING: {n_zero_dv} dual edges have length < 1 m (near-zero).")
            print(f"           Degenerate Delaunay triangles with near-coincident circumcenters.")
            print(f"           Increase lloyd_iters to fix this.")
        print(f"  cellArea min={ca.min():.3e}  max={ca.max():.3e}  CV={ca.std()/ca.mean():.3f}  (m^2)")
        n_tiny = int((ca < 1e6).sum())
        if n_tiny > 0:
            print(f"  WARNING: {n_tiny} cells have area < 1e6 m^2 (near-degenerate).")
        print(f"  cell_area sum / (4*pi*R^2) = {ca.sum() / (4*math.pi*R**2):.8f}")


# ---------------------------------------------------------------------------
# Post-init: build edge_side_area and cell_wind_vector weights
# These are separated from __init__ only because they need the buffers
# already registered; they are called at the end of __init__ via _finalize.
# ---------------------------------------------------------------------------

def _build_edge_side_area(mesh: VoronoiMesh) -> None:
    """
    For each edge, find the cell_edge_area contribution from each of its
    two adjacent cells. Stored as edge_side_area[nEdges, 2].

    edge_side_area[e, 0] = cell_edge_area[cellsOnEdge[e,0], local_pos_of_e]
    edge_side_area[e, 1] = cell_edge_area[cellsOnEdge[e,1], local_pos_of_e]
    """
    n_edges = mesh.nEdges
    side_area = torch.zeros((n_edges, 2), dtype=torch.float64)

    edges_on_cell = mesh.edgesOnCell          # [nCells, maxEdges]
    cells_on_edge = mesh.cellsOnEdge          # [nEdges, 2]
    cell_edge_area = mesh.cell_edge_area      # [nCells, maxEdges]
    n_edges_on_cell = mesh.nEdgesOnCell       # [nCells]

    for ci in range(mesh.nCells):
        ne = int(n_edges_on_cell[ci].item())
        for k in range(ne):
            e = int(edges_on_cell[ci, k].item())
            c0 = int(cells_on_edge[e, 0].item())
            side = 0 if ci == c0 else 1
            side_area[e, side] = cell_edge_area[ci, k]

    mesh.register_buffer("edge_side_area", side_area)


def _build_cell_wind_weights(mesh: VoronoiMesh, reg: float = 1e-10) -> None:
    """
    Precompute least-squares weights for cell_wind_vector reconstruction.

    For cell i, we solve:
        min || A w - e_k ||^2   where A[local_e, :] = edge_normal[e] projected
                                       onto the two tangent basis vectors.

    Stores:
      cell_rec_basis1  [nCells, 3]
      cell_rec_basis2  [nCells, 3]
      cell_rec_w1      [nCells, maxEdges]   weight for basis1 component
      cell_rec_w2      [nCells, maxEdges]   weight for basis2 component
    """
    n_cells = mesh.nCells
    max_edges = mesh.maxEdges
    cell_xyz = mesh.cell_xyz                    # [nCells, 3]

    # Build tangent bases at each cell center
    basis1 = torch.zeros((n_cells, 3), dtype=torch.float64)
    basis2 = torch.zeros((n_cells, 3), dtype=torch.float64)
    for ci in range(n_cells):
        cx = cell_xyz[ci]
        ref = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        if abs(float((ref * cx).sum())) > 0.9:
            ref = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        e1 = ref - (ref * cx).sum() * cx
        e1 = e1 / e1.norm().clamp_min(1e-15)
        e2 = torch.linalg.cross(cx, e1)
        e2 = e2 / e2.norm().clamp_min(1e-15)
        basis1[ci] = e1
        basis2[ci] = e2

    w1 = torch.zeros((n_cells, max_edges), dtype=torch.float64)
    w2 = torch.zeros((n_cells, max_edges), dtype=torch.float64)

    for ci in range(n_cells):
        ne = int(mesh.nEdgesOnCell[ci].item())
        edge_ids = mesh.edgesOnCell[ci, :ne]
        signs = mesh.edgeSignOnCell[ci, :ne]
        normals = signs[:, None] * mesh.edge_normal[edge_ids]  # [ne, 3]
        b1 = basis1[ci]
        b2 = basis2[ci]
        a1 = (normals * b1).sum(dim=1)  # [ne]
        a2 = (normals * b2).sum(dim=1)  # [ne]
        A = torch.stack([a1, a2], dim=1)   # [ne, 2]
        gram = A.T @ A + reg * torch.eye(2, dtype=torch.float64)
        sol = torch.linalg.solve(gram, A.T)   # [2, ne]
        w1[ci, :ne] = sol[0]
        w2[ci, :ne] = sol[1]

    mesh.register_buffer("cell_rec_basis1", basis1)
    mesh.register_buffer("cell_rec_basis2", basis2)
    mesh.register_buffer("cell_rec_w1", w1)
    mesh.register_buffer("cell_rec_w2", w2)


# Patch __init__ to call the finalizers after buffers are registered
_original_init = VoronoiMesh.__init__

def _patched_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    _build_edge_side_area(self)
    _build_cell_wind_weights(self)

VoronoiMesh.__init__ = _patched_init


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

# Buffers written verbatim to the netCDF file.
# Everything else is either trivially derived (safe/valid index arrays)
# or fast to recompute from these (cell wind reconstruction weights).
_FLOAT_BUFFERS = [
    # Unit-sphere / geometry
    "cell_xyz",                   # [nCells,    3]
    "vertex_xyz",                 # [nVertices, 3]
    "edge_xyz",                   # [nEdges,    3]  midpoint of dual edge, if present
    "edge_normal",                # [nEdges,    3]
    "edge_tangent",               # [nEdges,    3]

    # Metric geometry
    "dc_edge",                    # [nEdges]
    "dv_edge",                    # [nEdges]
    "cell_area",                  # [nCells]
    "dual_area",                  # [nVertices]
    "cell_edge_area",             # [nCells,    maxEdges]
    "edge_side_area",             # [nEdges,    2]

    # Tangential reconstruction
    "tan_weights",                # [nEdges,    tan_stencil_size]

    # Exact-Thuburn / projected rotation operator
    "rotation_weights",           # [nEdges,    rotation_stencil_size]
    "rotation_row_sum",           # [nEdges]
    "rotation_row_abs_sum",       # [nEdges]
    "rotation_skew_defect_max",   # scalar, optional diagnostic

    # Exact-Thuburn diagnostics
    "thuburn_corner_area",         # [nCells,    maxEdges]
    "thuburn_corner_fraction",     # [nCells,    maxEdges]
    "thuburn_corner_fraction_sum", # [nCells]
    "thuburn_endpoint_sign",       # [nEdges,    2]
    "thuburn_edge_sign",           # [nEdges]

    # Signs
    "edgeSignOnCell",             # [nCells,    maxEdges]
    "edgeSignOnVertex",           # [nVertices, 3]
]

_INT_BUFFERS = [
    # Connectivity
    "cellsOnEdge",                # [nEdges,    2]
    "verticesOnEdge",             # [nEdges,    2]
    "cellsOnVertex",              # [nVertices, 3]
    "edgesOnVertex",              # [nVertices, 3]
    "verticesOnCell",             # [nCells,    maxEdges]
    "edgesOnCell",                # [nCells,    maxEdges]
    "nEdgesOnCell",               # [nCells]

    # Tangential reconstruction
    "tan_stencil",                # [nEdges,    tan_stencil_size]

    # Projected rotation operator
    "rotation_stencil",           # [nEdges,    rotation_stencil_size]
    "rotation_stencil_count",     # [nEdges]
]
def _rebuild_derived_buffers(mesh: VoronoiMesh) -> None:
    """
    Rebuild derived buffers that are not saved to disk.

    Called after load() once all saved buffers are registered.

    Rebuilds:
      - safe_* / valid_* index arrays from padded stencil/connectivity arrays.
      - cell wind reconstruction weights from geometry/connectivity.
    """
    mesh.register_buffer(
        "safe_edges_on_cell",
        torch.where(
            mesh.edgesOnCell >= 0,
            mesh.edgesOnCell,
            torch.zeros_like(mesh.edgesOnCell),
        ),
    )
    mesh.register_buffer(
        "safe_cells_on_edge",
        torch.where(
            mesh.cellsOnEdge >= 0,
            mesh.cellsOnEdge,
            torch.zeros_like(mesh.cellsOnEdge),
        ),
    )
    mesh.register_buffer("valid_edges_on_cell", mesh.edgesOnCell >= 0)
    mesh.register_buffer(
        "cell_edge_area_masked",
        torch.where(
            mesh.edgesOnCell >= 0,
            mesh.cell_edge_area,
            torch.zeros_like(mesh.cell_edge_area),
        ),
    )

    # Tangential reconstruction derived masks
    mesh.register_buffer(
        "safe_tan_stencil",
        torch.where(
            mesh.tan_stencil >= 0,
            mesh.tan_stencil,
            torch.zeros_like(mesh.tan_stencil),
        ),
    )
    mesh.register_buffer("valid_tan_stencil", mesh.tan_stencil >= 0)
    mesh.register_buffer(
        "tan_weights_masked",
        torch.where(
            mesh.tan_stencil >= 0,
            mesh.tan_weights,
            torch.zeros_like(mesh.tan_weights),
        ),
    )

    # Projected rotation operator derived masks
    mesh.register_buffer(
        "safe_rotation_stencil",
        torch.where(
            mesh.rotation_stencil >= 0,
            mesh.rotation_stencil,
            torch.zeros_like(mesh.rotation_stencil),
        ),
    )
    mesh.register_buffer("valid_rotation_stencil", mesh.rotation_stencil >= 0)
    mesh.register_buffer(
        "rotation_weights_masked",
        torch.where(
            mesh.rotation_stencil >= 0,
            mesh.rotation_weights,
            torch.zeros_like(mesh.rotation_weights),
        ),
    )
    mesh.register_buffer(
        "coriolis_unit_vertex",
        2.0 * torch.sin(torch.asin(mesh.vertex_xyz[:, 2].clamp(-1.0, 1.0))),
    )

    # Rebuild cell-centered vector reconstruction.
    _build_cell_wind_weights(mesh)


def _dim_names(arr: torch.Tensor, name: str) -> tuple:
    """Assign human-readable netCDF dimension names."""
    s = tuple(arr.shape)

    if arr.ndim == 0:
        return ()

    d0 = (
        "cell" if s[0] == mesh.nCells
        else "edge" if s[0] == mesh.nEdges
        else "vertex" if s[0] == mesh.nVertices
        else f"d0_{name}"
    )

    if arr.ndim == 1:
        return (d0,)

    if arr.ndim != 2:
        return tuple(f"d{i}_{name}" for i in range(arr.ndim))

    s1 = s[1]
    if s1 == 3:
        d1 = "three"
    elif s1 == 2:
        d1 = "two"
    elif s1 == mesh.maxEdges:
        d1 = "max_edges"
    elif name in ("tan_stencil", "tan_weights"):
        d1 = "tan_stencil_size"
    elif name in ("rotation_stencil", "rotation_weights"):
        d1 = "rotation_stencil_size"
    else:
        d1 = f"d1_{name}"

    return (d0, d1)



def _save_mesh(mesh: VoronoiMesh, path) -> None:
    """
    Save mesh to a single netCDF4 file.

    Saves all available float and int buffers in _FLOAT_BUFFERS / _INT_BUFFERS
    plus scalar metadata.
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError(
            "xarray is required for mesh save/load: pip install xarray netcdf4"
        )
    from pathlib import Path as _Path

    def _dim_names(arr: torch.Tensor, name: str) -> tuple:
        s = tuple(arr.shape)

        if arr.ndim == 0:
            return ()

        d0 = (
            "cell" if s[0] == mesh.nCells
            else "edge" if s[0] == mesh.nEdges
            else "vertex" if s[0] == mesh.nVertices
            else f"d0_{name}"
        )

        if arr.ndim == 1:
            return (d0,)

        if arr.ndim != 2:
            return tuple(f"d{i}_{name}" for i in range(arr.ndim))

        s1 = s[1]
        if s1 == 3:
            d1 = "three"
        elif s1 == 2:
            d1 = "two"
        elif s1 == mesh.maxEdges:
            d1 = "max_edges"
        elif name in ("tan_stencil", "tan_weights"):
            d1 = "tan_stencil_size"
        elif name in ("rotation_stencil", "rotation_weights"):
            d1 = "rotation_stencil_size"
        else:
            d1 = f"d1_{name}"

        return (d0, d1)

    data_vars = {}

    for name in _FLOAT_BUFFERS + _INT_BUFFERS:
        if not hasattr(mesh, name):
            # Prefer strictness once all meshes are regenerated:
            # raise AttributeError(f"Mesh is missing required buffer {name!r}")
            continue

        arr = getattr(mesh, name).detach().cpu()
        data_vars[name] = xr.DataArray(
            arr.numpy(),
            dims=_dim_names(arr, name),
        )

    corner_sum = mesh.thuburn_corner_fraction_sum.detach().to(dtype=torch.float64)
    corner_err = (corner_sum - 1.0).abs()
    rotation_skew_defect = float(mesh.rotation_skew_defect_max.detach().reshape(-1)[0].item())

    ds = xr.Dataset(
        data_vars,
        attrs={
            "nCells": mesh.nCells,
            "nEdges": mesh.nEdges,
            "nVertices": mesh.nVertices,
            "maxEdges": mesh.maxEdges,
            "radius": mesh.radius,
            "mesh_schema_version": 2,
            "mesh_operator_version": MESH_OPERATOR_VERSION,
            "exact_thuburn_version": EXACT_THUBURN_VERSION,
            "thuburn_corner_convention": THUBURN_CORNER_CONVENTION,
            "thuburn_implementation": THUBURN_IMPLEMENTATION,
            "thuburn_corner_fraction_mean": float(corner_sum.mean().item()),
            "thuburn_corner_fraction_max_abs_err": float(corner_err.max().item()),
            "rotation_skew_defect_max": rotation_skew_defect,
        },
    )
    try:
        ds.to_netcdf(_Path(path), format="NETCDF4", engine="netcdf4")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Saving mesh caches requires the netCDF4 Python package. "
            "Install the project environment from environment.yml or run "
            "`pip install netCDF4`."
        ) from exc

def _load_mesh(path, *, allow_legacy_thuburn: bool = False) -> "VoronoiMesh":
    """
    Load a VoronoiMesh from a netCDF4 file written by mesh.save(path).

    Reads saved buffers then calls _rebuild_derived_buffers().
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError(
            "xarray is required for mesh save/load: pip install xarray netcdf4"
        )
    from pathlib import Path as _Path

    ds = xr.open_dataset(_Path(path))
    attrs = dict(ds.attrs)

    def _fail_stale(reason: str) -> None:
        if allow_legacy_thuburn:
            return
        raise RuntimeError(
            f"Refusing to load stale/unknown Thuburn mesh cache {path!s}: {reason}. "
            "Regenerate the mesh cache with "
            "python -m diagnostics.drivers.regenerate_mesh_cache, or "
            "call _load_mesh(..., allow_legacy_thuburn=True) only for legacy audits."
        )

    if int(attrs.get("mesh_operator_version", -1)) != MESH_OPERATOR_VERSION:
        _fail_stale(
            f"mesh_operator_version={attrs.get('mesh_operator_version')!r}, "
            f"expected {MESH_OPERATOR_VERSION}"
        )
    if int(attrs.get("exact_thuburn_version", -1)) != EXACT_THUBURN_VERSION:
        _fail_stale(
            f"exact_thuburn_version={attrs.get('exact_thuburn_version')!r}, "
            f"expected {EXACT_THUBURN_VERSION}"
        )
    if str(attrs.get("thuburn_corner_convention", "")) != THUBURN_CORNER_CONVENTION:
        _fail_stale(
            f"thuburn_corner_convention={attrs.get('thuburn_corner_convention')!r}, "
            f"expected {THUBURN_CORNER_CONVENTION!r}"
        )

    mesh = VoronoiMesh.__new__(VoronoiMesh)
    nn.Module.__init__(mesh)

    mesh.nCells = int(ds.attrs["nCells"])
    mesh.nEdges = int(ds.attrs["nEdges"])
    mesh.nVertices = int(ds.attrs["nVertices"])
    mesh.maxEdges = int(ds.attrs["maxEdges"])
    mesh.radius = float(ds.attrs["radius"])

    missing_float = []
    missing_int = []

    for name in _FLOAT_BUFFERS:
        if name not in ds:
            missing_float.append(name)
            continue
        mesh.register_buffer(
            name,
            torch.as_tensor(ds[name].values, dtype=torch.float64),
        )

    for name in _INT_BUFFERS:
        if name not in ds:
            missing_int.append(name)
            continue
        mesh.register_buffer(
            name,
            torch.as_tensor(ds[name].values, dtype=torch.int64),
        )

    ds.close()

    # Required for current dynamics. Old mesh files without these need regeneration.
    required_current = [
        "rotation_stencil",
        "rotation_weights",
        "rotation_stencil_count",
        "tan_stencil",
        "tan_weights",
        "edge_side_area",
    ]
    missing_required = [name for name in required_current if not hasattr(mesh, name)]
    if missing_required:
        raise RuntimeError(
            "Loaded mesh is missing buffers required by the projected-rotation "
            f"schema: {missing_required}. Regenerate and resave the mesh."
        )

    if hasattr(mesh, "thuburn_corner_fraction_sum"):
        max_corner_err = float((mesh.thuburn_corner_fraction_sum.detach().to(dtype=torch.float64) - 1.0).abs().max().item())
        if max_corner_err > 1.0e-10:
            _fail_stale(
                f"thuburn_corner_fraction_sum max abs error is {max_corner_err:.6e}; "
                "this cache may contain the previous/current corner overcount bug"
            )

    _rebuild_derived_buffers(mesh)

    # Optional metadata flags used by newer dynamics branches.
    mesh.rotation_scheme_name = "thuburn_exact_m_skew_projected"
    mesh.mpas_like_enabled = True
    mesh.pv_edge_reconstruction = "centered"
    mesh.mass_pv_flux_transport_space = "flux_space"
    mesh.dual_mass_mode = "equal_cell_average"
    mesh.mesh_operator_version = MESH_OPERATOR_VERSION
    mesh.exact_thuburn_version = EXACT_THUBURN_VERSION
    mesh.thuburn_corner_convention = THUBURN_CORNER_CONVENTION
    mesh.thuburn_implementation = THUBURN_IMPLEMENTATION

    return mesh


# Attach to the class.  save() is an instance method; load() is a static
# factory (classmethod-like) since it creates a new instance from a file.
VoronoiMesh.save = _save_mesh
VoronoiMesh.load = staticmethod(_load_mesh)
