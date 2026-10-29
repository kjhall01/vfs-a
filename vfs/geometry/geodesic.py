"""Deterministic class-I refined-icosahedral point sets.

These helpers construct only the static coarse meshes used by the IMEX
preconditioner.  They do not constrain the prognostic VFS mesh, which may be
an arbitrary spherical Voronoi/MPAS mesh.
"""

from __future__ import annotations

import math
from collections.abc import Hashable

import numpy as np


def icosahedral_cell_count(frequency: int) -> int:
    """Return the primal-point count for a class-I refined icosahedron."""

    frequency = int(frequency)
    if frequency <= 0:
        raise ValueError(f"frequency must be positive, got {frequency}")
    return 10 * frequency * frequency + 2


def refined_icosahedron(frequency: int) -> tuple[np.ndarray, np.ndarray]:
    """Return unit-sphere primal points and outward triangular faces.

    The construction is the donor's class-I refinement, retained here so a
    main-VFS unstructured fine mesh can use identical static coarse levels.
    """

    frequency = int(frequency)
    if frequency <= 0:
        raise ValueError(f"frequency must be positive, got {frequency}")

    base_vertices, base_faces = _icosahedron()
    vertices: list[np.ndarray] = []
    vertex_ids: dict[Hashable, int] = {}
    faces: list[tuple[int, int, int]] = []

    def vertex_id(key: Hashable, xyz: np.ndarray) -> int:
        existing = vertex_ids.get(key)
        if existing is not None:
            return existing
        vertices.append(_normalize(xyz))
        new_id = len(vertices) - 1
        vertex_ids[key] = new_id
        return new_id

    for face_id, (a_id, b_id, c_id) in enumerate(base_faces):
        a, b, c = base_vertices[a_id], base_vertices[b_id], base_vertices[c_id]
        local: dict[tuple[int, int], int] = {}
        for i in range(frequency + 1):
            for j in range(frequency + 1 - i):
                weights = (frequency - i - j, i, j)
                local[(i, j)] = vertex_id(
                    _barycentric_key(face_id, (a_id, b_id, c_id), weights),
                    weights[0] * a + weights[1] * b + weights[2] * c,
                )
        for i in range(frequency):
            for j in range(frequency - i):
                faces.append((local[(i, j)], local[(i + 1, j)], local[(i, j + 1)]))
                if j < frequency - i - 1:
                    faces.append(
                        (local[(i + 1, j)], local[(i + 1, j + 1)], local[(i, j + 1)])
                    )

    xyz = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if int(xyz.shape[0]) != icosahedral_cell_count(frequency):
        raise RuntimeError("refined-icosahedron construction returned an unexpected point count")
    return xyz, _orient_faces_outward(xyz, triangles)


def _icosahedron() -> tuple[np.ndarray, np.ndarray]:
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            (-1.0, golden_ratio, 0.0),
            (1.0, golden_ratio, 0.0),
            (-1.0, -golden_ratio, 0.0),
            (1.0, -golden_ratio, 0.0),
            (0.0, -1.0, golden_ratio),
            (0.0, 1.0, golden_ratio),
            (0.0, -1.0, -golden_ratio),
            (0.0, 1.0, -golden_ratio),
            (golden_ratio, 0.0, -1.0),
            (golden_ratio, 0.0, 1.0),
            (-golden_ratio, 0.0, -1.0),
            (-golden_ratio, 0.0, 1.0),
        ],
        dtype=np.float64,
    )
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = np.asarray(
        [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ],
        dtype=np.int64,
    )
    return vertices, _orient_faces_outward(vertices, faces)


def _normalize(xyz: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(xyz))
    if norm <= 0.0:
        raise ValueError("cannot normalize a zero-length point")
    return np.asarray(xyz, dtype=np.float64) / norm


def _barycentric_key(
    face_id: int,
    face: tuple[int, int, int],
    weights: tuple[int, int, int],
) -> Hashable:
    active = [(cell, weight) for cell, weight in zip(face, weights, strict=True) if weight > 0]
    if len(active) == 1:
        return ("v", active[0][0])
    if len(active) == 2:
        (left, left_weight), (right, right_weight) = active
        if left > right:
            left, right = right, left
            left_weight, right_weight = right_weight, left_weight
        return ("e", left, right, left_weight)
    return ("f", int(face_id), *map(int, weights))


def _orient_faces_outward(xyz: np.ndarray, faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64).copy()
    a, b, c = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
    inward = np.sum(np.cross(b - a, c - a) * (a + b + c), axis=1) < 0.0
    faces[inward, 1], faces[inward, 2] = faces[inward, 2].copy(), faces[inward, 1].copy()
    return faces


__all__ = ["icosahedral_cell_count", "refined_icosahedron"]
