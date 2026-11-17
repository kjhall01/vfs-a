"""Finite momentum exchange on the campaign mesh's very short Voronoi edges."""

from pathlib import Path
import sys

import pytest
import torch
from vfs import VoronoiMesh

sys.path.append(str(Path(__file__).resolve().parents[1]))
from workflows.momentum import KineticWindMap


@pytest.fixture(scope="module")
def mapping():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    return KineticWindMap(VoronoiMesh(n_cells=256, lloyd_iters=10, verbose=False))


def test_local_tangent_reproduction_and_independent_work_adjoint(mapping):
    mesh = mapping.mesh
    generator = torch.Generator().manual_seed(81)
    tangent = torch.randn(mesh.nCells, 3, generator=generator)
    tangent -= (tangent * mesh.cell_xyz).sum(-1, keepdim=True) * mesh.cell_xyz
    observations = (mesh.edge_normal[mesh.safe_edges_on_cell] * tangent[:, None]).sum(-1)
    fitted = (mapping.coefficients * observations[..., None]).sum(1)
    torch.testing.assert_close(fitted, tangent, rtol=4e-14, atol=4e-15)
    wind = torch.randn(mesh.nEdges, 3, generator=generator, requires_grad=True)
    force = torch.randn(mesh.nCells, 3, 3, generator=generator)
    work = (mapping.vector(wind) * force).sum()
    transpose = mapping.transpose(force)
    torch.testing.assert_close(work, (wind * transpose).sum(), rtol=2e-14, atol=1e-13)
    torch.testing.assert_close(torch.autograd.grad(work, wind)[0], transpose)


@pytest.mark.parametrize("variable_mass", [False, True])
def test_finite_drag_cannot_amplify_short_edge_mode(mapping, variable_mass):
    mesh = mapping.mesh
    # The actual campaign generators produce two ~1 km dual edges among
    # ~1000 km edges. Keep the geometry: do not remove these or alter its metric.
    assert mesh.dv_edge.max() / mesh.dv_edge.min() > 1000
    mass = torch.ones(mesh.nCells, 1)
    if variable_mass:
        mass *= torch.exp(0.5 * mesh.cell_xyz[:, 2:3])
    W = (mesh.cell_area[:, None] * mass).expand(-1, 3).flatten()
    M = (mesh.dc_edge * mesh.dv_edge) * mesh.cell_to_edge(mass)[:, 0]
    identity = torch.eye(mesh.nEdges)

    def gram(readout):
        R = readout(identity).permute(0, 2, 1).reshape(-1, mesh.nEdges)
        B = W.sqrt()[:, None] * R / M.sqrt()[None]
        return B.T @ B

    old = gram(mesh.cell_wind_vector)
    new = gram(mapping.vector)
    old_values, old_vectors = torch.linalg.eigh(old)
    new_values = torch.linalg.eigvalsh(new)
    assert old_values[-1] > 100  # reproduces the historical coupling defect
    # A complete cell-wind removal is u1=(I-M^-1 R^T W R)u. Its kinetic norm
    # cannot grow iff every eigenvalue of this symmetric Gram matrix is <=2.
    assert new_values[-1] < 2
    mode = old_vectors[:, -1]
    old_endpoint = mode - old @ mode
    new_endpoint = mode - new @ mode
    assert old_endpoint.square().sum() > 10000 * mode.square().sum()
    assert new_endpoint.square().sum() <= mode.square().sum()
