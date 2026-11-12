"""Topological curl/gradient identities; independent of rotation reconstruction."""
import pytest
import torch
from vfs import VoronoiMesh


@pytest.mark.parametrize('cells,lloyd', [(16,0),(64,0),(64,5),(256,2)])
def test_voronoi_curl_gradient_and_global_stokes(cells,lloyd):
    mesh=VoronoiMesh(n_cells=cells,lloyd_iters=lloyd,verbose=False).to(dtype=torch.float64)
    generator=torch.Generator().manual_seed(42)
    q=torch.randn(2,cells,3,dtype=torch.float64,generator=generator)
    circulation=mesh.curl(mesh.grad(q))*mesh.dual_area[:,None]
    torch.testing.assert_close(circulation,torch.zeros_like(circulation),atol=4e-15,rtol=0)
    u=torch.randn(2,mesh.nEdges,3,dtype=torch.float64,generator=generator)
    z=torch.randn(2,mesh.nVertices,3,dtype=torch.float64,generator=generator)
    curl=mesh.curl(u)
    total=(curl*mesh.dual_area[:,None]).sum(-2)
    scale=(curl.abs()*mesh.dual_area[:,None]).sum(-2)
    assert (total.abs() <= 64*torch.finfo(u.dtype).eps*scale).all()
    lhs=(curl*z*mesh.dual_area[:,None]).sum((-2,-1))
    rhs=(u*mesh.curl_adjoint(z)*(mesh.dc_edge*mesh.dv_edge)[:,None]).sum((-2,-1))
    torch.testing.assert_close(lhs,rhs,atol=1e-7,rtol=1e-13)
