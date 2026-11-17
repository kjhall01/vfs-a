"""Shared cell/edge momentum map weighted by native kinetic-energy ownership.

The package's diagnostic least-squares wind gives short edges the same weight
as long ones. Its transpose is unsuitable for finite physics impulses: dividing
by the native edge mass then amplifies a short edge by 1/dv. Here each observation
is weighted by its contribution to the cell's native kinetic energy instead.
This changes the external coupling map, not the packaged dynamics or diagnostics.
"""

import torch


class KineticWindMap:
    def __init__(self, mesh):
        self.mesh = mesh
        edges = mesh.safe_edges_on_cell
        # I_ce is exactly the ownership used by native cell_to_edge and KE.
        share = mesh.cell_edge_area / mesh.edge_side_area.sum(-1)[edges]
        weight = (
            (mesh.dc_edge * mesh.dv_edge)[edges]
            * share
            / mesh.cell_area[:, None]
            * mesh.valid_edges_on_cell
        )
        basis = torch.stack((mesh.cell_rec_basis1, mesh.cell_rec_basis2), -1)
        normal = torch.einsum("cem,cmk->cek", mesh.edge_normal[edges], basis)
        gram = torch.einsum("ce,cei,cej->cij", weight, normal, normal)
        # A two-dimensional weighted least-squares fit reproduces a locally
        # constant tangent vector. No diagonal regularizer or velocity cap.
        fit = torch.linalg.solve(gram, (weight[..., None] * normal).transpose(1, 2))
        self.coefficients = torch.einsum("cmk,cke->cem", basis, fit)

    def vector(self, wind):
        """[edge,layer] normal velocity -> [cell,layer,3] tangent velocity."""
        return torch.einsum(
            "cem,cel->clm", self.coefficients, wind[self.mesh.safe_edges_on_cell]
        )

    def transpose(self, covector):
        """Exact work-adjoint; [cell,layer,3] impulse -> [edge,layer]."""
        value = torch.einsum("cem,clm->cel", self.coefficients, covector)
        return value.new_zeros((self.mesh.nEdges, covector.shape[1])).index_add(
            0, self.mesh.safe_edges_on_cell.flatten(), value.flatten(0, 1)
        )
