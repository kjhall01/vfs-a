"""Apply split-face Redi diffusion to the coupled ocean's native inventories.

P averages native concentrations onto the partial-cell grid by physical depth.
The native operator is -(RP)^T M (RP), so a constant field has zero flux,
pairwise inventory sums close, and diffusion cannot increase tracer variance.
Only increments cross grids. The native state never undergoes a round-trip
remap. GM and the finite endpoint Hamiltonian settlement remain model-owned.
"""

import torch

from vfs.ocean.partial_cells import WetFaces, reference_thickness
from vfs.ocean.partial_cell_eddy import ShelfRedi
from eddy import GMRedi


class CoupledShelfRedi:
    def __init__(self, model, state, nominal, eddy):
        s = state.ocean
        mesh, domain = s.mesh, model.ocean.domain
        h = torch.where(domain.wet_cell[:, None], s.h, 0.0)
        depth = torch.where(domain.wet_cell, domain.depth[:, 0], 0.0)
        ref = reference_thickness(depth, nominal.to(h))
        shelf = ref * (h.sum(1) / torch.where(depth > 0, depth, 1.0))[:, None]
        self.shape = h.shape
        n = len(h)
        ids = torch.arange(n, device=h.device)
        # The same intersection routine also gives the vertical overlaps P.
        overlap = WetFaces.from_columns(
            depth.repeat(2), torch.cat((h, shelf)), torch.stack((ids, ids + n), 1)
        )
        self.native = overlap.left
        self.shelf = overlap.right - h.numel()
        self.fraction = overlap.opening / shelf.flatten()[self.shelf]
        faces = WetFaces.from_columns(depth, shelf, mesh.cellsOnEdge)
        self.left, self.right = faces.left, faces.right
        buoyancy = (
            -s.column.gravity
            / s.column.rho0
            * model.config.anomaly(s.temperature(), s.salinity(), torch.zeros_like(h))
        )
        self.rows = ShelfRedi.build(
            faces,
            depth,
            shelf,
            self.project(buoyancy[..., None])[..., 0],
            mesh.dc_edge,
            mesh.dv_edge,
            eddy.kappa_redi,
            max_slope=eddy.max_slope,
            min_n2=eddy.min_n2,
        )

    def project(self, x):
        flat = x.reshape(-1, x.shape[-1])
        return (
            torch.zeros_like(flat)
            .index_add(0, self.shelf, self.fraction[:, None] * flat[self.native])
            .reshape_as(x)
        )

    def face_mean(self, x):
        projected = self.project(x).reshape(-1, x.shape[-1])
        return 0.5 * (projected[self.left] + projected[self.right])

    def apply(self, x):
        q = self.project(x).reshape(-1, x.shape[-1])
        return (self.rows.weights[..., None] * q[self.rows.owners]).sum(1)

    def transpose(self, flux, *, absolute=False):
        weights = self.rows.weights.abs() if absolute else self.rows.weights
        zero = flux.new_zeros((self.shape.numel(), flux.shape[-1]))
        shelf = zero.index_add(
            0,
            self.rows.owners.flatten(),
            (weights[..., None] * flux[:, None, :]).reshape(-1, flux.shape[-1]),
        )
        return zero.index_add(
            0, self.native, self.fraction[:, None] * shelf[self.shelf]
        ).reshape(*self.shape, -1)

    def step_bound(self, measure):
        return self.transpose(
            (measure * self.rows.weights.abs().sum(1))[:, None], absolute=True
        )[..., 0]

    def positive_flux_scale(self, transfer, amounts):
        demand = self.transpose(transfer.abs(), absolute=True)
        ratio = (amounts / torch.where(demand > 0, demand, 1.0)).clamp(max=1.0)
        ratio = torch.where(demand > 0, ratio, 1.0)
        ratio = torch.where(
            ratio < 1, ratio * (1 - 64 * torch.finfo(ratio.dtype).eps), ratio
        )
        flat = ratio.reshape(-1, ratio.shape[-1])
        allowance = torch.ones_like(flat).scatter_reduce(
            0,
            self.shelf[:, None].expand(-1, flat.shape[1]),
            flat[self.native],
            reduce="amin",
        )
        donors = self.rows.weights[..., None] * transfer[:, None, :] < 0
        return torch.where(donors, allowance[self.rows.owners], 1.0).amin(1)


class ShelfGMRedi(GMRedi):
    def __init__(self, nominal, **kwargs):
        super().__init__(**kwargs)
        self.nominal = nominal

    def _geometry(self, model, state, *, part="all", **kwargs):
        if part != "redi":
            return super()._geometry(model, state, part=part, **kwargs)
        op = CoupledShelfRedi(model, state, self.nominal, self)
        return op.rows.slope, None, None, None, None, None, None, op, op.rows.measure
