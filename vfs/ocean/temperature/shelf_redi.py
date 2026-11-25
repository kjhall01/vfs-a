"""Apply split-face Redi diffusion to the coupled ocean's native inventories.

P averages native concentrations onto the partial-cell grid by physical depth.
The native operator is -(RP)^T M (RP), so a constant field has zero flux,
pairwise inventory sums close, and diffusion cannot increase tracer variance.
Only increments cross grids. The native state never undergoes a round-trip
remap. GM samples a closed shelf streamfunction at the physical interfaces
of the prognostic grid. Both exchanges use the model-owned endpoint budget.
"""

import torch

from vfs.ocean.partial_cells import WetFaces, reference_thickness
from vfs.ocean.partial_cell_eddy import ShelfRedi
from vfs.ocean.temperature.eddy import GMRedi


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
        self.faces = faces
        self.depth = depth
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


class _FaceSlots:
    """Index packed physical intersections into edge-by-segment diagnostic arrays."""

    def __init__(self, faces):
        self.faces = faces
        self.shape = (faces.n_edges, 2 * faces.cell_shape[1])
        count = faces.edge.bincount(minlength=faces.n_edges)
        start = count.cumsum(0) - count
        self.slot = (
            torch.arange(len(faces.edge), device=count.device) - start[faces.edge]
        )
        self.width = self.pack(faces.opening)

    def pack(self, values):
        out = values.new_zeros(self.shape)
        out[self.faces.edge, self.slot] = values
        return out


class _ShelfCirculation:
    def __init__(self, model, state, nominal, eddy):
        op = CoupledShelfRedi(model, state, nominal, eddy)
        self.operator = op
        self.shelf = _FaceSlots(op.faces)
        s = state.ocean
        h = torch.where(model.ocean.domain.wet_cell[:, None], s.h, 0.0)
        self.h = h
        self.length = s.mesh.dv_edge
        self.native = _FaceSlots(WetFaces.from_columns(op.depth, h, s.mesh.cellsOnEdge))
        # Both partitions start at the deeper free surface and stop at the
        # shallower bed. Interpolate Psi there before differencing it.
        self.source_bottom = self.shelf.width.cumsum(1)
        self.target_bottom = self.native.width.cumsum(1)
        self.bin = torch.searchsorted(self.source_bottom, self.target_bottom).clamp_max(
            self.source_bottom.shape[1] - 1
        )

    def streamfunction(self, transport):
        q = self.shelf.pack(transport)
        wet = self.shelf.width > 0
        zero = torch.zeros_like(q[:, :1])
        source = torch.cat(
            (zero, torch.where(wet[:, 1:], 0.5 * (q[:, :-1] + q[:, 1:]), 0.0), zero), 1
        )
        idx = self.bin
        width = self.shelf.width.gather(1, idx)
        remaining = self.source_bottom.gather(1, idx) - self.target_bottom
        weight = (1.0 - remaining / torch.where(width > 0, width, 1.0)).clamp(0.0, 1.0)
        sampled = torch.lerp(source.gather(1, idx), source.gather(1, idx + 1), weight)
        # Impose exact zero at each last wet interface, including padded coasts.
        return torch.cat(
            (
                zero,
                torch.where(self.native.width[:, 1:] > 0, sampled[:, :-1], 0.0),
                zero,
            ),
            1,
        )

    def interface_min(self, layer):
        faces = self.native.faces
        q = layer.flatten()
        values = torch.ones(self.native.shape, dtype=q.dtype, device=q.device)
        values[faces.edge, self.native.slot] = torch.minimum(
            q[faces.left], q[faces.right]
        )
        return torch.where(
            self.native.width[:, 1:] > 0,
            torch.minimum(values[:, :-1], values[:, 1:]),
            1,
        ).to(q.dtype)

    def carriers(self, psi):
        faces = self.native.faces
        horizontal = self.length[:, None] * psi.diff(dim=1)
        transport = horizontal[faces.edge, self.native.slot]
        # The discrete cumulative horizontal convergence gives the same
        # closed bolus volume circulation on every receiving column.
        net = faces.exchange(transport)
        wet = self.h[:, 1:] > 0
        down = torch.where(wet, net.cumsum(1)[:, :-1], 0.0)
        ids = torch.arange(self.h.numel(), device=net.device).reshape_as(net)
        i = torch.cat((faces.left, ids[:, :-1][wet]))
        j = torch.cat((faces.right, ids[:, 1:][wet]))
        zero = torch.zeros_like(net[:, :1])
        return (
            horizontal,
            torch.cat((zero, down, zero), 1),
            i,
            j,
            torch.cat((transport, down[wet])),
        )


class ShelfGMRedi(GMRedi):
    def __init__(self, nominal, **kwargs):
        super().__init__(**kwargs)
        self.nominal = nominal

    def _metrics(self, model, state):
        return _ShelfCirculation(model, state, self.nominal, self)

    def _gm_diffusivity(self, state, metrics):
        if self.energy_closure is None:
            return None
        edge_energy = metrics.operator.face_mean(
            state.specific_eddy_energy()[..., None]
        )[:, 0]
        return self.energy_closure.from_edge_energy(edge_energy)

    def _carriers(self, psi, metrics, h):
        return metrics.carriers(psi)

    def _geometry(self, model, state, *, part="all", metrics=None, gm_diffusivity=None):
        metrics = self._metrics(model, state) if metrics is None else metrics
        if part == "redi":
            op = CoupledShelfRedi(model, state, self.nominal, self)
            return (
                op.rows.slope,
                None,
                None,
                None,
                None,
                None,
                None,
                op,
                op.rows.measure,
            )
        op = metrics.operator
        diffusivity = (
            self._gm_diffusivity(state, metrics)
            if gm_diffusivity is None
            else gm_diffusivity
        )
        slope = op.rows.slope
        psi = metrics.streamfunction(
            slope * (self.kappa_gm if diffusivity is None else diffusivity)
        )
        return slope, psi, *metrics.carriers(psi), op, op.rows.measure
