"""Ocean domain masks and closed-coast semantics."""

from __future__ import annotations

import torch
import torch.nn as nn

from .constants import OCEAN_S0, OCEAN_T0
from .helpers import _normalize_tracer_tensor


class OceanDomain(nn.Module):
    """Static wet/land mask and closed-coast semantics for VFS-O.

    The mesh topology remains global and rectangular. Land cells keep finite
    dummy state values for tensor shape stability, while dynamics and physics
    operate only on wet cells and wet-wet edges. Wet-land edges are closed
    coasts with no normal transport in this first implementation.
    """

    def __init__(self, mesh, wet_cell: torch.Tensor, *, depth: torch.Tensor | None = None) -> None:
        super().__init__()
        n_cells = int(mesh.nCells)
        n_edges = int(mesh.nEdges)
        wet = torch.as_tensor(wet_cell, dtype=torch.bool).reshape(-1)
        if tuple(wet.shape) != (n_cells,):
            raise ValueError(f"wet_cell must have shape ({n_cells},), got {tuple(wet.shape)}")
        if not bool(wet.any().item()):
            raise ValueError("OceanDomain requires at least one wet cell")

        self.mesh = mesh
        cells_on_edge = mesh.cellsOnEdge.to(device=wet.device)
        c0 = cells_on_edge[:, 0]
        c1 = cells_on_edge[:, 1]
        valid0 = c0 >= 0
        valid1 = c1 >= 0
        wet0 = torch.zeros((n_edges,), dtype=torch.bool, device=wet.device)
        wet1 = torch.zeros((n_edges,), dtype=torch.bool, device=wet.device)
        wet0[valid0] = wet[c0[valid0]]
        wet1[valid1] = wet[c1[valid1]]
        wet_edge = wet0 & wet1
        coast_edge = wet0 ^ wet1
        land_edge = ~(wet_edge | coast_edge)

        self.n_cells = n_cells
        self.n_edges = n_edges
        self.register_buffer("wet_cell", wet)
        self.register_buffer("wet_edge", wet_edge)
        self.register_buffer("active_edge", wet_edge.clone())
        self.register_buffer("coast_edge", coast_edge)
        self.register_buffer("land_edge", land_edge)
        if depth is not None:
            depth_tensor = torch.as_tensor(depth, dtype=torch.float64)
            if depth_tensor.ndim == 1:
                depth_tensor = depth_tensor[:, None]
            if tuple(depth_tensor.shape) != (n_cells, 1):
                raise ValueError(f"depth must have shape ({n_cells},) or ({n_cells}, 1), got {tuple(depth_tensor.shape)}")
            if not torch.isfinite(depth_tensor).all().item() or float(depth_tensor.min().item()) <= 0.0:
                raise ValueError("depth must be finite and strictly positive")
            self.register_buffer("depth", depth_tensor)
        else:
            self.depth = None

    @classmethod
    def global_ocean(cls, mesh, *, depth: torch.Tensor | None = None) -> "OceanDomain":
        return cls(mesh, torch.ones(int(mesh.nCells), dtype=torch.bool), depth=depth)

    def check_compatible(self, state: "OceanState") -> None:
        mesh = state._require_mesh()
        if int(mesh.nCells) != self.n_cells or int(mesh.nEdges) != self.n_edges:
            raise ValueError(
                "OceanDomain mesh size does not match OceanState: "
                f"domain=({self.n_cells}, {self.n_edges}), state=({int(mesh.nCells)}, {int(mesh.nEdges)})"
            )

    def _mask_like(self, value: torch.Tensor, mask: torch.Tensor, *, name: str) -> torch.Tensor:
        axis_size = int(mask.numel())
        axis = None
        preferred_axes = []
        if value.ndim >= 3:
            preferred_axes.append(value.ndim - 3)
        if value.ndim >= 2:
            preferred_axes.append(value.ndim - 2)
        if value.ndim >= 1:
            preferred_axes.append(value.ndim - 1)
        for idx in preferred_axes:
            if int(value.shape[idx]) == axis_size:
                axis = idx
                break
        if axis is None:
            for idx in range(value.ndim):
                if int(value.shape[idx]) == axis_size:
                    axis = idx
                    break
        if axis is None:
            raise ValueError(f"cannot apply {name} mask to tensor with shape {tuple(value.shape)}")
        shape = [1] * value.ndim
        shape[axis] = axis_size
        return mask.to(device=value.device).reshape(shape)

    def cell_mask(self, value: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        mask = self._mask_like(value, self.wet_cell, name="cell")
        if dtype is None:
            dtype = value.dtype if value.is_floating_point() else torch.bool
        return mask.to(dtype=dtype, device=value.device)

    def edge_mask(self, value: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        mask = self._mask_like(value, self.active_edge, name="edge")
        if dtype is None:
            dtype = value.dtype if value.is_floating_point() else torch.bool
        return mask.to(dtype=dtype, device=value.device)

    def coast_mask(self, value: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        mask = self._mask_like(value, self.coast_edge, name="coast edge")
        if dtype is None:
            dtype = value.dtype if value.is_floating_point() else torch.bool
        return mask.to(dtype=dtype, device=value.device)

    def apply_cell(self, value: torch.Tensor) -> torch.Tensor:
        mask = self.cell_mask(value)
        if value.dtype == torch.bool:
            return value & mask
        return value * mask

    def apply_edge(self, value: torch.Tensor) -> torch.Tensor:
        mask = self.edge_mask(value)
        if value.dtype == torch.bool:
            return value & mask
        return value * mask

    def project_state(
        self,
        state: "OceanState",
        *,
        land_temperature: float = OCEAN_T0,
        land_salinity: float = OCEAN_S0,
    ) -> "OceanState":
        """Return a state projected onto the static closed-coast domain."""

        self.check_compatible(state)
        u_e, h, temp_h, salt_h = state._require_tensors()
        wet = self.cell_mask(h)
        edge = self.edge_mask(u_e)
        rest = state.column.rest_thickness_like(h)
        rest_fraction = rest / rest.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(h.dtype).tiny)
        dummy_h = rest_fraction * state.column.total_depth_like(h)
        h_proj = wet * h + (1.0 - wet) * dummy_h
        temp_proj = wet * temp_h + (1.0 - wet) * h_proj * float(land_temperature)
        salt_proj = wet * salt_h + (1.0 - wet) * h_proj * float(land_salinity)
        tracer_proj = None
        if state.tracer_h is not None:
            tracer_proj = self.apply_cell(_normalize_tracer_tensor(state.tracer_h, state.mesh.nCells, state.column.n_layers))
        return state.with_tensors(edge * u_e, h_proj, temp_proj, salt_proj, tracer_proj)

    def integrate_cell(self, state: "OceanState", cell_field: torch.Tensor) -> torch.Tensor:
        self.check_compatible(state)
        return state.mesh.integrate_cell(self.apply_cell(cell_field))

    def total_volume(self, state: "OceanState") -> torch.Tensor:
        return self.integrate_cell(state, state.h.sum(dim=-1, keepdim=True))

    def total_heat_content(self, state: "OceanState") -> torch.Tensor:
        return self.integrate_cell(state, state.temp_h.sum(dim=-1, keepdim=True))

    def total_salt_content(self, state: "OceanState") -> torch.Tensor:
        return self.integrate_cell(state, state.salt_h.sum(dim=-1, keepdim=True))

    def total_tracer_content(self, state: "OceanState") -> torch.Tensor | None:
        if state.tracer_h is None:
            return None
        tracer_h = _normalize_tracer_tensor(state.tracer_h, state.mesh.nCells, state.column.n_layers)
        return self.integrate_cell(state, tracer_h.sum(dim=-2))


_OCEAN_FIELD_TRAILING_SHAPES = {
    "u_e": ("nEdges", "nLayers"),
    "h": ("nCells", "nLayers"),
    "temp_h": ("nCells", "nLayers"),
    "salt_h": ("nCells", "nLayers"),
    "tracer_h": ("nCells", "nLayers", "nTracers"),
}
