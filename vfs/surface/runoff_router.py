"""Topography-derived runoff routing skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..core.registry import register_component
from vfs.field import Field
from .conservation import ConservativeTransfer
from .utils import as_cell_column, field_values


@dataclass(frozen=True)
class RunoffState:
    river_storage: Field | torch.Tensor

    @classmethod
    def zeros(
        cls,
        n_cells: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> "RunoffState":
        return cls(as_cell_column(0.0, n_cells=n_cells, dtype=dtype, device=device, name="river_storage"))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "river_storage",
            Field(
                as_cell_column(self.river_storage, n_cells=int(field_values(self.river_storage).numel()), name="river_storage").clamp_min(0.0),
                location="cell",
                vertical_location="surface",
                name="river_storage",
            ),
        )

    def to(self, *args, **kwargs) -> "RunoffState":
        return RunoffState(self.river_storage.to(*args, **kwargs))

    def clone(self) -> "RunoffState":
        return RunoffState(self.river_storage.clone())

    def detach(self) -> "RunoffState":
        return RunoffState(self.river_storage.detach())


@dataclass(frozen=True)
class RunoffRouteResult:
    state: RunoffState
    ocean_input_mass: torch.Tensor
    transfers: tuple[ConservativeTransfer, ...]
    diagnostics: dict[str, torch.Tensor]


class RunoffRouter(nn.Module):
    """Route dense per-cell runoff to terminal ocean outlets."""

    recipe_name = "runoff_router"

    def __init__(
        self,
        outlet_index: torch.Tensor,
        ocean_cell: torch.Tensor,
        *,
        downstream_index: torch.Tensor | None = None,
        mode: str = "instant_outlet",
        storage_timescale: float = 86400.0,
    ) -> None:
        super().__init__()
        outlet = torch.as_tensor(outlet_index, dtype=torch.long).reshape(-1)
        ocean = torch.as_tensor(ocean_cell, dtype=torch.bool).reshape(-1)
        if outlet.numel() != ocean.numel():
            raise ValueError("outlet_index and ocean_cell must have the same length")
        if mode not in ("instant_outlet", "routed_storage"):
            raise ValueError("mode must be 'instant_outlet' or 'routed_storage'")
        if bool(((outlet < 0) | (outlet >= int(outlet.numel()))).any().item()):
            raise ValueError("every cell must have a valid terminal ocean outlet")
        self.n_cells = int(outlet.numel())
        self.mode = str(mode)
        self.storage_timescale = float(storage_timescale)
        if self.storage_timescale <= 0.0:
            raise ValueError("storage_timescale must be positive")
        self.register_buffer("outlet_index", outlet)
        self.register_buffer("ocean_cell", ocean)
        if downstream_index is None:
            downstream_index = outlet
        self.register_buffer("downstream_index", torch.as_tensor(downstream_index, dtype=torch.long).reshape(-1))

    @classmethod
    def from_mesh_topography(
        cls,
        mesh: Any,
        topography: torch.Tensor,
        ocean_cell: torch.Tensor,
        *,
        mode: str = "instant_outlet",
        storage_timescale: float = 86400.0,
    ) -> "RunoffRouter":
        """Build a steepest-descent graph and terminal ocean outlet per cell."""

        topo = torch.as_tensor(topography, dtype=torch.float64)
        if topo.ndim == 2 and int(topo.shape[-1]) == 1:
            topo = topo[:, 0]
        topo = topo.reshape(-1)
        ocean = torch.as_tensor(ocean_cell, dtype=torch.bool).reshape(-1)
        n_cells = int(mesh.nCells)
        if topo.numel() != n_cells or ocean.numel() != n_cells:
            raise ValueError("topography and ocean_cell must match mesh.nCells")
        cells_on_edge = mesh.cellsOnEdge.detach().cpu().to(dtype=torch.long)
        neighbors: list[list[int]] = [[] for _ in range(n_cells)]
        for c0, c1 in cells_on_edge.tolist():
            if c0 >= 0 and c1 >= 0:
                neighbors[c0].append(c1)
                neighbors[c1].append(c0)
        downstream = torch.arange(n_cells, dtype=torch.long)
        for cell in range(n_cells):
            if bool(ocean[cell].item()):
                downstream[cell] = cell
                continue
            candidates = neighbors[cell]
            if not candidates:
                downstream[cell] = cell
                continue
            best = min(candidates, key=lambda idx: float(topo[idx].item()))
            downstream[cell] = int(best) if float(topo[best].item()) <= float(topo[cell].item()) else int(cell)

        outlet = torch.arange(n_cells, dtype=torch.long)
        ocean_indices = torch.nonzero(ocean, as_tuple=False).reshape(-1)
        for start in range(n_cells):
            if bool(ocean[start].item()):
                outlet[start] = start
                continue
            seen: set[int] = set()
            cell = start
            while cell not in seen:
                seen.add(cell)
                nxt = int(downstream[cell].item())
                if bool(ocean[nxt].item()):
                    cell = nxt
                    break
                if nxt == cell:
                    break
                cell = nxt
            if not bool(ocean[cell].item()):
                # Closed land depressions drain to the nearest ocean cell in
                # native-cell Euclidean distance. This preserves mass while
                # making unresolved lakes explicit future work.
                xyz = mesh.cell_xyz.detach().cpu().to(dtype=torch.float64)
                dist2 = (xyz[ocean_indices] - xyz[start]).square().sum(dim=-1)
                cell = int(ocean_indices[int(dist2.argmin().item())].item())
            outlet[start] = int(cell)
        return cls(outlet, ocean, downstream_index=downstream, mode=mode, storage_timescale=storage_timescale)

    def initial_state(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> RunoffState:
        return RunoffState.zeros(self.n_cells, dtype=dtype, device=device)

    def _scatter_to_outlet(self, mass: torch.Tensor) -> torch.Tensor:
        flat = as_cell_column(mass, n_cells=self.n_cells, dtype=mass.dtype, device=mass.device, name="runoff_mass")[:, 0]
        result = torch.zeros_like(flat)
        result.index_add_(0, self.outlet_index.to(device=flat.device), flat)
        return result[:, None]

    def step(self, state: RunoffState, runoff_mass: torch.Tensor, dt: float) -> RunoffRouteResult:
        runoff = as_cell_column(runoff_mass, like=state.river_storage, name="runoff_mass").clamp_min(0.0)
        if self.mode == "instant_outlet":
            ocean_input = self._scatter_to_outlet(runoff)
            next_state = state
        else:
            dt_t = torch.as_tensor(float(dt), dtype=state.river_storage.dtype, device=state.river_storage.device)
            storage = state.river_storage + runoff
            outflow = torch.minimum(storage, storage * (dt_t / float(self.storage_timescale)).clamp(0.0, 1.0))
            next_state = RunoffState(storage - outflow)
            ocean_input = self._scatter_to_outlet(outflow)
        transfers = (ConservativeTransfer("runoff_to_ocean", "runoff", "ocean", water_mass=ocean_input),)
        diagnostics = {
            "runoff_input_mass": runoff.sum(),
            "runoff_ocean_input_mass": ocean_input.sum(),
            "runoff_storage_mass": next_state.river_storage.sum(),
        }
        return RunoffRouteResult(next_state, ocean_input, transfers, diagnostics)


register_component("runoff_router", RunoffRouter)


__all__ = ["RunoffRouteResult", "RunoffRouter", "RunoffState"]
