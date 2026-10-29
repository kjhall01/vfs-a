"""Geometry-conditioned sparse mesh convolution layers.

These modules are intentionally small PyTorch utilities over the existing
VoronoiMesh buffers.  Kernel weights are conditioned only on local geometry;
source token values pass through value projections.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MeshKernelConfig:
    """Configuration for geometry-conditioned mesh kernels."""

    hidden_dim: int = 32
    blocks: int = 2
    geom_hidden_dim: int = 32
    prior: str = "uniform"
    feature_mode: str = "relative"


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)


def _local_basis(target_xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ref_z = torch.tensor([0.0, 0.0, 1.0], dtype=target_xyz.dtype, device=target_xyz.device)
    ref_x = torch.tensor([1.0, 0.0, 0.0], dtype=target_xyz.dtype, device=target_xyz.device)
    use_x = target_xyz[..., 2].abs() > 0.9
    ref = torch.where(use_x[..., None], ref_x, ref_z)
    east = _normalize(torch.cross(ref, target_xyz, dim=-1))
    north = _normalize(torch.cross(target_xyz, east, dim=-1))
    return east, north


def _relative_geometry(
    target_xyz: torch.Tensor,
    source_xyz: torch.Tensor,
    *,
    source_type: float,
    extra: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return local dimensionless target-source geometry features."""

    target = target_xyz[:, None, :].to(dtype=source_xyz.dtype, device=source_xyz.device)
    east, north = _local_basis(target_xyz.to(dtype=source_xyz.dtype, device=source_xyz.device))
    east = east[:, None, :]
    north = north[:, None, :]
    dot = (target * source_xyz).sum(dim=-1).clamp(-1.0, 1.0)
    delta = source_xyz - target
    tangent = source_xyz - dot[..., None] * target
    offset_e = (tangent * east).sum(dim=-1)
    offset_n = (tangent * north).sum(dim=-1)
    angular = torch.acos(dot)
    chord = delta.norm(dim=-1)
    bearing_norm = torch.sqrt(offset_e.square() + offset_n.square()).clamp_min(torch.finfo(source_xyz.dtype).tiny)
    geom = [
        offset_e,
        offset_n,
        angular,
        chord,
        dot,
        offset_e / bearing_norm,
        offset_n / bearing_norm,
        torch.full_like(dot, float(source_type)),
    ]
    if extra is not None:
        geom.append(extra.to(dtype=source_xyz.dtype, device=source_xyz.device).expand_as(dot))
    return torch.stack(geom, dim=-1)


def _cell_one_ring(mesh, *, device) -> tuple[torch.Tensor, torch.Tensor]:
    center = torch.arange(int(mesh.nCells), device=device, dtype=torch.long).reshape(-1, 1)
    edges = mesh.safe_edges_on_cell.to(device=device).long()
    valid = mesh.valid_edges_on_cell.to(device=device)
    edge_cells = mesh.safe_cells_on_edge.to(device=device).long()[edges]
    c0 = edge_cells[..., 0]
    c1 = edge_cells[..., 1]
    neighbor = torch.where(c0 == center, c1, c0)
    indices = torch.cat([center, neighbor], dim=1)
    mask = torch.cat([torch.ones_like(center, dtype=torch.bool), valid], dim=1)
    return indices, mask


def _edge_cell_one_ring(mesh, *, device) -> tuple[torch.Tensor, torch.Tensor]:
    cells = mesh.safe_cells_on_edge.to(device=device).long()
    c0 = cells[:, 0]
    c1 = cells[:, 1]
    edges = mesh.safe_edges_on_cell.to(device=device).long()
    valid = mesh.valid_edges_on_cell.to(device=device)
    edge_cells = mesh.safe_cells_on_edge.to(device=device).long()[edges]
    center0 = c0[:, None]
    center1 = c1[:, None]
    neighbors0 = torch.where(edge_cells[c0, :, 0] == center0, edge_cells[c0, :, 1], edge_cells[c0, :, 0])
    neighbors1 = torch.where(edge_cells[c1, :, 0] == center1, edge_cells[c1, :, 1], edge_cells[c1, :, 0])
    indices = torch.cat([cells, neighbors0, neighbors1], dim=1)
    mask = torch.cat(
        [
            torch.ones((int(mesh.nEdges), 2), dtype=torch.bool, device=device),
            valid[c0],
            valid[c1],
        ],
        dim=1,
    )
    return indices, mask


def _edge_edge_one_ring(mesh, *, device) -> tuple[torch.Tensor, torch.Tensor]:
    edge = torch.arange(int(mesh.nEdges), device=device, dtype=torch.long).reshape(-1, 1)
    cells = mesh.safe_cells_on_edge.to(device=device).long()
    c0 = cells[:, 0]
    c1 = cells[:, 1]
    edges = mesh.safe_edges_on_cell.to(device=device).long()
    valid = mesh.valid_edges_on_cell.to(device=device)
    indices = torch.cat([edge, edges[c0], edges[c1]], dim=1)
    mask = torch.cat(
        [
            torch.ones_like(edge, dtype=torch.bool),
            valid[c0],
            valid[c1],
        ],
        dim=1,
    )
    return indices, mask


def _vertical_slots(
    source: torch.Tensor,
    *,
    source_location: str,
    target_location: str,
    n_target_levels: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map source vertical levels to target levels with tiny CP-aware stencils.

    Returns source slots ``[nSource, nTargetLevels, nSlots, channels]``, relative
    vertical offsets ``[nTargetLevels, nSlots]``, and a valid mask with the same
    last two dimensions.
    """

    if source.ndim != 3:
        raise ValueError("mesh convolution source must have shape [n, levels, channels]")
    device = source.device
    dtype = source.dtype
    n_source_levels = int(source.shape[1])
    target_levels = int(n_target_levels or n_source_levels)
    source_location = str(source_location)
    target_location = str(target_location)

    if n_source_levels == target_levels and source_location == target_location:
        idx = torch.arange(target_levels, device=device).reshape(target_levels, 1)
        offsets = torch.zeros(target_levels, 1, dtype=dtype, device=device)
        valid = torch.ones(target_levels, 1, dtype=torch.bool, device=device)
    elif target_location == "full" and source_location == "interface" and n_source_levels == target_levels + 1:
        base = torch.arange(target_levels, device=device)
        idx = torch.stack([base, base + 1], dim=1)
        offsets = torch.stack(
            [
                torch.full((target_levels,), -0.5, dtype=dtype, device=device),
                torch.full((target_levels,), 0.5, dtype=dtype, device=device),
            ],
            dim=1,
        )
        valid = torch.ones(target_levels, 2, dtype=torch.bool, device=device)
    elif target_location == "interface" and source_location == "full" and n_source_levels + 1 == target_levels:
        interface = torch.arange(target_levels, device=device)
        lower = (interface - 1).clamp(0, n_source_levels - 1)
        upper = interface.clamp(0, n_source_levels - 1)
        idx = torch.stack([lower, upper], dim=1)
        valid = torch.stack([interface > 0, interface < n_source_levels], dim=1)
        offsets = torch.stack(
            [
                torch.full((target_levels,), -0.5, dtype=dtype, device=device),
                torch.full((target_levels,), 0.5, dtype=dtype, device=device),
            ],
            dim=1,
        )
    elif n_source_levels == target_levels:
        idx = torch.arange(target_levels, device=device).reshape(target_levels, 1)
        offsets = torch.zeros(target_levels, 1, dtype=dtype, device=device)
        valid = torch.ones(target_levels, 1, dtype=torch.bool, device=device)
    else:
        raise ValueError(
            "unsupported source/target vertical locations or level counts: "
            f"{source_location}({n_source_levels}) -> {target_location}({target_levels})"
        )

    slots = source[:, idx, :]
    return slots, offsets, valid


class GeometryKernelMLP(nn.Module):
    """Small MLP that maps geometry features to candidate logits."""

    def __init__(self, geom_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(geom_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        return self.net(geometry).squeeze(-1)


class CellMeshKernelConv(nn.Module):
    """Cell-target one-ring geometry-kernel convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        geom_hidden_dim: int = 32,
        source_location: str = "full",
        target_location: str = "full",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.source_location = str(source_location)
        self.target_location = str(target_location)
        self.value = nn.Linear(self.in_channels, self.out_channels)
        self.kernel = GeometryKernelMLP(geom_dim=9, hidden_dim=int(geom_hidden_dim))

    def forward(self, mesh, source: torch.Tensor, *, target_levels: int | None = None) -> torch.Tensor:
        indices, spatial_valid = _cell_one_ring(mesh, device=source.device)
        slots, vertical_offsets, vertical_valid = _vertical_slots(
            source,
            source_location=self.source_location,
            target_location=self.target_location,
            n_target_levels=target_levels,
        )
        source_slots = slots[indices]
        values = self.value(source_slots)
        n_target, n_spatial, n_levels, n_vertical, _channels = values.shape
        source_xyz = mesh.cell_xyz.to(dtype=source.dtype, device=source.device)[indices]
        target_xyz = mesh.cell_xyz.to(dtype=source.dtype, device=source.device)
        geom_spatial = _relative_geometry(target_xyz, source_xyz, source_type=0.0)
        geom = geom_spatial[:, :, None, None, :].expand(n_target, n_spatial, n_levels, n_vertical, -1)
        zoff = vertical_offsets[None, None, :, :, None].expand(n_target, n_spatial, n_levels, n_vertical, 1)
        geom = torch.cat([geom[..., :-1], zoff, geom[..., -1:]], dim=-1)
        logits = self.kernel(geom).reshape(n_target, n_spatial * n_vertical, n_levels)
        values = values.reshape(n_target, n_spatial * n_vertical, n_levels, self.out_channels)
        valid = (spatial_valid[:, :, None, None] & vertical_valid[None, None, :, :]).reshape(
            n_target,
            n_spatial * n_vertical,
            n_levels,
        )
        low = -1.0e30 if source.dtype == torch.float64 else -1.0e20
        logits = torch.where(valid, logits, torch.full_like(logits, low))
        weights = torch.softmax(logits, dim=1)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(source.dtype).tiny)
        return (weights[..., None] * values).sum(dim=1)


class EdgeMeshKernelConv(nn.Module):
    """Edge-target geometry-kernel convolution over cell and/or edge sources."""

    def __init__(
        self,
        out_channels: int,
        *,
        cell_in_channels: int | None = None,
        edge_in_channels: int | None = None,
        geom_hidden_dim: int = 32,
        source_location: str = "full",
        target_location: str = "full",
    ) -> None:
        super().__init__()
        if cell_in_channels is None and edge_in_channels is None:
            raise ValueError("at least one of cell_in_channels or edge_in_channels is required")
        self.out_channels = int(out_channels)
        self.source_location = str(source_location)
        self.target_location = str(target_location)
        self.cell_value = None if cell_in_channels is None else nn.Linear(int(cell_in_channels), self.out_channels)
        self.edge_value = None if edge_in_channels is None else nn.Linear(int(edge_in_channels), self.out_channels)
        self.kernel = GeometryKernelMLP(geom_dim=9, hidden_dim=int(geom_hidden_dim))

    def _candidate_pack(
        self,
        mesh,
        source: torch.Tensor,
        *,
        target_xyz: torch.Tensor,
        source_xyz_all: torch.Tensor,
        indices: torch.Tensor,
        spatial_valid: torch.Tensor,
        value: nn.Linear,
        source_type: float,
        target_levels: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slots, vertical_offsets, vertical_valid = _vertical_slots(
            source,
            source_location=self.source_location,
            target_location=self.target_location,
            n_target_levels=target_levels,
        )
        source_slots = slots[indices]
        values = value(source_slots)
        n_target, n_spatial, n_levels, n_vertical, _channels = values.shape
        source_xyz = source_xyz_all[indices]
        geom_spatial = _relative_geometry(target_xyz, source_xyz, source_type=source_type)
        geom = geom_spatial[:, :, None, None, :].expand(n_target, n_spatial, n_levels, n_vertical, -1)
        zoff = vertical_offsets[None, None, :, :, None].expand(n_target, n_spatial, n_levels, n_vertical, 1)
        geom = torch.cat([geom[..., :-1], zoff, geom[..., -1:]], dim=-1)
        valid = (spatial_valid[:, :, None, None] & vertical_valid[None, None, :, :]).reshape(
            n_target,
            n_spatial * n_vertical,
            n_levels,
        )
        return (
            values.reshape(n_target, n_spatial * n_vertical, n_levels, self.out_channels),
            geom.reshape(n_target, n_spatial * n_vertical, n_levels, geom.shape[-1]),
            valid,
        )

    def forward(
        self,
        mesh,
        *,
        cell_source: torch.Tensor | None = None,
        edge_source: torch.Tensor | None = None,
        target_levels: int | None = None,
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        geometries: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        dtype = (cell_source if cell_source is not None else edge_source).dtype  # type: ignore[union-attr]
        device = (cell_source if cell_source is not None else edge_source).device  # type: ignore[union-attr]
        target_xyz = mesh.edge_xyz.to(dtype=dtype, device=device)
        if cell_source is not None:
            if self.cell_value is None:
                raise ValueError("cell_source was provided but cell_value projection is absent")
            indices, valid = _edge_cell_one_ring(mesh, device=device)
            values, geom, mask = self._candidate_pack(
                mesh,
                cell_source,
                target_xyz=target_xyz,
                source_xyz_all=mesh.cell_xyz.to(dtype=dtype, device=device),
                indices=indices,
                spatial_valid=valid,
                value=self.cell_value,
                source_type=0.0,
                target_levels=target_levels,
            )
            pieces.append(values)
            geometries.append(geom)
            masks.append(mask)
        if edge_source is not None:
            if self.edge_value is None:
                raise ValueError("edge_source was provided but edge_value projection is absent")
            indices, valid = _edge_edge_one_ring(mesh, device=device)
            values, geom, mask = self._candidate_pack(
                mesh,
                edge_source,
                target_xyz=target_xyz,
                source_xyz_all=mesh.edge_xyz.to(dtype=dtype, device=device),
                indices=indices,
                spatial_valid=valid,
                value=self.edge_value,
                source_type=1.0,
                target_levels=target_levels,
            )
            pieces.append(values)
            geometries.append(geom)
            masks.append(mask)
        values = torch.cat(pieces, dim=1)
        geometry = torch.cat(geometries, dim=1)
        valid = torch.cat(masks, dim=1)
        logits = self.kernel(geometry)
        low = -1.0e30 if dtype == torch.float64 else -1.0e20
        logits = torch.where(valid, logits, torch.full_like(logits, low))
        weights = torch.softmax(logits, dim=1)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
        return (weights[..., None] * values).sum(dim=1)


class CellMeshKernelNetwork(nn.Module):
    """Residual stack of cell-target geometry-kernel convolutions."""

    def __init__(
        self,
        in_channels: int,
        *,
        hidden_dim: int = 32,
        blocks: int = 2,
        geom_hidden_dim: int = 32,
        source_location: str = "full",
        target_location: str = "full",
    ) -> None:
        super().__init__()
        self.input_dim = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.blocks = int(blocks)
        self.input = nn.Linear(self.input_dim, self.hidden_dim)
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(max(0, self.blocks))])
        self.convs = nn.ModuleList(
            [
                CellMeshKernelConv(
                    self.hidden_dim,
                    self.hidden_dim,
                    geom_hidden_dim=int(geom_hidden_dim),
                    source_location=source_location,
                    target_location=target_location,
                )
                for _ in range(max(0, self.blocks))
            ]
        )

    def forward(self, mesh, source: torch.Tensor, *, target_levels: int | None = None) -> torch.Tensor:
        x = self.input(source)
        for norm, conv in zip(self.norms, self.convs):
            x = x + conv(mesh, norm(x), target_levels=target_levels)
        return x


class EdgeMeshKernelNetwork(nn.Module):
    """Residual stack of edge-target geometry-kernel convolutions."""

    def __init__(
        self,
        *,
        hidden_dim: int = 32,
        blocks: int = 2,
        geom_hidden_dim: int = 32,
        cell_in_channels: int | None = None,
        edge_in_channels: int | None = None,
        source_location: str = "full",
        target_location: str = "full",
    ) -> None:
        super().__init__()
        if cell_in_channels is None and edge_in_channels is None:
            raise ValueError("at least one source channel count is required")
        self.hidden_dim = int(hidden_dim)
        self.blocks = int(blocks)
        self.cell_input = None if cell_in_channels is None else nn.Linear(int(cell_in_channels), self.hidden_dim)
        self.edge_input = None if edge_in_channels is None else nn.Linear(int(edge_in_channels), self.hidden_dim)
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(max(0, self.blocks))])
        self.convs = nn.ModuleList(
            [
                EdgeMeshKernelConv(
                    self.hidden_dim,
                    cell_in_channels=self.hidden_dim if cell_in_channels is not None else None,
                    edge_in_channels=self.hidden_dim if edge_in_channels is not None else None,
                    geom_hidden_dim=int(geom_hidden_dim),
                    source_location=source_location,
                    target_location=target_location,
                )
                for _ in range(max(0, self.blocks))
            ]
        )

    def forward(
        self,
        mesh,
        *,
        cell_source: torch.Tensor | None = None,
        edge_source: torch.Tensor | None = None,
        target_levels: int | None = None,
    ) -> torch.Tensor:
        cell_h = None if cell_source is None or self.cell_input is None else self.cell_input(cell_source)
        edge_h = None if edge_source is None or self.edge_input is None else self.edge_input(edge_source)
        if edge_h is not None:
            x = edge_h
        elif cell_h is not None:
            x = self.convs[0](mesh, cell_source=cell_h, target_levels=target_levels) if self.convs else cell_h
            start = 1
            for norm, conv in zip(self.norms[start:], self.convs[start:]):
                x = x + conv(mesh, cell_source=norm(x), target_levels=target_levels)
            return x
        else:
            raise ValueError("at least one source tensor must be provided")
        for norm, conv in zip(self.norms, self.convs):
            x_norm = norm(x)
            x = x + conv(mesh, cell_source=cell_h, edge_source=x_norm, target_levels=target_levels)
        return x


__all__ = [
    "CellMeshKernelConv",
    "CellMeshKernelNetwork",
    "EdgeMeshKernelConv",
    "EdgeMeshKernelNetwork",
    "MeshKernelConfig",
]
