"""Lightweight neural edge reconstruction for conservative passive transport."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from vfs.core.state import State
from vfs.geometry.mesh.kernels import EdgeMeshKernelNetwork


@dataclass(frozen=True)
class NeuralReconstructionConfig:
    """Admissibility controls for neural passive-tracer reconstruction."""

    mode: str = "strict"
    relaxed_eps_minus: float = 0.0
    relaxed_eps_plus: float = 0.0
    donor_export_safety: float = 1.0


@dataclass(frozen=True)
class NeuralAllocationConfig:
    """Controls for donor-cell neural export allocation."""

    mode: str = "strict"
    relaxed_eps: float = 0.0
    envelope: str = "scalar"
    envelope_max: float = 0.05
    envelope_eps_floor: float = 1.0e-6
    residual_alpha_max: float = 1.0
    zero_flux_tolerance: float = 1.0e-14
    eps: float = 1.0e-12


@dataclass(frozen=True)
class SectorRemapConfig:
    """Controls for neural conservative sector remap reconstruction."""

    logit_scale: float = 1.0
    min_prior: float = 1.0e-8
    zero_flux_tolerance: float = 1.0e-14
    eps: float = 1.0e-12


@dataclass(frozen=True)
class InvariantFeatureConfig:
    """Controls for dimensionless neural edge feature construction."""

    enabled: bool = False
    eps: float = 1.0e-8
    clip: float = 8.0
    range_floor_mode: str = "relative_global"
    range_floor: float = 1.0e-6
    include_raw_legacy_features: bool = False


NEURAL_FEATURE_MODES = ("raw", "invariant", "hybrid")
NEURAL_MODEL_TYPES = ("mlp", "meshconv")


def invariant_feature_config(
    mode: str = "raw",
    *,
    clip: float = 8.0,
    range_floor: float = 1.0e-6,
    eps: float = 1.0e-8,
) -> InvariantFeatureConfig:
    """Return a feature config for the public raw/invariant/hybrid modes."""

    name = str(mode).lower()
    if name not in NEURAL_FEATURE_MODES:
        raise ValueError(f"neural feature mode must be one of {NEURAL_FEATURE_MODES}, got {mode!r}")
    return InvariantFeatureConfig(
        enabled=name in {"invariant", "hybrid"},
        eps=float(eps),
        clip=float(clip),
        range_floor=float(range_floor),
        include_raw_legacy_features=name == "hybrid",
    )


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1.0e-8)
    return math.log(math.expm1(value))


def _logit(value: float) -> float:
    value = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(value / (1.0 - value))


class EdgeReconstructionMLP(nn.Module):
    """Tiny MLP that predicts edge correction weights ``gamma in [0, 1]``."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 32,
        depth: int = 2,
        initial_gamma: float = 0.05,
    ) -> None:
        super().__init__()
        if int(input_dim) < 1:
            raise ValueError("input_dim must be positive")
        layers: list[nn.Module] = []
        dim = int(input_dim)
        for _ in range(max(1, int(depth))):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.SiLU())
            dim = int(hidden_dim)
        out = nn.Linear(dim, 1)
        gamma = min(max(float(initial_gamma), 1.0e-4), 1.0 - 1.0e-4)
        nn.init.zeros_(out.weight)
        nn.init.constant_(out.bias, math.log(gamma / (1.0 - gamma)))
        layers.append(out)
        self.net = nn.Sequential(*layers)
        self.input_dim = int(input_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flat = features.reshape(-1, features.shape[-1])
        gamma = torch.sigmoid(self.net(flat))
        return gamma.reshape(*features.shape[:-1], 1)


class EdgeAllocationMLP(nn.Module):
    """Tiny MLP that predicts edge preferences and optional envelope weights."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 32,
        depth: int = 2,
        initial_edge_preference: float = 1.0,
        initial_stay_bias: float = 1.0,
        initial_envelope_eta: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if int(input_dim) < 1:
            raise ValueError("input_dim must be positive")

        def mlp(final_bias: float, *, sigmoid_output: bool) -> nn.Sequential:
            layers: list[nn.Module] = []
            dim = int(input_dim)
            for _ in range(max(1, int(depth))):
                layers.append(nn.Linear(dim, int(hidden_dim)))
                layers.append(nn.SiLU())
                dim = int(hidden_dim)
            out = nn.Linear(dim, 1)
            nn.init.zeros_(out.weight)
            nn.init.constant_(out.bias, final_bias)
            layers.append(out)
            if sigmoid_output:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.Softplus())
            return nn.Sequential(*layers)

        self.net = mlp(_inverse_softplus(float(initial_edge_preference)), sigmoid_output=False)
        self.envelope_net = mlp(_logit(float(initial_envelope_eta)), sigmoid_output=True)
        self.stay_logit = nn.Parameter(torch.tensor(_inverse_softplus(float(initial_stay_bias))))
        self.input_dim = int(input_dim)
        self.initial_edge_preference = float(initial_edge_preference)
        self.initial_stay_bias = float(initial_stay_bias)
        self.initial_envelope_eta = float(initial_envelope_eta)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flat = features.reshape(-1, features.shape[-1])
        pref = self.net(flat)
        return pref.reshape(*features.shape[:-1], 1)

    def envelope_eta(self, features: torch.Tensor) -> torch.Tensor:
        flat = features.reshape(-1, features.shape[-1])
        eta = self.envelope_net(flat)
        return eta.reshape(*features.shape[:-1], 1)

    def stay_preference(self, *, dtype: torch.dtype, device) -> torch.Tensor:
        return torch.nn.functional.softplus(self.stay_logit.to(dtype=dtype, device=device))


class SectorRemapMLP(nn.Module):
    """MLP that predicts bounded residual logits for cell edge-sector masses."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 32,
        depth: int = 2,
    ) -> None:
        super().__init__()
        if int(input_dim) < 1:
            raise ValueError("input_dim must be positive")
        layers: list[nn.Module] = []
        dim = int(input_dim)
        for _ in range(max(1, int(depth))):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.SiLU())
            dim = int(hidden_dim)
        out = nn.Linear(dim, 1)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)
        self.input_dim = int(input_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flat = features.reshape(-1, features.shape[-1])
        logits = self.net(flat)
        return logits.reshape(*features.shape[:-1], 1)


class EdgeAllocationMeshConv(nn.Module):
    """Edge-target mesh-convolution readout for allocation preferences."""

    uses_mesh_convolution = True
    model_type = "allocation_meshconv"

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 32,
        depth: int = 2,
        geom_hidden_dim: int = 32,
        initial_edge_preference: float = 1.0,
        initial_stay_bias: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.geom_hidden_dim = int(geom_hidden_dim)
        self.encoder = EdgeMeshKernelNetwork(
            cell_in_channels=self.input_dim,
            edge_in_channels=self.input_dim,
            hidden_dim=self.hidden_dim,
            blocks=self.depth,
            geom_hidden_dim=self.geom_hidden_dim,
            source_location="full",
            target_location="full",
        )
        self.pref_head = nn.Linear(self.hidden_dim, 1)
        self.eta_head = nn.Linear(self.hidden_dim, 1)
        self.stay_logit = nn.Parameter(torch.tensor(_inverse_softplus(float(initial_stay_bias))))
        self.initial_edge_preference = float(initial_edge_preference)
        self.initial_stay_bias = float(initial_stay_bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.pref_head.weight)
        nn.init.constant_(self.pref_head.bias, _inverse_softplus(float(self.initial_edge_preference)))
        nn.init.zeros_(self.eta_head.weight)
        nn.init.zeros_(self.eta_head.bias)

    def _cell_tokens(self, state: State, features: torch.Tensor) -> torch.Tensor:
        n_edges, n_levels, n_channels = features.shape
        return state.edge_to_cell(features.reshape(n_edges, n_levels * n_channels)).reshape(
            state.mesh.nCells,
            n_levels,
            n_channels,
        )

    def latent(self, features: torch.Tensor, *, state: State) -> torch.Tensor:
        cell_tokens = self._cell_tokens(state, features)
        return self.encoder(
            state.mesh,
            cell_source=cell_tokens,
            edge_source=features,
            target_levels=features.shape[1],
        )

    def forward(self, features: torch.Tensor, *, state: State | None = None) -> torch.Tensor:
        if state is None:
            raise ValueError("EdgeAllocationMeshConv.forward requires state=")
        pref = torch.nn.functional.softplus(self.pref_head(self.latent(features, state=state)))
        return pref

    def envelope_eta(self, features: torch.Tensor, *, state: State | None = None) -> torch.Tensor:
        if state is None:
            raise ValueError("EdgeAllocationMeshConv.envelope_eta requires state=")
        raw = self.eta_head(self.latent(features, state=state))
        eta = torch.nn.functional.softplus(raw) - math.log(2.0)
        return eta.clamp(0.0, 1.0)

    def stay_preference(self, *, dtype: torch.dtype, device) -> torch.Tensor:
        return torch.nn.functional.softplus(self.stay_logit.to(dtype=dtype, device=device))


class SectorRemapMeshConv(nn.Module):
    """Edge-target mesh-convolution residual-logit model for sector remap."""

    uses_mesh_convolution = True
    model_type = "sector_remap_meshconv"

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 32,
        depth: int = 2,
        geom_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.geom_hidden_dim = int(geom_hidden_dim)
        self.encoder = EdgeMeshKernelNetwork(
            cell_in_channels=self.input_dim,
            edge_in_channels=self.input_dim,
            hidden_dim=self.hidden_dim,
            blocks=self.depth,
            geom_hidden_dim=self.geom_hidden_dim,
            source_location="full",
            target_location="full",
        )
        self.head = nn.Linear(self.hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _cell_tokens(self, state: State, features: torch.Tensor) -> torch.Tensor:
        n_edges, n_levels, n_channels = features.shape
        return state.edge_to_cell(features.reshape(n_edges, n_levels * n_channels)).reshape(
            state.mesh.nCells,
            n_levels,
            n_channels,
        )

    def forward(self, features: torch.Tensor, *, state: State | None = None) -> torch.Tensor:
        if state is None:
            raise ValueError("SectorRemapMeshConv.forward requires state=")
        latent = self.encoder(
            state.mesh,
            cell_source=self._cell_tokens(state, features),
            edge_source=features,
            target_levels=features.shape[1],
        )
        return self.head(latent)


def _model_forward(model: nn.Module, features: torch.Tensor, *, state: State) -> torch.Tensor:
    if getattr(model, "uses_mesh_convolution", False):
        return model(features, state=state)
    return model(features)


def _model_envelope_eta(model: nn.Module, features: torch.Tensor, *, state: State) -> torch.Tensor:
    if getattr(model, "uses_mesh_convolution", False):
        return model.envelope_eta(features, state=state)  # type: ignore[attr-defined]
    return model.envelope_eta(features)  # type: ignore[attr-defined]


def _cell_bounds(state: State, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mesh = state.mesh
    n_cells = mesh.nCells
    cell_index = torch.arange(n_cells, device=q.device, dtype=torch.long).view(n_cells, 1)
    safe_edges = mesh.safe_edges_on_cell.to(device=q.device).long()
    edge_cells = mesh.safe_cells_on_edge.to(device=q.device).long()
    neighbor0 = edge_cells[safe_edges, 0]
    neighbor1 = edge_cells[safe_edges, 1]
    neighbor = torch.where(neighbor0 == cell_index, neighbor1, neighbor0)
    valid = mesh.valid_edges_on_cell.to(device=q.device)
    neighbor = torch.where(valid, neighbor, cell_index)
    neighbor_values = q[neighbor]
    center = q[:, None, :, :]
    neighbor_values = torch.where(valid[:, :, None, None], neighbor_values, center)
    q_min = torch.minimum(q, neighbor_values.amin(dim=1))
    q_max = torch.maximum(q, neighbor_values.amax(dim=1))
    return q_min, q_max


def _gather_cell_level(cell_values: torch.Tensor, cell_index: torch.Tensor) -> torch.Tensor:
    n_levels = cell_values.shape[1]
    level_index = torch.arange(n_levels, device=cell_values.device).reshape(1, n_levels)
    return cell_values[cell_index, level_index]


def _edge_geometry_features(mesh, q: torch.Tensor, u_e: torch.Tensor, dt: float) -> torch.Tensor:
    dtype = q.dtype
    device = q.device
    dc = mesh.dc_edge.to(dtype=dtype, device=device)
    dv = mesh.dv_edge.to(dtype=dtype, device=device)
    area = mesh.cell_area.to(dtype=dtype, device=device)
    c0 = mesh.cellsOnEdge[:, 0]
    c1 = mesh.cellsOnEdge[:, 1]
    area0 = area[c0]
    area1 = area[c1]
    edge_xyz = mesh.edge_xyz.to(dtype=dtype, device=device)
    edge_normal = mesh.edge_normal.to(dtype=dtype, device=device)
    scale_l = dc.mean().clamp_min(torch.finfo(dtype).tiny)
    scale_a = area.mean().clamp_min(torch.finfo(dtype).tiny)
    cfl = float(dt) * u_e.abs() / dc[:, None].clamp_min(torch.finfo(dtype).tiny)
    n_levels = q.shape[1]
    ones = torch.ones_like(u_e)
    def feature(value: torch.Tensor) -> torch.Tensor:
        return value[..., None]

    geom = torch.cat(
        [
            feature(u_e / u_e.abs().amax().clamp_min(torch.finfo(dtype).tiny)),
            feature(cfl),
            feature(ones * (dc[:, None] / scale_l)),
            feature(ones * (dv[:, None] / scale_l)),
            feature(ones * (area0[:, None] / scale_a)),
            feature(ones * (area1[:, None] / scale_a)),
            feature(ones * edge_normal[:, 0:1]),
            feature(ones * edge_normal[:, 1:2]),
            feature(ones * edge_normal[:, 2:3]),
            feature(ones * edge_xyz[:, 0:1]),
            feature(ones * edge_xyz[:, 1:2]),
            feature(ones * edge_xyz[:, 2:3]),
        ],
        dim=-1,
    )
    return geom.reshape(mesh.nEdges, n_levels, -1)


def _raw_edge_reconstruction_features(
    state: State,
    tracer_m: torch.Tensor,
    u_e: torch.Tensor,
    diagnostics: torch.Tensor,
    shadow_z: torch.Tensor,
    *,
    dt: float,
) -> torch.Tensor:
    """Build explicit local edge features for neural reconstruction."""

    mesh = state.mesh
    q = tracer_m / state.layer_dp()[..., None].clamp_min(torch.finfo(tracer_m.dtype).tiny)
    q_scalar = q[..., :1]
    c0 = mesh.cellsOnEdge[:, 0]
    c1 = mesh.cellsOnEdge[:, 1]
    q0 = q_scalar[c0]
    q1 = q_scalar[c1]
    q_min, q_max = _cell_bounds(state, q_scalar)
    edge_min = torch.minimum(q_min[c0], q_min[c1])
    edge_max = torch.maximum(q_max[c0], q_max[c1])
    grad_e = mesh.grad(q_scalar[..., 0])[..., None]
    lap = mesh.div(mesh.grad(q_scalar[..., 0]))[..., None]
    geom = _edge_geometry_features(mesh, q, u_e, dt)
    D0 = diagnostics[c0]
    D1 = diagnostics[c1]
    z0 = shadow_z[c0]
    z1 = shadow_z[c1]
    pieces = [
        q0,
        q1,
        q1 - q0,
        edge_min,
        edge_max,
        edge_max - edge_min,
        grad_e,
        lap[c0],
        lap[c1],
        geom,
        D0,
        D1,
        z0,
        z1,
        z0 - D0,
        z1 - D1,
    ]
    return torch.cat(pieces, dim=-1)


def _feature_clip(values: torch.Tensor, config: InvariantFeatureConfig) -> torch.Tensor:
    clipped = torch.clamp(values, -float(config.clip), float(config.clip))
    return torch.nan_to_num(clipped, nan=0.0, posinf=float(config.clip), neginf=-float(config.clip))


def _range_floor(q_scalar: torch.Tensor, config: InvariantFeatureConfig) -> torch.Tensor:
    floor = torch.as_tensor(float(config.range_floor), dtype=q_scalar.dtype, device=q_scalar.device)
    if str(config.range_floor_mode) == "relative_global":
        global_range = (q_scalar.amax() - q_scalar.amin()).detach()
        amplitude = q_scalar.detach().abs().amax()
        scale = torch.maximum(torch.maximum(global_range, amplitude), torch.ones_like(floor))
        return (floor * scale).clamp_min(floor)
    if str(config.range_floor_mode) == "absolute":
        return floor
    raise ValueError("range_floor_mode must be 'relative_global' or 'absolute'")


def _invariant_feature_context(
    state: State,
    tracer_m: torch.Tensor,
    u_e: torch.Tensor,
    diagnostics: torch.Tensor,
    shadow_z: torch.Tensor,
    *,
    dt: float,
    config: InvariantFeatureConfig,
) -> dict[str, torch.Tensor]:
    mesh = state.mesh
    dtype = tracer_m.dtype
    device = tracer_m.device
    q = tracer_m / state.layer_dp()[..., None].clamp_min(torch.finfo(dtype).tiny)
    q_scalar = q[..., :1]
    c0 = mesh.cellsOnEdge[:, 0].to(device=device)
    c1 = mesh.cellsOnEdge[:, 1].to(device=device)
    q0 = q_scalar[c0]
    q1 = q_scalar[c1]
    positive = u_e >= 0.0
    donor = torch.where(positive, c0[:, None], c1[:, None])
    receiver = torch.where(positive, c1[:, None], c0[:, None])
    q_donor = _gather_cell_level(q_scalar, donor)
    q_receiver = _gather_cell_level(q_scalar, receiver)
    q_min, q_max = _cell_bounds(state, q_scalar)
    stencil_min = torch.minimum(_gather_cell_level(q_min, donor), _gather_cell_level(q_min, receiver))
    stencil_max = torch.maximum(_gather_cell_level(q_max, donor), _gather_cell_level(q_max, receiver))
    q_range = (stencil_max - stencil_min).clamp_min(0.0)
    floor = _range_floor(q_scalar, config)
    q_range_safe = torch.maximum(q_range, floor)
    q_mid = 0.5 * (stencil_min + stencil_max)
    dc = mesh.dc_edge.to(dtype=dtype, device=device)
    dv = mesh.dv_edge.to(dtype=dtype, device=device)
    area = mesh.cell_area.to(dtype=dtype, device=device)
    area_donor = _gather_cell_level(area[:, None, None].expand(-1, q_scalar.shape[1], 1), donor)
    area_receiver = _gather_cell_level(area[:, None, None].expand(-1, q_scalar.shape[1], 1), receiver)
    L_cell = area_donor.sqrt().clamp_min(torch.finfo(dtype).tiny)
    L_edge = dc[:, None, None].clamp_min(torch.finfo(dtype).tiny)
    grad_e = mesh.grad(q_scalar[..., 0])[..., None]
    lap = mesh.div(mesh.grad(q_scalar[..., 0]))[..., None]
    lap_donor = _gather_cell_level(lap, donor)
    lap_receiver = _gather_cell_level(lap, receiver)
    dp = state.layer_dp()[..., None]
    dp_donor = _gather_cell_level(dp, donor)
    m_donor = (area_donor * dp_donor).clamp_min(torch.finfo(dtype).tiny)
    M = (float(dt) * dv[:, None, None] * (mesh.cell_to_edge(state.layer_dp()) * u_e).abs()[..., None]).clamp_min(0.0)
    q_plus = stencil_max.clamp_min(0.0)
    C = M * q_plus
    capacity_q_scale = (q_donor.clamp_min(0.0) + q_range_safe).clamp_min(floor)
    capacity_scale = (m_donor * capacity_q_scale).clamp_min(torch.as_tensor(float(config.eps), dtype=dtype, device=device))
    D_donor = _gather_cell_level(diagnostics, donor)
    D_receiver = _gather_cell_level(diagnostics, receiver)
    z_donor = _gather_cell_level(shadow_z, donor)
    z_receiver = _gather_cell_level(shadow_z, receiver)
    edge_xyz = mesh.edge_xyz.to(dtype=dtype, device=device)
    edge_normal = mesh.edge_normal.to(dtype=dtype, device=device)
    ones = torch.ones_like(q_donor)
    return {
        "q0": q0,
        "q1": q1,
        "q_donor": q_donor,
        "q_receiver": q_receiver,
        "q_min": stencil_min,
        "q_max": stencil_max,
        "q_mid": q_mid,
        "q_range": q_range,
        "q_range_safe": q_range_safe,
        "range_floor": floor,
        "flatness": q_range / (q_range + floor),
        "dc": dc,
        "dv": dv,
        "L_cell": L_cell,
        "L_edge": L_edge,
        "area_donor": area_donor,
        "area_receiver": area_receiver,
        "grad_e": grad_e,
        "lap_donor": lap_donor,
        "lap_receiver": lap_receiver,
        "signed_cfl": float(dt) * u_e[..., None] / L_edge,
        "cfl": float(dt) * u_e.abs()[..., None] / L_edge,
        "mass_export_fraction": M / m_donor,
        "capacity_fraction": C / capacity_scale,
        "q_plus_relative": (q_plus - stencil_min) / q_range_safe,
        "q_plus_excess": (q_plus - stencil_max) / q_range_safe,
        "D_donor": D_donor,
        "D_receiver": D_receiver,
        "z_donor": z_donor,
        "z_receiver": z_receiver,
        "edge_xyz": edge_xyz,
        "edge_normal": edge_normal,
        "ones": ones,
    }


def _append_channel(pieces: list[torch.Tensor], names: list[str], value: torch.Tensor, name: str) -> None:
    pieces.append(value)
    if value.shape[-1] == 1:
        names.append(name)
    else:
        names.extend(f"{name}_{index}" for index in range(value.shape[-1]))


def _invariant_edge_reconstruction_features(
    state: State,
    tracer_m: torch.Tensor,
    u_e: torch.Tensor,
    diagnostics: torch.Tensor,
    shadow_z: torch.Tensor,
    *,
    dt: float,
    config: InvariantFeatureConfig,
    return_names: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[str]]:
    ctx = _invariant_feature_context(state, tracer_m, u_e, diagnostics, shadow_z, dt=dt, config=config)
    q_range_safe = ctx["q_range_safe"]
    floor = ctx["range_floor"]
    pieces: list[torch.Tensor] = []
    names: list[str] = []
    add = lambda value, name: _append_channel(pieces, names, _feature_clip(value, config), name)

    add((ctx["q_donor"] - ctx["q_min"]) / q_range_safe, "q_donor_norm")
    add((ctx["q_receiver"] - ctx["q_min"]) / q_range_safe, "q_receiver_norm")
    add((ctx["q_receiver"] - ctx["q_donor"]) / q_range_safe, "q_jump_norm")
    add((ctx["q_mid"] - ctx["q_min"]) / q_range_safe, "q_mid_norm")
    add(ctx["L_edge"] * ctx["grad_e"] / q_range_safe, "edge_gradient_norm")
    add(ctx["L_cell"].square() * ctx["lap_donor"] / q_range_safe, "laplacian_donor_norm")
    add(ctx["L_cell"].square() * ctx["lap_receiver"] / q_range_safe, "laplacian_receiver_norm")
    add(torch.log1p(ctx["q_range"] / floor.clamp_min(torch.finfo(tracer_m.dtype).tiny)), "log_local_range_over_floor")
    add(ctx["flatness"], "flatness")
    add(ctx["signed_cfl"], "signed_cfl")
    add(ctx["cfl"], "cfl")
    add(ctx["mass_export_fraction"], "mass_export_fraction")
    add(ctx["capacity_fraction"], "capacity_fraction")
    add(ctx["q_plus_relative"], "q_plus_relative")
    add(ctx["q_plus_excess"], "q_plus_excess")
    add(ctx["dv"][:, None, None] / ctx["L_cell"], "edge_length_over_L")
    add(ctx["dc"][:, None, None] / ctx["L_cell"], "dual_length_over_L")
    add(ctx["area_receiver"] / ctx["area_donor"].clamp_min(torch.finfo(tracer_m.dtype).tiny), "area_receiver_over_donor")
    add(ctx["ones"] * ctx["edge_normal"][:, None, :], "edge_normal")
    add(ctx["ones"] * ctx["edge_xyz"][:, None, :], "edge_xyz")
    add(ctx["D_donor"], "D_donor")
    add(ctx["D_receiver"], "D_receiver")
    add(ctx["z_donor"], "z_donor")
    add(ctx["z_receiver"], "z_receiver")
    add(torch.tanh(ctx["z_donor"] - ctx["D_donor"]), "z_donor_lag_tanh")
    add(torch.tanh(ctx["z_receiver"] - ctx["D_receiver"]), "z_receiver_lag_tanh")

    features = torch.cat(pieces, dim=-1)
    if return_names:
        return features, names
    return features


def edge_reconstruction_features(
    state: State,
    tracer_m: torch.Tensor,
    u_e: torch.Tensor,
    diagnostics: torch.Tensor,
    shadow_z: torch.Tensor,
    *,
    dt: float,
    feature_config: InvariantFeatureConfig | None = None,
) -> torch.Tensor:
    """Build explicit local edge features for neural reconstruction."""

    cfg = feature_config or InvariantFeatureConfig()
    if not cfg.enabled:
        return _raw_edge_reconstruction_features(state, tracer_m, u_e, diagnostics, shadow_z, dt=dt)
    invariant = _invariant_edge_reconstruction_features(
        state,
        tracer_m,
        u_e,
        diagnostics,
        shadow_z,
        dt=dt,
        config=cfg,
    )
    if cfg.include_raw_legacy_features:
        raw = _raw_edge_reconstruction_features(state, tracer_m, u_e, diagnostics, shadow_z, dt=dt)
        return torch.cat([raw, invariant], dim=-1)
    return invariant


def edge_reconstruction_feature_names(
    state: State,
    tracer_m: torch.Tensor,
    u_e: torch.Tensor,
    diagnostics: torch.Tensor,
    shadow_z: torch.Tensor,
    *,
    dt: float,
    feature_config: InvariantFeatureConfig | None = None,
) -> list[str]:
    cfg = feature_config or InvariantFeatureConfig()
    if not cfg.enabled:
        dim = edge_reconstruction_features(state, tracer_m, u_e, diagnostics, shadow_z, dt=dt, feature_config=cfg).shape[-1]
        return [f"raw_{index}" for index in range(int(dim))]
    invariant, names = _invariant_edge_reconstruction_features(
        state,
        tracer_m,
        u_e,
        diagnostics,
        shadow_z,
        dt=dt,
        config=cfg,
        return_names=True,
    )
    if cfg.include_raw_legacy_features:
        raw_dim = _raw_edge_reconstruction_features(state, tracer_m, u_e, diagnostics, shadow_z, dt=dt).shape[-1]
        return [f"raw_{index}" for index in range(int(raw_dim))] + names
    return names


def feature_audit_summary(
    features: torch.Tensor,
    *,
    feature_config: InvariantFeatureConfig | None = None,
) -> dict[str, Any]:
    cfg = feature_config or InvariantFeatureConfig()
    finite = torch.isfinite(features)
    abs_features = features.detach().abs()
    clipped = abs_features >= (float(cfg.clip) - 1.0e-12) if cfg.enabled else torch.zeros_like(finite)
    safe = torch.where(finite, features.detach(), torch.zeros_like(features.detach()))
    mode = "raw"
    if cfg.enabled:
        mode = "hybrid" if cfg.include_raw_legacy_features else "invariant"
    return {
        "feature_mode": mode,
        "feature_dim": int(features.shape[-1]),
        "feature_clip": float(cfg.clip),
        "feature_range_floor": float(cfg.range_floor),
        "feature_raw_legacy_included": bool(cfg.include_raw_legacy_features),
        "feature_mean": float(safe.mean().cpu()),
        "feature_std": float(safe.std(unbiased=False).cpu()),
        "feature_min": float(safe.amin().cpu()),
        "feature_max": float(safe.amax().cpu()),
        "feature_nan_count": int(torch.isnan(features).sum().detach().cpu()),
        "feature_inf_count": int(torch.isinf(features).sum().detach().cpu()),
        "feature_fraction_clipped": float(clipped.to(dtype=features.dtype).mean().detach().cpu()) if cfg.enabled else 0.0,
    }


def invariant_feature_scale_summary(
    state: State,
    tracer_m: torch.Tensor,
    u_e: torch.Tensor,
    diagnostics: torch.Tensor,
    shadow_z: torch.Tensor,
    *,
    dt: float,
    feature_config: InvariantFeatureConfig | None = None,
) -> dict[str, Any]:
    cfg = feature_config or InvariantFeatureConfig(enabled=True)
    if not cfg.enabled:
        return {}
    ctx = _invariant_feature_context(state, tracer_m, u_e, diagnostics, shadow_z, dt=dt, config=cfg)

    def row(prefix: str, value: torch.Tensor) -> dict[str, float]:
        return {
            f"{prefix}_mean": float(value.detach().mean().cpu()),
            f"{prefix}_min": float(value.detach().amin().cpu()),
            f"{prefix}_max": float(value.detach().amax().cpu()),
        }

    out: dict[str, Any] = {}
    out.update(row("feature_q_range", ctx["q_range"]))
    out.update(row("feature_flatness", ctx["flatness"]))
    out.update(row("feature_cfl", ctx["cfl"]))
    out.update(row("feature_mass_export_fraction", ctx["mass_export_fraction"]))
    out.update(row("feature_capacity_fraction", ctx["capacity_fraction"]))
    return out


def _sector_invariant_config(feature_config: InvariantFeatureConfig | None) -> InvariantFeatureConfig:
    cfg = feature_config or InvariantFeatureConfig(enabled=True)
    return InvariantFeatureConfig(
        enabled=True,
        eps=float(cfg.eps),
        clip=float(cfg.clip),
        range_floor_mode=str(cfg.range_floor_mode),
        range_floor=float(cfg.range_floor),
        include_raw_legacy_features=False,
    )


def _sector_slot_geometry(state: State, dtype: torch.dtype, device) -> dict[str, torch.Tensor]:
    mesh = state.mesh
    edges = mesh.safe_edges_on_cell.to(device=device).long()
    valid = mesh.valid_edges_on_cell.to(device=device)
    edge_cells = mesh.safe_cells_on_edge.to(device=device).long()[edges]
    cell_index = torch.arange(mesh.nCells, device=device, dtype=torch.long).view(mesh.nCells, 1)
    c0 = edge_cells[..., 0]
    c1 = edge_cells[..., 1]
    neighbor = torch.where(c0 == cell_index, c1, c0)
    neighbor = torch.where(valid, neighbor, cell_index)
    dv = mesh.dv_edge.to(dtype=dtype, device=device)[edges]
    dv = torch.where(valid, dv, torch.zeros_like(dv))
    valid_count = valid.to(dtype=dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
    uniform = valid.to(dtype=dtype) / valid_count
    prior = dv / dv.sum(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
    prior = torch.where(valid, prior, uniform)
    prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
    return {
        "edges": edges,
        "valid": valid,
        "neighbor": neighbor,
        "prior": prior,
        "edge_cells": edge_cells,
    }


def sector_remap_features(
    state: State,
    tracer_m: torch.Tensor,
    u_e: torch.Tensor,
    diagnostics: torch.Tensor,
    shadow_z: torch.Tensor,
    *,
    dt: float,
    feature_config: InvariantFeatureConfig | None = None,
) -> torch.Tensor:
    """Build invariant cell-edge-sector features for neural subcell remap."""

    cfg = _sector_invariant_config(feature_config)
    mesh = state.mesh
    dtype = tracer_m.dtype
    device = tracer_m.device
    q = tracer_m / state.layer_dp()[..., None].clamp_min(torch.finfo(dtype).tiny)
    if q.shape[-1] != 1:
        raise ValueError("sector remap features currently support one passive tracer channel")
    q_scalar = q[..., :1]
    slots = _sector_slot_geometry(state, dtype, device)
    edges = slots["edges"]
    valid = slots["valid"]
    neighbor = slots["neighbor"]
    prior = slots["prior"]
    edge_cells = slots["edge_cells"]
    n_levels = q_scalar.shape[1]
    cell_index = torch.arange(mesh.nCells, device=device, dtype=torch.long).view(mesh.nCells, 1)

    q_cell = q_scalar[:, None, :, :].expand(-1, edges.shape[1], -1, -1)
    q_neighbor = q_scalar[neighbor]
    q_min, q_max = _cell_bounds(state, q_scalar)
    stencil_min = torch.minimum(q_min[:, None, :, :], q_min[neighbor])
    stencil_max = torch.maximum(q_max[:, None, :, :], q_max[neighbor])
    q_range = (stencil_max - stencil_min).clamp_min(0.0)
    floor = _range_floor(q_scalar, cfg)
    q_range_safe = torch.maximum(q_range, floor)

    grad_e = mesh.grad(q_scalar[..., 0])[..., None][edges]
    lap = mesh.div(mesh.grad(q_scalar[..., 0]))[..., None]
    lap_cell = lap[:, None, :, :].expand(-1, edges.shape[1], -1, -1)
    lap_neighbor = lap[neighbor]
    area = mesh.cell_area.to(dtype=dtype, device=device)
    area_cell = area[:, None, None, None]
    area_neighbor = area[neighbor][..., None, None]
    L_cell = area_cell.sqrt().clamp_min(torch.finfo(dtype).tiny)
    dc = mesh.dc_edge.to(dtype=dtype, device=device)[edges]
    dv = mesh.dv_edge.to(dtype=dtype, device=device)[edges]
    L_edge = dc[..., None, None].clamp_min(torch.finfo(dtype).tiny)

    dp = state.layer_dp()
    dp_cell = dp[:, None, :, None]
    mass_cell = (area_cell * dp_cell).clamp_min(torch.finfo(dtype).tiny)
    mass_flux = mesh.cell_to_edge(dp) * u_e
    edge_flux = mass_flux[edges]
    cell_is_c0 = edge_cells[..., 0] == cell_index
    outward_flux = torch.where(cell_is_c0[..., None], edge_flux, -edge_flux)
    swept_fraction = float(dt) * dv[..., None] * outward_flux.abs() / mass_cell[..., 0].clamp_min(torch.finfo(dtype).tiny)
    signed_swept_fraction = float(dt) * dv[..., None] * outward_flux / mass_cell[..., 0].clamp_min(torch.finfo(dtype).tiny)

    D_cell = diagnostics[:, None, :, :].expand(-1, edges.shape[1], -1, -1)
    D_neighbor = diagnostics[neighbor]
    z_cell = shadow_z[:, None, :, :].expand(-1, edges.shape[1], -1, -1)
    z_neighbor = shadow_z[neighbor]
    edge_xyz = mesh.edge_xyz.to(dtype=dtype, device=device)[edges]
    edge_normal = mesh.edge_normal.to(dtype=dtype, device=device)[edges]
    cell_xyz = mesh.cell_xyz.to(dtype=dtype, device=device)[:, None, :].expand(-1, edges.shape[1], -1)
    rel_xyz = edge_xyz - cell_xyz
    ones = torch.ones_like(q_cell)
    pieces: list[torch.Tensor] = []

    def add(value: torch.Tensor) -> None:
        if value.shape[2] == 1 and n_levels != 1:
            value = value.expand(-1, -1, n_levels, -1)
        pieces.append(_feature_clip(value, cfg))

    add((q_cell - stencil_min) / q_range_safe)
    add((q_neighbor - stencil_min) / q_range_safe)
    add((q_neighbor - q_cell) / q_range_safe)
    add(L_edge * grad_e / q_range_safe)
    add(L_cell.square() * lap_cell / q_range_safe)
    add(L_cell.square() * lap_neighbor / q_range_safe)
    add(torch.log1p(q_range / floor.clamp_min(torch.finfo(dtype).tiny)))
    add(q_range / (q_range + floor))
    add(signed_swept_fraction[..., None])
    add(swept_fraction[..., None])
    add(prior[..., None, None].expand(-1, -1, n_levels, 1))
    add(torch.log(prior.clamp_min(float(cfg.eps)))[..., None, None].expand(-1, -1, n_levels, 1))
    add(dv[..., None, None] / L_cell)
    add(dc[..., None, None] / L_cell)
    add(area_neighbor / area_cell.clamp_min(torch.finfo(dtype).tiny))
    add(ones * edge_normal[:, :, None, :])
    add(ones * edge_xyz[:, :, None, :])
    add(ones * rel_xyz[:, :, None, :])
    add(D_cell)
    add(D_neighbor)
    add(z_cell)
    add(z_neighbor)
    add(torch.tanh(z_cell - D_cell))
    add(torch.tanh(z_neighbor - D_neighbor))

    features = torch.cat(pieces, dim=-1)
    return torch.where(valid[:, :, None, None], features, torch.zeros_like(features))


def infer_sector_feature_dim(
    state: State,
    tracer_m: torch.Tensor,
    u_e: torch.Tensor,
    diagnostics: torch.Tensor,
    shadow_z: torch.Tensor,
    *,
    dt: float,
    feature_config: InvariantFeatureConfig | None = None,
) -> int:
    return int(sector_remap_features(state, tracer_m, u_e, diagnostics, shadow_z, dt=dt, feature_config=feature_config).shape[-1])


def infer_feature_dim(
    state: State,
    tracer_m: torch.Tensor,
    u_e: torch.Tensor,
    diagnostics: torch.Tensor,
    shadow_z: torch.Tensor,
    *,
    dt: float,
    feature_config: InvariantFeatureConfig | None = None,
) -> int:
    return int(edge_reconstruction_features(state, tracer_m, u_e, diagnostics, shadow_z, dt=dt, feature_config=feature_config).shape[-1])


def _edge_bounds(
    state: State,
    q: torch.Tensor,
    *,
    mode: str,
    relaxed_eps_minus: float,
    relaxed_eps_plus: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mesh = state.mesh
    c0 = mesh.cellsOnEdge[:, 0]
    c1 = mesh.cellsOnEdge[:, 1]
    q_min, q_max = _cell_bounds(state, q[..., :1])
    lower = torch.minimum(q_min[c0], q_min[c1])
    upper = torch.maximum(q_max[c0], q_max[c1])
    if str(mode) == "relaxed":
        lower = lower - float(relaxed_eps_minus)
        upper = upper + float(relaxed_eps_plus)
    elif str(mode) != "strict":
        raise ValueError("neural admissibility mode must be strict or relaxed")
    lower = torch.maximum(lower, torch.zeros_like(lower))
    return lower, torch.maximum(upper, lower)


def _donor_export_scale(
    state: State,
    q: torch.Tensor,
    q_edge: torch.Tensor,
    mass_flux: torch.Tensor,
    *,
    dt: float,
    safety: float,
) -> torch.Tensor:
    mesh = state.mesh
    dtype = q.dtype
    device = q.device
    c0 = mesh.cellsOnEdge[:, 0].to(device=device)
    c1 = mesh.cellsOnEdge[:, 1].to(device=device)
    positive = mass_flux >= 0.0
    donor = torch.where(positive, c0[:, None], c1[:, None])
    export = (
        float(dt)
        * mesh.dv_edge.to(dtype=dtype, device=device)[:, None, None]
        * mass_flux.abs()[..., None]
        * q_edge.clamp_min(0.0)
    )
    n_cells, n_levels, n_tracers = q.shape
    flat_export = torch.zeros(n_cells * n_levels * n_tracers, dtype=dtype, device=device)
    level = torch.arange(n_levels, device=device).reshape(1, n_levels, 1)
    tracer = torch.arange(n_tracers, device=device).reshape(1, 1, n_tracers)
    flat_index = (donor[..., None] * n_levels * n_tracers + level * n_tracers + tracer).reshape(-1)
    flat_export.scatter_add_(0, flat_index, export.reshape(-1))
    total_export = flat_export.reshape(n_cells, n_levels, n_tracers)
    available = (
        float(safety)
        * state.mesh.cell_area.to(dtype=dtype, device=device)[:, None, None]
        * state.layer_dp()[..., None]
        * q.clamp_min(0.0)
    )
    cell_scale = torch.where(
        total_export > 0.0,
        torch.minimum(torch.ones_like(total_export), available / total_export.clamp_min(torch.finfo(dtype).tiny)),
        torch.ones_like(total_export),
    )
    edge_scale = cell_scale.reshape(-1)[flat_index].reshape(mesh.nEdges, n_levels, n_tracers)
    return edge_scale


def _scatter_edge_to_cell(state: State, values: torch.Tensor, donor: torch.Tensor) -> torch.Tensor:
    n_cells = state.mesh.nCells
    n_levels = values.shape[1]
    n_tracers = values.shape[2]
    device = values.device
    level = torch.arange(n_levels, device=device).reshape(1, n_levels, 1)
    tracer = torch.arange(n_tracers, device=device).reshape(1, 1, n_tracers)
    flat_index = (donor[..., None] * n_levels * n_tracers + level * n_tracers + tracer).reshape(-1)
    out = torch.zeros(n_cells * n_levels * n_tracers, dtype=values.dtype, device=device)
    out.scatter_add_(0, flat_index, values.reshape(-1))
    return out.reshape(n_cells, n_levels, n_tracers)


def _masked_corr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if int(mask.sum().detach().cpu()) < 2:
        return torch.zeros((), dtype=a.dtype, device=a.device)
    av = a[mask]
    bv = b[mask]
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = av.std(unbiased=False) * bv.std(unbiased=False)
    eps = torch.as_tensor(1.0e-12, dtype=a.dtype, device=a.device)
    return torch.where(denom > eps, (av * bv).mean() / denom, torch.zeros((), dtype=a.dtype, device=a.device))


def _channel_or_zero(values: torch.Tensor, index: int) -> torch.Tensor:
    if values.shape[-1] > int(index):
        return values[..., int(index) : int(index) + 1]
    return torch.zeros_like(values[..., :1])


def _allocation_envelope_terms(
    state: State,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    donor: torch.Tensor,
    model: EdgeAllocationMLP,
    features: torch.Tensor,
    *,
    config: NeuralAllocationConfig,
) -> dict[str, torch.Tensor]:
    q_stencil_max_e = _gather_cell_level(q_max, donor).clamp_min(0.0)
    local_range_e = _gather_cell_level((q_max - q_min).clamp_min(0.0), donor)
    zero = torch.zeros_like(q_stencil_max_e)
    envelope = str(config.envelope).lower()
    if envelope == "scalar":
        if str(config.mode) == "relaxed":
            delta = torch.full_like(q_stencil_max_e, float(config.relaxed_eps))
            q_plus_e = q_stencil_max_e + delta
        else:
            delta = zero
            q_plus_e = q_stencil_max_e
        eta = zero
        scale = local_range_e + float(config.envelope_eps_floor)
        pref_bound = q_plus_e
    elif envelope == "anisotropic":
        eta = _model_envelope_eta(model, features, state=state).clamp(0.0, 1.0)
        scale = local_range_e + float(config.envelope_eps_floor)
        delta = eta * float(config.envelope_max) * scale
        q_plus_e = q_stencil_max_e + delta
        # In anisotropic mode the direction-dependent envelope affects both
        # feasibility and allocation preference. Scalar mode keeps the old
        # upwind-compatible cancellation exactly.
        pref_bound = q_stencil_max_e
    else:
        raise ValueError("neural allocation envelope must be 'scalar' or 'anisotropic'")
    return {
        "q_plus_e": q_plus_e.clamp_min(0.0),
        "q_stencil_max_e": q_stencil_max_e,
        "local_range_e": local_range_e,
        "eta": eta,
        "delta": delta.clamp_min(0.0),
        "scale": scale,
        "pref_bound": pref_bound.clamp_min(0.0),
    }


def neural_allocation_tracer_tendency(
    state: State,
    tracer_m: torch.Tensor,
    u_e: torch.Tensor,
    model: EdgeAllocationMLP,
    diagnostics: torch.Tensor,
    shadow_z: torch.Tensor,
    *,
    dt: float,
    config: NeuralAllocationConfig | None = None,
    feature_config: InvariantFeatureConfig | None = None,
    return_info: bool = False,
    include_tensors: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Conservative tracer tendency from donor-cell export allocation."""

    config = config or NeuralAllocationConfig()
    mesh = state.mesh
    dtype = tracer_m.dtype
    device = tracer_m.device
    q = tracer_m / state.layer_dp()[..., None].clamp_min(torch.finfo(dtype).tiny)
    if q.shape[-1] != 1:
        raise ValueError("neural_allocation_tracer_tendency currently supports one passive tracer channel")
    q_scalar = q[..., :1]
    n_levels = q_scalar.shape[1]

    dp_e = mesh.cell_to_edge(state.layer_dp())
    mass_flux = dp_e * u_e
    c0 = mesh.cellsOnEdge[:, 0].to(device=device)
    c1 = mesh.cellsOnEdge[:, 1].to(device=device)
    positive = mass_flux >= 0.0
    donor = torch.where(positive, c0[:, None], c1[:, None])
    sign = torch.where(positive, torch.ones_like(mass_flux), -torch.ones_like(mass_flux))[..., None]
    dv = mesh.dv_edge.to(dtype=dtype, device=device)
    M = float(dt) * dv[:, None] * mass_flux.abs()
    outgoing = M > float(config.zero_flux_tolerance)

    if str(config.mode) not in {"strict", "relaxed"}:
        raise ValueError("neural allocation mode must be strict or relaxed")
    q_min, q_max = _cell_bounds(state, q_scalar)
    features = edge_reconstruction_features(state, tracer_m, u_e, diagnostics, shadow_z, dt=dt, feature_config=feature_config)
    envelope = _allocation_envelope_terms(
        state,
        q_min,
        q_max,
        donor,
        model,
        features,
        config=config,
    )
    q_plus_e = envelope["q_plus_e"]
    q_stencil_max_e = envelope["q_stencil_max_e"]
    q_envelope_delta = envelope["delta"]
    q_envelope_eta = envelope["eta"]
    pref_bound = envelope["pref_bound"]
    q_donor = _gather_cell_level(q_scalar, donor)

    edge_capacity = M[..., None] * q_plus_e
    base_edge_pref = torch.where(
        pref_bound > float(config.eps),
        (q_donor / pref_bound.clamp_min(float(config.eps))).clamp_min(0.0),
        torch.zeros_like(q_donor),
    )
    edge_multiplier = _model_forward(model, features, state=state)
    edge_weight = base_edge_pref * edge_multiplier * edge_capacity
    edge_weight = torch.where(outgoing[..., None], edge_weight, torch.zeros_like(edge_weight))

    area = mesh.cell_area.to(dtype=dtype, device=device)[:, None, None]
    available = area * tracer_m[..., :1].clamp_min(0.0)
    base_export = torch.where(outgoing[..., None], M[..., None] * q_donor.clamp_min(0.0), torch.zeros_like(edge_weight))
    base_export_sum = _scatter_edge_to_cell(state, base_export, donor)
    base_stay_weight = torch.relu(available - base_export_sum)
    stay_weight = base_stay_weight * model.stay_preference(dtype=dtype, device=device)
    edge_weight_sum = _scatter_edge_to_cell(state, edge_weight, donor)
    denom_cell = edge_weight_sum + stay_weight + float(config.eps)
    denom_edge = _gather_cell_level(denom_cell, donor)
    available_edge = _gather_cell_level(available, donor)

    raw_export = available_edge * edge_weight / denom_edge.clamp_min(float(config.eps))
    export = torch.minimum(raw_export, edge_capacity)
    export = torch.where(outgoing[..., None], export, torch.zeros_like(export))

    flux = sign * export / (float(dt) * dv[:, None, None]).clamp_min(float(config.eps))
    tendency = -mesh.div(flux.reshape(mesh.nEdges, -1)).reshape(mesh.nCells, n_levels, 1)
    if not return_info:
        return tendency

    export_sum = _scatter_edge_to_cell(state, export, donor)
    export_fraction = torch.where(
        available > float(config.eps),
        export_sum / available.clamp_min(float(config.eps)),
        torch.zeros_like(available),
    )
    stay_fraction = torch.where(
        available > float(config.eps),
        torch.relu(available - export_sum) / available.clamp_min(float(config.eps)),
        torch.ones_like(available),
    )
    utilization = torch.where(
        edge_capacity > float(config.eps),
        export / edge_capacity.clamp_min(float(config.eps)),
        torch.zeros_like(export),
    )
    q_edge = torch.where(M[..., None] > float(config.zero_flux_tolerance), export / M[..., None].clamp_min(float(config.eps)), torch.zeros_like(export))
    outgoing_count_cell = _scatter_edge_to_cell(state, outgoing[..., None].to(dtype=dtype), donor).clamp_min(1.0)
    utilization_cell = _scatter_edge_to_cell(state, utilization, donor) / outgoing_count_cell
    edge_prob = edge_weight / denom_edge.clamp_min(float(config.eps))
    edge_entropy = _scatter_edge_to_cell(
        state,
        torch.where(edge_prob > 0.0, -edge_prob * torch.log(edge_prob.clamp_min(float(config.eps))), torch.zeros_like(edge_prob)),
        donor,
    )
    stay_prob = stay_weight / denom_cell.clamp_min(float(config.eps))
    entropy_cell = edge_entropy + torch.where(
        stay_prob > 0.0,
        -stay_prob * torch.log(stay_prob.clamp_min(float(config.eps))),
        torch.zeros_like(stay_prob),
    )
    active_edge = outgoing[..., None] & (edge_capacity > float(config.eps))
    active_cell = available > float(config.eps)
    q_edge_active = q_edge[active_edge]
    utilization_active = utilization[active_edge]
    multiplier_active = edge_multiplier[active_edge]
    eta_active = q_envelope_eta[active_edge]
    delta_active = q_envelope_delta[active_edge]
    q_plus_active = q_plus_e[active_edge]
    q_stencil_active = q_stencil_max_e[active_edge]
    D_donor = _gather_cell_level(diagnostics, donor)
    z_donor = _gather_cell_level(shadow_z, donor)
    z_lag_mag = (z_donor - D_donor).square().mean(dim=-1, keepdim=True).sqrt()
    shadow_grad = _channel_or_zero(z_donor, 0)
    deformation = _channel_or_zero(D_donor, 3)
    local_range = envelope["local_range_e"]
    edge_velocity = u_e.abs()[..., None]
    scalar_info: dict[str, Any] = {
        "alloc_export_fraction_mean": float(export_fraction[active_cell].detach().mean().cpu()) if active_cell.any() else 0.0,
        "alloc_export_fraction_max": float(export_fraction.detach().amax().cpu()),
        "alloc_stay_fraction_mean": float(stay_fraction[active_cell].detach().mean().cpu()) if active_cell.any() else 1.0,
        "alloc_stay_fraction_min": float(stay_fraction.detach().amin().cpu()),
        "alloc_capacity_utilization_mean": float(utilization_active.detach().mean().cpu()) if active_edge.any() else 0.0,
        "alloc_capacity_utilization_max": float(utilization.detach().amax().cpu()),
        "alloc_capacity_saturation_rate": float((utilization_active > 1.0 - 1.0e-6).to(dtype=dtype).mean().detach().cpu()) if active_edge.any() else 0.0,
        "alloc_q_edge_min": float(q_edge_active.detach().amin().cpu()) if q_edge_active.numel() else 0.0,
        "alloc_q_edge_max": float(q_edge_active.detach().amax().cpu()) if q_edge_active.numel() else 0.0,
        "alloc_near_full_export_cell_count": int(((export_fraction > 0.99) & active_cell).sum().detach().cpu()),
        "alloc_zero_flux_edge_count": int((~outgoing).sum().detach().cpu()),
        "alloc_donor_overexport_max": float(torch.relu(export_sum - available).detach().amax().cpu()),
        "alloc_preference_entropy_mean": float(entropy_cell[active_cell].detach().mean().cpu()) if active_cell.any() else 0.0,
        "alloc_edge_preference_mean": float(multiplier_active.detach().mean().cpu()) if active_edge.any() else 0.0,
        "alloc_edge_preference_max": float(edge_multiplier.detach().amax().cpu()),
        "alloc_stay_preference": float(model.stay_preference(dtype=dtype, device=device).detach().cpu()),
        "alloc_pref_shadow0_correlation": float(_masked_corr(edge_multiplier, shadow_grad, active_edge).detach().cpu()) if active_edge.any() else 0.0,
        "alloc_envelope_mode": str(config.envelope),
        "alloc_envelope_eta_mean": float(eta_active.detach().mean().cpu()) if eta_active.numel() else 0.0,
        "alloc_envelope_eta_max": float(q_envelope_eta.detach().amax().cpu()),
        "alloc_envelope_eta_std": float(eta_active.detach().std(unbiased=False).cpu()) if eta_active.numel() else 0.0,
        "alloc_envelope_delta_mean": float(delta_active.detach().mean().cpu()) if delta_active.numel() else 0.0,
        "alloc_envelope_delta_max": float(q_envelope_delta.detach().amax().cpu()),
        "alloc_q_plus_mean": float(q_plus_active.detach().mean().cpu()) if q_plus_active.numel() else 0.0,
        "alloc_q_plus_max": float(q_plus_e.detach().amax().cpu()),
        "alloc_q_plus_minus_stencil_max_mean": float((q_plus_active - q_stencil_active).detach().mean().cpu()) if q_plus_active.numel() else 0.0,
        "alloc_q_plus_minus_stencil_max_max": float((q_plus_e - q_stencil_max_e).detach().amax().cpu()),
        "alloc_fraction_edges_with_nonzero_envelope": float((delta_active > float(config.eps)).to(dtype=dtype).mean().detach().cpu()) if delta_active.numel() else 0.0,
        "alloc_eta_shadow_grad_correlation": float(_masked_corr(q_envelope_eta, shadow_grad, active_edge).detach().cpu()) if active_edge.any() else 0.0,
        "alloc_eta_z_d_lag_correlation": float(_masked_corr(q_envelope_eta, z_lag_mag, active_edge).detach().cpu()) if active_edge.any() else 0.0,
        "alloc_eta_deformation_correlation": float(_masked_corr(q_envelope_eta, deformation, active_edge).detach().cpu()) if active_edge.any() else 0.0,
        "alloc_eta_edge_velocity_correlation": float(_masked_corr(q_envelope_eta, edge_velocity, active_edge).detach().cpu()) if active_edge.any() else 0.0,
        "alloc_eta_local_q_range_correlation": float(_masked_corr(q_envelope_eta, local_range, active_edge).detach().cpu()) if active_edge.any() else 0.0,
    }
    scalar_info.update(feature_audit_summary(features, feature_config=feature_config))
    scalar_info.update(
        invariant_feature_scale_summary(
            state,
            tracer_m,
            u_e,
            diagnostics,
            shadow_z,
            dt=dt,
            feature_config=feature_config,
        )
    )
    if include_tensors:
        count = _scatter_edge_to_cell(state, outgoing[..., None].to(dtype=dtype), donor).clamp_min(1.0)
        scalar_info.update(
            {
                "alloc_stay_fraction": stay_fraction.detach().cpu(),
                "alloc_export_fraction": export_fraction.detach().cpu(),
                "alloc_capacity_utilization_cell": utilization_cell.detach().cpu(),
                "alloc_envelope_eta_cell": (_scatter_edge_to_cell(state, torch.where(outgoing[..., None], q_envelope_eta, torch.zeros_like(q_envelope_eta)), donor) / count).detach().cpu(),
                "alloc_envelope_delta_cell": (_scatter_edge_to_cell(state, torch.where(outgoing[..., None], q_envelope_delta, torch.zeros_like(q_envelope_delta)), donor) / count).detach().cpu(),
                "alloc_q_plus_cell": (_scatter_edge_to_cell(state, torch.where(outgoing[..., None], q_plus_e, torch.zeros_like(q_plus_e)), donor) / count).detach().cpu(),
            }
        )
    return tendency, scalar_info


def _residual_allocation_edge_terms(
    state: State,
    tracer_m: torch.Tensor,
    u_e: torch.Tensor,
    model: EdgeAllocationMLP,
    diagnostics: torch.Tensor,
    shadow_z: torch.Tensor,
    *,
    dt: float,
    config: NeuralAllocationConfig,
    feature_config: InvariantFeatureConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Build edge terms for the residual-upwind allocator path."""

    mesh = state.mesh
    dtype = tracer_m.dtype
    device = tracer_m.device
    q = tracer_m / state.layer_dp()[..., None].clamp_min(torch.finfo(dtype).tiny)
    if q.shape[-1] != 1:
        raise ValueError("neural residual allocation currently supports one passive tracer channel")
    if str(config.mode) not in {"strict", "relaxed"}:
        raise ValueError("neural allocation mode must be strict or relaxed")

    q_scalar = q[..., :1]
    dp_e = mesh.cell_to_edge(state.layer_dp())
    mass_flux = dp_e * u_e
    c0 = mesh.cellsOnEdge[:, 0].to(device=device)
    c1 = mesh.cellsOnEdge[:, 1].to(device=device)
    q0 = q_scalar[c0]
    q1 = q_scalar[c1]
    positive = mass_flux >= 0.0
    donor = torch.where(positive, c0[:, None], c1[:, None])
    q_donor = _gather_cell_level(q_scalar, donor)
    q_centered = 0.5 * (q0 + q1)
    dv = mesh.dv_edge.to(dtype=dtype, device=device)
    M = float(dt) * dv[:, None] * mass_flux.abs()
    outgoing = M > float(config.zero_flux_tolerance)

    features = edge_reconstruction_features(state, tracer_m, u_e, diagnostics, shadow_z, dt=dt, feature_config=feature_config)
    alpha = float(config.residual_alpha_max) * _model_envelope_eta(model, features, state=state).clamp(0.0, 1.0)
    alpha = alpha.clamp_min(0.0)
    q_edge_raw = q_donor + alpha * (q_centered - q_donor)
    lower, upper = _edge_bounds(
        state,
        q,
        mode=config.mode,
        relaxed_eps_minus=config.relaxed_eps,
        relaxed_eps_plus=config.relaxed_eps,
    )
    q_edge_bounded = q_edge_raw.clamp(min=lower, max=upper)
    scale = _donor_export_scale(
        state,
        q_scalar,
        q_edge_bounded,
        mass_flux,
        dt=dt,
        safety=1.0,
    )
    q_edge = torch.where(outgoing[..., None], q_edge_bounded * scale, torch.zeros_like(q_edge_bounded))
    flux = mass_flux[..., None] * q_edge
    export = torch.where(outgoing[..., None], M[..., None] * q_edge.clamp_min(0.0), torch.zeros_like(q_edge))
    edge_capacity = torch.where(outgoing[..., None], M[..., None] * upper.clamp_min(0.0), torch.zeros_like(q_edge))
    return {
        "q": q_scalar,
        "mass_flux": mass_flux,
        "donor": donor,
        "outgoing": outgoing,
        "M": M,
        "q_donor": q_donor,
        "q_centered": q_centered,
        "q_edge_raw": q_edge_raw,
        "q_edge_bounded": q_edge_bounded,
        "q_edge": q_edge,
        "q_lower": lower,
        "q_upper": upper,
        "scale": scale,
        "alpha": alpha,
        "features": features,
        "flux": flux,
        "export": export,
        "edge_capacity": edge_capacity,
    }


def neural_residual_allocation_tracer_tendency(
    state: State,
    tracer_m: torch.Tensor,
    u_e: torch.Tensor,
    model: EdgeAllocationMLP,
    diagnostics: torch.Tensor,
    shadow_z: torch.Tensor,
    *,
    dt: float,
    config: NeuralAllocationConfig | None = None,
    feature_config: InvariantFeatureConfig | None = None,
    return_info: bool = False,
    include_tensors: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Conservative upwind transport plus a bounded learned edge correction.

    Unlike the full donor-cell allocator, this path cannot choose to keep all
    tracer mass stationary.  With a near-zero residual alpha it reduces to the
    ordinary upwind flux, while training can learn a limited correction toward
    centered edge concentrations.
    """

    config = config or NeuralAllocationConfig()
    mesh = state.mesh
    dtype = tracer_m.dtype
    device = tracer_m.device
    terms = _residual_allocation_edge_terms(
        state,
        tracer_m,
        u_e,
        model,
        diagnostics,
        shadow_z,
        dt=dt,
        config=config,
        feature_config=feature_config,
    )
    q = terms["q"]
    n_levels = q.shape[1]
    tendency = -mesh.div(terms["flux"].reshape(mesh.nEdges, -1)).reshape(mesh.nCells, n_levels, 1)
    if not return_info:
        return tendency

    donor = terms["donor"]
    outgoing = terms["outgoing"]
    export = terms["export"]
    edge_capacity = terms["edge_capacity"]
    q_edge = terms["q_edge"]
    q_donor = terms["q_donor"]
    q_centered = terms["q_centered"]
    alpha = terms["alpha"]
    scale = terms["scale"]
    M = terms["M"]
    features = terms["features"]
    area = mesh.cell_area.to(dtype=dtype, device=device)[:, None, None]
    available = area * tracer_m[..., :1].clamp_min(0.0)
    export_sum = _scatter_edge_to_cell(state, export, donor)
    export_fraction = torch.where(
        available > float(config.eps),
        export_sum / available.clamp_min(float(config.eps)),
        torch.zeros_like(available),
    )
    stay_fraction = torch.where(
        available > float(config.eps),
        torch.relu(available - export_sum) / available.clamp_min(float(config.eps)),
        torch.ones_like(available),
    )
    utilization = torch.where(
        edge_capacity > float(config.eps),
        export / edge_capacity.clamp_min(float(config.eps)),
        torch.zeros_like(export),
    )
    outgoing_count_cell = _scatter_edge_to_cell(state, outgoing[..., None].to(dtype=dtype), donor).clamp_min(1.0)
    utilization_cell = _scatter_edge_to_cell(state, utilization, donor) / outgoing_count_cell
    active_edge = outgoing[..., None]
    active_cell = available > float(config.eps)
    q_edge_active = q_edge[active_edge]
    utilization_active = utilization[active_edge]
    alpha_active = alpha[active_edge]
    residual_delta = q_edge - q_donor
    raw_residual_delta = terms["q_edge_raw"] - q_donor
    centered_delta = q_centered - q_donor
    D_donor = _gather_cell_level(diagnostics, donor)
    z_donor = _gather_cell_level(shadow_z, donor)
    z_lag_mag = (z_donor - D_donor).square().mean(dim=-1, keepdim=True).sqrt()
    shadow_grad = _channel_or_zero(z_donor, 0)
    deformation = _channel_or_zero(D_donor, 3)
    edge_velocity = u_e.abs()[..., None]

    scalar_info: dict[str, Any] = {
        "alloc_mode": "residual_upwind",
        "alloc_export_fraction_mean": float(export_fraction[active_cell].detach().mean().cpu()) if active_cell.any() else 0.0,
        "alloc_export_fraction_max": float(export_fraction.detach().amax().cpu()),
        "alloc_stay_fraction_mean": float(stay_fraction[active_cell].detach().mean().cpu()) if active_cell.any() else 1.0,
        "alloc_stay_fraction_min": float(stay_fraction.detach().amin().cpu()),
        "alloc_capacity_utilization_mean": float(utilization_active.detach().mean().cpu()) if active_edge.any() else 0.0,
        "alloc_capacity_utilization_max": float(utilization.detach().amax().cpu()),
        "alloc_capacity_saturation_rate": float((utilization_active > 1.0 - 1.0e-6).to(dtype=dtype).mean().detach().cpu()) if active_edge.any() else 0.0,
        "alloc_q_edge_min": float(q_edge_active.detach().amin().cpu()) if q_edge_active.numel() else 0.0,
        "alloc_q_edge_max": float(q_edge_active.detach().amax().cpu()) if q_edge_active.numel() else 0.0,
        "alloc_near_full_export_cell_count": int(((export_fraction > 0.99) & active_cell).sum().detach().cpu()),
        "alloc_zero_flux_edge_count": int((~outgoing).sum().detach().cpu()),
        "alloc_donor_overexport_max": float(torch.relu(export_sum - available).detach().amax().cpu()),
        "alloc_preference_entropy_mean": 0.0,
        "alloc_edge_preference_mean": float(alpha_active.detach().mean().cpu()) if alpha_active.numel() else 0.0,
        "alloc_edge_preference_max": float(alpha.detach().amax().cpu()),
        "alloc_stay_preference": 0.0,
        "alloc_pref_shadow0_correlation": float(_masked_corr(alpha, shadow_grad, active_edge).detach().cpu()) if active_edge.any() else 0.0,
        "alloc_residual_alpha_mean": float(alpha_active.detach().mean().cpu()) if alpha_active.numel() else 0.0,
        "alloc_residual_alpha_max": float(alpha.detach().amax().cpu()),
        "alloc_residual_alpha_l2": float(alpha_active.square().mean().detach().cpu()) if alpha_active.numel() else 0.0,
        "alloc_residual_q_delta_abs_mean": float(residual_delta[active_edge].detach().abs().mean().cpu()) if active_edge.any() else 0.0,
        "alloc_residual_q_delta_abs_max": float(residual_delta.detach().abs().amax().cpu()),
        "alloc_residual_raw_q_delta_abs_mean": float(raw_residual_delta[active_edge].detach().abs().mean().cpu()) if active_edge.any() else 0.0,
        "alloc_residual_centered_delta_abs_mean": float(centered_delta[active_edge].detach().abs().mean().cpu()) if active_edge.any() else 0.0,
        "alloc_residual_edge_scale_min": float(scale.detach().amin().cpu()),
        "alloc_residual_edge_scale_mean": float(scale.detach().mean().cpu()),
        "alloc_residual_shadow_grad_correlation": float(_masked_corr(alpha, shadow_grad, active_edge).detach().cpu()) if active_edge.any() else 0.0,
        "alloc_residual_z_d_lag_correlation": float(_masked_corr(alpha, z_lag_mag, active_edge).detach().cpu()) if active_edge.any() else 0.0,
        "alloc_residual_deformation_correlation": float(_masked_corr(alpha, deformation, active_edge).detach().cpu()) if active_edge.any() else 0.0,
        "alloc_residual_edge_velocity_correlation": float(_masked_corr(alpha, edge_velocity, active_edge).detach().cpu()) if active_edge.any() else 0.0,
    }
    scalar_info.update(feature_audit_summary(features, feature_config=feature_config))
    scalar_info.update(
        invariant_feature_scale_summary(
            state,
            tracer_m,
            u_e,
            diagnostics,
            shadow_z,
            dt=dt,
            feature_config=feature_config,
        )
    )
    if include_tensors:
        alpha_cell = _scatter_edge_to_cell(state, torch.where(active_edge, alpha, torch.zeros_like(alpha)), donor) / outgoing_count_cell
        delta_cell = _scatter_edge_to_cell(
            state,
            torch.where(active_edge, residual_delta.abs(), torch.zeros_like(residual_delta)),
            donor,
        ) / outgoing_count_cell
        scalar_info.update(
            {
                "alloc_stay_fraction": stay_fraction.detach().cpu(),
                "alloc_export_fraction": export_fraction.detach().cpu(),
                "alloc_capacity_utilization_cell": utilization_cell.detach().cpu(),
                "alloc_residual_alpha_cell": alpha_cell.detach().cpu(),
                "alloc_residual_q_delta_abs_cell": delta_cell.detach().cpu(),
            }
        )
    return tendency, scalar_info


def _edge_sector_indices(state: State, edges_on_cell: torch.Tensor, valid_edges: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mesh = state.mesh
    device = edges_on_cell.device
    edge_ids = torch.arange(mesh.nEdges, device=device, dtype=torch.long).view(mesh.nEdges, 1)
    c0 = mesh.cellsOnEdge[:, 0].to(device=device).long()
    c1 = mesh.cellsOnEdge[:, 1].to(device=device).long()
    mask0 = (edges_on_cell[c0] == edge_ids) & valid_edges[c0]
    mask1 = (edges_on_cell[c1] == edge_ids) & valid_edges[c1]
    return mask0.to(dtype=torch.long).argmax(dim=1), mask1.to(dtype=torch.long).argmax(dim=1)


def _sector_remap_terms(
    state: State,
    tracer_m: torch.Tensor,
    u_e: torch.Tensor,
    model: SectorRemapMLP,
    diagnostics: torch.Tensor,
    shadow_z: torch.Tensor,
    *,
    dt: float,
    config: SectorRemapConfig,
    feature_config: InvariantFeatureConfig | None = None,
) -> dict[str, torch.Tensor]:
    mesh = state.mesh
    dtype = tracer_m.dtype
    device = tracer_m.device
    q = tracer_m / state.layer_dp()[..., None].clamp_min(torch.finfo(dtype).tiny)
    if q.shape[-1] != 1:
        raise ValueError("neural sector remap currently supports one passive tracer channel")
    q_scalar = q[..., :1]
    slots = _sector_slot_geometry(state, dtype, device)
    edges = slots["edges"]
    valid = slots["valid"]
    prior = slots["prior"].clamp_min(float(config.min_prior))
    prior = torch.where(valid, prior, torch.zeros_like(prior))
    prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(float(config.eps))
    if getattr(model, "uses_mesh_convolution", False):
        features = edge_reconstruction_features(
            state,
            tracer_m,
            u_e,
            diagnostics,
            shadow_z,
            dt=dt,
            feature_config=feature_config,
        )
        raw_edge_logits = _model_forward(model, features, state=state)
        raw_logits = raw_edge_logits[edges]
    else:
        features = sector_remap_features(state, tracer_m, u_e, diagnostics, shadow_z, dt=dt, feature_config=feature_config)
        raw_logits = _model_forward(model, features, state=state)
    residual_logits = float(config.logit_scale) * torch.tanh(raw_logits)
    log_prior = torch.log(prior.clamp_min(float(config.min_prior)))[..., None, None]
    invalid = ~valid[:, :, None, None]
    low = -1.0e30 if dtype == torch.float64 else -1.0e20
    logits = torch.where(invalid, torch.full_like(residual_logits, low), log_prior + residual_logits)
    prob = torch.softmax(logits, dim=1)
    prob = torch.where(invalid, torch.zeros_like(prob), prob)
    prob = prob / prob.sum(dim=1, keepdim=True).clamp_min(float(config.eps))
    ratio = prob / prior[..., None, None].clamp_min(float(config.min_prior))
    q_sector = q_scalar[:, None, :, :] * ratio

    slot0, slot1 = _edge_sector_indices(state, edges, valid)
    c0 = mesh.cellsOnEdge[:, 0].to(device=device).long()
    c1 = mesh.cellsOnEdge[:, 1].to(device=device).long()
    q0 = q_sector[c0, slot0]
    q1 = q_sector[c1, slot1]
    dp_e = mesh.cell_to_edge(state.layer_dp())
    mass_flux = dp_e * u_e
    positive = mass_flux >= 0.0
    donor = torch.where(positive, c0[:, None], c1[:, None])
    q_edge = torch.where(positive[..., None], q0, q1)
    flux = mass_flux[..., None] * q_edge
    dv = mesh.dv_edge.to(dtype=dtype, device=device)
    swept_mass = float(dt) * dv[:, None] * mass_flux.abs()
    outgoing = swept_mass > float(config.zero_flux_tolerance)
    export = torch.where(outgoing[..., None], swept_mass[..., None] * q_edge, torch.zeros_like(q_edge))
    return {
        "q": q_scalar,
        "features": features,
        "raw_logits": raw_logits,
        "residual_logits": residual_logits,
        "prob": prob,
        "prior": prior,
        "ratio": ratio,
        "q_sector": q_sector,
        "q_edge": q_edge,
        "flux": flux,
        "mass_flux": mass_flux,
        "donor": donor,
        "outgoing": outgoing,
        "swept_mass": swept_mass,
        "export": export,
        "valid": valid,
    }


def neural_sector_remap_tracer_tendency(
    state: State,
    tracer_m: torch.Tensor,
    u_e: torch.Tensor,
    model: SectorRemapMLP,
    diagnostics: torch.Tensor,
    shadow_z: torch.Tensor,
    *,
    dt: float,
    config: SectorRemapConfig | None = None,
    feature_config: InvariantFeatureConfig | None = None,
    return_info: bool = False,
    include_tensors: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Conservative FV tendency from a learned subcell edge-sector remap.

    The model redistributes each cell's tracer inventory across one sector per
    incident edge, then the fixed upwind carrier-mass flux samples the upwind
    sector concentration. No local extrema limiter is applied in this first
    experimental path.
    """

    config = config or SectorRemapConfig()
    mesh = state.mesh
    dtype = tracer_m.dtype
    device = tracer_m.device
    terms = _sector_remap_terms(
        state,
        tracer_m,
        u_e,
        model,
        diagnostics,
        shadow_z,
        dt=dt,
        config=config,
        feature_config=feature_config,
    )
    q = terms["q"]
    tendency = -mesh.div(terms["flux"].reshape(mesh.nEdges, -1)).reshape(mesh.nCells, q.shape[1], 1)
    if not return_info:
        return tendency

    donor = terms["donor"]
    export = terms["export"]
    outgoing = terms["outgoing"]
    q_edge = terms["q_edge"]
    prob = terms["prob"]
    prior = terms["prior"]
    ratio = terms["ratio"]
    q_sector = terms["q_sector"]
    residual_logits = terms["residual_logits"]
    features = terms["features"]
    valid = terms["valid"]
    area = mesh.cell_area.to(dtype=dtype, device=device)[:, None, None]
    available = area * tracer_m[..., :1]
    export_sum = _scatter_edge_to_cell(state, export, donor)
    eps = float(config.eps)
    export_fraction = torch.where(
        available.abs() > eps,
        export_sum / available.clamp_min(eps),
        torch.zeros_like(available),
    )
    stay_fraction = 1.0 - export_fraction
    active_edge = outgoing[..., None]
    active_sector = valid[:, :, None, None].expand_as(prob)
    entropy = torch.where(prob > 0.0, -prob * torch.log(prob.clamp_min(eps)), torch.zeros_like(prob)).sum(dim=1)
    q_edge_active = q_edge[active_edge]
    ratio_active = ratio[active_sector]
    logit_active = residual_logits[active_sector]
    q_sector_active = q_sector[active_sector]
    export_active = export[active_edge]
    shadow_cell = _channel_or_zero(shadow_z, 0)
    sector_pref_cell = (ratio * prob).sum(dim=1)
    scalar_info: dict[str, Any] = {
        "alloc_mode": "sector_remap",
        "alloc_export_fraction_mean": float(export_fraction.detach().mean().cpu()),
        "alloc_export_fraction_max": float(export_fraction.detach().amax().cpu()),
        "alloc_stay_fraction_mean": float(stay_fraction.detach().mean().cpu()),
        "alloc_stay_fraction_min": float(stay_fraction.detach().amin().cpu()),
        "alloc_donor_overexport_max": float(torch.relu(export_sum - available).detach().amax().cpu()),
        "alloc_zero_flux_edge_count": int((~outgoing).sum().detach().cpu()),
        "alloc_preference_entropy_mean": float(entropy.detach().mean().cpu()),
        "alloc_edge_preference_mean": float(ratio_active.detach().mean().cpu()) if ratio_active.numel() else 0.0,
        "alloc_edge_preference_max": float(ratio.detach().amax().cpu()),
        "alloc_pref_shadow0_correlation": float(_masked_corr(sector_pref_cell, shadow_cell, torch.ones_like(sector_pref_cell, dtype=torch.bool)).detach().cpu()),
        "sector_logit_scale": float(config.logit_scale),
        "sector_residual_logit_abs_mean": float(logit_active.detach().abs().mean().cpu()) if logit_active.numel() else 0.0,
        "sector_residual_logit_abs_max": float(residual_logits.detach().abs().amax().cpu()),
        "sector_prior_min": float(prior[valid].detach().amin().cpu()) if bool(valid.any()) else 0.0,
        "sector_prior_max": float(prior.detach().amax().cpu()),
        "sector_ratio_mean": float(ratio_active.detach().mean().cpu()) if ratio_active.numel() else 0.0,
        "sector_ratio_min": float(ratio_active.detach().amin().cpu()) if ratio_active.numel() else 0.0,
        "sector_ratio_max": float(ratio.detach().amax().cpu()),
        "sector_q_min": float(q_sector_active.detach().amin().cpu()) if q_sector_active.numel() else 0.0,
        "sector_q_max": float(q_sector_active.detach().amax().cpu()) if q_sector_active.numel() else 0.0,
        "sector_q_edge_min": float(q_edge_active.detach().amin().cpu()) if q_edge_active.numel() else 0.0,
        "sector_q_edge_max": float(q_edge_active.detach().amax().cpu()) if q_edge_active.numel() else 0.0,
        "sector_export_mean": float(export_active.detach().mean().cpu()) if export_active.numel() else 0.0,
        "sector_export_max": float(export.detach().amax().cpu()),
    }
    sector_cfg = _sector_invariant_config(feature_config)
    scalar_info.update(feature_audit_summary(features, feature_config=sector_cfg))
    if include_tensors:
        scalar_info.update(
            {
                "alloc_stay_fraction": stay_fraction.detach().cpu(),
                "alloc_export_fraction": export_fraction.detach().cpu(),
                "sector_ratio_cell": sector_pref_cell.detach().cpu(),
            }
        )
    return tendency, scalar_info


def neural_allocation_envelope_penalty(
    state: State,
    tracer_m: torch.Tensor,
    u_e: torch.Tensor,
    model: EdgeAllocationMLP,
    diagnostics: torch.Tensor,
    shadow_z: torch.Tensor,
    *,
    dt: float,
    config: NeuralAllocationConfig | None = None,
    feature_config: InvariantFeatureConfig | None = None,
) -> torch.Tensor:
    """Differentiable mean-square anisotropic envelope size penalty."""

    config = config or NeuralAllocationConfig()
    dtype = tracer_m.dtype
    device = tracer_m.device
    if str(config.envelope).lower() != "anisotropic":
        return torch.zeros((), dtype=dtype, device=device)
    q = tracer_m / state.layer_dp()[..., None].clamp_min(torch.finfo(dtype).tiny)
    q_scalar = q[..., :1]
    q_min, q_max = _cell_bounds(state, q_scalar)
    dp_e = state.mesh.cell_to_edge(state.layer_dp())
    mass_flux = dp_e * u_e
    c0 = state.mesh.cellsOnEdge[:, 0].to(device=device)
    c1 = state.mesh.cellsOnEdge[:, 1].to(device=device)
    donor = torch.where(mass_flux >= 0.0, c0[:, None], c1[:, None])
    outgoing = (
        float(dt)
        * state.mesh.dv_edge.to(dtype=dtype, device=device)[:, None]
        * mass_flux.abs()
    ) > float(config.zero_flux_tolerance)
    features = edge_reconstruction_features(state, tracer_m, u_e, diagnostics, shadow_z, dt=dt, feature_config=feature_config)
    terms = _allocation_envelope_terms(
        state,
        q_min,
        q_max,
        donor,
        model,
        features,
        config=config,
    )
    delta = torch.where(outgoing[..., None], terms["delta"], torch.zeros_like(terms["delta"]))
    active = outgoing[..., None]
    if not bool(active.any()):
        return torch.zeros((), dtype=dtype, device=device)
    return delta[active].square().mean()


def neural_allocation_residual_penalty(
    state: State,
    tracer_m: torch.Tensor,
    u_e: torch.Tensor,
    model: EdgeAllocationMLP,
    diagnostics: torch.Tensor,
    shadow_z: torch.Tensor,
    *,
    dt: float,
    config: NeuralAllocationConfig | None = None,
    feature_config: InvariantFeatureConfig | None = None,
) -> torch.Tensor:
    """Differentiable mean-square correction-size penalty for residual mode."""

    config = config or NeuralAllocationConfig()
    dtype = tracer_m.dtype
    device = tracer_m.device
    terms = _residual_allocation_edge_terms(
        state,
        tracer_m,
        u_e,
        model,
        diagnostics,
        shadow_z,
        dt=dt,
        config=config,
        feature_config=feature_config,
    )
    active = terms["outgoing"][..., None]
    if not bool(active.any()):
        return torch.zeros((), dtype=dtype, device=device)
    return terms["alpha"][active].square().mean()


def neural_tracer_tendency(
    state: State,
    tracer_m: torch.Tensor,
    u_e: torch.Tensor,
    model: nn.Module,
    diagnostics: torch.Tensor,
    shadow_z: torch.Tensor,
    *,
    dt: float,
    config: NeuralReconstructionConfig | None = None,
    feature_config: InvariantFeatureConfig | None = None,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Conservative finite-volume tracer tendency with neural edge gamma."""

    config = config or NeuralReconstructionConfig()
    mesh = state.mesh
    q = tracer_m / state.layer_dp()[..., None].clamp_min(torch.finfo(tracer_m.dtype).tiny)
    if q.shape[-1] != 1:
        raise ValueError("neural_tracer_tendency currently supports one passive tracer channel")
    q_scalar = q[..., :1]
    dp_e = mesh.cell_to_edge(state.layer_dp())
    mass_flux = dp_e * u_e
    c0 = mesh.cellsOnEdge[:, 0]
    c1 = mesh.cellsOnEdge[:, 1]
    q0 = q_scalar[c0]
    q1 = q_scalar[c1]
    q_base = torch.where(mass_flux[..., None] >= 0.0, q0, q1)
    q_high = 0.5 * (q0 + q1)
    features = edge_reconstruction_features(state, tracer_m, u_e, diagnostics, shadow_z, dt=dt, feature_config=feature_config)
    gamma = _model_forward(model, features, state=state).clamp(0.0, 1.0)
    q_edge = q_base + gamma * (q_high - q_base)
    lower, upper = _edge_bounds(
        state,
        q,
        mode=config.mode,
        relaxed_eps_minus=config.relaxed_eps_minus,
        relaxed_eps_plus=config.relaxed_eps_plus,
    )
    q_edge = q_edge.clamp(min=lower, max=upper)
    scale = _donor_export_scale(
        state,
        q_scalar,
        q_edge,
        mass_flux,
        dt=dt,
        safety=config.donor_export_safety,
    )
    q_edge = q_edge * scale
    flux = mass_flux[..., None] * q_edge
    tendency = -mesh.div(flux.reshape(mesh.nEdges, -1)).reshape(mesh.nCells, state.column.n_levels, 1)
    if not return_info:
        return tendency
    info = {
        "gamma_mean": float(gamma.detach().mean().cpu()),
        "gamma_max": float(gamma.detach().amax().cpu()),
        "edge_scale_min": float(scale.detach().amin().cpu()),
        "edge_scale_mean": float(scale.detach().mean().cpu()),
    }
    info.update(feature_audit_summary(features, feature_config=feature_config))
    info.update(
        invariant_feature_scale_summary(
            state,
            tracer_m,
            u_e,
            diagnostics,
            shadow_z,
            dt=dt,
            feature_config=feature_config,
        )
    )
    return tendency, info


__all__ = [
    "EdgeAllocationMLP",
    "EdgeAllocationMeshConv",
    "EdgeReconstructionMLP",
    "InvariantFeatureConfig",
    "NEURAL_FEATURE_MODES",
    "NEURAL_MODEL_TYPES",
    "NeuralAllocationConfig",
    "NeuralReconstructionConfig",
    "SectorRemapConfig",
    "SectorRemapMLP",
    "SectorRemapMeshConv",
    "edge_reconstruction_feature_names",
    "edge_reconstruction_features",
    "feature_audit_summary",
    "invariant_feature_config",
    "invariant_feature_scale_summary",
    "infer_feature_dim",
    "infer_sector_feature_dim",
    "neural_allocation_envelope_penalty",
    "neural_allocation_residual_penalty",
    "neural_allocation_tracer_tendency",
    "neural_residual_allocation_tracer_tendency",
    "neural_sector_remap_tracer_tendency",
    "neural_tracer_tendency",
    "sector_remap_features",
]
