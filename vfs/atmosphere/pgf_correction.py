"""Experimental wind-excluded learned PGF defect correction.

This module intentionally lives behind an opt-in pressure-gradient strategy
wrapper.  The default dry core never constructs it, and the feature path uses
only thermodynamic, pressure, geopotential, and mesh-geometry fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..core.state import State
from ..geometry.mesh.kernels import CellMeshKernelNetwork


PGF_CORRECTION_FEATURE_NAMES = (
    "log_ps_over_p0",
    "log_p_full_over_p0",
    "sigma_full",
    "theta_anom_100K",
    "theta_vertical_jump_50K",
    "exner_over_cp",
    "phi_over_gH",
    "phi_s_over_gH",
    "B_full",
    "alpha_B",
    "cell_x",
    "cell_y",
    "cell_z",
    "latitude_sin",
    "latitude_cos",
    "log_cell_area_rel",
    "cell_degree_rel",
    "L_grad_log_ps",
    "L_grad_log_p_full",
    "L_grad_theta_100K",
    "grad_phi_over_g",
    "grad_phi_s_over_g",
)


@dataclass
class PotentialPGFCorrectionConfig:
    """Runtime bounds and guards for the learned PGF potential correction."""

    max_fraction: float = 0.02
    scale_floor: float = 1.0e-8
    stop_grad_inputs: bool = False
    zero_rest_guard: bool = True
    feature_clip: float = 8.0
    feature_eps: float = 1.0e-8


class PGFPotentialCorrectionMLP(nn.Module):
    """Small cell/layer scalar-potential model for PGF defect correction."""

    def __init__(
        self,
        input_dim: int | None = None,
        *,
        hidden_dim: int = 64,
        depth: int = 3,
        potential_scale: float = 1.0e4,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim or len(PGF_CORRECTION_FEATURE_NAMES))
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.potential_scale = float(potential_scale)

        layers: list[nn.Module] = []
        width_in = self.input_dim
        for _ in range(max(0, self.depth)):
            layers.append(nn.Linear(width_in, self.hidden_dim))
            layers.append(nn.SiLU())
            width_in = self.hidden_dim
        layers.append(nn.Linear(width_in, 1))
        self.net = nn.Sequential(*layers)
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

    def config_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "potential_scale": self.potential_scale,
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.potential_scale * self.net(features).squeeze(-1)


class PGFPotentialCorrectionMeshConv(nn.Module):
    """Cell one-ring mesh-convolution scalar-potential model."""

    model_type = "meshconv"

    def __init__(
        self,
        input_dim: int | None = None,
        *,
        hidden_dim: int = 64,
        depth: int = 3,
        geom_hidden_dim: int = 32,
        potential_scale: float = 1.0e4,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim or len(PGF_CORRECTION_FEATURE_NAMES))
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.geom_hidden_dim = int(geom_hidden_dim)
        self.potential_scale = float(potential_scale)
        self.encoder = CellMeshKernelNetwork(
            self.input_dim,
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

    def config_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "geom_hidden_dim": self.geom_hidden_dim,
            "potential_scale": self.potential_scale,
            "model_type": self.model_type,
        }

    def forward(self, features: torch.Tensor, *, mesh=None) -> torch.Tensor:
        if mesh is None:
            raise ValueError("PGFPotentialCorrectionMeshConv.forward requires mesh=")
        latent = self.encoder(mesh, features, target_levels=features.shape[-2])
        return self.potential_scale * self.head(latent).squeeze(-1)


def pgf_correction_feature_dim() -> int:
    return len(PGF_CORRECTION_FEATURE_NAMES)


def _dataclass_from_mapping(cls, data: dict[str, Any] | None):
    if data is None:
        return cls()
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in data.items() if key in allowed})


def _as_cell_level(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    value = value.to(dtype=like.dtype, device=like.device)
    if value.shape == like.shape:
        return value
    if value.ndim == 0:
        return value.reshape((1,) * like.ndim).expand_as(like)
    if value.ndim == 1 and value.shape[0] == like.shape[-2]:
        shape = (1,) * (like.ndim - 2) + (like.shape[-2], 1)
        return value.reshape(shape).expand_as(like)
    if value.ndim == 1 and value.shape[0] == like.shape[-1]:
        shape = (1,) * (like.ndim - 1) + (like.shape[-1],)
        return value.reshape(shape).expand_as(like)
    if value.shape[-2:] == (like.shape[-2], 1):
        return value.expand(*like.shape[:-1], like.shape[-1])
    if value.shape[-2:] == (1, like.shape[-1]):
        return value.expand_as(like)
    return value.expand_as(like)


def _cell_length_scale(state: State, like: torch.Tensor) -> torch.Tensor:
    area = state.mesh.cell_area.to(dtype=like.dtype, device=like.device).clamp_min(
        torch.finfo(like.dtype).tiny
    )
    return _as_cell_level(torch.sqrt(area), like)


def _edge_abs_to_cell(state: State, edge_value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    cell_value = state.edge_to_cell(edge_value.abs())
    return _as_cell_level(cell_value, like)


def _clip_features(features: torch.Tensor, clip: float) -> torch.Tensor:
    features = torch.nan_to_num(features, nan=0.0, posinf=float(clip), neginf=-float(clip))
    return torch.clamp(features, -float(clip), float(clip))


def build_pgf_correction_features(
    dyn: Any,
    state: State,
    theta_i: torch.Tensor,
    *,
    phi_full: torch.Tensor | None = None,
    alpha_B: torch.Tensor | None = None,
    config: PotentialPGFCorrectionConfig | None = None,
) -> torch.Tensor:
    """Return wind-free dimensionless features shaped ``[..., nCells, nLevels, C]``."""

    cfg = PotentialPGFCorrectionConfig() if config is None else config
    if cfg.stop_grad_inputs:
        theta_i = theta_i.detach()
        ps = state.ps.detach()
    else:
        ps = state.ps

    if phi_full is None or alpha_B is None:
        phi_full, alpha_B = dyn.pressure_gradient.pgf_components(dyn, state, theta_i)
    if cfg.stop_grad_inputs:
        phi_full = phi_full.detach()
        alpha_B = alpha_B.detach()

    dtype = phi_full.dtype
    device = phi_full.device
    tiny = torch.finfo(dtype).tiny
    eps = float(cfg.feature_eps)
    column = state.column
    mesh = state.mesh

    theta_full = 0.5 * (theta_i[..., :-1] + theta_i[..., 1:])
    p_full = column.full_pressure(ps).to(dtype=dtype, device=device)
    exner_full = column.exner_full(ps).to(dtype=dtype, device=device)
    ps_safe = ps.to(dtype=dtype, device=device).clamp_min(tiny)
    p_full_safe = p_full.clamp_min(tiny)
    p0 = float(getattr(column, "p0", 1.0e5))
    cp = float(getattr(column, "c_p", 1004.0))
    g = float(getattr(dyn, "gravity", getattr(column, "g", 9.80665)))
    height_scale = 1.0e4

    like = theta_full
    L_cell = _cell_length_scale(state, like)

    phi_s = column.phi_s.to(dtype=dtype, device=device)
    phi_s_cell = _as_cell_level(phi_s, like)
    B_full = _as_cell_level(column.B_full.to(dtype=dtype, device=device), like)

    xyz = mesh.cell_xyz.to(dtype=dtype, device=device)
    x = _as_cell_level(xyz[:, 0], like)
    y = _as_cell_level(xyz[:, 1], like)
    z = _as_cell_level(xyz[:, 2].clamp(-1.0, 1.0), like)
    lat_cos = torch.sqrt((1.0 - z.square()).clamp_min(0.0))

    area = mesh.cell_area.to(dtype=dtype, device=device).clamp_min(tiny)
    log_area = torch.log(area / area.mean().clamp_min(tiny))
    degree = mesh.nEdgesOnCell.to(dtype=dtype, device=device)
    degree_rel = (degree - degree.mean()) / degree.std(unbiased=False).clamp_min(1.0)

    log_ps = torch.log(ps_safe / p0)
    log_p_full = torch.log(p_full_safe / p0)
    sigma_full = p_full_safe / ps_safe.expand_as(p_full_safe).clamp_min(tiny)
    theta_vertical_jump = theta_i[..., 1:] - theta_i[..., :-1]

    grad_log_ps_cell = _edge_abs_to_cell(state, state.grad(log_ps), like) * L_cell
    grad_log_p_cell = _edge_abs_to_cell(state, state.grad(log_p_full), like) * L_cell
    grad_theta_cell = _edge_abs_to_cell(state, state.grad(theta_full), like) * L_cell / 100.0
    grad_phi_cell = _edge_abs_to_cell(state, state.grad(phi_full), like) / max(g, eps)
    grad_phi_s_cell = _edge_abs_to_cell(state, state.grad(phi_s_cell), like) / max(g, eps)

    feature_list = [
        _as_cell_level(log_ps, like),
        log_p_full,
        sigma_full,
        (theta_full - 300.0) / 100.0,
        theta_vertical_jump / 50.0,
        exner_full / cp,
        phi_full / (max(g, eps) * height_scale),
        phi_s_cell / (max(g, eps) * height_scale),
        B_full,
        alpha_B,
        x,
        y,
        z,
        z,
        lat_cos,
        _as_cell_level(log_area, like),
        _as_cell_level(degree_rel, like),
        grad_log_ps_cell,
        grad_log_p_cell,
        grad_theta_cell,
        grad_phi_cell,
        grad_phi_s_cell,
    ]
    features = torch.stack(feature_list, dim=-1)
    return _clip_features(features, cfg.feature_clip)


def _thermo_rest_guard(
    state: State,
    theta_i: torch.Tensor,
    like: torch.Tensor,
    config: PotentialPGFCorrectionConfig,
) -> torch.Tensor:
    if not config.zero_rest_guard:
        return torch.ones_like(like)
    theta_full = 0.5 * (theta_i[..., :-1] + theta_i[..., 1:])
    L_cell = _cell_length_scale(state, like)
    ps_safe = state.ps.clamp_min(torch.finfo(like.dtype).tiny)
    grad_measure = (
        _edge_abs_to_cell(state, state.grad(torch.log(ps_safe)), like) * L_cell
        + _edge_abs_to_cell(state, state.grad(theta_full), like) * L_cell / 100.0
    )
    return torch.tanh(grad_measure)


class LearnedPotentialPressureGradientCorrection(nn.Module):
    """Pressure-gradient strategy wrapper adding ``-grad(psi_corr)``.

    The wrapped base pressure-gradient strategy remains responsible for the
    physical PGF.  This class only adds a bounded scalar-potential gradient
    correction built from wind-excluded local features.
    """

    name = "learned_potential_pgf_correction"

    def __init__(
        self,
        base_pressure_gradient: Any,
        *,
        model: PGFPotentialCorrectionMLP | None = None,
        config: PotentialPGFCorrectionConfig | None = None,
    ) -> None:
        super().__init__()
        self.base_pressure_gradient = base_pressure_gradient
        self.model = model if model is not None else PGFPotentialCorrectionMLP()
        self.config = PotentialPGFCorrectionConfig() if config is None else config

    def pgf_components(self, dyn: Any, state: State, theta_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.base_pressure_gradient.pgf_components(dyn, state, theta_i)

    def sigma_correction_term(self, dyn: Any, state: State, theta_i: torch.Tensor) -> torch.Tensor:
        sigma = getattr(self.base_pressure_gradient, "sigma_correction_term", None)
        if callable(sigma):
            return sigma(dyn, state, theta_i)
        _phi_full, alpha_B = self.pgf_components(dyn, state, theta_i)
        return -state.cell_to_edge(alpha_B) * state.grad(state.ps)

    def base_pgf_term(self, dyn: Any, state: State, theta_i: torch.Tensor) -> torch.Tensor:
        pgf_term = getattr(self.base_pressure_gradient, "pgf_term", None)
        if callable(pgf_term):
            return pgf_term(dyn, state, theta_i)
        phi_full, alpha_B = self.pgf_components(dyn, state, theta_i)
        return -state.grad(phi_full) - state.cell_to_edge(alpha_B) * state.grad(state.ps)

    def _model_for_features(self, features: torch.Tensor) -> PGFPotentialCorrectionMLP:
        try:
            parameter = next(self.model.parameters())
        except StopIteration:  # pragma: no cover - defensive for custom empty modules.
            return self.model
        if parameter.dtype != features.dtype or parameter.device != features.device:
            self.model.to(dtype=features.dtype, device=features.device)
        return self.model

    def potential(self, dyn: Any, state: State, theta_i: torch.Tensor) -> torch.Tensor:
        phi_full, alpha_B = self.pgf_components(dyn, state, theta_i)
        features = build_pgf_correction_features(
            dyn,
            state,
            theta_i,
            phi_full=phi_full,
            alpha_B=alpha_B,
            config=self.config,
        )
        model = self._model_for_features(features)
        if isinstance(model, PGFPotentialCorrectionMeshConv):
            psi = model(features, mesh=state.mesh)
        else:
            psi = model(features)
        psi = psi - psi.mean(dim=-2, keepdim=True)
        return psi * _thermo_rest_guard(state, theta_i, psi, self.config)

    def pgf_terms(self, dyn: Any, state: State, theta_i: torch.Tensor) -> dict[str, torch.Tensor]:
        base = self.base_pgf_term(dyn, state, theta_i)
        psi = self.potential(dyn, state, theta_i)
        raw_correction = -state.grad(psi)
        denom = base.abs() + float(self.config.scale_floor)
        cap = float(self.config.max_fraction) * denom
        correction = cap * torch.tanh(raw_correction / cap.clamp_min(torch.finfo(base.dtype).tiny))
        return {
            "pgf": base + correction,
            "pgf_base": base,
            "pgf_correction": correction,
            "pgf_correction_raw": raw_correction,
            "pgf_correction_cap": cap,
            "psi_corr": psi,
        }

    def pgf_term(self, dyn: Any, state: State, theta_i: torch.Tensor) -> torch.Tensor:
        return self.pgf_terms(dyn, state, theta_i)["pgf"]

    def diagnostic_summary(self, dyn: Any, state: State, theta_i: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            terms = self.pgf_terms(dyn, state, theta_i)
            correction = terms["pgf_correction"]
            base = terms["pgf_base"]
            denom = base.abs() + float(self.config.scale_floor)
            ratio = correction.abs() / denom
            curl_proxy = state.curl(correction)
            return {
                "pgf_correction_rms": float(correction.square().mean().sqrt().detach().cpu()),
                "pgf_base_rms": float(base.square().mean().sqrt().detach().cpu()),
                "pgf_correction_ratio_rms": float(ratio.square().mean().sqrt().detach().cpu()),
                "pgf_correction_ratio_max": float(ratio.max().detach().cpu()),
                "pgf_correction_max_abs": float(correction.abs().max().detach().cpu()),
                "pgf_correction_curl_proxy_rms": float(curl_proxy.square().mean().sqrt().detach().cpu()),
                "pgf_correction_cap_fraction_mean": float(
                    (correction.abs() / terms["pgf_correction_cap"].clamp_min(torch.finfo(base.dtype).tiny))
                    .mean()
                    .detach()
                    .cpu()
                ),
            }


def save_pgf_correction_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: PotentialPGFCorrectionConfig,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "learned_potential_pgf_correction",
            "model_type": str(getattr(model, "model_type", "mlp")),
            "model_config": model.config_dict(),
            "model_state_dict": model.state_dict(),
            "correction_config": asdict(config),
            "feature_names": list(PGF_CORRECTION_FEATURE_NAMES),
            "metadata": dict(metadata or {}),
        },
        path,
    )
    return path


def load_pgf_correction_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> tuple[nn.Module, PotentialPGFCorrectionConfig, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location)
    if "model_state_dict" not in payload:
        raise ValueError(f"{path} is not a PGF correction checkpoint")
    model_config = dict(payload.get("model_config", {}))
    model_type = str(payload.get("model_type", model_config.pop("model_type", "mlp")))
    if model_type == "meshconv":
        model = PGFPotentialCorrectionMeshConv(**model_config)
    elif model_type == "mlp":
        model = PGFPotentialCorrectionMLP(**model_config)
    else:
        raise ValueError(f"unknown PGF correction model_type {model_type!r}")
    model.load_state_dict(payload["model_state_dict"])
    config = _dataclass_from_mapping(PotentialPGFCorrectionConfig, payload.get("correction_config"))
    metadata = dict(payload.get("metadata", {}))
    return model, config, metadata


__all__ = [
    "LearnedPotentialPressureGradientCorrection",
    "PGF_CORRECTION_FEATURE_NAMES",
    "PGFPotentialCorrectionMeshConv",
    "PGFPotentialCorrectionMLP",
    "PotentialPGFCorrectionConfig",
    "build_pgf_correction_features",
    "load_pgf_correction_checkpoint",
    "pgf_correction_feature_dim",
    "save_pgf_correction_checkpoint",
]
