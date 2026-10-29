"""Shadow-tracer state evolution and diagnostics for passive transport."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vfs.atmosphere.dry_strategies import UpwindTracerTransport
from vfs.core.state import State
from vfs.transport.shadow_diagnostics import DEFAULT_SHADOW_DIAGNOSTIC_NAMES, compute_shadow_diagnostics


@dataclass(frozen=True)
class ShadowTracerConfig:
    """Configuration for diagnostic-memory shadow tracers."""

    n_shadow: int = len(DEFAULT_SHADOW_DIAGNOSTIC_NAMES)
    source_mode: str = "relaxation"
    tau_seconds: float | Sequence[float] | None = 3600.0
    diagnostic_names: tuple[str, ...] = DEFAULT_SHADOW_DIAGNOSTIC_NAMES

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostic_names", tuple(self.diagnostic_names))
        if int(self.n_shadow) != len(self.diagnostic_names):
            object.__setattr__(self, "n_shadow", len(self.diagnostic_names))

    def tau_tensor(self, *, dtype: torch.dtype, device) -> torch.Tensor | None:
        if self.tau_seconds is None:
            return None
        if isinstance(self.tau_seconds, Sequence) and not isinstance(self.tau_seconds, (str, bytes)):
            tau = torch.tensor(self.tau_seconds, dtype=dtype, device=device)
        else:
            tau = torch.full((int(self.n_shadow),), float(self.tau_seconds), dtype=dtype, device=device)
        if tau.numel() != int(self.n_shadow):
            raise ValueError("shadow tau_seconds must be scalar or have one value per shadow channel")
        return tau.clamp_min(torch.finfo(dtype).tiny)


class _TendencyContext:
    def __init__(self, dt: float) -> None:
        self._current_tendency_dt = float(dt)


def shadow_mass_from_diagnostics(state: State, diagnostics: torch.Tensor) -> torch.Tensor:
    """Pack diagnostic shadow concentration ``z`` as conservative ``dp*z``."""

    return state.layer_dp()[..., None] * diagnostics


def shadow_concentration(state: State, shadow_m: torch.Tensor) -> torch.Tensor:
    """Recover shadow concentration from conservative shadow content."""

    return shadow_m / state.layer_dp()[..., None].clamp_min(torch.finfo(shadow_m.dtype).tiny)


def initialize_shadow_mass(
    state: State,
    q: torch.Tensor,
    velocity: torch.Tensor,
    config: ShadowTracerConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize shadow content from instantaneous diagnostic targets."""

    config = config or ShadowTracerConfig()
    diagnostics = compute_shadow_diagnostics(
        state.mesh,
        q,
        velocity,
        names=config.diagnostic_names,
    )
    return shadow_mass_from_diagnostics(state, diagnostics), diagnostics


def shadow_tendency(
    state: State,
    shadow_m: torch.Tensor,
    velocity: torch.Tensor,
    diagnostics: torch.Tensor,
    *,
    config: ShadowTracerConfig,
    dt: float,
) -> torch.Tensor:
    """Return conservative shadow-tracer tendency plus optional source."""

    name = str(config.source_mode)
    if name not in {"none", "relaxation", "relaxation_plus_residual"}:
        raise ValueError("shadow source_mode must be none, relaxation, or relaxation_plus_residual")
    z = shadow_concentration(state, shadow_m)
    mu = torch.zeros(
        state.mesh.nCells,
        state.column.n_levels + 1,
        dtype=shadow_m.dtype,
        device=shadow_m.device,
    )
    local_state = State(
        velocity,
        state.mtheta_i,
        state.ps,
        shadow_m,
        mesh=state.mesh,
        column=state.column,
        attrs=state.attrs,
    )
    tendency = UpwindTracerTransport().tendency(
        _TendencyContext(dt),
        local_state,
        velocity,
        z,
        local_state.layer_dp(),
        mu,
    )
    if name == "none" or config.tau_seconds is None:
        return tendency
    tau = config.tau_tensor(dtype=shadow_m.dtype, device=shadow_m.device).reshape(1, 1, -1)
    source = local_state.layer_dp()[..., None] * (diagnostics - z) / tau
    if name == "relaxation_plus_residual":
        source = source + torch.zeros_like(source)
    return tendency + source


def shadow_memory_diagnostics(q: torch.Tensor, z: torch.Tensor, diagnostics: torch.Tensor) -> dict[str, Any]:
    """Detached scalar diagnostics for shadow-memory behavior."""

    q0 = q[..., 0] if q.ndim == 3 else q
    q_flat = q0.reshape(-1)
    rows: dict[str, Any] = {}
    for k in range(z.shape[-1]):
        zk = z[..., k].reshape(-1)
        dk = diagnostics[..., k].reshape(-1)
        z_var = zk.var(unbiased=False)
        d_var = dk.var(unbiased=False)
        lag = (zk - dk).square().mean().sqrt()
        q_std = q_flat.std(unbiased=False).clamp_min(torch.finfo(q.dtype).tiny)
        z_std = zk.std(unbiased=False).clamp_min(torch.finfo(q.dtype).tiny)
        d_std = dk.std(unbiased=False).clamp_min(torch.finfo(q.dtype).tiny)
        qz = ((q_flat - q_flat.mean()) * (zk - zk.mean())).mean() / (q_std * z_std)
        zd = ((zk - zk.mean()) * (dk - dk.mean())).mean() / (z_std * d_std)
        rows[f"shadow_{k}_z_variance"] = float(z_var.detach().cpu())
        rows[f"shadow_{k}_diagnostic_variance"] = float(d_var.detach().cpu())
        rows[f"shadow_{k}_lag_rms"] = float(lag.detach().cpu())
        rows[f"shadow_{k}_q_correlation"] = float(qz.detach().cpu())
        rows[f"shadow_{k}_diagnostic_correlation"] = float(zd.detach().cpu())
    rows["shadow_lag_rms_mean"] = float((z - diagnostics).square().mean().sqrt().detach().cpu())
    rows["shadow_z_variance_mean"] = float(z.var(dim=(0, 1), unbiased=False).mean().detach().cpu())
    return rows


__all__ = [
    "ShadowTracerConfig",
    "initialize_shadow_mass",
    "shadow_concentration",
    "shadow_mass_from_diagnostics",
    "shadow_memory_diagnostics",
    "shadow_tendency",
]
