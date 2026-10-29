"""Reversible passive-tracer advection harness primitives."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vfs.atmosphere import FCTTracerTransport
from vfs.atmosphere.dry_strategies import UpwindTracerTransport
from vfs.core.state import State
from vfs.geometry.column import CharneyPhillipsColumn, make_hybrid_coefficients
from vfs.transport.neural_reconstruction import (
    EdgeAllocationMLP,
    EdgeReconstructionMLP,
    InvariantFeatureConfig,
    NeuralAllocationConfig,
    NeuralReconstructionConfig,
    SectorRemapConfig,
    SectorRemapMLP,
    infer_feature_dim,
    infer_sector_feature_dim,
    neural_allocation_tracer_tendency,
    neural_residual_allocation_tracer_tendency,
    neural_sector_remap_tracer_tendency,
    neural_tracer_tendency,
)
from vfs.transport.shadow_diagnostics import compute_shadow_diagnostics
from vfs.transport.shadow_tracers import (
    ShadowTracerConfig,
    initialize_shadow_mass,
    shadow_mass_from_diagnostics,
    shadow_concentration,
    shadow_memory_diagnostics,
    shadow_tendency,
)


@dataclass(frozen=True)
class PrescribedFlow:
    """Parameters needed to regenerate a prescribed reversible flow."""

    name: str = "deformational"
    velocity_scale: float = 20.0
    duration: float = 86400.0
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    wave_number: int = 2


@dataclass
class ReversibleAdvectionSample:
    """Tensor payload for future learned conservative-transport training."""

    q0: torch.Tensor
    qT: torch.Tensor
    qR: torch.Tensor
    snapshots: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "q0": self.q0,
            "qT": self.qT,
            "qR": self.qR,
            "snapshots": self.snapshots,
            "metadata": self.metadata,
        }


class _TendencyContext:
    def __init__(self, dt: float) -> None:
        self._current_tendency_dt = float(dt)


def _is_neural_scheme(scheme: str) -> bool:
    return str(scheme).lower() in {
        "neural",
        "neural_strict",
        "neural_relaxed",
        "shadow_neural_strict",
        "shadow_neural_relaxed",
        "neural_alloc",
        "neural_alloc_strict",
        "neural_alloc_relaxed",
        "neural_alloc_residual",
        "neural_alloc_residual_strict",
        "neural_alloc_residual_relaxed",
        "neural_sector_remap",
    }


def _is_allocation_scheme(scheme: str) -> bool:
    return str(scheme).lower() in {
        "neural_alloc",
        "neural_alloc_strict",
        "neural_alloc_relaxed",
        "neural_alloc_residual",
        "neural_alloc_residual_strict",
        "neural_alloc_residual_relaxed",
    }


def _is_sector_remap_scheme(scheme: str) -> bool:
    return str(scheme).lower() in {"neural_sector_remap"}


def _is_residual_allocation_scheme(scheme: str) -> bool:
    return str(scheme).lower() in {
        "neural_alloc_residual",
        "neural_alloc_residual_strict",
        "neural_alloc_residual_relaxed",
    }


def _neural_mode_from_scheme(scheme: str) -> str:
    return "relaxed" if "relaxed" in str(scheme).lower() else "strict"


def _lat_lon(xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.asin(xyz[:, 2].clamp(-1.0, 1.0)), torch.atan2(xyz[:, 1], xyz[:, 0])


def _normalize_vector(values: torch.Tensor) -> torch.Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(values.dtype).tiny)


def _streamfunction_solid_body(mesh, *, dtype: torch.dtype, device, axis: tuple[float, float, float]) -> torch.Tensor:
    vertex_xyz = mesh.vertex_xyz.to(dtype=dtype, device=device)
    axis_t = torch.tensor(axis, dtype=dtype, device=device).reshape(1, 3)
    axis_t = _normalize_vector(axis_t)[0]
    return float(mesh.radius) * (vertex_xyz @ axis_t)


def _streamfunction_deformational(
    mesh,
    *,
    dtype: torch.dtype,
    device,
    time_seconds: float,
    duration: float,
    wave_number: int,
) -> torch.Tensor:
    vertex_xyz = mesh.vertex_xyz.to(dtype=dtype, device=device)
    lat, lon = _lat_lon(vertex_xyz)
    phase = 2.0 * math.pi * float(time_seconds) / max(float(duration), 1.0)
    m = int(wave_number)
    pattern = (
        torch.sin(m * (lon + phase)) * torch.cos(lat).pow(2)
        + 0.35 * torch.cos((m + 1) * (lon - 0.5 * phase)) * torch.sin(2.0 * lat)
    )
    return float(mesh.radius) * pattern


def prescribed_edge_velocity(
    mesh,
    flow: str | PrescribedFlow,
    *,
    time_seconds: float,
    duration: float | None = None,
    reverse: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    velocity_scale: float | None = None,
    wave_number: int | None = None,
) -> torch.Tensor:
    """Return a divergence-free prescribed edge-normal velocity field."""

    if isinstance(flow, PrescribedFlow):
        flow_cfg = flow
    else:
        flow_cfg = PrescribedFlow(name=str(flow))
    name = flow_cfg.name
    dtype = dtype or mesh.cell_xyz.dtype
    device = device or mesh.cell_xyz.device
    total_duration = float(flow_cfg.duration if duration is None else duration)
    t = float(time_seconds)
    if reverse:
        t = total_duration - t
    if name == "solid_body":
        psi = _streamfunction_solid_body(mesh, dtype=dtype, device=device, axis=flow_cfg.axis)
    elif name == "deformational":
        psi = _streamfunction_deformational(
            mesh,
            dtype=dtype,
            device=device,
            time_seconds=t,
            duration=total_duration,
            wave_number=flow_cfg.wave_number if wave_number is None else int(wave_number),
        )
    else:
        raise ValueError(f"flow must be 'solid_body' or 'deformational', got {name!r}")
    edge = mesh.curl_adjoint(psi[:, None]).squeeze(-1)
    scale = edge.abs().amax().clamp_min(torch.finfo(edge.dtype).tiny)
    target = float(flow_cfg.velocity_scale if velocity_scale is None else velocity_scale)
    edge = edge * (target / scale)
    if reverse:
        edge = -edge
    return edge[:, None]


def build_passive_transport_state(
    mesh,
    *,
    n_levels: int,
    q: torch.Tensor,
    ps_value: float = 100000.0,
    p_top: float = 5000.0,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> State:
    """Build a simple dry State carrying passive tracer mass only."""

    dtype = dtype or q.dtype
    device = device or q.device
    A, B = make_hybrid_coefficients(int(n_levels), p_top=float(p_top), sigma_power=1.0)
    column = CharneyPhillipsColumn(int(n_levels), A, B)
    column = column.to(dtype=dtype, device=device)
    ps = torch.full((mesh.nCells, 1), float(ps_value), dtype=dtype, device=device)
    u_e = torch.zeros(mesh.nEdges, int(n_levels), dtype=dtype, device=device)
    theta_i = torch.full((mesh.nCells, int(n_levels) + 1), 300.0, dtype=dtype, device=device)
    return State.from_theta(u_e, theta_i, ps, mesh=mesh, column=column, q_tracers=q, attrs={"gravity": 9.80665})


def _transport_tendency(
    state: State,
    tracer_m: torch.Tensor,
    u_e: torch.Tensor,
    *,
    scheme: str,
    dt: float,
) -> torch.Tensor:
    q = tracer_m / state.layer_dp()[..., None].clamp_min(torch.finfo(tracer_m.dtype).tiny)
    mu = torch.zeros(
        state.mesh.nCells,
        state.column.n_levels + 1,
        dtype=tracer_m.dtype,
        device=tracer_m.device,
    )
    local_state = State(
        u_e,
        state.mtheta_i,
        state.ps,
        tracer_m,
        mesh=state.mesh,
        column=state.column,
        attrs=state.attrs,
    )
    name = str(scheme).lower()
    if name == "upwind":
        return UpwindTracerTransport().tendency(_TendencyContext(dt), local_state, u_e, q, local_state.layer_dp(), mu)
    if name == "fct":
        return FCTTracerTransport(tendency_dt=float(dt)).tendency(
            _TendencyContext(dt),
            local_state,
            u_e,
            q,
            local_state.layer_dp(),
            mu,
        )
    raise ValueError(f"transport scheme must be 'upwind' or 'fct', got {scheme!r}")


def _stage_velocity(
    mesh,
    flow: str | PrescribedFlow,
    *,
    time_seconds: float,
    duration: float,
    reverse: bool,
    n_levels: int,
    dtype: torch.dtype,
    device,
    velocity_scale: float,
) -> torch.Tensor:
    u = prescribed_edge_velocity(
        mesh,
        flow,
        time_seconds=float(time_seconds),
        duration=float(duration),
        reverse=bool(reverse),
        dtype=dtype,
        device=device,
        velocity_scale=float(velocity_scale),
    )
    return u.expand(mesh.nEdges, int(n_levels)).contiguous()


def _advance_ssprk3(
    state: State,
    tracer_m: torch.Tensor,
    *,
    flow: str | PrescribedFlow,
    t: float,
    dt: float,
    duration: float,
    reverse: bool,
    scheme: str,
    velocity_scale: float,
) -> torch.Tensor:
    mesh = state.mesh
    dtype = tracer_m.dtype
    device = tracer_m.device
    n_levels = int(state.column.n_levels)
    u0 = _stage_velocity(mesh, flow, time_seconds=t, duration=duration, reverse=reverse, n_levels=n_levels, dtype=dtype, device=device, velocity_scale=velocity_scale)
    k0 = _transport_tendency(state, tracer_m, u0, scheme=scheme, dt=dt)
    y1 = tracer_m + float(dt) * k0

    u1 = _stage_velocity(mesh, flow, time_seconds=t + dt, duration=duration, reverse=reverse, n_levels=n_levels, dtype=dtype, device=device, velocity_scale=velocity_scale)
    k1 = _transport_tendency(state, y1, u1, scheme=scheme, dt=dt)
    y2 = 0.75 * tracer_m + 0.25 * (y1 + float(dt) * k1)

    uh = _stage_velocity(mesh, flow, time_seconds=t + 0.5 * dt, duration=duration, reverse=reverse, n_levels=n_levels, dtype=dtype, device=device, velocity_scale=velocity_scale)
    k2 = _transport_tendency(state, y2, uh, scheme=scheme, dt=dt)
    return (1.0 / 3.0) * tracer_m + (2.0 / 3.0) * (y2 + float(dt) * k2)


def _neural_tendencies(
    state: State,
    tracer_m: torch.Tensor,
    shadow_m: torch.Tensor,
    u_e: torch.Tensor,
    *,
    scheme: str,
    dt: float,
    neural_model: nn.Module,
    shadow_config: ShadowTracerConfig,
    neural_config: NeuralReconstructionConfig,
    allocation_config: NeuralAllocationConfig | None = None,
    sector_config: SectorRemapConfig | None = None,
    feature_config: InvariantFeatureConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    q = q_from_tracer_m(state, tracer_m)
    diagnostics = compute_shadow_diagnostics(
        state.mesh,
        q,
        u_e,
        names=shadow_config.diagnostic_names,
    )
    z = shadow_concentration(state, shadow_m)
    mode = _neural_mode_from_scheme(scheme)
    if _is_sector_remap_scheme(scheme):
        dq, info = neural_sector_remap_tracer_tendency(
            state,
            tracer_m,
            u_e,
            neural_model,  # type: ignore[arg-type]
            diagnostics,
            z,
            dt=dt,
            config=sector_config or SectorRemapConfig(),
            feature_config=feature_config,
            return_info=True,
        )
    elif _is_allocation_scheme(scheme):
        allocation_config = allocation_config or NeuralAllocationConfig(mode=mode)
        if mode != allocation_config.mode:
            allocation_config = NeuralAllocationConfig(
                mode=mode,
                relaxed_eps=allocation_config.relaxed_eps,
                envelope=allocation_config.envelope,
                envelope_max=allocation_config.envelope_max,
                envelope_eps_floor=allocation_config.envelope_eps_floor,
                residual_alpha_max=allocation_config.residual_alpha_max,
                zero_flux_tolerance=allocation_config.zero_flux_tolerance,
                eps=allocation_config.eps,
            )
        tendency_fn = neural_residual_allocation_tracer_tendency if _is_residual_allocation_scheme(scheme) else neural_allocation_tracer_tendency
        dq, info = tendency_fn(
            state,
            tracer_m,
            u_e,
            neural_model,  # type: ignore[arg-type]
            diagnostics,
            z,
            dt=dt,
            config=allocation_config,
            feature_config=feature_config,
            return_info=True,
        )
    else:
        if mode != neural_config.mode:
            neural_config = NeuralReconstructionConfig(
                mode=mode,
                relaxed_eps_minus=neural_config.relaxed_eps_minus,
                relaxed_eps_plus=neural_config.relaxed_eps_plus,
                donor_export_safety=neural_config.donor_export_safety,
            )
        dq, info = neural_tracer_tendency(
            state,
            tracer_m,
            u_e,
            neural_model,
            diagnostics,
            z,
            dt=dt,
            config=neural_config,
            feature_config=feature_config,
            return_info=True,
        )
    dz = shadow_tendency(
        state,
        shadow_m,
        u_e,
        diagnostics,
        config=shadow_config,
        dt=dt,
    )
    return dq, dz, diagnostics, info


def _advance_shadow_ssprk3(
    state: State,
    tracer_m: torch.Tensor,
    shadow_m: torch.Tensor,
    *,
    flow: str | PrescribedFlow,
    t: float,
    dt: float,
    duration: float,
    reverse: bool,
    scheme: str,
    velocity_scale: float,
    neural_model: nn.Module,
    shadow_config: ShadowTracerConfig,
    neural_config: NeuralReconstructionConfig,
    allocation_config: NeuralAllocationConfig | None = None,
    sector_config: SectorRemapConfig | None = None,
    feature_config: InvariantFeatureConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mesh = state.mesh
    dtype = tracer_m.dtype
    device = tracer_m.device
    n_levels = int(state.column.n_levels)
    u0 = _stage_velocity(mesh, flow, time_seconds=t, duration=duration, reverse=reverse, n_levels=n_levels, dtype=dtype, device=device, velocity_scale=velocity_scale)
    kq0, kz0, _D0, _info0 = _neural_tendencies(
        state,
        tracer_m,
        shadow_m,
        u0,
        scheme=scheme,
        dt=dt,
        neural_model=neural_model,
        shadow_config=shadow_config,
        neural_config=neural_config,
        allocation_config=allocation_config,
        sector_config=sector_config,
        feature_config=feature_config,
    )
    q1 = tracer_m + float(dt) * kq0
    z1 = shadow_m + float(dt) * kz0

    u1 = _stage_velocity(mesh, flow, time_seconds=t + dt, duration=duration, reverse=reverse, n_levels=n_levels, dtype=dtype, device=device, velocity_scale=velocity_scale)
    kq1, kz1, _D1, _info1 = _neural_tendencies(
        state,
        q1,
        z1,
        u1,
        scheme=scheme,
        dt=dt,
        neural_model=neural_model,
        shadow_config=shadow_config,
        neural_config=neural_config,
        allocation_config=allocation_config,
        sector_config=sector_config,
        feature_config=feature_config,
    )
    q2 = 0.75 * tracer_m + 0.25 * (q1 + float(dt) * kq1)
    z2 = 0.75 * shadow_m + 0.25 * (z1 + float(dt) * kz1)

    uh = _stage_velocity(mesh, flow, time_seconds=t + 0.5 * dt, duration=duration, reverse=reverse, n_levels=n_levels, dtype=dtype, device=device, velocity_scale=velocity_scale)
    kq2, kz2, _Dh, _infoh = _neural_tendencies(
        state,
        q2,
        z2,
        uh,
        scheme=scheme,
        dt=dt,
        neural_model=neural_model,
        shadow_config=shadow_config,
        neural_config=neural_config,
        allocation_config=allocation_config,
        sector_config=sector_config,
        feature_config=feature_config,
    )
    return (
        (1.0 / 3.0) * tracer_m + (2.0 / 3.0) * (q2 + float(dt) * kq2),
        (1.0 / 3.0) * shadow_m + (2.0 / 3.0) * (z2 + float(dt) * kz2),
    )


def _weights(state: State) -> torch.Tensor:
    dp = state.layer_dp()
    area = state.mesh.cell_area.to(dtype=dp.dtype, device=dp.device)[:, None, None]
    return area * dp[..., None]


def _weighted_mean(q: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (q * weights).sum() / weights.sum().clamp_min(torch.finfo(q.dtype).tiny)


def _weighted_variance(q: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    mean = _weighted_mean(q, weights)
    return ((q - mean).square() * weights).sum() / weights.sum().clamp_min(torch.finfo(q.dtype).tiny)


def _weighted_l2(q: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return ((q.square() * weights).sum() / weights.sum().clamp_min(torch.finfo(q.dtype).tiny)).sqrt()


def edge_variation_metrics(mesh, q: torch.Tensor) -> dict[str, torch.Tensor]:
    c0 = mesh.cellsOnEdge[:, 0]
    c1 = mesh.cellsOnEdge[:, 1]
    delta = q[c1] - q[c0]
    dv = mesh.dv_edge.to(dtype=q.dtype, device=q.device)[:, None, None]
    dc = mesh.dc_edge.to(dtype=q.dtype, device=q.device)[:, None, None]
    edge_area = (dv * dc).clamp_min(torch.finfo(q.dtype).tiny)
    tv = (delta.abs() * dv).sum()
    grad_l2 = ((delta / dc).square() * edge_area).sum().sqrt() / edge_area.sum().sqrt().clamp_min(
        torch.finfo(q.dtype).tiny
    )
    return {"tv_proxy": tv, "edge_gradient_l2": grad_l2}


def q_from_tracer_m(state: State, tracer_m: torch.Tensor) -> torch.Tensor:
    return tracer_m / state.layer_dp()[..., None].clamp_min(torch.finfo(tracer_m.dtype).tiny)


def q_diagnostics(state: State, tracer_m: torch.Tensor, *, prefix: str) -> dict[str, Any]:
    q = q_from_tracer_m(state, tracer_m)
    w = _weights(state)
    variation = edge_variation_metrics(state.mesh, q)
    return {
        f"{prefix}_q_min": float(q.amin().detach().cpu()),
        f"{prefix}_q_max": float(q.amax().detach().cpu()),
        f"{prefix}_q_mean": float(_weighted_mean(q, w).detach().cpu()),
        f"{prefix}_q_variance": float(_weighted_variance(q, w).detach().cpu()),
        f"{prefix}_q_tv_proxy": float(variation["tv_proxy"].detach().cpu()),
        f"{prefix}_q_edge_gradient_l2": float(variation["edge_gradient_l2"].detach().cpu()),
        f"{prefix}_negative_count": int((q < 0.0).sum().detach().cpu()),
        f"{prefix}_negative_magnitude": float(torch.relu(-q).amax().detach().cpu()),
    }


def comparison_diagnostics(state: State, reference_m: torch.Tensor, candidate_m: torch.Tensor) -> dict[str, Any]:
    q0 = q_from_tracer_m(state, reference_m)
    q = q_from_tracer_m(state, candidate_m)
    w = _weights(state)
    diff = q - q0
    l2_abs = _weighted_l2(diff, w)
    l2_ref = _weighted_l2(q0, w).clamp_min(torch.finfo(q.dtype).tiny)
    l1_abs = (diff.abs() * w).sum() / w.sum().clamp_min(torch.finfo(q.dtype).tiny)
    q0_mean = _weighted_mean(q0, w)
    q_mean = _weighted_mean(q, w)
    cov = ((q0 - q0_mean) * (q - q_mean) * w).sum() / w.sum().clamp_min(torch.finfo(q.dtype).tiny)
    std0 = _weighted_variance(q0, w).sqrt()
    std = _weighted_variance(q, w).sqrt()
    corr = cov / (std0 * std).clamp_min(torch.finfo(q.dtype).tiny)
    q0_min = q0.amin()
    q0_max = q0.amax()
    return {
        "recovery_l2_relative": float((l2_abs / l2_ref).detach().cpu()),
        "recovery_l1_abs": float(l1_abs.detach().cpu()),
        "recovery_linf_abs": float(diff.abs().amax().detach().cpu()),
        "recovery_correlation": float(corr.detach().cpu()),
        "recovery_variance_retention": float((_weighted_variance(q, w) / _weighted_variance(q0, w).clamp_min(torch.finfo(q.dtype).tiny)).detach().cpu()),
        "recovery_overshoot_above_q0_global_max": float(torch.relu(q.amax() - q0_max).detach().cpu()),
        "recovery_undershoot_below_q0_global_min": float(torch.relu(q0_min - q.amin()).detach().cpu()),
    }


def tracer_mass(state: State, tracer_m: torch.Tensor) -> torch.Tensor:
    gravity = float(state.attrs.get("gravity", 9.80665))
    return (tracer_m * state.mesh.cell_area.to(dtype=tracer_m.dtype, device=tracer_m.device)[:, None, None]).sum(dim=(0, 1)) / gravity


def cfl_diagnostics(mesh, u_e: torch.Tensor, dt: float) -> dict[str, float]:
    speed = u_e.abs()
    local = speed[mesh.safe_edges_on_cell, :] * mesh.dv_edge.to(dtype=u_e.dtype, device=u_e.device)[mesh.safe_edges_on_cell, None]
    local = torch.where(mesh.valid_edges_on_cell[..., None].to(device=u_e.device), local, torch.zeros_like(local))
    cell_cfl = float(dt) * local.sum(dim=1) / mesh.cell_area.to(dtype=u_e.dtype, device=u_e.device)[:, None]
    edge_cfl = float(dt) * speed / mesh.dc_edge.to(dtype=u_e.dtype, device=u_e.device)[:, None].clamp_min(torch.finfo(u_e.dtype).tiny)
    return {
        "cfl_cell_max": float(cell_cfl.amax().detach().cpu()),
        "cfl_cell_mean": float(cell_cfl.mean().detach().cpu()),
        "cfl_edge_max": float(edge_cfl.amax().detach().cpu()),
        "velocity_abs_max": float(speed.amax().detach().cpu()),
    }


def _sample_row(
    state: State,
    tracer_m: torch.Tensor,
    *,
    step: int,
    phase: str,
    elapsed_seconds: float,
    flow_time_seconds: float,
    phase_time_seconds: float,
    mass0: torch.Tensor,
    cfl: dict[str, float],
    shadow_m: torch.Tensor | None = None,
    shadow_diagnostics_target: torch.Tensor | None = None,
    transport_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mass = tracer_mass(state, tracer_m)
    mass_rel = ((mass - mass0).abs() / mass0.abs().clamp_min(torch.finfo(mass.dtype).tiny)).amax()
    row = {
        "step": int(step),
        "phase": str(phase),
        "time_seconds": float(elapsed_seconds),
        "elapsed_seconds": float(elapsed_seconds),
        "elapsed_days": float(elapsed_seconds) / 86400.0,
        "flow_time_seconds": float(flow_time_seconds),
        "flow_time_days": float(flow_time_seconds) / 86400.0,
        "phase_time_seconds": float(phase_time_seconds),
        "phase_time_days": float(phase_time_seconds) / 86400.0,
        "tracer_mass_rel_error_max_abs": float(mass_rel.detach().cpu()),
        **q_diagnostics(state, tracer_m, prefix="q"),
        **cfl,
    }
    if shadow_m is not None and shadow_diagnostics_target is not None:
        q = q_from_tracer_m(state, tracer_m)
        z = shadow_concentration(state, shadow_m)
        row.update(shadow_memory_diagnostics(q, z, shadow_diagnostics_target))
    if transport_info is not None:
        row.update(
            {
                str(key): value
                for key, value in transport_info.items()
                if not isinstance(value, torch.Tensor)
            }
        )
    return row


@torch.no_grad()
def rollout_reversible_advection(
    state: State,
    *,
    flow: str | PrescribedFlow = "deformational",
    scheme: str = "upwind",
    nsteps: int = 200,
    dt: float = 300.0,
    velocity_scale: float = 20.0,
    snapshot_cadence: int = 0,
    neural_model: nn.Module | None = None,
    shadow_config: ShadowTracerConfig | None = None,
    neural_config: NeuralReconstructionConfig | None = None,
    allocation_config: NeuralAllocationConfig | None = None,
    sector_config: SectorRemapConfig | None = None,
    feature_config: InvariantFeatureConfig | None = None,
) -> dict[str, Any]:
    """Run forward-to-T and reversed-back-to-0 passive tracer transport."""

    if state.tracer_m is None:
        raise ValueError("reversible advection requires state.tracer_m")
    if int(nsteps) < 1:
        raise ValueError("nsteps must be positive")
    if float(dt) <= 0.0:
        raise ValueError("dt must be positive")
    duration = float(nsteps) * float(dt)
    t0 = time.perf_counter()
    if isinstance(flow, PrescribedFlow):
        flow_cfg = PrescribedFlow(
            name=flow.name,
            velocity_scale=float(velocity_scale),
            duration=duration,
            axis=flow.axis,
            wave_number=flow.wave_number,
        )
    else:
        flow_cfg = PrescribedFlow(name=str(flow), velocity_scale=float(velocity_scale), duration=duration)
    tracer0 = state.tracer_m.detach().clone()
    mass0 = tracer_mass(state, tracer0)
    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    neural_scheme = _is_neural_scheme(str(scheme))
    allocation_scheme = _is_allocation_scheme(str(scheme))
    sector_scheme = _is_sector_remap_scheme(str(scheme))
    shadow_m: torch.Tensor | None = None
    if neural_scheme:
        shadow_config = shadow_config or ShadowTracerConfig()
        neural_config = neural_config or NeuralReconstructionConfig(mode=_neural_mode_from_scheme(str(scheme)))
        allocation_config = allocation_config or NeuralAllocationConfig(mode=_neural_mode_from_scheme(str(scheme)))
        u_init = _stage_velocity(
            state.mesh,
            flow_cfg,
            time_seconds=0.0,
            duration=duration,
            reverse=False,
            n_levels=state.column.n_levels,
            dtype=tracer0.dtype,
            device=tracer0.device,
            velocity_scale=velocity_scale,
        )
        shadow_m, D_init = initialize_shadow_mass(
            state,
            q_from_tracer_m(state, tracer0),
            u_init,
            shadow_config,
        )
        if neural_model is None:
            z_init = shadow_concentration(state, shadow_m)
            if sector_scheme:
                input_dim = infer_sector_feature_dim(state, tracer0, u_init, D_init, z_init, dt=dt, feature_config=feature_config)
                neural_model = SectorRemapMLP(input_dim).to(dtype=tracer0.dtype, device=tracer0.device)
            else:
                input_dim = infer_feature_dim(state, tracer0, u_init, D_init, z_init, dt=dt, feature_config=feature_config)
            if allocation_scheme:
                neural_model = EdgeAllocationMLP(input_dim).to(dtype=tracer0.dtype, device=tracer0.device)
            elif not sector_scheme:
                neural_model = EdgeReconstructionMLP(input_dim).to(dtype=tracer0.dtype, device=tracer0.device)
        else:
            neural_model = neural_model.to(dtype=tracer0.dtype, device=tracer0.device)

    def maybe_snapshot(
        phase: str,
        step: int,
        *,
        elapsed_seconds: float,
        flow_time_seconds: float,
        phase_time_seconds: float,
        tracer_m: torch.Tensor,
        shadow_m: torch.Tensor | None = None,
        shadow_diagnostics_target: torch.Tensor | None = None,
        transport_info: dict[str, Any] | None = None,
    ) -> None:
        if snapshot_cadence <= 0:
            return
        if step % int(snapshot_cadence) == 0 or step == int(nsteps):
            item = {
                "phase": phase,
                "step": int(step),
                "time_seconds": float(elapsed_seconds),
                "elapsed_seconds": float(elapsed_seconds),
                "flow_time_seconds": float(flow_time_seconds),
                "phase_time_seconds": float(phase_time_seconds),
                "q": q_from_tracer_m(state, tracer_m).detach().cpu(),
            }
            if shadow_m is not None and shadow_diagnostics_target is not None:
                item["z"] = shadow_concentration(state, shadow_m).detach().cpu()
                item["D"] = shadow_diagnostics_target.detach().cpu()
            if transport_info is not None:
                for key, value in transport_info.items():
                    if isinstance(value, torch.Tensor):
                        item[str(key)] = value.detach().cpu()
            snapshots.append(item)

    def transport_info_for_row(
        tracer_m: torch.Tensor,
        shadow_m: torch.Tensor | None,
        u: torch.Tensor,
        D_row: torch.Tensor | None,
        *,
        include_tensors: bool,
    ) -> dict[str, Any] | None:
        if shadow_m is None or D_row is None or neural_model is None:
            return None
        if allocation_scheme:
            tendency_fn = neural_residual_allocation_tracer_tendency if _is_residual_allocation_scheme(str(scheme)) else neural_allocation_tracer_tendency
            _dq, info = tendency_fn(
                state,
                tracer_m,
                u,
                neural_model,  # type: ignore[arg-type]
                D_row,
                shadow_concentration(state, shadow_m),
                dt=dt,
                config=allocation_config or NeuralAllocationConfig(mode=_neural_mode_from_scheme(str(scheme))),
                feature_config=feature_config,
                return_info=True,
                include_tensors=include_tensors,
            )
            return info
        if sector_scheme:
            _dq, info = neural_sector_remap_tracer_tendency(
                state,
                tracer_m,
                u,
                neural_model,  # type: ignore[arg-type]
                D_row,
                shadow_concentration(state, shadow_m),
                dt=dt,
                config=sector_config or SectorRemapConfig(),
                feature_config=feature_config,
                return_info=True,
                include_tensors=include_tensors,
            )
            return info
        return None

    tracer = tracer0
    for step in range(int(nsteps) + 1):
        phase_t = float(step) * float(dt)
        elapsed = phase_t
        flow_t = phase_t
        u = _stage_velocity(
            state.mesh,
            flow_cfg,
            time_seconds=flow_t,
            duration=duration,
            reverse=False,
            n_levels=state.column.n_levels,
            dtype=tracer.dtype,
            device=tracer.device,
            velocity_scale=velocity_scale,
        )
        D_row = None
        if shadow_m is not None:
            D_row = compute_shadow_diagnostics(
                state.mesh,
                q_from_tracer_m(state, tracer),
                u,
                names=(shadow_config or ShadowTracerConfig()).diagnostic_names,
            )
        snapshot_due = snapshot_cadence > 0 and (step % int(snapshot_cadence) == 0 or step == int(nsteps))
        transport_info = transport_info_for_row(
            tracer,
            shadow_m,
            u,
            D_row,
            include_tensors=bool(snapshot_due),
        )
        rows.append(
            _sample_row(
                state,
                tracer,
                step=step,
                phase="forward",
                elapsed_seconds=elapsed,
                flow_time_seconds=flow_t,
                phase_time_seconds=phase_t,
                mass0=mass0,
                cfl=cfl_diagnostics(state.mesh, u, dt),
                shadow_m=shadow_m,
                shadow_diagnostics_target=D_row,
                transport_info=transport_info,
            )
        )
        maybe_snapshot("forward", step, elapsed_seconds=elapsed, flow_time_seconds=flow_t, phase_time_seconds=phase_t, tracer_m=tracer, shadow_m=shadow_m, shadow_diagnostics_target=D_row, transport_info=transport_info)
        if step == int(nsteps):
            break
        if shadow_m is None:
            tracer = _advance_ssprk3(
                state,
                tracer,
                flow=flow_cfg,
                t=phase_t,
                dt=dt,
                duration=duration,
                reverse=False,
                scheme=scheme,
                velocity_scale=velocity_scale,
            )
        else:
            assert neural_model is not None
            tracer, shadow_m = _advance_shadow_ssprk3(
                state,
                tracer,
                shadow_m,
                flow=flow_cfg,
                t=phase_t,
                dt=dt,
                duration=duration,
                reverse=False,
                scheme=scheme,
                velocity_scale=velocity_scale,
                neural_model=neural_model,
                shadow_config=shadow_config or ShadowTracerConfig(),
                neural_config=neural_config or NeuralReconstructionConfig(mode=_neural_mode_from_scheme(str(scheme))),
                allocation_config=allocation_config,
                sector_config=sector_config,
                feature_config=feature_config,
            )
    tracer_t = tracer.detach().clone()
    shadow_t = None if shadow_m is None else shadow_m.detach().clone()

    for step in range(1, int(nsteps) + 1):
        stage_t = float(step - 1) * float(dt)
        if shadow_m is None:
            tracer = _advance_ssprk3(
                state,
                tracer,
                flow=flow_cfg,
                t=stage_t,
                dt=dt,
                duration=duration,
                reverse=True,
                scheme=scheme,
                velocity_scale=velocity_scale,
            )
        else:
            assert neural_model is not None
            tracer, shadow_m = _advance_shadow_ssprk3(
                state,
                tracer,
                shadow_m,
                flow=flow_cfg,
                t=stage_t,
                dt=dt,
                duration=duration,
                reverse=True,
                scheme=scheme,
                velocity_scale=velocity_scale,
                neural_model=neural_model,
                shadow_config=shadow_config or ShadowTracerConfig(),
                neural_config=neural_config or NeuralReconstructionConfig(mode=_neural_mode_from_scheme(str(scheme))),
                allocation_config=allocation_config,
                sector_config=sector_config,
                feature_config=feature_config,
            )
        phase_t = float(step) * float(dt)
        elapsed = duration + phase_t
        flow_t = duration - phase_t
        u = _stage_velocity(
            state.mesh,
            flow_cfg,
            time_seconds=phase_t,
            duration=duration,
            reverse=True,
            n_levels=state.column.n_levels,
            dtype=tracer.dtype,
            device=tracer.device,
            velocity_scale=velocity_scale,
        )
        D_row = None
        if shadow_m is not None:
            D_row = compute_shadow_diagnostics(
                state.mesh,
                q_from_tracer_m(state, tracer),
                u,
                names=(shadow_config or ShadowTracerConfig()).diagnostic_names,
            )
        snapshot_due = snapshot_cadence > 0 and (step % int(snapshot_cadence) == 0 or step == int(nsteps))
        transport_info = transport_info_for_row(
            tracer,
            shadow_m,
            u,
            D_row,
            include_tensors=bool(snapshot_due),
        )
        rows.append(
            _sample_row(
                state,
                tracer,
                step=step,
                phase="reverse",
                elapsed_seconds=elapsed,
                flow_time_seconds=flow_t,
                phase_time_seconds=phase_t,
                mass0=mass0,
                cfl=cfl_diagnostics(state.mesh, u, dt),
                shadow_m=shadow_m,
                shadow_diagnostics_target=D_row,
                transport_info=transport_info,
            )
        )
        maybe_snapshot("reverse", step, elapsed_seconds=elapsed, flow_time_seconds=flow_t, phase_time_seconds=phase_t, tracer_m=tracer, shadow_m=shadow_m, shadow_diagnostics_target=D_row, transport_info=transport_info)
    tracer_r = tracer.detach().clone()
    shadow_r = None if shadow_m is None else shadow_m.detach().clone()

    q0_metrics = q_diagnostics(state, tracer0, prefix="initial")
    qt_metrics = q_diagnostics(state, tracer_t, prefix="forward")
    qr_metrics = q_diagnostics(state, tracer_r, prefix="recovered")
    recovery = comparison_diagnostics(state, tracer0, tracer_r)
    forward_shape = comparison_diagnostics(state, tracer0, tracer_t)
    q0_var = q0_metrics["initial_q_variance"]
    qt_var = qt_metrics["forward_q_variance"]
    q0_grad = q0_metrics["initial_q_edge_gradient_l2"]
    qt_grad = qt_metrics["forward_q_edge_gradient_l2"]
    summary = {
        "scheme": str(scheme),
        "flow": flow_cfg.name,
        "nsteps": int(nsteps),
        "dt": float(dt),
        "duration_seconds": duration,
        "velocity_scale": float(velocity_scale),
        "runtime_seconds": float(time.perf_counter() - t0),
        "max_abs_tracer_mass_rel_error": max(float(r["tracer_mass_rel_error_max_abs"]) for r in rows),
        "max_cfl_cell": max(float(r["cfl_cell_max"]) for r in rows),
        "mean_cfl_cell": float(np.mean([float(r["cfl_cell_mean"]) for r in rows])),
        "forward_variance_retention": float(qt_var / q0_var) if q0_var else None,
        "forward_variance_loss": float(1.0 - qt_var / q0_var) if q0_var else None,
        "forward_edge_gradient_ratio": float(qt_grad / q0_grad) if q0_grad else None,
        "forward_shape_l2_relative": forward_shape["recovery_l2_relative"],
        **q0_metrics,
        **qt_metrics,
        **qr_metrics,
        **recovery,
    }
    if shadow_m is not None:
        shadow_rows = [row for row in rows if "shadow_lag_rms_mean" in row]
        if shadow_rows:
            summary["shadow_final_lag_rms_mean"] = float(shadow_rows[-1]["shadow_lag_rms_mean"])
            summary["shadow_final_z_variance_mean"] = float(shadow_rows[-1]["shadow_z_variance_mean"])
    alloc_keys = sorted({key for row in rows for key in row if str(key).startswith("alloc_")})
    for key in alloc_keys:
        values = [
            float(row[key])
            for row in rows
            if key in row and isinstance(row[key], (int, float)) and math.isfinite(float(row[key]))
        ]
        if values:
            summary[f"{key}_final"] = float(values[-1])
            summary[f"{key}_max"] = float(max(values))
            summary[f"{key}_mean"] = float(np.mean(values))
    feature_keys = sorted({key for row in rows for key in row if str(key).startswith("feature_")})
    for key in feature_keys:
        values = [
            float(row[key])
            for row in rows
            if key in row and isinstance(row[key], (int, float)) and math.isfinite(float(row[key]))
        ]
        if values:
            summary[f"{key}_final"] = float(values[-1])
            summary[f"{key}_min"] = float(min(values))
            summary[f"{key}_max"] = float(max(values))
            summary[f"{key}_mean"] = float(np.mean(values))
    sector_keys = sorted({key for row in rows for key in row if str(key).startswith("sector_")})
    for key in sector_keys:
        values = [
            float(row[key])
            for row in rows
            if key in row and isinstance(row[key], (int, float)) and math.isfinite(float(row[key]))
        ]
        if values:
            summary[f"{key}_final"] = float(values[-1])
            summary[f"{key}_min"] = float(min(values))
            summary[f"{key}_max"] = float(max(values))
            summary[f"{key}_mean"] = float(np.mean(values))
    return {
        "summary": summary,
        "rows": rows,
        "sample": ReversibleAdvectionSample(
            q0=q_from_tracer_m(state, tracer0).detach().cpu(),
            qT=q_from_tracer_m(state, tracer_t).detach().cpu(),
            qR=q_from_tracer_m(state, tracer_r).detach().cpu(),
            snapshots=snapshots,
            metadata={
                "scheme": str(scheme),
                "flow": flow_cfg.__dict__,
                "nsteps": int(nsteps),
                "dt": float(dt),
                "snapshot_cadence": int(snapshot_cadence),
                "shadow_diagnostic_names": None if shadow_config is None else tuple(shadow_config.diagnostic_names),
                "allocation_config": None if allocation_config is None else allocation_config.__dict__,
                "sector_config": None if sector_config is None else sector_config.__dict__,
                "feature_config": None if feature_config is None else feature_config.__dict__,
            },
        ),
    }


def reversible_advection_sample(
    mesh,
    q0: torch.Tensor,
    *,
    flow: str | PrescribedFlow = "deformational",
    scheme: str = "upwind",
    nsteps: int = 200,
    dt: float = 300.0,
    velocity_scale: float = 20.0,
    snapshot_cadence: int = 0,
) -> ReversibleAdvectionSample:
    """Generate a single training-ready reversible-advection sample."""

    state = build_passive_transport_state(
        mesh,
        n_levels=int(q0.shape[1]),
        q=q0,
        dtype=q0.dtype,
        device=q0.device,
    )
    return rollout_reversible_advection(
        state,
        flow=flow,
        scheme=scheme,
        nsteps=nsteps,
        dt=dt,
        velocity_scale=velocity_scale,
        snapshot_cadence=snapshot_cadence,
    )["sample"]


def differentiable_neural_reversible_rollout(
    state: State,
    neural_model: nn.Module,
    *,
    flow: str | PrescribedFlow = "deformational",
    scheme: str = "neural_strict",
    nsteps: int = 12,
    dt: float = 300.0,
    velocity_scale: float = 20.0,
    shadow_config: ShadowTracerConfig | None = None,
    neural_config: NeuralReconstructionConfig | None = None,
    allocation_config: NeuralAllocationConfig | None = None,
    sector_config: SectorRemapConfig | None = None,
    feature_config: InvariantFeatureConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable forward/backward neural-shadow tracer rollout.

    This is intentionally narrower than ``rollout_reversible_advection``:
    it only supports the conservative neural reconstruction schemes and
    returns tensors for training losses instead of detached report payloads.
    """

    if state.tracer_m is None:
        raise ValueError("differentiable rollout requires state.tracer_m")
    if not _is_neural_scheme(str(scheme)):
        raise ValueError("differentiable neural rollout requires a neural scheme")
    if int(nsteps) < 1:
        raise ValueError("nsteps must be positive")
    if float(dt) <= 0.0:
        raise ValueError("dt must be positive")

    duration = float(nsteps) * float(dt)
    if isinstance(flow, PrescribedFlow):
        flow_cfg = PrescribedFlow(
            name=flow.name,
            velocity_scale=float(velocity_scale),
            duration=duration,
            axis=flow.axis,
            wave_number=flow.wave_number,
        )
    else:
        flow_cfg = PrescribedFlow(name=str(flow), velocity_scale=float(velocity_scale), duration=duration)
    tracer0 = state.tracer_m
    shadow_config = shadow_config or ShadowTracerConfig()
    neural_config = neural_config or NeuralReconstructionConfig(mode=_neural_mode_from_scheme(str(scheme)))
    allocation_config = allocation_config or NeuralAllocationConfig(mode=_neural_mode_from_scheme(str(scheme)))
    sector_config = sector_config or SectorRemapConfig()

    u_init = _stage_velocity(
        state.mesh,
        flow_cfg,
        time_seconds=0.0,
        duration=duration,
        reverse=False,
        n_levels=state.column.n_levels,
        dtype=tracer0.dtype,
        device=tracer0.device,
        velocity_scale=velocity_scale,
    )
    shadow_m, diagnostics0 = initialize_shadow_mass(
        state,
        q_from_tracer_m(state, tracer0),
        u_init,
        shadow_config,
    )
    tracer = tracer0
    for step in range(int(nsteps)):
        tracer, shadow_m = _advance_shadow_ssprk3(
            state,
            tracer,
            shadow_m,
            flow=flow_cfg,
            t=float(step) * float(dt),
            dt=dt,
            duration=duration,
            reverse=False,
            scheme=scheme,
            velocity_scale=velocity_scale,
            neural_model=neural_model,
            shadow_config=shadow_config,
            neural_config=neural_config,
            allocation_config=allocation_config,
            sector_config=sector_config,
            feature_config=feature_config,
        )
    tracer_t = tracer
    shadow_t = shadow_m

    for step in range(int(nsteps)):
        tracer, shadow_m = _advance_shadow_ssprk3(
            state,
            tracer,
            shadow_m,
            flow=flow_cfg,
            t=float(step) * float(dt),
            dt=dt,
            duration=duration,
            reverse=True,
            scheme=scheme,
            velocity_scale=velocity_scale,
            neural_model=neural_model,
            shadow_config=shadow_config,
            neural_config=neural_config,
            allocation_config=allocation_config,
            sector_config=sector_config,
            feature_config=feature_config,
        )
    u_final = _stage_velocity(
        state.mesh,
        flow_cfg,
        time_seconds=duration,
        duration=duration,
        reverse=True,
        n_levels=state.column.n_levels,
        dtype=tracer.dtype,
        device=tracer.device,
        velocity_scale=velocity_scale,
    )
    diagnostics_r = compute_shadow_diagnostics(
        state.mesh,
        q_from_tracer_m(state, tracer),
        u_final,
        names=shadow_config.diagnostic_names,
    )
    return {
        "tracer0": tracer0,
        "tracerT": tracer_t,
        "tracerR": tracer,
        "shadow0": shadow_mass_from_diagnostics(state, diagnostics0),
        "shadowT": shadow_t,
        "shadowR": shadow_m,
        "diagnostics0": diagnostics0,
        "diagnosticsR": diagnostics_r,
    }


def neural_recovery_loss_terms(
    state: State,
    tracer0: torch.Tensor,
    tracer_r: torch.Tensor,
    *,
    overshoot_eps: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Differentiable loss terms for recovery-trained reconstruction."""

    q0 = q_from_tracer_m(state, tracer0)
    qr = q_from_tracer_m(state, tracer_r)
    w = _weights(state)
    eps = torch.finfo(qr.dtype).tiny
    diff = qr - q0
    l2 = ((diff.square() * w).sum() / (q0.square() * w).sum().clamp_min(eps)).sqrt()
    l1 = (diff.abs() * w).sum() / w.sum().clamp_min(eps)
    q0_mean = _weighted_mean(q0, w)
    qr_mean = _weighted_mean(qr, w)
    q0_var = _weighted_variance(q0, w).clamp_min(eps)
    qr_var = _weighted_variance(qr, w).clamp_min(eps)
    cov = ((q0 - q0_mean) * (qr - qr_mean) * w).sum() / w.sum().clamp_min(eps)
    corr = cov / (q0_var.sqrt() * qr_var.sqrt()).clamp_min(eps)
    variance_loss = (qr_var / q0_var - 1.0).abs()
    variation0 = edge_variation_metrics(state.mesh, q0)["edge_gradient_l2"].clamp_min(eps)
    variation_r = edge_variation_metrics(state.mesh, qr)["edge_gradient_l2"]
    tv_loss = (variation_r / variation0 - 1.0).abs()
    negativity = torch.relu(-qr).mean()
    overshoot = torch.relu(qr - (q0.amax() + float(overshoot_eps))).mean()
    undershoot = torch.relu((q0.amin() - float(overshoot_eps)) - qr).mean()
    extrema_penalty = overshoot + undershoot + negativity
    loss = l2 + 0.2 * (1.0 - corr) + 0.1 * variance_loss + 0.05 * tv_loss + 10.0 * extrema_penalty
    return {
        "loss": loss,
        "l2": l2,
        "l1": l1,
        "correlation": corr,
        "correlation_loss": 1.0 - corr,
        "variance_loss": variance_loss,
        "tv_loss": tv_loss,
        "negativity_penalty": negativity,
        "overshoot_penalty": overshoot,
        "undershoot_penalty": undershoot,
        "extrema_penalty": extrema_penalty,
    }


__all__ = [
    "PrescribedFlow",
    "ReversibleAdvectionSample",
    "build_passive_transport_state",
    "cfl_diagnostics",
    "comparison_diagnostics",
    "differentiable_neural_reversible_rollout",
    "edge_variation_metrics",
    "neural_recovery_loss_terms",
    "prescribed_edge_velocity",
    "q_diagnostics",
    "q_from_tracer_m",
    "reversible_advection_sample",
    "rollout_reversible_advection",
    "tracer_mass",
]
