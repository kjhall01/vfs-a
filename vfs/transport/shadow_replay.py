"""Replay buffers for passive shadow-tracer reversible training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from vfs.core.state import State
from vfs.transport.neural_reconstruction import (
    EdgeAllocationMLP,
    InvariantFeatureConfig,
    NeuralAllocationConfig,
    NeuralReconstructionConfig,
    SectorRemapConfig,
    neural_allocation_envelope_penalty,
    neural_allocation_residual_penalty,
    neural_allocation_tracer_tendency,
    neural_residual_allocation_tracer_tendency,
    neural_sector_remap_tracer_tendency,
    neural_tracer_tendency,
)
from vfs.transport.reversible_advection import (
    PrescribedFlow,
    _transport_tendency,
    build_passive_transport_state,
    edge_variation_metrics,
    prescribed_edge_velocity,
    q_from_tracer_m,
    tracer_mass,
)
from vfs.transport.shadow_diagnostics import compute_shadow_diagnostics
from vfs.transport.shadow_tracers import (
    ShadowTracerConfig,
    initialize_shadow_mass,
    shadow_concentration,
    shadow_tendency,
)
from vfs.transport.synthetic_tracers import make_tracer_field


REPLAY_FORMAT = "vfs_shadow_replay_buffer_v1"


@dataclass(frozen=True)
class ReplayWindowRef:
    """Reference to a saved state that can support a training window."""

    trajectory_index: int
    state_index: int
    step: int
    time_seconds: float


@dataclass(frozen=True)
class ReplayWindow:
    """A sampled replay state plus trajectory metadata."""

    trajectory_index: int
    state_index: int
    step: int
    time_seconds: float
    q: torch.Tensor
    z: torch.Tensor
    D: torch.Tensor | None
    velocity: torch.Tensor | None
    trajectory_metadata: dict[str, Any]
    buffer_metadata: dict[str, Any]


@dataclass(frozen=True)
class ReplayLossConfig:
    """Loss coefficients for truncated reversible replay windows."""

    lambda_step: float = 0.5
    lambda_corr: float = 0.2
    lambda_var: float = 0.1
    lambda_grad: float = 0.05
    lambda_extrema: float = 10.0
    lambda_env: float = 0.0
    lambda_residual: float = 0.0
    lambda_forward_reference: float = 0.0
    overshoot_eps: float = 0.0


def _flow_from_metadata(metadata: dict[str, Any]) -> PrescribedFlow:
    flow = metadata.get("flow", "deformational")
    if isinstance(flow, dict):
        return PrescribedFlow(
            name=str(flow.get("name", "deformational")),
            velocity_scale=float(flow.get("velocity_scale", metadata.get("velocity_scale", 20.0))),
            duration=float(flow.get("duration", metadata.get("duration_seconds", 86400.0))),
            axis=tuple(float(x) for x in flow.get("axis", (0.0, 0.0, 1.0))),  # type: ignore[arg-type]
            wave_number=int(flow.get("wave_number", 2)),
        )
    return PrescribedFlow(
        name=str(flow),
        velocity_scale=float(metadata.get("velocity_scale", 20.0)),
        duration=float(metadata.get("duration_seconds", metadata.get("nsteps", 1) * metadata.get("dt", 300.0))),
        wave_number=int(metadata.get("wave_number", 2)),
    )


def _stage_velocity_at(
    mesh,
    flow: PrescribedFlow,
    *,
    original_time_seconds: float,
    reverse: bool,
    n_levels: int,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    u = prescribed_edge_velocity(
        mesh,
        flow,
        time_seconds=float(original_time_seconds),
        duration=float(flow.duration),
        reverse=False,
        dtype=dtype,
        device=device,
        velocity_scale=float(flow.velocity_scale),
        wave_number=int(flow.wave_number),
    )
    if reverse:
        u = -u
    return u.expand(mesh.nEdges, int(n_levels)).contiguous()


def _mode_from_scheme(scheme: str) -> str:
    return "relaxed" if "relaxed" in str(scheme).lower() else "strict"


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


def _neural_tendency(
    state: State,
    tracer_m: torch.Tensor,
    shadow_m: torch.Tensor,
    u_e: torch.Tensor,
    model: nn.Module,
    *,
    scheme: str,
    dt: float,
    shadow_config: ShadowTracerConfig,
    neural_config: NeuralReconstructionConfig,
    allocation_config: NeuralAllocationConfig | None,
    sector_config: SectorRemapConfig | None = None,
    feature_config: InvariantFeatureConfig | None = None,
    return_info: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    q = q_from_tracer_m(state, tracer_m)
    diagnostics = compute_shadow_diagnostics(
        state.mesh,
        q,
        u_e,
        names=shadow_config.diagnostic_names,
    )
    z = shadow_concentration(state, shadow_m)
    mode = _mode_from_scheme(scheme)
    if _is_sector_remap_scheme(scheme):
        dq_info = neural_sector_remap_tracer_tendency(
            state,
            tracer_m,
            u_e,
            model,  # type: ignore[arg-type]
            diagnostics,
            z,
            dt=dt,
            config=sector_config or SectorRemapConfig(),
            feature_config=feature_config,
            return_info=return_info,
        )
    elif _is_allocation_scheme(scheme):
        allocation_config = allocation_config or NeuralAllocationConfig(mode=mode)
        if allocation_config.mode != mode:
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
        dq_info = tendency_fn(
            state,
            tracer_m,
            u_e,
            model,  # type: ignore[arg-type]
            diagnostics,
            z,
            dt=dt,
            config=allocation_config,
            feature_config=feature_config,
            return_info=return_info,
        )
    else:
        if neural_config.mode != mode:
            neural_config = NeuralReconstructionConfig(
                mode=mode,
                relaxed_eps_minus=neural_config.relaxed_eps_minus,
                relaxed_eps_plus=neural_config.relaxed_eps_plus,
                donor_export_safety=neural_config.donor_export_safety,
            )
        dq_info = neural_tracer_tendency(
            state,
            tracer_m,
            u_e,
            model,
            diagnostics,
            z,
            dt=dt,
            config=neural_config,
            feature_config=feature_config,
            return_info=return_info,
        )
    if return_info:
        dq, info = dq_info  # type: ignore[misc]
    else:
        dq = dq_info  # type: ignore[assignment]
        info = {}
    dz = shadow_tendency(
        state,
        shadow_m,
        u_e,
        diagnostics,
        config=shadow_config,
        dt=dt,
    )
    return dq, dz, diagnostics, info


def shadow_transport_step(
    state: State,
    tracer_m: torch.Tensor,
    shadow_m: torch.Tensor,
    *,
    flow: PrescribedFlow,
    time_seconds: float,
    dt: float,
    reverse: bool,
    scheme: str,
    velocity_scale: float,
    shadow_config: ShadowTracerConfig,
    neural_model: nn.Module | None = None,
    neural_config: NeuralReconstructionConfig | None = None,
    allocation_config: NeuralAllocationConfig | None = None,
    sector_config: SectorRemapConfig | None = None,
    feature_config: InvariantFeatureConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance tracer and shadow memory one SSPRK3 step in a replay window."""

    del velocity_scale  # velocity is carried by the PrescribedFlow object.
    mesh = state.mesh
    dtype = tracer_m.dtype
    device = tracer_m.device
    n_levels = int(state.column.n_levels)
    neural_config = neural_config or NeuralReconstructionConfig(mode=_mode_from_scheme(scheme))
    allocation_config = allocation_config or NeuralAllocationConfig(mode=_mode_from_scheme(scheme))

    def tendencies(qm: torch.Tensor, zm: torch.Tensor, original_time: float) -> tuple[torch.Tensor, torch.Tensor]:
        u = _stage_velocity_at(
            mesh,
            flow,
            original_time_seconds=original_time,
            reverse=reverse,
            n_levels=n_levels,
            dtype=dtype,
            device=device,
        )
        if _is_neural_scheme(scheme):
            if neural_model is None:
                raise ValueError("neural replay transport requires a neural_model")
            dq, dz, _D, _info = _neural_tendency(
                state,
                qm,
                zm,
                u,
                neural_model,
                scheme=scheme,
                dt=dt,
                shadow_config=shadow_config,
                neural_config=neural_config,
                allocation_config=allocation_config,
                sector_config=sector_config,
                feature_config=feature_config,
            )
            return dq, dz
        q = q_from_tracer_m(state, qm)
        D = compute_shadow_diagnostics(state.mesh, q, u, names=shadow_config.diagnostic_names)
        return (
            _transport_tendency(state, qm, u, scheme=scheme, dt=dt),
            shadow_tendency(state, zm, u, D, config=shadow_config, dt=dt),
        )

    if reverse:
        t0 = float(time_seconds)
        t1 = float(time_seconds) - float(dt)
        th = float(time_seconds) - 0.5 * float(dt)
    else:
        t0 = float(time_seconds)
        t1 = float(time_seconds) + float(dt)
        th = float(time_seconds) + 0.5 * float(dt)

    kq0, kz0 = tendencies(tracer_m, shadow_m, t0)
    q1 = tracer_m + float(dt) * kq0
    z1 = shadow_m + float(dt) * kz0
    kq1, kz1 = tendencies(q1, z1, t1)
    q2 = 0.75 * tracer_m + 0.25 * (q1 + float(dt) * kq1)
    z2 = 0.75 * shadow_m + 0.25 * (z1 + float(dt) * kz1)
    kq2, kz2 = tendencies(q2, z2, th)
    return (
        (1.0 / 3.0) * tracer_m + (2.0 / 3.0) * (q2 + float(dt) * kq2),
        (1.0 / 3.0) * shadow_m + (2.0 / 3.0) * (z2 + float(dt) * kz2),
    )


def generate_shadow_replay_trajectory(
    mesh,
    *,
    tracer: str,
    seed: int,
    n_levels: int,
    flow: str | PrescribedFlow,
    nsteps: int,
    dt: float,
    save_every: int,
    scheme: str,
    velocity_scale: float,
    dtype: torch.dtype,
    device,
    shadow_config: ShadowTracerConfig,
    neural_model: nn.Module | None = None,
    neural_config: NeuralReconstructionConfig | None = None,
    allocation_config: NeuralAllocationConfig | None = None,
    sector_config: SectorRemapConfig | None = None,
    feature_config: InvariantFeatureConfig | None = None,
) -> dict[str, Any]:
    """Generate one non-differentiable replay trajectory."""

    if int(save_every) < 1:
        raise ValueError("save_every must be positive")
    q0 = make_tracer_field(
        mesh,
        str(tracer),
        seed=int(seed),
        n_levels=int(n_levels),
        n_tracers=1,
        dtype=dtype,
        device=device,
    )
    state = build_passive_transport_state(mesh, n_levels=int(n_levels), q=q0, dtype=dtype, device=device)
    duration = float(nsteps) * float(dt)
    flow_cfg = (
        flow
        if isinstance(flow, PrescribedFlow)
        else PrescribedFlow(name=str(flow), velocity_scale=float(velocity_scale), duration=duration)
    )
    flow_cfg = PrescribedFlow(
        name=flow_cfg.name,
        velocity_scale=float(velocity_scale),
        duration=duration,
        axis=flow_cfg.axis,
        wave_number=flow_cfg.wave_number,
    )
    u0 = _stage_velocity_at(
        mesh,
        flow_cfg,
        original_time_seconds=0.0,
        reverse=False,
        n_levels=int(n_levels),
        dtype=dtype,
        device=device,
    )
    shadow_m, _D0 = initialize_shadow_mass(state, q0, u0, shadow_config)
    tracer_m = state.tracer_m
    states: list[dict[str, Any]] = []

    def save_state(step: int, q_m: torch.Tensor, z_m: torch.Tensor) -> None:
        t = float(step) * float(dt)
        u = _stage_velocity_at(
            mesh,
            flow_cfg,
            original_time_seconds=t,
            reverse=False,
            n_levels=int(n_levels),
            dtype=dtype,
            device=device,
        )
        q = q_from_tracer_m(state, q_m)
        D = compute_shadow_diagnostics(mesh, q, u, names=shadow_config.diagnostic_names)
        states.append(
            {
                "step": int(step),
                "time_seconds": t,
                "q": q.detach().cpu(),
                "z": shadow_concentration(state, z_m).detach().cpu(),
                "D": D.detach().cpu(),
                "velocity": u.detach().cpu(),
            }
        )

    with torch.no_grad():
        save_state(0, tracer_m, shadow_m)
        for step in range(int(nsteps)):
            tracer_m, shadow_m = shadow_transport_step(
                state,
                tracer_m,
                shadow_m,
                flow=flow_cfg,
                time_seconds=float(step) * float(dt),
                dt=float(dt),
                reverse=False,
                scheme=str(scheme),
                velocity_scale=float(velocity_scale),
                shadow_config=shadow_config,
                neural_model=neural_model,
                neural_config=neural_config,
                allocation_config=allocation_config,
                sector_config=sector_config,
                feature_config=feature_config,
            )
            next_step = step + 1
            if next_step % int(save_every) == 0 or next_step == int(nsteps):
                save_state(next_step, tracer_m, shadow_m)

    return {
        "metadata": {
            "tracer": str(tracer),
            "seed": int(seed),
            "n_levels": int(n_levels),
            "flow": flow_cfg.__dict__,
            "nsteps": int(nsteps),
            "dt": float(dt),
            "duration_seconds": duration,
            "save_every": int(save_every),
            "scheme": str(scheme),
            "shadow_config": {
                "source_mode": shadow_config.source_mode,
                "tau_seconds": shadow_config.tau_seconds,
                "diagnostic_names": tuple(shadow_config.diagnostic_names),
            },
        },
        "states": states,
    }


class ShadowReplayBuffer:
    """Lightweight replay-buffer payload loader and window sampler."""

    def __init__(self, payload: dict[str, Any], *, dtype: torch.dtype | None = None, device=None) -> None:
        if payload.get("format") != REPLAY_FORMAT:
            raise ValueError(f"unsupported replay buffer format {payload.get('format')!r}")
        self.payload = payload
        self.metadata = dict(payload.get("metadata", {}))
        self.trajectories = list(payload.get("trajectories", []))
        self.dtype = dtype
        self.device = device

    @classmethod
    def load(cls, path: str | Path, *, dtype: torch.dtype | None = None, device=None) -> "ShadowReplayBuffer":
        return cls(torch.load(path, map_location="cpu"), dtype=dtype, device=device)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.payload, path)

    def _tensor(self, value: Any) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if self.dtype is not None or self.device is not None:
            tensor = tensor.to(dtype=self.dtype or tensor.dtype, device=self.device or tensor.device)
        return tensor

    def state_count(self) -> int:
        return sum(len(traj.get("states", [])) for traj in self.trajectories)

    def eligible_windows(
        self,
        *,
        window_steps: int,
        time_range: str = "mixed",
        trajectory_indices: Iterable[int] | None = None,
    ) -> list[ReplayWindowRef]:
        allowed = set(trajectory_indices) if trajectory_indices is not None else None
        refs: list[ReplayWindowRef] = []
        for ti, traj in enumerate(self.trajectories):
            if allowed is not None and ti not in allowed:
                continue
            metadata = traj.get("metadata", {})
            nsteps = int(metadata.get("nsteps", self.metadata.get("nsteps", 0)))
            states = traj.get("states", [])
            for si, state in enumerate(states):
                step = int(state["step"])
                if step + int(window_steps) > nsteps:
                    continue
                frac = step / max(float(nsteps), 1.0)
                if time_range == "early" and frac > 1.0 / 3.0:
                    continue
                if time_range in {"mid", "mid-deformation", "middle"} and not (1.0 / 3.0 <= frac <= 2.0 / 3.0):
                    continue
                if time_range == "late" and frac < 2.0 / 3.0:
                    continue
                refs.append(ReplayWindowRef(ti, si, step, float(state["time_seconds"])))
        if not refs:
            raise ValueError("no replay states can support the requested window")
        return refs

    def window_from_ref(self, ref: ReplayWindowRef) -> ReplayWindow:
        traj = self.trajectories[int(ref.trajectory_index)]
        state = traj["states"][int(ref.state_index)]
        return ReplayWindow(
            trajectory_index=int(ref.trajectory_index),
            state_index=int(ref.state_index),
            step=int(state["step"]),
            time_seconds=float(state["time_seconds"]),
            q=self._tensor(state["q"]),
            z=self._tensor(state["z"]),
            D=None if "D" not in state else self._tensor(state["D"]),
            velocity=None if "velocity" not in state else self._tensor(state["velocity"]),
            trajectory_metadata=dict(traj.get("metadata", {})),
            buffer_metadata=self.metadata,
        )

    def sample_window(
        self,
        *,
        window_steps: int,
        rng: random.Random,
        time_range: str = "mixed",
        candidates: list[ReplayWindowRef] | None = None,
    ) -> ReplayWindow:
        refs = candidates if candidates is not None else self.eligible_windows(window_steps=window_steps, time_range=time_range)
        return self.window_from_ref(rng.choice(refs))

    def find_saved_state(self, trajectory_index: int, step: int) -> torch.Tensor | None:
        traj = self.trajectories[int(trajectory_index)]
        for state in traj.get("states", []):
            if int(state.get("step", -1)) == int(step):
                return self._tensor(state["q"])
        return None

    def summary(self) -> dict[str, Any]:
        tracers = sorted({str(traj.get("metadata", {}).get("tracer", "")) for traj in self.trajectories})
        flows = sorted({str(_flow_from_metadata(traj.get("metadata", {})).name) for traj in self.trajectories})
        q_min: list[float] = []
        q_max: list[float] = []
        q_var: list[float] = []
        z_min: list[float] = []
        z_max: list[float] = []
        z_var: list[float] = []
        lag: list[float] = []
        steps: list[int] = []
        for traj in self.trajectories:
            for state in traj.get("states", []):
                q = state["q"]
                z = state["z"]
                q_min.append(float(q.amin()))
                q_max.append(float(q.amax()))
                q_var.append(float(q.var(unbiased=False)))
                z_min.append(float(z.amin()))
                z_max.append(float(z.amax()))
                z_var.append(float(z.var(unbiased=False)))
                steps.append(int(state["step"]))
                if "D" in state:
                    lag.append(float((z - state["D"]).square().mean().sqrt()))
        return {
            "format": self.payload.get("format"),
            "trajectory_count": len(self.trajectories),
            "saved_state_count": self.state_count(),
            "tracers": tracers,
            "flows": flows,
            "step_min": min(steps) if steps else None,
            "step_max": max(steps) if steps else None,
            "q_min": min(q_min) if q_min else None,
            "q_max": max(q_max) if q_max else None,
            "q_variance_min": min(q_var) if q_var else None,
            "q_variance_max": max(q_var) if q_var else None,
            "z_min": min(z_min) if z_min else None,
            "z_max": max(z_max) if z_max else None,
            "z_variance_min": min(z_var) if z_var else None,
            "z_variance_max": max(z_var) if z_var else None,
            "shadow_lag_rms_mean": sum(lag) / len(lag) if lag else None,
            "metadata": self.metadata,
        }


def make_replay_payload(
    *,
    trajectories: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": REPLAY_FORMAT,
        "metadata": dict(metadata),
        "trajectories": trajectories,
    }


def replay_window_state(
    mesh,
    window: ReplayWindow,
    *,
    dtype: torch.dtype,
    device,
    shadow_config: ShadowTracerConfig | None = None,
    shadow_mode: str = "saved",
) -> tuple[State, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a sampled replay window into State, tracer mass, shadow mass, D."""

    q = window.q.to(dtype=dtype, device=device)
    state = build_passive_transport_state(mesh, n_levels=int(q.shape[1]), q=q, dtype=dtype, device=device)
    flow = _flow_from_metadata(window.trajectory_metadata)
    u = _stage_velocity_at(
        mesh,
        flow,
        original_time_seconds=float(window.time_seconds),
        reverse=False,
        n_levels=int(q.shape[1]),
        dtype=dtype,
        device=device,
    )
    shadow_config = shadow_config or ShadowTracerConfig()
    D = compute_shadow_diagnostics(mesh, q, u, names=shadow_config.diagnostic_names)
    mode = str(shadow_mode)
    if mode == "saved":
        z = window.z.to(dtype=dtype, device=device)
    elif mode == "reset":
        z = D
    elif mode == "zero":
        z = torch.zeros_like(D)
    else:
        raise ValueError("shadow_mode must be saved, reset, or zero")
    return state, state.tracer_m, state.layer_dp()[..., None] * z, D


def _loss_shape_terms(state: State, reference_m: torch.Tensor, candidate_m: torch.Tensor, *, overshoot_eps: float) -> dict[str, torch.Tensor]:
    q0 = q_from_tracer_m(state, reference_m)
    q = q_from_tracer_m(state, candidate_m)
    weights = state.mesh.cell_area.to(dtype=q.dtype, device=q.device)[:, None, None] * state.layer_dp()[..., None]
    eps = torch.finfo(q.dtype).tiny
    diff = q - q0
    l2 = ((diff.square() * weights).sum() / (q0.square() * weights).sum().clamp_min(eps)).sqrt()
    q0_mean = (q0 * weights).sum() / weights.sum().clamp_min(eps)
    q_mean = (q * weights).sum() / weights.sum().clamp_min(eps)
    q0_var = ((q0 - q0_mean).square() * weights).sum() / weights.sum().clamp_min(eps)
    q_var = ((q - q_mean).square() * weights).sum() / weights.sum().clamp_min(eps)
    cov = ((q0 - q0_mean) * (q - q_mean) * weights).sum() / weights.sum().clamp_min(eps)
    corr = cov / (q0_var.sqrt() * q_var.sqrt()).clamp_min(eps)
    grad0 = edge_variation_metrics(state.mesh, q0)["edge_gradient_l2"].clamp_min(eps)
    grad = edge_variation_metrics(state.mesh, q)["edge_gradient_l2"]
    variance_retention = q_var / q0_var.clamp_min(eps)
    gradient_retention = grad / grad0
    extrema_penalty = (
        torch.relu(-q).mean()
        + torch.relu(q - (q0.amax() + float(overshoot_eps))).mean()
        + torch.relu((q0.amin() - float(overshoot_eps)) - q).mean()
    )
    return {
        "l2": l2,
        "correlation": corr,
        "correlation_loss": 1.0 - corr,
        "variance_retention": variance_retention,
        "variance_loss": (variance_retention - 1.0).abs(),
        "edge_gradient_retention": gradient_retention,
        "edge_gradient_loss": (gradient_retention - 1.0).abs(),
        "extrema_penalty": extrema_penalty,
    }


def replay_truncated_reversible_loss(
    mesh,
    window: ReplayWindow,
    model: nn.Module,
    *,
    scheme: str,
    window_steps: int,
    dt: float,
    velocity_scale: float | None,
    shadow_config: ShadowTracerConfig,
    neural_config: NeuralReconstructionConfig,
    allocation_config: NeuralAllocationConfig | None,
    loss_config: ReplayLossConfig,
    sector_config: SectorRemapConfig | None = None,
    feature_config: InvariantFeatureConfig | None = None,
    shadow_mode: str = "saved",
    forward_reference_q: torch.Tensor | None = None,
    dtype: torch.dtype,
    device,
) -> dict[str, torch.Tensor]:
    """Run a short reversible replay window and return differentiable losses."""

    state, tracer0, shadow0, D0 = replay_window_state(
        mesh,
        window,
        dtype=dtype,
        device=device,
        shadow_config=shadow_config,
        shadow_mode=shadow_mode,
    )
    flow = _flow_from_metadata(window.trajectory_metadata)
    flow = PrescribedFlow(
        name=flow.name,
        velocity_scale=float(flow.velocity_scale if velocity_scale is None else velocity_scale),
        duration=float(flow.duration),
        axis=flow.axis,
        wave_number=flow.wave_number,
    )
    tracer = tracer0
    shadow = shadow0
    forward: list[tuple[torch.Tensor, torch.Tensor]] = [(tracer, shadow)]
    start = float(window.time_seconds)
    for step in range(int(window_steps)):
        tracer, shadow = shadow_transport_step(
            state,
            tracer,
            shadow,
            flow=flow,
            time_seconds=start + float(step) * float(dt),
            dt=float(dt),
            reverse=False,
            scheme=str(scheme),
            velocity_scale=float(flow.velocity_scale),
            shadow_config=shadow_config,
            neural_model=model,
            neural_config=neural_config,
            allocation_config=allocation_config,
            sector_config=sector_config,
            feature_config=feature_config,
        )
        forward.append((tracer, shadow))

    forward_end = tracer
    step_l2: list[torch.Tensor] = []
    for reverse_step in range(int(window_steps)):
        original_time = start + float(window_steps - reverse_step) * float(dt)
        tracer, shadow = shadow_transport_step(
            state,
            tracer,
            shadow,
            flow=flow,
            time_seconds=original_time,
            dt=float(dt),
            reverse=True,
            scheme=str(scheme),
            velocity_scale=float(flow.velocity_scale),
            shadow_config=shadow_config,
            neural_model=model,
            neural_config=neural_config,
            allocation_config=allocation_config,
            sector_config=sector_config,
            feature_config=feature_config,
        )
        ref = forward[int(window_steps) - reverse_step - 1][0]
        step_l2.append(_loss_shape_terms(state, ref, tracer, overshoot_eps=loss_config.overshoot_eps)["l2"])

    final_terms = _loss_shape_terms(state, tracer0, tracer, overshoot_eps=loss_config.overshoot_eps)
    step_l2_tensor = torch.stack(step_l2) if step_l2 else torch.zeros(1, dtype=dtype, device=device)
    env_penalty = torch.zeros((), dtype=dtype, device=device)
    if (
        allocation_config is not None
        and _is_allocation_scheme(scheme)
        and not _is_residual_allocation_scheme(scheme)
        and str(allocation_config.envelope).lower() == "anisotropic"
    ):
        u0 = _stage_velocity_at(
            mesh,
            flow,
            original_time_seconds=start,
            reverse=False,
            n_levels=int(state.column.n_levels),
            dtype=dtype,
            device=device,
        )
        env_penalty = neural_allocation_envelope_penalty(
            state,
            tracer0,
            u0,
            model,  # type: ignore[arg-type]
            D0,
            shadow_concentration(state, shadow0),
            dt=float(dt),
            config=allocation_config,
            feature_config=feature_config,
        )
    residual_penalty = torch.zeros((), dtype=dtype, device=device)
    if (
        allocation_config is not None
        and _is_residual_allocation_scheme(scheme)
        and float(loss_config.lambda_residual) != 0.0
    ):
        u0 = _stage_velocity_at(
            mesh,
            flow,
            original_time_seconds=start,
            reverse=False,
            n_levels=int(state.column.n_levels),
            dtype=dtype,
            device=device,
        )
        residual_penalty = neural_allocation_residual_penalty(
            state,
            tracer0,
            u0,
            model,  # type: ignore[arg-type]
            D0,
            shadow_concentration(state, shadow0),
            dt=float(dt),
            config=allocation_config,
            feature_config=feature_config,
        )

    forward_reference_loss = torch.zeros((), dtype=dtype, device=device)
    if forward_reference_q is not None and float(loss_config.lambda_forward_reference) != 0.0:
        ref_state = build_passive_transport_state(
            mesh,
            n_levels=int(forward_reference_q.shape[1]),
            q=forward_reference_q.to(dtype=dtype, device=device),
            dtype=dtype,
            device=device,
        )
        forward_reference_loss = _loss_shape_terms(state, ref_state.tracer_m, forward_end, overshoot_eps=loss_config.overshoot_eps)["l2"]

    mass0 = tracer_mass(state, tracer0).abs().clamp_min(torch.finfo(dtype).tiny)
    mass_drift = ((tracer_mass(state, tracer) - tracer_mass(state, tracer0)).abs() / mass0).amax()
    q_start = q_from_tracer_m(state, tracer0)
    q_end = q_from_tracer_m(state, tracer)
    shadow_lag = (shadow_concentration(state, shadow0) - D0).square().mean().sqrt()

    alloc_info: dict[str, Any] = {}
    if allocation_config is not None and _is_allocation_scheme(scheme):
        u0 = _stage_velocity_at(
            mesh,
            flow,
            original_time_seconds=start,
            reverse=False,
            n_levels=int(state.column.n_levels),
            dtype=dtype,
            device=device,
        )
        tendency_fn = neural_residual_allocation_tracer_tendency if _is_residual_allocation_scheme(scheme) else neural_allocation_tracer_tendency
        _dq, alloc_info = tendency_fn(
            state,
            tracer0,
            u0,
            model,  # type: ignore[arg-type]
            D0,
            shadow_concentration(state, shadow0),
            dt=float(dt),
            config=allocation_config,
            feature_config=feature_config,
            return_info=True,
        )
    elif _is_sector_remap_scheme(scheme):
        u0 = _stage_velocity_at(
            mesh,
            flow,
            original_time_seconds=start,
            reverse=False,
            n_levels=int(state.column.n_levels),
            dtype=dtype,
            device=device,
        )
        _dq, alloc_info = neural_sector_remap_tracer_tendency(
            state,
            tracer0,
            u0,
            model,  # type: ignore[arg-type]
            D0,
            shadow_concentration(state, shadow0),
            dt=float(dt),
            config=sector_config or SectorRemapConfig(),
            feature_config=feature_config,
            return_info=True,
        )

    loss = (
        final_terms["l2"]
        + float(loss_config.lambda_step) * step_l2_tensor.mean()
        + float(loss_config.lambda_corr) * final_terms["correlation_loss"]
        + float(loss_config.lambda_var) * final_terms["variance_loss"]
        + float(loss_config.lambda_grad) * final_terms["edge_gradient_loss"]
        + float(loss_config.lambda_extrema) * final_terms["extrema_penalty"]
        + float(loss_config.lambda_env) * env_penalty
        + float(loss_config.lambda_residual) * residual_penalty
        + float(loss_config.lambda_forward_reference) * forward_reference_loss
    )
    out: dict[str, torch.Tensor] = {
        "loss": loss,
        "final_recovery_l2": final_terms["l2"],
        "stepwise_recovery_l2_mean": step_l2_tensor.mean(),
        "recovery_correlation": final_terms["correlation"],
        "recovery_correlation_loss": final_terms["correlation_loss"],
        "variance_retention": final_terms["variance_retention"],
        "variance_loss": final_terms["variance_loss"],
        "edge_gradient_retention": final_terms["edge_gradient_retention"],
        "edge_gradient_loss": final_terms["edge_gradient_loss"],
        "extrema_penalty": final_terms["extrema_penalty"],
        "envelope_penalty": env_penalty,
        "residual_penalty": residual_penalty,
        "forward_reference_loss": forward_reference_loss,
        "q_min": q_end.amin(),
        "q_max": q_end.amax(),
        "q_start_min": q_start.amin(),
        "q_start_max": q_start.amax(),
        "mass_drift": mass_drift,
        "shadow_lag_rms": shadow_lag,
    }
    for index, value in enumerate(step_l2_tensor, start=1):
        out[f"stepwise_recovery_l2_{index}"] = value
    for key, value in alloc_info.items():
        if isinstance(value, (int, float)):
            out[str(key)] = torch.as_tensor(float(value), dtype=dtype, device=device)
    return out


__all__ = [
    "REPLAY_FORMAT",
    "ReplayLossConfig",
    "ReplayWindow",
    "ReplayWindowRef",
    "ShadowReplayBuffer",
    "generate_shadow_replay_trajectory",
    "make_replay_payload",
    "replay_truncated_reversible_loss",
    "replay_window_state",
    "shadow_transport_step",
]
