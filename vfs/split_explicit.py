"""Experimental split-explicit atmosphere fast-mode steppers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .atmosphere import DryHydrostaticDynamics
from .core.registry import register_model
from .core.state import State


def _rms(value: torch.Tensor) -> torch.Tensor:
    return value.detach().square().mean().sqrt()


def _float_info(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


@register_model(
    "dry_core_split_explicit_fastmode",
    status="negative_result",
    component="throughput",
    description="Split-explicit dry-atmosphere fast-mode prototype; failed current dry science gates.",
    default_safe=False,
)
class DryCoreSplitExplicitFastMode(nn.Module):
    """Minimal dry-atmosphere split-explicit fast-mode prototype.

    The canonical dry ``tendency(state)`` path is unchanged.  This object is an
    opt-in stepper: it holds slow non-PGF tendencies fixed over one large step
    and advances the pressure-gradient/continuity fast mode with a
    forward-backward subcycle.  It is intended for throughput diagnostics, not
    as a validated production timestepper.
    """

    scheme_name = "split_explicit_fastmode"

    def __init__(
        self,
        dynamics: DryHydrostaticDynamics | None = None,
        *,
        fast_dt: float = 360.0,
        mesh: Any | None = None,
        column: Any | None = None,
        distribute_slow_u: bool = True,
        include_fast_rotation: bool = False,
    ) -> None:
        super().__init__()
        self.dynamics = DryHydrostaticDynamics(validate_state=False) if dynamics is None else dynamics
        self.fast_dt = float(fast_dt)
        self.mesh = mesh
        self.column = column
        self.distribute_slow_u = bool(distribute_slow_u)
        self.include_fast_rotation = bool(include_fast_rotation)
        self.last_step_info: dict[str, Any] = {}

    def bind_state(self, state: State) -> State:
        if state.mesh is None and self.mesh is not None:
            state = State(fields=state.fields, mesh=self.mesh, column=state.column, attrs=state._attrs_copy())
        if state.column is None and self.column is not None:
            state = State(fields=state.fields, mesh=state.mesh, column=self.column, attrs=state._attrs_copy())
        self.dynamics.check_state_compatible(state)
        return state

    def _fast_step_sizes(self, slow_dt: float | torch.Tensor) -> tuple[float, ...]:
        slow = float(torch.as_tensor(slow_dt).detach().cpu())
        if slow <= 0.0 or self.fast_dt <= 0.0:
            raise ValueError("slow_dt and fast_dt must be positive")
        sizes: list[float] = []
        remaining = slow
        while remaining > 1.0e-9 * max(1.0, slow):
            step = min(self.fast_dt, remaining)
            sizes.append(step)
            remaining -= step
        return tuple(sizes)

    def slow_tendency_parts(self, state: State) -> dict[str, torch.Tensor | None]:
        """Return one large-step slow package with PGF excluded from momentum."""

        terms = self.dynamics.tendency_terms(state)
        u_terms = terms["u_e"]
        rotation = torch.zeros_like(u_terms["rotation"]) if self.include_fast_rotation else u_terms["rotation"]
        slow_u = rotation + u_terms["bernoulli"] + u_terms["vertical"]
        tracer = terms["tracer_m"]["total"] if "tracer_m" in terms else None
        return {
            "u_e": slow_u,
            "mtheta_i": terms["mtheta_i"]["total"],
            "tracer_m": tracer,
            "pgf_initial": u_terms["pgf"],
        }

    def fast_pressure_gradient(self, state: State, theta_lagged: torch.Tensor) -> torch.Tensor:
        """Pressure-gradient acceleration with theta held fixed inside subcycles."""

        phi_full, alpha_B = self.dynamics._pgf_components_for_state(state, theta_lagged)
        return -state.grad(phi_full) - state.cell_to_edge(alpha_B) * state.grad(state.ps)

    def fast_substep(
        self,
        *,
        state: State,
        u_e: torch.Tensor,
        ps: torch.Tensor,
        theta_lagged: torch.Tensor,
        slow_u: torch.Tensor,
        dt_fast: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One forward-backward pressure/divergence update."""

        fast_state = State(
            u_e,
            state.mtheta_i,
            ps,
            state.tracer_m,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        pgf = self.fast_pressure_gradient(fast_state, theta_lagged).to(dtype=u_e.dtype, device=u_e.device)
        rotation = (
            self.dynamics._trisk_rotation(fast_state, u_e).to(dtype=u_e.dtype, device=u_e.device)
            if self.include_fast_rotation
            else torch.zeros_like(u_e)
        )
        u_next = u_e + dt_fast.to(dtype=u_e.dtype, device=u_e.device) * (slow_u + rotation + pgf)
        dp = fast_state.layer_dp()
        layer_div = self.dynamics._layer_divergence(fast_state, u_next, dp)
        dps_dt = self.dynamics._ps_tendency(layer_div)
        ps_increment = dt_fast.to(dtype=ps.dtype, device=ps.device) * dps_dt.to(dtype=ps.dtype, device=ps.device)
        ps_next = ps + ps_increment
        mass_flux = u_next * fast_state.cell_to_edge(dp).to(dtype=u_next.dtype, device=u_next.device)
        return u_next, ps_next, ps_increment, mass_flux

    def step_with_info(self, state: State, dt: float | torch.Tensor) -> tuple[State, dict[str, Any]]:
        state = self.bind_state(state)
        fast_step_sizes = self._fast_step_sizes(dt)
        n_fast = len(fast_step_sizes)
        dt_slow_u = torch.as_tensor(dt, dtype=state.u_e.dtype, device=state.u_e.device)
        dt_slow_scalar = torch.as_tensor(dt, dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)

        slow = self.slow_tendency_parts(state)
        slow_u = slow["u_e"].to(dtype=state.u_e.dtype, device=state.u_e.device)
        if not self.distribute_slow_u:
            slow_u = torch.zeros_like(slow_u)
        theta_lagged = state.theta()
        u = state.u_e
        ps = state.ps
        accumulated_ps = torch.zeros_like(state.ps)
        accumulated_mass_flux = torch.zeros_like(state.u_e)
        for sub_dt in fast_step_sizes:
            dt_fast_ps = torch.as_tensor(sub_dt, dtype=state.ps.dtype, device=state.ps.device)
            u, ps, ps_inc, mass_flux = self.fast_substep(
                state=state,
                u_e=u,
                ps=ps,
                theta_lagged=theta_lagged,
                slow_u=slow_u,
                dt_fast=dt_fast_ps,
            )
            accumulated_ps = accumulated_ps + ps_inc
            accumulated_mass_flux = accumulated_mass_flux + mass_flux.to(dtype=accumulated_mass_flux.dtype)

        if not self.distribute_slow_u:
            u = u + dt_slow_u * slow["u_e"].to(dtype=state.u_e.dtype, device=state.u_e.device)

        mtheta_new = state.mtheta_i + dt_slow_scalar * slow["mtheta_i"].to(dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)
        tracer_new = None
        if state.tracer_m is not None:
            tracer_tendency = slow["tracer_m"]
            tracer_new = state.tracer_m if tracer_tendency is None else state.tracer_m + dt_slow_scalar * tracer_tendency
        out = State(
            u,
            mtheta_new,
            ps,
            tracer_new,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        ps_mass_mismatch = (ps - state.ps) - accumulated_ps
        slow_u_increment = dt_slow_u * slow["u_e"].to(dtype=state.u_e.dtype, device=state.u_e.device)
        info = {
            "split_scheme": self.scheme_name,
            "split_slow_dt": float(torch.as_tensor(dt).detach().cpu()),
            "split_fast_dt": self.fast_dt,
            "split_n_fast_substeps": int(n_fast),
            "split_fast_dt_min": float(min(fast_step_sizes)),
            "split_fast_dt_max": float(max(fast_step_sizes)),
            "split_distribute_slow_u": self.distribute_slow_u,
            "split_include_fast_rotation": self.include_fast_rotation,
            "split_delta_u_total_rms": _float_info(_rms(out.u_e - state.u_e)),
            "split_delta_ps_total_rms": _float_info(_rms(out.ps - state.ps)),
            "split_delta_mtheta_total_rms": _float_info(_rms(out.mtheta_i - state.mtheta_i)),
            "split_slow_u_tendency_rms": _float_info(_rms(slow["u_e"])),
            "split_slow_mtheta_tendency_rms": _float_info(_rms(slow["mtheta_i"])),
            "split_initial_pgf_tendency_rms": _float_info(_rms(slow["pgf_initial"])),
            "split_slow_u_increment_rms": _float_info(_rms(slow_u_increment)),
            "split_fast_loop_u_increment_rms": _float_info(_rms((out.u_e - state.u_e) - slow_u_increment)),
            "split_accumulated_mass_flux_rms": _float_info(_rms(accumulated_mass_flux / float(n_fast))),
            "split_continuity_ps_increment_rms": _float_info(_rms(accumulated_ps)),
            "split_ps_mass_consistency_mismatch_rms": _float_info(_rms(ps_mass_mismatch)),
            "split_ps_mass_consistency_mismatch_max_abs": _float_info(ps_mass_mismatch.detach().abs().max()),
            "split_ps_mass_consistency_mismatch_over_ps_increment": _float_info(
                _rms(ps_mass_mismatch) / _rms(accumulated_ps).clamp_min(1.0e-30)
            ),
        }
        self.last_step_info = info
        return out, info

    def step(self, state: State, dt: float | torch.Tensor) -> State:
        return self.step_with_info(state, dt)[0]


@register_model(
    "dry_core_split_explicit_fastmode_fb",
    status="negative_result",
    component="throughput",
    description="Split-explicit forward-backward diagnostic variant with fast rotation.",
    default_safe=False,
)
class DryCoreSplitExplicitFastModeFB(DryCoreSplitExplicitFastMode):
    """Diagnostic forward-backward variant with rotation inside fast substeps."""

    scheme_name = "split_explicit_fastmode_fb"

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("include_fast_rotation", True)
        super().__init__(*args, **kwargs)


__all__ = ["DryCoreSplitExplicitFastMode", "DryCoreSplitExplicitFastModeFB"]
