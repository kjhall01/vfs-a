"""Experimental matrix-free pressure-IMEX stepping prototypes."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ...atmosphere import DryHydrostaticDynamics
from ...core.registry import register_model
from ...core.state import State, interface_mass


def _cell_area_for(field: torch.Tensor, state: State) -> torch.Tensor:
    """Return cell areas reshaped for a cell-centered field."""

    if state.mesh is None:
        raise ValueError("pressure IMEX solve requires State.mesh")
    shape = [1] * field.ndim
    shape[-2] = state.mesh.nCells
    shape[-1] = 1
    return state.mesh.cell_area.to(dtype=field.dtype, device=field.device).reshape(shape)


def _area_mean(field: torch.Tensor, state: State) -> torch.Tensor:
    area = _cell_area_for(field, state)
    return (field * area).sum(dim=-2, keepdim=True) / area.sum(dim=-2, keepdim=True).clamp_min(1.0e-30)


def _zero_area_mean(field: torch.Tensor, state: State) -> torch.Tensor:
    return field - _area_mean(field, state)


def _rms(field: torch.Tensor) -> torch.Tensor:
    return field.square().mean().sqrt()


def _float_info(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def _theta_full_for_column(state: State, theta_i: torch.Tensor) -> torch.Tensor:
    """Full-level placement tangent for an interface-theta-like field."""

    column = state._require_column()
    if column.phi_full_mode == "direct_u_gl2_adjoint":
        return column.theta_full_pmid_linear_exner(theta_i, state.ps)
    return 0.5 * (theta_i[..., :-1] + theta_i[..., 1:])


def _virtual_factor_interface(dynamics: DryHydrostaticDynamics, state: State) -> torch.Tensor | None:
    method = getattr(getattr(dynamics, "thermodynamics", None), "virtual_temperature_factor_interface", None)
    if not callable(method):
        return None
    return method(state)


def _active_specific_volume_full(dynamics: DryHydrostaticDynamics, state: State, theta_i: torch.Tensor) -> torch.Tensor:
    method = getattr(getattr(dynamics, "thermodynamics", None), "virtual_specific_volume_full", None)
    if callable(method):
        return method(state, theta_i)
    return state._require_column().specific_volume_full(theta_i, state.ps)


def _active_theta_full_for_alpha(dynamics: DryHydrostaticDynamics, state: State, theta_i: torch.Tensor) -> torch.Tensor:
    method = getattr(getattr(dynamics, "thermodynamics", None), "theta_full_for_column", None)
    if callable(method):
        return method(state, theta_i)
    return _theta_full_for_column(state, theta_i)


def _active_pressure_gradient_response(
    q: torch.Tensor,
    state: State,
    dynamics: DryHydrostaticDynamics,
) -> torch.Tensor:
    """Pressure-gradient tangent for ps increments using active thermodynamics.

    The original ``PressureCorrectionCG`` tangent is dry-only.  This helper
    keeps the same local algebra, but applies virtual-temperature factors when
    the wrapped dynamics owns them.  Vapor/tracer factors are held fixed.
    """

    if q.shape != state.ps.shape:
        raise ValueError(f"pressure increment q shape {tuple(q.shape)} must match state.ps {tuple(state.ps.shape)}")
    column = state._require_column()
    theta = state.theta()
    factor_i = _virtual_factor_interface(dynamics, state)
    theta_hydro = theta if factor_i is None else theta * factor_i
    ps = state.ps
    p_i = column.interface_pressure(ps)
    pi_i = column.exner_interface(ps)
    tiny = torch.finfo(ps.dtype).tiny
    delta_pi_i = column.kappa * pi_i / p_i.clamp_min(tiny) * column.B_interface * q
    d_delta_pi = delta_pi_i[..., 1:] - delta_pi_i[..., :-1]
    theta_bar = 0.5 * (theta_hydro[..., :-1] + theta_hydro[..., 1:])
    layer_dphi = theta_bar * d_delta_pi
    suffix = torch.flip(torch.cumsum(torch.flip(layer_dphi, dims=[-1]), dim=-1), dims=[-1])
    delta_phi_i = torch.cat([suffix, torch.zeros_like(suffix[..., :1])], dim=-1)

    mode = getattr(column, "phi_full_mode", "interface_mean")
    if mode == "interface_mean":
        delta_phi_full = 0.5 * (delta_phi_i[..., :-1] + delta_phi_i[..., 1:])
        delta_pi_full = 0.5 * (delta_pi_i[..., :-1] + delta_pi_i[..., 1:])
        pi_full = column.exner_full(ps)
    else:
        p_full = column.full_pressure(ps)
        pi_full = column.exner_full_pmid(ps)
        delta_pi_full = column.kappa * pi_full / p_full.clamp_min(tiny) * column.B_full * q
        delta_phi_full = delta_phi_i[..., 1:] + theta_bar * (delta_pi_i[..., 1:] - delta_pi_full)

    alpha_full = _active_specific_volume_full(dynamics, state, theta)
    p_full = column.full_pressure(ps)
    alpha_B = alpha_full * column.B_full
    delta_p_full = column.B_full * q
    delta_alpha = alpha_full * (
        delta_pi_full / pi_full.clamp_min(tiny) - delta_p_full / p_full.clamp_min(tiny)
    )
    delta_alpha_B = delta_alpha * column.B_full
    grad_q = state.grad(q)
    grad_ps = state.grad(state.ps)
    return (
        state.grad(delta_phi_full)
        + state.cell_to_edge(delta_alpha_B) * grad_ps
        + state.cell_to_edge(alpha_B) * grad_q
    )


def _theta_pressure_gradient_response(
    eta_i: torch.Tensor,
    state: State,
    dynamics: DryHydrostaticDynamics,
) -> torch.Tensor:
    """Pressure-gradient tangent for interface-theta increments."""

    if eta_i.shape != state.mtheta_i.shape:
        raise ValueError(
            f"theta increment eta_i shape {tuple(eta_i.shape)} must match state.mtheta_i {tuple(state.mtheta_i.shape)}"
        )
    column = state._require_column()
    factor_i = _virtual_factor_interface(dynamics, state)
    eta_hydro = eta_i if factor_i is None else eta_i * factor_i
    zeros = torch.zeros_like(eta_hydro)
    delta_phi_full = column.geopotential_full(eta_hydro, state.ps) - column.geopotential_full(zeros, state.ps)

    theta = state.theta()
    theta_full = _active_theta_full_for_alpha(dynamics, state, theta)
    delta_theta_full = _theta_full_for_column(state, eta_i)
    alpha_full = _active_specific_volume_full(dynamics, state, theta)
    tiny = torch.finfo(theta_full.dtype).tiny
    delta_alpha = alpha_full * delta_theta_full / theta_full.abs().clamp_min(tiny)
    delta_alpha_B = delta_alpha * column.B_full
    return state.grad(delta_phi_full) + state.cell_to_edge(delta_alpha_B) * state.grad(state.ps)


class BlockBiCGSTAB(nn.Module):
    """Fixed-iteration matrix-free BiCGSTAB for tensor blocks.

    This is intended for experimental IMEX block systems whose operator may be
    nonsymmetric. It is deliberately small: callers provide a tensor-only
    ``matvec`` and may optionally provide geometry-aware ``dot`` and
    ``project`` functions. The returned solution remains connected to the
    PyTorch graph through the unrolled iterations.
    """

    def __init__(self, max_iter: int = 30, tol: float = 1.0e-6, eps: float = 1.0e-30) -> None:
        super().__init__()
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.eps = float(eps)

    def _blocks(self, value: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        return (value,) if torch.is_tensor(value) else tuple(value)

    def _restore(self, blocks: tuple[torch.Tensor, ...], template):
        return blocks[0] if torch.is_tensor(template) else blocks

    def _zeros_like(self, blocks: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(torch.zeros_like(block) for block in blocks)

    def _add(self, a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(ai + bi for ai, bi in zip(a, b, strict=True))

    def _sub(self, a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(ai - bi for ai, bi in zip(a, b, strict=True))

    def _mul(self, scalar: torch.Tensor, a: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(scalar.to(dtype=ai.dtype, device=ai.device) * ai for ai in a)

    def _axpy(
        self,
        a: tuple[torch.Tensor, ...],
        scalar: torch.Tensor,
        b: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        return self._add(a, self._mul(scalar, b))

    def _dot_raw(self, a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return sum((ai * bi).sum() for ai, bi in zip(a, b, strict=True))

    def _safe_div(self, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        safe = torch.where(
            denominator.abs() > self.eps,
            denominator,
            torch.full_like(denominator, self.eps),
        )
        return numerator / safe

    def solve(
        self,
        matvec,
        rhs: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
        *,
        x0: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor] | None = None,
        dot=None,
        project=None,
    ):
        """Solve a matrix-free block system with fixed-iteration BiCGSTAB.

        ``matvec`` receives and returns the same block structure as ``rhs``.
        ``dot(a, b)`` may be supplied for geometry-aware inner products.
        ``project(blocks)`` may enforce constraints such as zero-mean pressure.
        """

        template = rhs
        b = self._blocks(rhs)
        if x0 is None:
            x = self._zeros_like(b)
        else:
            x = self._blocks(x0)
        if len(x) != len(b):
            raise ValueError("BlockBiCGSTAB x0 must have the same number of blocks as rhs")

        dot_fn = self._dot_raw if dot is None else dot
        project_fn = (lambda blocks: blocks) if project is None else project
        b = project_fn(b)
        x = project_fn(x)

        def apply(blocks: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            out = self._blocks(matvec(self._restore(blocks, template)))
            if len(out) != len(b):
                raise ValueError("BlockBiCGSTAB matvec returned the wrong number of blocks")
            return project_fn(out)

        r = project_fn(self._sub(b, apply(x)))
        r_hat = tuple(ri.clone() for ri in r)
        p = self._zeros_like(b)
        v = self._zeros_like(b)
        first = b[0]
        one = torch.ones((), dtype=first.dtype, device=first.device)
        rho_old = one
        alpha = one
        omega = one
        rr0 = dot_fn(r, r).clamp_min(self.eps)
        initial_residual = rr0.sqrt()

        for _ in range(self.max_iter):
            rho = dot_fn(r_hat, r)
            beta = self._safe_div(rho, rho_old) * self._safe_div(alpha, omega)
            p = self._add(r, self._mul(beta, self._sub(p, self._mul(omega, v))))
            v = apply(p)
            alpha = self._safe_div(rho, dot_fn(r_hat, v))
            s = self._sub(r, self._mul(alpha, v))
            t = apply(s)
            omega = self._safe_div(dot_fn(t, s), dot_fn(t, t))
            x = project_fn(self._axpy(self._axpy(x, alpha, p), omega, s))
            r = project_fn(self._sub(s, self._mul(omega, t)))
            rho_old = rho

        final_residual = dot_fn(r, r).clamp_min(self.eps).sqrt()
        relative_residual = final_residual / initial_residual.clamp_min(self.eps)
        info = {
            "iterations": self.max_iter,
            "initial_residual": _float_info(initial_residual),
            "final_residual": _float_info(final_residual),
            "relative_residual": _float_info(relative_residual),
            "converged": bool(relative_residual.detach().cpu() <= self.tol),
        }
        return self._restore(x, template), info


class PressureCorrectionCG(nn.Module):
    """Small fixed-iteration CG solver for a pressure-correction prototype.

    The operator is a matrix-free pressure-increment approximation:

        A q = q - damping * dt**2 * sum_k div(dp_edge_k * P_k(q))

    Here ``P_k(q)`` is a local algebraic tangent response of the active
    hydrostatic pressure-gradient term to a surface-pressure increment, and
    ``dp_edge`` is the positive layer pressure thickness at edges. The response
    is applied directly with existing mesh/column operators and does not use
    autograd/JVP linearization.
    """

    def __init__(
        self,
        max_iter: int = 30,
        tol: float = 1.0e-5,
        damping: float = 1.0,
        remove_mean: bool = True,
        eps: float = 1.0e-30,
    ) -> None:
        super().__init__()
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.damping = float(damping)
        self.remove_mean = bool(remove_mean)
        self.eps = float(eps)

    def edge_mass_weight(self, state: State) -> torch.Tensor:
        """Positive edge-layer pressure mass used by the pressure correction."""

        return state.cell_to_edge(state.layer_dp()).clamp_min(torch.finfo(state.ps.dtype).tiny)

    def pressure_tangent_components(self, q: torch.Tensor, state: State) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Local tangent pieces of the hydrostatic PGF for a ps increment.

        This is an explicit algebraic column response, not an autograd/JVP
        linearization. It treats theta as fixed and differentiates the active
        hydrostatic pressure reconstruction and sigma-slope correction with
        respect to surface pressure.
        """

        if q.shape != state.ps.shape:
            raise ValueError(f"pressure increment q shape {tuple(q.shape)} must match state.ps {tuple(state.ps.shape)}")
        column = state._require_column()
        theta = state.theta()
        ps = state.ps
        p_i = column.interface_pressure(ps)
        pi_i = column.exner_interface(ps)
        tiny = torch.finfo(ps.dtype).tiny
        delta_pi_i = column.kappa * pi_i / p_i.clamp_min(tiny) * column.B_interface * q
        d_delta_pi = delta_pi_i[..., 1:] - delta_pi_i[..., :-1]
        theta_bar = 0.5 * (theta[..., :-1] + theta[..., 1:])
        layer_dphi = theta_bar * d_delta_pi
        suffix = torch.flip(torch.cumsum(torch.flip(layer_dphi, dims=[-1]), dim=-1), dims=[-1])
        delta_phi_i = torch.cat([suffix, torch.zeros_like(suffix[..., :1])], dim=-1)

        mode = getattr(column, "phi_full_mode", "interface_mean")
        if mode == "interface_mean":
            delta_phi_full = 0.5 * (delta_phi_i[..., :-1] + delta_phi_i[..., 1:])
            delta_pi_full = 0.5 * (delta_pi_i[..., :-1] + delta_pi_i[..., 1:])
            pi_full = column.exner_full(ps)
        else:
            p_full = column.full_pressure(ps)
            pi_full = column.exner_full_pmid(ps)
            delta_pi_full = column.kappa * pi_full / p_full.clamp_min(tiny) * column.B_full * q
            delta_phi_full = delta_phi_i[..., 1:] + theta_bar * (delta_pi_i[..., 1:] - delta_pi_full)

        alpha_full = column.specific_volume_full(theta, ps)
        p_full = column.full_pressure(ps)
        alpha_B = alpha_full * column.B_full
        delta_p_full = column.B_full * q
        # alpha = R_d * theta_full * pi_full / (c_p * p_full), with theta
        # held fixed for this tangent prototype.
        delta_alpha = alpha_full * (
            delta_pi_full / pi_full.clamp_min(tiny) - delta_p_full / p_full.clamp_min(tiny)
        )
        delta_alpha_B = delta_alpha * column.B_full
        return delta_phi_full, delta_alpha_B, alpha_B

    def pressure_response_cell(self, state: State) -> torch.Tensor:
        """Local positive scale of the pressure response, for diagnostics."""

        unit = torch.ones_like(state.ps)
        delta_phi_full, _delta_alpha_B, alpha_B = self.pressure_tangent_components(unit, state)
        return (delta_phi_full + alpha_B).clamp_min(torch.finfo(state.ps.dtype).tiny)

    def pressure_response_level(self, state: State) -> torch.Tensor:
        """Area-mean positive column response, one scalar per full level."""

        response = self.pressure_response_cell(state)
        area = _cell_area_for(response, state)
        return (response * area).sum(dim=-2, keepdim=True) / area.sum(dim=-2, keepdim=True).clamp_min(self.eps)

    def pressure_gradient(self, q: torch.Tensor, state: State) -> torch.Tensor:
        """Local tangent pressure-gradient response to a surface-pressure increment."""

        delta_phi_full, delta_alpha_B, alpha_B = self.pressure_tangent_components(q, state)
        grad_q = state.grad(q)
        grad_ps = state.grad(state.ps)
        return (
            state.grad(delta_phi_full)
            + state.cell_to_edge(delta_alpha_B) * grad_ps
            + state.cell_to_edge(alpha_B) * grad_q
        )

    def dot(self, a: torch.Tensor, b: torch.Tensor, state: State) -> torch.Tensor:
        """Area-weighted cell dot product for pressure-like cell fields."""

        area = _cell_area_for(a, state)
        return (a * b * area).sum()

    def project(self, q: torch.Tensor, state: State) -> torch.Tensor:
        """Remove the area-weighted mean pressure correction when requested."""

        if not self.remove_mean:
            return q
        return _zero_area_mean(q, state)

    def default_operator_probes(self, state: State) -> tuple[torch.Tensor, ...]:
        """Deterministic pressure-like probes for operator sanity diagnostics."""

        mesh = state._require_mesh()
        shape = [1] * state.ps.ndim
        shape[-2] = mesh.nCells
        shape[-1] = 1
        xyz = mesh.cell_xyz.to(dtype=state.ps.dtype, device=state.ps.device)
        probes = [
            xyz[:, 0].reshape(shape).expand_as(state.ps),
            xyz[:, 1].reshape(shape).expand_as(state.ps),
            xyz[:, 2].reshape(shape).expand_as(state.ps),
            state.ps / state.ps.detach().abs().mean().clamp_min(1.0),
            state.theta().mean(dim=-1, keepdim=True) / state.theta().detach().abs().mean().clamp_min(1.0),
        ]
        return tuple(probes)

    def operator_diagnostics(
        self,
        state: State,
        dt: float | torch.Tensor,
        probes: tuple[torch.Tensor, ...] | list[torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Report symmetry and positive-definiteness probes for ``matvec``.

        These are sanity diagnostics for deciding whether fixed-iteration CG is
        a defensible solver for a given state. They do not prove that the IMEX
        pressure operator is scientifically validated.
        """

        raw_probes = self.default_operator_probes(state) if probes is None else tuple(probes)
        projected: list[torch.Tensor] = []
        for probe in raw_probes:
            if probe.shape != state.ps.shape:
                raise ValueError(f"operator probe shape {tuple(probe.shape)} must match state.ps {tuple(state.ps.shape)}")
            q = self.project(probe.to(dtype=state.ps.dtype, device=state.ps.device), state)
            if bool(torch.isfinite(q).all()) and bool(self.dot(q, q, state).detach().cpu() > self.eps):
                projected.append(q)

        if not projected:
            zero = torch.zeros((), dtype=state.ps.dtype, device=state.ps.device)
            return {
                "imex_operator_n_probes": 0,
                "imex_operator_symmetry_abs_max": 0.0,
                "imex_operator_symmetry_rel_max": 0.0,
                "imex_operator_qAq_min": 0.0,
                "imex_operator_qAq_max": 0.0,
                "imex_operator_qAq_rel_min": 0.0,
                "imex_operator_qAq_rel_max": 0.0,
                "imex_operator_nonpositive_count": 0,
                "imex_operator_finite": bool(torch.isfinite(zero)),
            }

        applied = [self.matvec(q, state, dt) for q in projected]
        qAq = torch.stack([self.dot(q, Aq, state) for q, Aq in zip(projected, applied, strict=True)])
        qq = torch.stack([self.dot(q, q, state).clamp_min(self.eps) for q in projected])
        qAq_rel = qAq / qq

        sym_abs_values = []
        sym_rel_values = []
        for i, q in enumerate(projected):
            Aq = applied[i]
            for j in range(i + 1, len(projected)):
                r = projected[j]
                Ar = applied[j]
                lhs = self.dot(q, Ar, state)
                rhs = self.dot(Aq, r, state)
                diff = (lhs - rhs).abs()
                scale = torch.maximum(lhs.abs(), rhs.abs()).clamp_min(self.eps)
                sym_abs_values.append(diff)
                sym_rel_values.append(diff / scale)
        if sym_abs_values:
            sym_abs = torch.stack(sym_abs_values)
            sym_rel = torch.stack(sym_rel_values)
            sym_abs_max = sym_abs.max()
            sym_rel_max = sym_rel.max()
        else:
            sym_abs_max = torch.zeros((), dtype=state.ps.dtype, device=state.ps.device)
            sym_rel_max = torch.zeros((), dtype=state.ps.dtype, device=state.ps.device)

        finite = torch.isfinite(qAq).all() & torch.isfinite(qAq_rel).all() & torch.isfinite(sym_abs_max) & torch.isfinite(sym_rel_max)
        return {
            "imex_operator_n_probes": len(projected),
            "imex_operator_symmetry_abs_max": _float_info(sym_abs_max),
            "imex_operator_symmetry_rel_max": _float_info(sym_rel_max),
            "imex_operator_qAq_min": _float_info(qAq.min()),
            "imex_operator_qAq_max": _float_info(qAq.max()),
            "imex_operator_qAq_rel_min": _float_info(qAq_rel.min()),
            "imex_operator_qAq_rel_max": _float_info(qAq_rel.max()),
            "imex_operator_nonpositive_count": int((qAq <= 0.0).detach().sum().cpu()),
            "imex_operator_finite": bool(finite.detach().cpu()),
        }

    def matvec(self, q: torch.Tensor, state: State, dt: float | torch.Tensor) -> torch.Tensor:
        """Apply ``A q`` without forming a matrix."""

        if q.shape != state.ps.shape:
            raise ValueError(f"pressure correction q shape {tuple(q.shape)} must match state.ps {tuple(state.ps.shape)}")
        dt_t = torch.as_tensor(dt, dtype=q.dtype, device=q.device)
        pressure_grad = self.pressure_gradient(q, state)
        weighted_div = state.div(self.edge_mass_weight(state).to(dtype=q.dtype, device=q.device) * pressure_grad).sum(dim=-1, keepdim=True)
        return q - float(self.damping) * dt_t.square() * weighted_div

    def solve(
        self,
        rhs: torch.Tensor,
        state: State,
        dt: float | torch.Tensor,
        x0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Solve ``A x = rhs`` with a fixed-iteration matrix-free CG loop."""

        if rhs.shape != state.ps.shape:
            raise ValueError(f"pressure correction rhs shape {tuple(rhs.shape)} must match state.ps {tuple(state.ps.shape)}")
        b = self.project(rhs, state)
        x = torch.zeros_like(b) if x0 is None else self.project(x0.to(dtype=b.dtype, device=b.device), state)
        r = self.project(b - self.matvec(x, state, dt), state)
        p = r
        rs = self.dot(r, r, state)
        initial_residual = rs.clamp_min(self.eps).sqrt()

        for _ in range(self.max_iter):
            Ap = self.matvec(p, state, dt)
            pAp = self.dot(p, Ap, state)
            denom = torch.where(pAp.abs() > self.eps, pAp, torch.full_like(pAp, self.eps))
            alpha = rs / denom
            x = self.project(x + alpha * p, state)
            r_next = self.project(r - alpha * Ap, state)
            rs_next = self.dot(r_next, r_next, state)
            beta = rs_next / rs.clamp_min(self.eps)
            p = r_next + beta * p
            r = r_next
            rs = rs_next

        final_residual = rs.clamp_min(self.eps).sqrt()
        relative_residual = final_residual / initial_residual.clamp_min(self.eps)
        info = {
            "iterations": self.max_iter,
            "initial_residual": _float_info(initial_residual),
            "final_residual": _float_info(final_residual),
            "relative_residual": _float_info(relative_residual),
            "converged": bool(relative_residual.detach().cpu() <= self.tol),
        }
        return self.project(x, state), info


@register_model("dry_core_pressure_imex_euler", status="negative_result", component="throughput", description="Experimental pressure-correction IMEX Euler prototype; not a default timestepper.", default_safe=False)
class DryCorePressureIMEXEuler(nn.Module):
    """Experimental dry-core pressure-correction Euler recipe."""

    def __init__(
        self,
        dynamics: DryHydrostaticDynamics | None = None,
        solver: PressureCorrectionCG | None = None,
        *,
        mesh: Any | None = None,
        column: Any | None = None,
        remap_mtheta: bool = True,
    ) -> None:
        super().__init__()
        self.mesh = mesh
        self.column = column
        self.dynamics = dynamics or DryHydrostaticDynamics()
        self.solver = solver or PressureCorrectionCG()
        self.remap_mtheta = bool(remap_mtheta)
        self.last_solver_info: dict[str, Any] | None = None

    def bind_state(self, state: State) -> State:
        mesh = self.mesh if self.mesh is not None else state.mesh
        column = self.column if self.column is not None else state.column
        if mesh is state.mesh and column is state.column:
            return state
        return State(fields=state.fields, mesh=mesh, column=column, attrs=state._attrs_copy())

    def operator_diagnostics(self, state: State, dt: float | torch.Tensor) -> dict[str, Any]:
        """Diagnostics for the current pressure-correction operator."""

        return self.solver.operator_diagnostics(self.bind_state(state), dt)

    def correction_budget(
        self,
        state: State,
        explicit: State,
        corrected: State,
        *,
        q: torch.Tensor,
        u_star: torch.Tensor,
        u_corr: torch.Tensor,
        mtheta_explicit: torch.Tensor,
        mtheta_remap_rms: torch.Tensor,
        dt: float | torch.Tensor,
    ) -> dict[str, Any]:
        """Detached mass/energy/correction budget for one IMEX step.

        This is reporting for the experimental recipe. It compares the old
        state, explicit predictor, and corrected state; it does not define the
        canonical dry-core energy diagnostic.
        """

        mass0 = self.dynamics.total_mass(state)
        mass_explicit = self.dynamics.total_mass(explicit)
        mass_final = self.dynamics.total_mass(corrected)
        e0 = self.dynamics.energy_terms(state)
        e_explicit = self.dynamics.energy_terms(explicit)
        e_final = self.dynamics.energy_terms(corrected)
        energy0 = e0["total"]
        energy_explicit = e_explicit["total"]
        energy_final = e_final["total"]
        energy_scale = energy0.detach().abs().clamp_min(1.0)
        mass_scale = mass0.detach().abs().clamp_min(1.0)

        dt_u = torch.as_tensor(dt, dtype=state.u_e.dtype, device=state.u_e.device)
        tiny = torch.finfo(state.u_e.dtype).tiny
        accel = u_corr / dt_u.abs().clamp_min(tiny)
        metric = self.dynamics.edge_metric(state).to(dtype=state.u_e.dtype, device=state.u_e.device)
        u_mid = u_star + 0.5 * u_corr
        correction_work_rate = (u_mid * accel * metric).sum() / float(self.dynamics.gravity)
        correction_work = correction_work_rate * dt_u
        energy_correction_delta = energy_final - energy_explicit
        work_residual = energy_correction_delta - correction_work.to(dtype=energy_correction_delta.dtype, device=energy_correction_delta.device)

        ps_predictor_rms = _rms(explicit.ps - state.ps)
        u_predictor_rms = _rms(u_star - state.u_e)
        mtheta_predictor_rms = _rms(mtheta_explicit - state.mtheta_i)
        ps_correction_rms = _rms(q)
        u_correction_rms = _rms(u_corr)

        return {
            "imex_mass_initial": _float_info(mass0),
            "imex_mass_explicit": _float_info(mass_explicit),
            "imex_mass_final": _float_info(mass_final),
            "imex_mass_explicit_delta": _float_info(mass_explicit - mass0),
            "imex_mass_final_delta": _float_info(mass_final - mass0),
            "imex_mass_correction_delta": _float_info(mass_final - mass_explicit),
            "imex_mass_explicit_delta_rel": _float_info((mass_explicit - mass0) / mass_scale),
            "imex_mass_final_delta_rel": _float_info((mass_final - mass0) / mass_scale),
            "imex_mass_correction_delta_rel": _float_info((mass_final - mass_explicit) / mass_scale),
            "imex_energy_initial": _float_info(energy0),
            "imex_energy_explicit": _float_info(energy_explicit),
            "imex_energy_final": _float_info(energy_final),
            "imex_energy_explicit_delta": _float_info(energy_explicit - energy0),
            "imex_energy_final_delta": _float_info(energy_final - energy0),
            "imex_energy_correction_delta": _float_info(energy_correction_delta),
            "imex_energy_explicit_delta_rel": _float_info((energy_explicit - energy0) / energy_scale),
            "imex_energy_final_delta_rel": _float_info((energy_final - energy0) / energy_scale),
            "imex_energy_correction_delta_rel": _float_info(energy_correction_delta / energy_scale),
            "imex_kinetic_energy_explicit_delta": _float_info(e_explicit["kinetic"] - e0["kinetic"]),
            "imex_kinetic_energy_final_delta": _float_info(e_final["kinetic"] - e0["kinetic"]),
            "imex_kinetic_energy_correction_delta": _float_info(e_final["kinetic"] - e_explicit["kinetic"]),
            "imex_potential_energy_explicit_delta": _float_info(e_explicit["potential"] - e0["potential"]),
            "imex_potential_energy_final_delta": _float_info(e_final["potential"] - e0["potential"]),
            "imex_potential_energy_correction_delta": _float_info(e_final["potential"] - e_explicit["potential"]),
            "imex_internal_energy_explicit_delta": _float_info(e_explicit["internal"] - e0["internal"]),
            "imex_internal_energy_final_delta": _float_info(e_final["internal"] - e0["internal"]),
            "imex_internal_energy_correction_delta": _float_info(e_final["internal"] - e_explicit["internal"]),
            "imex_pressure_correction_work_rate_midpoint": _float_info(correction_work_rate),
            "imex_pressure_correction_work_estimate": _float_info(correction_work),
            "imex_pressure_correction_work_residual": _float_info(work_residual),
            "imex_pressure_correction_work_residual_rel": _float_info(work_residual.abs() / energy_scale),
            "imex_ps_predictor_rms": _float_info(ps_predictor_rms),
            "imex_u_predictor_rms": _float_info(u_predictor_rms),
            "imex_mtheta_predictor_rms": _float_info(mtheta_predictor_rms),
            "imex_ps_correction_over_predictor": _float_info(ps_correction_rms / ps_predictor_rms.detach().clamp_min(1.0e-30)),
            "imex_u_correction_over_predictor": _float_info(u_correction_rms / u_predictor_rms.detach().clamp_min(1.0e-30)),
            "imex_mtheta_remap_over_predictor": _float_info(mtheta_remap_rms / mtheta_predictor_rms.detach().clamp_min(1.0e-30)),
        }

    def step_with_info(self, state: State, dt: float | torch.Tensor) -> tuple[State, dict[str, Any]]:
        """Take one experimental pressure-IMEX Euler step and return diagnostics."""

        state = self.bind_state(state)
        terms = self.dynamics.tendency_terms(state)
        rhs_explicit = self.dynamics.terms_total(terms, like=state)
        dt_ps = torch.as_tensor(dt, dtype=state.ps.dtype, device=state.ps.device)
        dt_u = torch.as_tensor(dt, dtype=state.u_e.dtype, device=state.u_e.device)
        dt_scalar = torch.as_tensor(dt, dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)

        # Pressure split: q is the actual ps increment from the old state. The
        # RHS is the continuity tendency from an explicit velocity predictor,
        # then the matrix-free solve damps the fast pressure-gradient part of
        # that pressure increment. Because ps_new = ps_old + q, this no longer
        # cancels the explicit pressure update by construction.
        u_star = state.u_e + dt_u * terms["u_e"]["total"]
        dp_edge = self.solver.edge_mass_weight(state).to(dtype=state.u_e.dtype, device=state.u_e.device)
        dps_star = -state.div(u_star * dp_edge).sum(dim=-1, keepdim=True)
        rhs_q = dt_ps * dps_star
        q, info = self.solver.solve(rhs_q, state=state, dt=dt)

        pressure_grad_q = self.solver.pressure_gradient(q, state).to(dtype=state.u_e.dtype, device=state.u_e.device)
        u_corr = -float(self.solver.damping) * dt_u * pressure_grad_q
        ps_explicit = state.ps + dt_ps * rhs_explicit.ps
        ps_new = state.ps + q
        u_new = u_star + u_corr
        mtheta_explicit = state.mtheta_i + dt_scalar * rhs_explicit.mtheta_i
        if self.remap_mtheta:
            dm_explicit = interface_mass(state.column.layer_thickness(ps_explicit))
            theta_explicit = mtheta_explicit / dm_explicit.clamp_min(torch.finfo(mtheta_explicit.dtype).tiny)
            dm_new = interface_mass(state.column.layer_thickness(ps_new))
            mtheta_new = dm_new * theta_explicit
            mtheta_remap_rms = _rms(mtheta_new - mtheta_explicit)
        else:
            mtheta_new = mtheta_explicit
            mtheta_remap_rms = torch.zeros((), dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)
        tracer_new = None if state.tracer_m is None else state.tracer_m + dt_scalar * rhs_explicit.tracer_m
        explicit = State(
            u_star,
            mtheta_explicit,
            ps_explicit,
            tracer_new,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        out = State(
            u_new,
            mtheta_new,
            ps_new,
            tracer_new,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        info = {
            **info,
            "imex_pressure_iterations": info["iterations"],
            "imex_pressure_initial_residual": info["initial_residual"],
            "imex_pressure_final_residual": info["final_residual"],
            "imex_pressure_relative_residual": info["relative_residual"],
            "imex_pressure_converged": info["converged"],
            "imex_pressure_correction_rms": _float_info(_rms(q)),
            "imex_u_correction_rms": _float_info(_rms(u_corr)),
            "imex_ps_correction_rms": _float_info(_rms(q)),
            "imex_pressure_response_min": _float_info(self.solver.pressure_response_level(state).min()),
            "imex_pressure_response_max": _float_info(self.solver.pressure_response_level(state).max()),
            "imex_edge_mass_min": _float_info(self.solver.edge_mass_weight(state).min()),
            "imex_edge_mass_max": _float_info(self.solver.edge_mass_weight(state).max()),
            "imex_mtheta_pressure_remap_rms": _float_info(mtheta_remap_rms),
            "imex_mtheta_pressure_remap": self.remap_mtheta,
            **self.correction_budget(
                state,
                explicit,
                out,
                q=q,
                u_star=u_star,
                u_corr=u_corr,
                mtheta_explicit=mtheta_explicit,
                mtheta_remap_rms=mtheta_remap_rms,
                dt=dt,
            ),
        }
        self.last_solver_info = info
        return out, info

    def step(self, state: State, dt: float | torch.Tensor) -> State:
        state_next, _info = self.step_with_info(state, dt)
        return state_next

    def forward(self, state: State, dt: float | torch.Tensor) -> State:
        return self.step(state, dt)


@register_model("dry_core_pressure_si_midpoint", status="negative_result", component="throughput", description="Experimental pressure semi-implicit midpoint prototype; not a default timestepper.", default_safe=False)
class DryCorePressureSIMidpoint(DryCorePressureIMEXEuler):
    """Experimental linearly implicit midpoint pressure-correction recipe.

    This keeps the same matrix-free hydrostatic pressure tangent as
    ``DryCorePressureIMEXEuler`` but changes the time split.  The solve uses
    the midpoint pressure operator:

        A q = q - damping * (0.5 * dt)**2 * sum_k div(dp_edge_k * P_k(q))

    with a RHS from midpoint continuity:

        b = dt * F_ps - 0.5 * dt**2 * sum_k div(dp_edge_k * F_u_k)

    It is still experimental and keeps theta frozen inside the pressure
    tangent; the scalar update is the same pressure-mass remap as the Euler
    prototype.
    """

    def operator_diagnostics(self, state: State, dt: float | torch.Tensor) -> dict[str, Any]:
        state = self.bind_state(state)
        dt_half = 0.5 * torch.as_tensor(dt, dtype=state.ps.dtype, device=state.ps.device)
        out = self.solver.operator_diagnostics(state, dt_half)
        out["imex_operator_effective_dt_factor"] = 0.5
        return out

    def step_with_info(self, state: State, dt: float | torch.Tensor) -> tuple[State, dict[str, Any]]:
        """Take one experimental linearly implicit midpoint pressure step."""

        state = self.bind_state(state)
        terms = self.dynamics.tendency_terms(state)
        rhs_explicit = self.dynamics.terms_total(terms, like=state)
        dt_ps = torch.as_tensor(dt, dtype=state.ps.dtype, device=state.ps.device)
        dt_u = torch.as_tensor(dt, dtype=state.u_e.dtype, device=state.u_e.device)
        dt_scalar = torch.as_tensor(dt, dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)

        f_u = terms["u_e"]["total"]
        f_ps = terms["ps"]["total"]
        u_star = state.u_e + dt_u * f_u
        dp_edge = self.solver.edge_mass_weight(state).to(dtype=state.u_e.dtype, device=state.u_e.device)

        # Midpoint continuity with frozen layer mass:
        #   q = dt F_ps - 0.5 dt div(dp_edge * delta_u)
        #   delta_u = dt F_u - 0.5 damping dt P(q)
        # which gives:
        #   [I - damping * (0.5 dt)^2 D M P] q
        #     = dt F_ps - 0.5 dt^2 D M F_u.
        div_f_u = state.div(dp_edge * f_u).sum(dim=-1, keepdim=True).to(dtype=state.ps.dtype, device=state.ps.device)
        rhs_q = dt_ps * f_ps - 0.5 * dt_ps.square() * div_f_u
        q, info = self.solver.solve(rhs_q, state=state, dt=0.5 * dt_ps)

        pressure_grad_q = self.solver.pressure_gradient(q, state).to(dtype=state.u_e.dtype, device=state.u_e.device)
        u_corr = -0.5 * float(self.solver.damping) * dt_u * pressure_grad_q
        ps_explicit = state.ps + dt_ps * rhs_explicit.ps
        ps_new = state.ps + q
        u_new = u_star + u_corr
        mtheta_explicit = state.mtheta_i + dt_scalar * rhs_explicit.mtheta_i
        if self.remap_mtheta:
            dm_explicit = interface_mass(state.column.layer_thickness(ps_explicit))
            theta_explicit = mtheta_explicit / dm_explicit.clamp_min(torch.finfo(mtheta_explicit.dtype).tiny)
            dm_new = interface_mass(state.column.layer_thickness(ps_new))
            mtheta_new = dm_new * theta_explicit
            mtheta_remap_rms = _rms(mtheta_new - mtheta_explicit)
        else:
            mtheta_new = mtheta_explicit
            mtheta_remap_rms = torch.zeros((), dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)
        tracer_new = None if state.tracer_m is None else state.tracer_m + dt_scalar * rhs_explicit.tracer_m
        explicit = State(
            u_star,
            mtheta_explicit,
            ps_explicit,
            tracer_new,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        out = State(
            u_new,
            mtheta_new,
            ps_new,
            tracer_new,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        info = {
            **info,
            "imex_scheme": "pressure_si_midpoint",
            "imex_pressure_iterations": info["iterations"],
            "imex_pressure_initial_residual": info["initial_residual"],
            "imex_pressure_final_residual": info["final_residual"],
            "imex_pressure_relative_residual": info["relative_residual"],
            "imex_pressure_converged": info["converged"],
            "imex_pressure_correction_rms": _float_info(_rms(q)),
            "imex_u_correction_rms": _float_info(_rms(u_corr)),
            "imex_ps_correction_rms": _float_info(_rms(q)),
            "imex_pressure_response_min": _float_info(self.solver.pressure_response_level(state).min()),
            "imex_pressure_response_max": _float_info(self.solver.pressure_response_level(state).max()),
            "imex_edge_mass_min": _float_info(self.solver.edge_mass_weight(state).min()),
            "imex_edge_mass_max": _float_info(self.solver.edge_mass_weight(state).max()),
            "imex_mtheta_pressure_remap_rms": _float_info(mtheta_remap_rms),
            "imex_mtheta_pressure_remap": self.remap_mtheta,
            **self.correction_budget(
                state,
                explicit,
                out,
                q=q,
                u_star=u_star,
                u_corr=u_corr,
                mtheta_explicit=mtheta_explicit,
                mtheta_remap_rms=mtheta_remap_rms,
                dt=dt,
            ),
        }
        self.last_solver_info = info
        return out, info


@register_model("dry_core_pressure_si_thermo_midpoint", status="negative_result", component="throughput", description="Experimental pressure/thermodynamic midpoint prototype; documented unhealthy formulation.", default_safe=False)
class DryCorePressureSIThermoMidpoint(DryCorePressureIMEXEuler):
    """Experimental midpoint recipe with half-step thermodynamic coefficients.

    This variant is still a pressure-correction prototype.  It freezes the
    matrix-free pressure operator on an explicit half-step pressure/scalar
    state, then solves the same midpoint pressure-increment equation with
    half-step edge mass and hydrostatic coefficients.  The goal is to test
    whether a more consistent pressure/scalar linearization improves the
    long-timestep behavior without changing canonical dry dynamics.
    """

    def _tangent_state(
        self,
        state: State,
        terms: dict[str, dict[str, torch.Tensor]],
        *,
        dt: float | torch.Tensor,
    ) -> State:
        """Explicit half-step state used only for frozen IMEX coefficients."""

        dt_ps = torch.as_tensor(dt, dtype=state.ps.dtype, device=state.ps.device)
        dt_u = torch.as_tensor(dt, dtype=state.u_e.dtype, device=state.u_e.device)
        dt_scalar = torch.as_tensor(dt, dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)
        tracer_half = None
        if state.tracer_m is not None:
            tracer_half = state.tracer_m + 0.5 * dt_scalar * terms["tracer_m"]["total"]
        return State(
            u_e=state.u_e + 0.5 * dt_u * terms["u_e"]["total"],
            mtheta_i=state.mtheta_i + 0.5 * dt_scalar * terms["mtheta_i"]["total"],
            ps=state.ps + 0.5 * dt_ps * terms["ps"]["total"],
            tracer_m=tracer_half,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )

    def operator_diagnostics(self, state: State, dt: float | torch.Tensor) -> dict[str, Any]:
        state = self.bind_state(state)
        terms = self.dynamics.tendency_terms(state)
        tangent = self._tangent_state(state, terms, dt=dt)
        dt_half = 0.5 * torch.as_tensor(dt, dtype=state.ps.dtype, device=state.ps.device)
        out = self.solver.operator_diagnostics(tangent, dt_half)
        out["imex_operator_effective_dt_factor"] = 0.5
        out["imex_operator_tangent_state"] = "explicit_half_step"
        return out

    def step_with_info(self, state: State, dt: float | torch.Tensor) -> tuple[State, dict[str, Any]]:
        """Take one experimental thermodynamic-coefficient midpoint step."""

        state = self.bind_state(state)
        terms = self.dynamics.tendency_terms(state)
        rhs_explicit = self.dynamics.terms_total(terms, like=state)
        tangent = self._tangent_state(state, terms, dt=dt)
        dt_ps = torch.as_tensor(dt, dtype=state.ps.dtype, device=state.ps.device)
        dt_u = torch.as_tensor(dt, dtype=state.u_e.dtype, device=state.u_e.device)
        dt_scalar = torch.as_tensor(dt, dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)

        f_u_pressure = terms["u_e"]["pgf"]
        f_u_nonpressure = terms["u_e"]["rotation"] + terms["u_e"]["bernoulli"] + terms["u_e"]["vertical"]
        f_u_explicit = f_u_nonpressure + f_u_pressure
        u_explicit = state.u_e + dt_u * f_u_explicit
        u_mid_explicit = state.u_e + 0.5 * dt_u * f_u_explicit
        dp_edge_half = self.solver.edge_mass_weight(tangent).to(dtype=state.u_e.dtype, device=state.u_e.device)

        # Midpoint continuity evaluated with explicit half-step mass and
        # velocity.  The pressure-correction part is handled by the same
        # half-dt matrix-free operator, frozen on the half-step scalar state.
        dps_mid_explicit = -tangent.div(dp_edge_half * u_mid_explicit).sum(dim=-1, keepdim=True)
        rhs_q = dt_ps * dps_mid_explicit.to(dtype=state.ps.dtype, device=state.ps.device)
        q, info = self.solver.solve(rhs_q, state=tangent, dt=0.5 * dt_ps)

        pressure_grad_q = self.solver.pressure_gradient(q, tangent).to(dtype=state.u_e.dtype, device=state.u_e.device)
        u_corr = -0.5 * float(self.solver.damping) * dt_u * pressure_grad_q
        ps_explicit = state.ps + dt_ps * rhs_explicit.ps
        ps_new = state.ps + q
        u_new = u_explicit + u_corr
        mtheta_explicit = state.mtheta_i + dt_scalar * rhs_explicit.mtheta_i
        if self.remap_mtheta:
            dm_explicit = interface_mass(state.column.layer_thickness(ps_explicit))
            theta_explicit = mtheta_explicit / dm_explicit.clamp_min(torch.finfo(mtheta_explicit.dtype).tiny)
            dm_new = interface_mass(state.column.layer_thickness(ps_new))
            mtheta_new = dm_new * theta_explicit
            mtheta_remap_rms = _rms(mtheta_new - mtheta_explicit)
        else:
            mtheta_new = mtheta_explicit
            mtheta_remap_rms = torch.zeros((), dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)
        tracer_new = None if state.tracer_m is None else state.tracer_m + dt_scalar * rhs_explicit.tracer_m
        explicit = State(
            u_explicit,
            mtheta_explicit,
            ps_explicit,
            tracer_new,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        out = State(
            u_new,
            mtheta_new,
            ps_new,
            tracer_new,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        tangent_theta_delta = _rms(tangent.theta() - state.theta())
        info = {
            **info,
            "imex_scheme": "pressure_si_thermo_midpoint",
            "imex_pressure_iterations": info["iterations"],
            "imex_pressure_initial_residual": info["initial_residual"],
            "imex_pressure_final_residual": info["final_residual"],
            "imex_pressure_relative_residual": info["relative_residual"],
            "imex_pressure_converged": info["converged"],
            "imex_pressure_correction_rms": _float_info(_rms(q)),
            "imex_u_correction_rms": _float_info(_rms(u_corr)),
            "imex_ps_correction_rms": _float_info(_rms(q)),
            "imex_pressure_response_min": _float_info(self.solver.pressure_response_level(tangent).min()),
            "imex_pressure_response_max": _float_info(self.solver.pressure_response_level(tangent).max()),
            "imex_edge_mass_min": _float_info(self.solver.edge_mass_weight(tangent).min()),
            "imex_edge_mass_max": _float_info(self.solver.edge_mass_weight(tangent).max()),
            "imex_mtheta_pressure_remap_rms": _float_info(mtheta_remap_rms),
            "imex_mtheta_pressure_remap": self.remap_mtheta,
            "imex_tangent_state": "explicit_half_step",
            "imex_tangent_ps_delta_rms": _float_info(_rms(tangent.ps - state.ps)),
            "imex_tangent_mtheta_delta_rms": _float_info(_rms(tangent.mtheta_i - state.mtheta_i)),
            "imex_tangent_theta_delta_rms": _float_info(tangent_theta_delta),
            "imex_nonpressure_split": True,
            **self.correction_budget(
                state,
                explicit,
                out,
                q=q,
                u_star=u_explicit,
                u_corr=u_corr,
                mtheta_explicit=mtheta_explicit,
                mtheta_remap_rms=mtheta_remap_rms,
                dt=dt,
            ),
        }
        self.last_solver_info = info
        return out, info


@register_model("dry_core_pressure_scalar_si_thermo_midpoint", status="negative_result", component="throughput", description="Scalar pressure/interface-theta midpoint IMEX negative JW-B result.", default_safe=False)
class DryCorePressureScalarSIThermoMidpoint(DryCorePressureSIThermoMidpoint):
    """Experimental midpoint pressure/scalar block solve.

    This is the first prototype that solves pressure and interface-theta
    increments together.  The block operator is deliberately local and frozen
    on the same explicit half-step state as ``DryCorePressureSIThermoMidpoint``:

    ``q``
        surface-pressure increment from the old state.

    ``eta_i``
        interface-theta increment from the old state.

    The pressure row sees both the active-thermodynamic pressure response
    ``P(q)`` and the hydrostatic/specific-volume response ``P_theta(eta_i)``.
    The scalar row relaxes ``eta_i`` toward the explicit scalar increment plus
    the pressure-mass remap tangent.  This is still a research prototype, not a
    derived production semi-implicit hydrostatic scheme.
    """

    def __init__(
        self,
        dynamics: DryHydrostaticDynamics | None = None,
        solver: PressureCorrectionCG | None = None,
        *,
        block_solver: BlockBiCGSTAB | None = None,
        mesh: Any | None = None,
        column: Any | None = None,
        remap_mtheta: bool = True,
        scalar_pressure_coupling: float = 1.0,
    ) -> None:
        super().__init__(
            dynamics=dynamics,
            solver=solver,
            mesh=mesh,
            column=column,
            remap_mtheta=remap_mtheta,
        )
        self.block_solver = block_solver or BlockBiCGSTAB(max_iter=self.solver.max_iter, tol=self.solver.tol)
        self.scalar_pressure_coupling = float(scalar_pressure_coupling)

    def _pressure_to_theta_tangent(self, q: torch.Tensor, state: State) -> torch.Tensor:
        """Theta change implied by conserving mtheta under a ps increment."""

        column = state._require_column()
        theta = state.theta()
        dm = interface_mass(column.layer_thickness(state.ps)).clamp_min(torch.finfo(theta.dtype).tiny)
        delta_dp = (column.B_interface[..., 1:] - column.B_interface[..., :-1]) * q
        delta_dm = interface_mass(delta_dp)
        return -theta * delta_dm / dm

    def _block_project(self, blocks: tuple[torch.Tensor, torch.Tensor], state: State) -> tuple[torch.Tensor, torch.Tensor]:
        q, eta = blocks
        return self.solver.project(q, state), eta

    def _block_dot(
        self,
        blocks_a: tuple[torch.Tensor, torch.Tensor],
        blocks_b: tuple[torch.Tensor, torch.Tensor],
        state: State,
    ) -> torch.Tensor:
        q_a, eta_a = blocks_a
        q_b, eta_b = blocks_b
        ps_scale = state.ps.detach().abs().mean().clamp_min(1.0)
        theta_scale = state.theta().detach().abs().mean().clamp_min(1.0)
        area = _cell_area_for(eta_a, state)
        q_part = self.solver.dot(q_a, q_b, state) / ps_scale.square()
        eta_part = (eta_a * eta_b * area).sum() / theta_scale.square()
        return q_part + eta_part

    def operator_diagnostics(self, state: State, dt: float | torch.Tensor) -> dict[str, Any]:
        state = self.bind_state(state)
        terms = self.dynamics.tendency_terms(state)
        tangent = self._tangent_state(state, terms, dt=dt)
        dt_half = 0.5 * torch.as_tensor(dt, dtype=state.ps.dtype, device=state.ps.device)
        out = self.solver.operator_diagnostics(tangent, dt_half)
        out["imex_operator_effective_dt_factor"] = 0.5
        out["imex_operator_tangent_state"] = "explicit_half_step"
        out["imex_operator_block_system"] = True
        out["imex_operator_block_solver"] = "bicgstab"
        return out

    def step_with_info(self, state: State, dt: float | torch.Tensor) -> tuple[State, dict[str, Any]]:
        """Take one experimental pressure/interface-theta block IMEX step."""

        state = self.bind_state(state)
        terms = self.dynamics.tendency_terms(state)
        rhs_explicit = self.dynamics.terms_total(terms, like=state)
        tangent = self._tangent_state(state, terms, dt=dt)
        dt_ps = torch.as_tensor(dt, dtype=state.ps.dtype, device=state.ps.device)
        dt_u = torch.as_tensor(dt, dtype=state.u_e.dtype, device=state.u_e.device)
        dt_scalar = torch.as_tensor(dt, dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)

        f_u_pressure = terms["u_e"]["pgf"]
        f_u_nonpressure = terms["u_e"]["rotation"] + terms["u_e"]["bernoulli"] + terms["u_e"]["vertical"]
        f_u_explicit = f_u_nonpressure + f_u_pressure
        u_explicit = state.u_e + dt_u * f_u_explicit
        u_mid_explicit = state.u_e + 0.5 * dt_u * f_u_explicit
        dp_edge_half = self.solver.edge_mass_weight(tangent).to(dtype=state.u_e.dtype, device=state.u_e.device)

        dps_mid_explicit = -tangent.div(dp_edge_half * u_mid_explicit).sum(dim=-1, keepdim=True)
        rhs_q = dt_ps * dps_mid_explicit.to(dtype=state.ps.dtype, device=state.ps.device)

        mtheta_explicit = state.mtheta_i + dt_scalar * rhs_explicit.mtheta_i
        ps_explicit = state.ps + dt_ps * rhs_explicit.ps
        dm_explicit = interface_mass(state.column.layer_thickness(ps_explicit))
        theta_explicit = mtheta_explicit / dm_explicit.clamp_min(torch.finfo(mtheta_explicit.dtype).tiny)
        rhs_eta = theta_explicit - state.theta()
        if not self.remap_mtheta:
            rhs_eta = torch.zeros_like(rhs_eta)

        dt_half = 0.5 * dt_ps

        def matvec(blocks: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            q, eta = blocks
            pressure_grad = _active_pressure_gradient_response(q, tangent, self.dynamics)
            pressure_grad = pressure_grad + _theta_pressure_gradient_response(eta, tangent, self.dynamics)
            weighted_div = tangent.div(
                dp_edge_half.to(dtype=pressure_grad.dtype, device=pressure_grad.device) * pressure_grad
            ).sum(dim=-1, keepdim=True)
            aq = q - float(self.solver.damping) * dt_half.square() * weighted_div.to(dtype=q.dtype, device=q.device)
            pressure_theta = self._pressure_to_theta_tangent(q, tangent)
            aeta = eta - float(self.scalar_pressure_coupling) * pressure_theta.to(dtype=eta.dtype, device=eta.device)
            return aq, aeta

        rhs = (rhs_q, rhs_eta)
        (q, eta), info = self.block_solver.solve(
            matvec,
            rhs,
            dot=lambda a, b: self._block_dot(a, b, tangent),
            project=lambda blocks: self._block_project(blocks, tangent),
        )

        pressure_grad_q = _active_pressure_gradient_response(q, tangent, self.dynamics)
        pressure_grad_eta = _theta_pressure_gradient_response(eta, tangent, self.dynamics)
        pressure_grad_total = (pressure_grad_q + pressure_grad_eta).to(dtype=state.u_e.dtype, device=state.u_e.device)
        u_corr = -0.5 * float(self.solver.damping) * dt_u * pressure_grad_total
        ps_new = state.ps + q
        u_new = u_explicit + u_corr
        if self.remap_mtheta:
            theta_new = state.theta() + eta.to(dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)
            dm_new = interface_mass(state.column.layer_thickness(ps_new))
            mtheta_new = dm_new * theta_new
            mtheta_remap_rms = _rms(mtheta_new - mtheta_explicit)
        else:
            mtheta_new = mtheta_explicit
            mtheta_remap_rms = torch.zeros((), dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)
        tracer_new = None if state.tracer_m is None else state.tracer_m + dt_scalar * rhs_explicit.tracer_m
        explicit = State(
            u_explicit,
            mtheta_explicit,
            ps_explicit,
            tracer_new,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        out = State(
            u_new,
            mtheta_new,
            ps_new,
            tracer_new,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        pressure_theta = self._pressure_to_theta_tangent(q, tangent)
        tangent_theta_delta = _rms(tangent.theta() - state.theta())
        info = {
            **info,
            "imex_scheme": "pressure_scalar_si_thermo_midpoint",
            "imex_block_solver": "bicgstab",
            "imex_block_iterations": info["iterations"],
            "imex_block_initial_residual": info["initial_residual"],
            "imex_block_final_residual": info["final_residual"],
            "imex_block_relative_residual": info["relative_residual"],
            "imex_block_converged": info["converged"],
            "imex_pressure_iterations": info["iterations"],
            "imex_pressure_initial_residual": info["initial_residual"],
            "imex_pressure_final_residual": info["final_residual"],
            "imex_pressure_relative_residual": info["relative_residual"],
            "imex_pressure_converged": info["converged"],
            "imex_pressure_correction_rms": _float_info(_rms(q)),
            "imex_u_correction_rms": _float_info(_rms(u_corr)),
            "imex_ps_correction_rms": _float_info(_rms(q)),
            "imex_scalar_correction_rms": _float_info(_rms(eta)),
            "imex_scalar_rhs_rms": _float_info(_rms(rhs_eta)),
            "imex_scalar_pressure_coupling_rms": _float_info(_rms(pressure_theta)),
            "imex_scalar_pressure_coupling": self.remap_mtheta,
            "imex_scalar_pressure_coupling_factor": float(self.scalar_pressure_coupling),
            "imex_pressure_response_min": _float_info(self.solver.pressure_response_level(tangent).min()),
            "imex_pressure_response_max": _float_info(self.solver.pressure_response_level(tangent).max()),
            "imex_edge_mass_min": _float_info(self.solver.edge_mass_weight(tangent).min()),
            "imex_edge_mass_max": _float_info(self.solver.edge_mass_weight(tangent).max()),
            "imex_mtheta_pressure_remap_rms": _float_info(mtheta_remap_rms),
            "imex_mtheta_pressure_remap": self.remap_mtheta,
            "imex_tangent_state": "explicit_half_step",
            "imex_tangent_ps_delta_rms": _float_info(_rms(tangent.ps - state.ps)),
            "imex_tangent_mtheta_delta_rms": _float_info(_rms(tangent.mtheta_i - state.mtheta_i)),
            "imex_tangent_theta_delta_rms": _float_info(tangent_theta_delta),
            "imex_nonpressure_split": True,
            "imex_scalar_block": True,
            **self.correction_budget(
                state,
                explicit,
                out,
                q=q,
                u_star=u_explicit,
                u_corr=u_corr,
                mtheta_explicit=mtheta_explicit,
                mtheta_remap_rms=mtheta_remap_rms,
                dt=dt,
            ),
        }
        self.last_solver_info = info
        return out, info


@register_model("dry_core_pressure_scalar_si_pressure_only_centered", status="negative_result", component="throughput", description="Pressure-only scalar IMEX diagnostic control; mechanically stable but distorted JW-B.", default_safe=False)
class DryCorePressureScalarSIPressureOnlyCentered(DryCorePressureSIThermoMidpoint):
    """Pressure/velocity-only centered correction with lagged thermodynamics.

    This candidate deliberately removes theta/mtheta from the implicit unknown
    set.  It freezes thermodynamic coefficients on an explicit lagged state,
    solves only for the pressure increment, corrects velocity, and leaves
    mtheta on the explicit conservative tendency path by default.
    """

    scheme_name = "pressure_scalar_si_pressure_only_centered"
    pressure_update_mode = "pressure_increment"

    def __init__(
        self,
        dynamics: DryHydrostaticDynamics | None = None,
        solver: PressureCorrectionCG | None = None,
        *,
        mesh: Any | None = None,
        column: Any | None = None,
        remap_mtheta: bool = False,
        centering_alpha: float = 0.5,
    ) -> None:
        super().__init__(
            dynamics=dynamics,
            solver=solver,
            mesh=mesh,
            column=column,
            remap_mtheta=remap_mtheta,
        )
        self.centering_alpha = float(centering_alpha)
        if not (0.0 < self.centering_alpha <= 1.0):
            raise ValueError("centering_alpha must be in (0, 1]")

    def _centered_tangent_state(
        self,
        state: State,
        terms: dict[str, dict[str, torch.Tensor]],
        *,
        dt: float | torch.Tensor,
    ) -> State:
        alpha = float(self.centering_alpha)
        dt_ps = torch.as_tensor(dt, dtype=state.ps.dtype, device=state.ps.device)
        dt_u = torch.as_tensor(dt, dtype=state.u_e.dtype, device=state.u_e.device)
        dt_scalar = torch.as_tensor(dt, dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)
        tracer_lag = None
        if state.tracer_m is not None:
            tracer_lag = state.tracer_m + alpha * dt_scalar * terms["tracer_m"]["total"]
        return State(
            u_e=state.u_e + alpha * dt_u * terms["u_e"]["total"],
            mtheta_i=state.mtheta_i + alpha * dt_scalar * terms["mtheta_i"]["total"],
            ps=state.ps + alpha * dt_ps * terms["ps"]["total"],
            tracer_m=tracer_lag,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )

    def operator_diagnostics(self, state: State, dt: float | torch.Tensor) -> dict[str, Any]:
        state = self.bind_state(state)
        terms = self.dynamics.tendency_terms(state)
        tangent = self._centered_tangent_state(state, terms, dt=dt)
        dt_eff = self.centering_alpha * torch.as_tensor(dt, dtype=state.ps.dtype, device=state.ps.device)
        out = self.solver.operator_diagnostics(tangent, dt_eff)
        out["imex_operator_effective_dt_factor"] = float(self.centering_alpha)
        out["imex_operator_tangent_state"] = "explicit_lagged"
        out["imex_operator_pressure_only"] = True
        return out

    def _pressure_update(
        self,
        *,
        state: State,
        tangent: State,
        q: torch.Tensor,
        u_new: torch.Tensor,
        dt: torch.Tensor,
        dp_edge_lag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return state.ps + q, torch.zeros((), dtype=state.ps.dtype, device=state.ps.device)

    def step_with_info(self, state: State, dt: float | torch.Tensor) -> tuple[State, dict[str, Any]]:
        """Take one pressure-only lagged-thermodynamic centered IMEX step."""

        state = self.bind_state(state)
        terms = self.dynamics.tendency_terms(state)
        rhs_explicit = self.dynamics.terms_total(terms, like=state)
        tangent = self._centered_tangent_state(state, terms, dt=dt)
        alpha = float(self.centering_alpha)
        dt_ps = torch.as_tensor(dt, dtype=state.ps.dtype, device=state.ps.device)
        dt_u = torch.as_tensor(dt, dtype=state.u_e.dtype, device=state.u_e.device)
        dt_scalar = torch.as_tensor(dt, dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)

        f_u = terms["u_e"]["total"]
        u_explicit = state.u_e + dt_u * f_u
        u_center_explicit = state.u_e + alpha * dt_u * f_u
        dp_edge_lag = self.solver.edge_mass_weight(tangent).to(dtype=state.u_e.dtype, device=state.u_e.device)
        dps_center_explicit = -tangent.div(dp_edge_lag * u_center_explicit).sum(dim=-1, keepdim=True)
        rhs_q = dt_ps * dps_center_explicit.to(dtype=state.ps.dtype, device=state.ps.device)
        q, info = self.solver.solve(rhs_q, state=tangent, dt=alpha * dt_ps)

        pressure_grad_q = self.solver.pressure_gradient(q, tangent).to(dtype=state.u_e.dtype, device=state.u_e.device)
        u_corr = -alpha * float(self.solver.damping) * dt_u * pressure_grad_q
        u_new = u_explicit + u_corr
        ps_explicit = state.ps + dt_ps * rhs_explicit.ps
        ps_new, ps_projection_delta_rms = self._pressure_update(
            state=state,
            tangent=tangent,
            q=q,
            u_new=u_new,
            dt=dt_ps,
            dp_edge_lag=dp_edge_lag,
        )
        dps_corrected_flux = -tangent.div(dp_edge_lag.to(dtype=u_new.dtype, device=u_new.device) * u_new).sum(
            dim=-1,
            keepdim=True,
        )
        continuity_increment = dt_ps * dps_corrected_flux.to(dtype=state.ps.dtype, device=state.ps.device)
        ps_mass_mismatch = (ps_new - state.ps) - continuity_increment
        mtheta_explicit = state.mtheta_i + dt_scalar * rhs_explicit.mtheta_i
        if self.remap_mtheta:
            dm_explicit = interface_mass(state.column.layer_thickness(ps_explicit))
            theta_explicit = mtheta_explicit / dm_explicit.clamp_min(torch.finfo(mtheta_explicit.dtype).tiny)
            dm_new = interface_mass(state.column.layer_thickness(ps_new))
            mtheta_new = dm_new * theta_explicit
            mtheta_remap_rms = _rms(mtheta_new - mtheta_explicit)
        else:
            mtheta_new = mtheta_explicit
            mtheta_remap_rms = torch.zeros((), dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)
        tracer_new = None if state.tracer_m is None else state.tracer_m + dt_scalar * rhs_explicit.tracer_m
        explicit = State(
            u_explicit,
            mtheta_explicit,
            ps_explicit,
            tracer_new,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        out = State(
            u_new,
            mtheta_new,
            ps_new,
            tracer_new,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        tangent_theta_delta = _rms(tangent.theta() - state.theta())
        continuity_rms = _rms(continuity_increment)
        u_corr_rms = _rms(u_corr)
        info = {
            **info,
            "imex_scheme": self.scheme_name,
            "imex_pressure_iterations": info["iterations"],
            "imex_pressure_initial_residual": info["initial_residual"],
            "imex_pressure_final_residual": info["final_residual"],
            "imex_pressure_relative_residual": info["relative_residual"],
            "imex_pressure_converged": info["converged"],
            "imex_pressure_correction_rms": _float_info(_rms(q)),
            "imex_u_correction_rms": _float_info(u_corr_rms),
            "imex_ps_correction_rms": _float_info(_rms(ps_new - state.ps)),
            "imex_raw_q_correction_rms": _float_info(_rms(q)),
            "imex_pressure_response_min": _float_info(self.solver.pressure_response_level(tangent).min()),
            "imex_pressure_response_max": _float_info(self.solver.pressure_response_level(tangent).max()),
            "imex_edge_mass_min": _float_info(dp_edge_lag.min()),
            "imex_edge_mass_max": _float_info(dp_edge_lag.max()),
            "imex_mtheta_pressure_remap_rms": _float_info(mtheta_remap_rms),
            "imex_mtheta_pressure_remap": self.remap_mtheta,
            "imex_tangent_state": "explicit_lagged",
            "imex_tangent_ps_delta_rms": _float_info(_rms(tangent.ps - state.ps)),
            "imex_tangent_mtheta_delta_rms": _float_info(_rms(tangent.mtheta_i - state.mtheta_i)),
            "imex_tangent_theta_delta_rms": _float_info(tangent_theta_delta),
            "imex_nonpressure_split": True,
            "imex_pressure_only": True,
            "imex_lagged_thermo": True,
            "imex_centering_alpha": float(self.centering_alpha),
            "imex_pressure_update_mode": self.pressure_update_mode,
            "imex_continuity_ps_increment_rms": _float_info(continuity_rms),
            "imex_continuity_ps_increment_max_abs": _float_info(continuity_increment.detach().abs().max()),
            "imex_ps_mass_consistency_mismatch_rms": _float_info(_rms(ps_mass_mismatch)),
            "imex_ps_mass_consistency_mismatch_max_abs": _float_info(ps_mass_mismatch.detach().abs().max()),
            "imex_ps_mass_consistency_mismatch_over_ps_correction": _float_info(
                _rms(ps_mass_mismatch) / _rms(ps_new - state.ps).detach().clamp_min(1.0e-30)
            ),
            "imex_continuity_ps_increment_over_u_correction": _float_info(
                continuity_rms / u_corr_rms.detach().clamp_min(1.0e-30)
            ),
            "imex_ps_projection_delta_rms": _float_info(ps_projection_delta_rms),
            **self.correction_budget(
                state,
                explicit,
                out,
                q=ps_new - state.ps,
                u_star=u_explicit,
                u_corr=u_corr,
                mtheta_explicit=mtheta_explicit,
                mtheta_remap_rms=mtheta_remap_rms,
                dt=dt,
            ),
        }
        self.last_solver_info = info
        return out, info


@register_model("dry_core_pressure_scalar_si_pressure_only_lagged_thermo", status="negative_result", component="throughput", description="Pressure-only scalar IMEX diagnostic control with lagged thermodynamics.", default_safe=False)
class DryCorePressureScalarSIPressureOnlyLaggedThermo(DryCorePressureScalarSIPressureOnlyCentered):
    """Named alpha=0.5 pressure-only lagged-thermodynamic candidate."""

    scheme_name = "pressure_scalar_si_pressure_only_lagged_thermo"

    def __init__(
        self,
        dynamics: DryHydrostaticDynamics | None = None,
        solver: PressureCorrectionCG | None = None,
        *,
        mesh: Any | None = None,
        column: Any | None = None,
        remap_mtheta: bool = False,
    ) -> None:
        super().__init__(
            dynamics=dynamics,
            solver=solver,
            mesh=mesh,
            column=column,
            remap_mtheta=remap_mtheta,
            centering_alpha=0.5,
        )


@register_model("dry_core_pressure_scalar_si_projection_probe", status="negative_result", component="throughput", description="Projection-style pressure IMEX probe; diagnostic only.", default_safe=False)
class DryCorePressureScalarSIProjectionProbe(DryCorePressureScalarSIPressureOnlyCentered):
    """Diagnostic pressure-gradient projection probe.

    The pressure increment is used to correct velocity, but final surface
    pressure is recomputed from the same frozen-mass discrete continuity used
    by the pressure solve rather than copied directly from ``q``.
    """

    scheme_name = "pressure_scalar_si_projection_probe"
    pressure_update_mode = "continuity_projection"

    def _pressure_update(
        self,
        *,
        state: State,
        tangent: State,
        q: torch.Tensor,
        u_new: torch.Tensor,
        dt: torch.Tensor,
        dp_edge_lag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dps_projected = -tangent.div(dp_edge_lag.to(dtype=u_new.dtype, device=u_new.device) * u_new).sum(
            dim=-1,
            keepdim=True,
        )
        ps_projected = state.ps + dt * dps_projected.to(dtype=state.ps.dtype, device=state.ps.device)
        ps_direct = state.ps + q
        return ps_projected, _rms(ps_projected - ps_direct)


@register_model("dry_core_pressure_projection_si_conservative", status="negative_result", component="throughput", description="Conservative pressure-projection IMEX branch; stable but phase/amplitude-distorting.", default_safe=False)
class DryCorePressureProjectionSIConservative(DryCorePressureScalarSIPressureOnlyCentered):
    """Conservative pressure-projection candidate with explicit thermodynamics.

    This is the production-shaped successor to the diagnostic projection probe:
    solve a scalar pressure-like correction, correct edge velocity through the
    existing gradient response, then derive final surface pressure from the
    corrected frozen-mass continuity update.  The pressure solve variable is
    therefore not independently copied into ``ps``.
    """

    scheme_name = "pressure_projection_si_conservative"
    pressure_update_mode = "conservative_continuity"

    def __init__(
        self,
        dynamics: DryHydrostaticDynamics | None = None,
        solver: PressureCorrectionCG | None = None,
        *,
        mesh: Any | None = None,
        column: Any | None = None,
        remap_mtheta: bool = False,
        centering_alpha: float = 1.0,
    ) -> None:
        super().__init__(
            dynamics=dynamics,
            solver=solver,
            mesh=mesh,
            column=column,
            remap_mtheta=remap_mtheta,
            centering_alpha=centering_alpha,
        )

    def _continuity_increment(
        self,
        *,
        state: State,
        tangent: State,
        dp_edge_lag: torch.Tensor,
        u_flux: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        dps_dt = -tangent.div(dp_edge_lag.to(dtype=u_flux.dtype, device=u_flux.device) * u_flux).sum(
            dim=-1,
            keepdim=True,
        )
        return dt * dps_dt.to(dtype=state.ps.dtype, device=state.ps.device)

    def _mtheta_from_corrected_flux(
        self,
        *,
        state: State,
        tangent: State,
        theta_i: torch.Tensor,
        u_flux: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        dp = tangent.layer_dp()
        layer_div = self.dynamics._layer_divergence(tangent, u_flux, dp)
        dps_dt = self.dynamics._ps_tendency(layer_div)
        mu = self.dynamics._vertical_mass_flux(tangent, layer_div, dps_dt)
        omega = self.dynamics._omega_full(mu)
        horizontal = self.dynamics._mtheta_tendency(tangent, u_flux, theta_i, dp)
        vertical = self.dynamics._mtheta_vertical(theta_i, omega)
        coord_work = self.dynamics._canonical_mtheta_coordinate_work(tangent, theta_i, omega)
        return state.mtheta_i + dt * (horizontal + vertical + coord_work)

    def operator_diagnostics(self, state: State, dt: float | torch.Tensor) -> dict[str, Any]:
        state = self.bind_state(state)
        terms = self.dynamics.tendency_terms(state)
        tangent = self._centered_tangent_state(state, terms, dt=dt)
        dt_eff = torch.as_tensor(dt, dtype=state.ps.dtype, device=state.ps.device)
        out = self.solver.operator_diagnostics(tangent, dt_eff)
        out["imex_operator_effective_dt_factor"] = 1.0
        out["imex_operator_tangent_state"] = "explicit_lagged"
        out["imex_operator_pressure_only"] = True
        out["imex_operator_conservative_projection"] = True
        return out

    def step_with_info(self, state: State, dt: float | torch.Tensor) -> tuple[State, dict[str, Any]]:
        """Take one conservative pressure-projection IMEX step."""

        state = self.bind_state(state)
        terms = self.dynamics.tendency_terms(state)
        rhs_explicit = self.dynamics.terms_total(terms, like=state)
        tangent = self._centered_tangent_state(state, terms, dt=dt)
        dt_ps = torch.as_tensor(dt, dtype=state.ps.dtype, device=state.ps.device)
        dt_u = torch.as_tensor(dt, dtype=state.u_e.dtype, device=state.u_e.device)
        dt_scalar = torch.as_tensor(dt, dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)

        f_u = terms["u_e"]["total"]
        u_explicit = state.u_e + dt_u * f_u
        dp_edge_lag = self.solver.edge_mass_weight(tangent).to(dtype=state.u_e.dtype, device=state.u_e.device)
        dps_explicit_flux = -tangent.div(dp_edge_lag * u_explicit).sum(dim=-1, keepdim=True)
        rhs_q = dt_ps * dps_explicit_flux.to(dtype=state.ps.dtype, device=state.ps.device)
        q, info = self.solver.solve(rhs_q, state=tangent, dt=dt_ps)

        pressure_grad_q = self.solver.pressure_gradient(q, tangent).to(dtype=state.u_e.dtype, device=state.u_e.device)
        u_corr = -float(self.solver.damping) * dt_u * pressure_grad_q
        u_new = u_explicit + u_corr
        dps_corrected_flux = -tangent.div(dp_edge_lag.to(dtype=u_new.dtype, device=u_new.device) * u_new).sum(
            dim=-1,
            keepdim=True,
        )
        continuity_increment = dt_ps * dps_corrected_flux.to(dtype=state.ps.dtype, device=state.ps.device)
        ps_new = state.ps + continuity_increment
        ps_direct = state.ps + q
        ps_projection_delta_rms = _rms(ps_new - ps_direct)
        ps_mass_mismatch = (ps_new - state.ps) - continuity_increment
        ps_explicit = state.ps + dt_ps * rhs_explicit.ps

        mtheta_explicit = state.mtheta_i + dt_scalar * rhs_explicit.mtheta_i
        if self.remap_mtheta:
            dm_explicit = interface_mass(state.column.layer_thickness(ps_explicit))
            theta_explicit = mtheta_explicit / dm_explicit.clamp_min(torch.finfo(mtheta_explicit.dtype).tiny)
            dm_new = interface_mass(state.column.layer_thickness(ps_new))
            mtheta_new = dm_new * theta_explicit
            mtheta_remap_rms = _rms(mtheta_new - mtheta_explicit)
        else:
            mtheta_new = mtheta_explicit
            mtheta_remap_rms = torch.zeros((), dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)
        tracer_new = None if state.tracer_m is None else state.tracer_m + dt_scalar * rhs_explicit.tracer_m
        explicit = State(
            u_explicit,
            mtheta_explicit,
            ps_explicit,
            tracer_new,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        out = State(
            u_new,
            mtheta_new,
            ps_new,
            tracer_new,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        tangent_theta_delta = _rms(tangent.theta() - state.theta())
        continuity_rms = _rms(continuity_increment)
        u_corr_rms = _rms(u_corr)
        info = {
            **info,
            "imex_scheme": self.scheme_name,
            "imex_pressure_iterations": info["iterations"],
            "imex_pressure_initial_residual": info["initial_residual"],
            "imex_pressure_final_residual": info["final_residual"],
            "imex_pressure_relative_residual": info["relative_residual"],
            "imex_pressure_converged": info["converged"],
            "imex_pressure_correction_rms": _float_info(_rms(q)),
            "imex_u_correction_rms": _float_info(u_corr_rms),
            "imex_ps_correction_rms": _float_info(_rms(ps_new - state.ps)),
            "imex_raw_q_correction_rms": _float_info(_rms(q)),
            "imex_pressure_response_min": _float_info(self.solver.pressure_response_level(tangent).min()),
            "imex_pressure_response_max": _float_info(self.solver.pressure_response_level(tangent).max()),
            "imex_edge_mass_min": _float_info(dp_edge_lag.min()),
            "imex_edge_mass_max": _float_info(dp_edge_lag.max()),
            "imex_mtheta_pressure_remap_rms": _float_info(mtheta_remap_rms),
            "imex_mtheta_pressure_remap": self.remap_mtheta,
            "imex_tangent_state": "explicit_lagged",
            "imex_tangent_ps_delta_rms": _float_info(_rms(tangent.ps - state.ps)),
            "imex_tangent_mtheta_delta_rms": _float_info(_rms(tangent.mtheta_i - state.mtheta_i)),
            "imex_tangent_theta_delta_rms": _float_info(tangent_theta_delta),
            "imex_nonpressure_split": True,
            "imex_pressure_only": True,
            "imex_lagged_thermo": True,
            "imex_centering_alpha": float(self.centering_alpha),
            "imex_pressure_update_mode": self.pressure_update_mode,
            "imex_conservative_projection": True,
            "imex_continuity_ps_increment_rms": _float_info(continuity_rms),
            "imex_continuity_ps_increment_max_abs": _float_info(continuity_increment.detach().abs().max()),
            "imex_ps_mass_consistency_mismatch_rms": _float_info(_rms(ps_mass_mismatch)),
            "imex_ps_mass_consistency_mismatch_max_abs": _float_info(ps_mass_mismatch.detach().abs().max()),
            "imex_ps_mass_consistency_mismatch_over_ps_correction": _float_info(
                _rms(ps_mass_mismatch) / _rms(ps_new - state.ps).detach().clamp_min(1.0e-30)
            ),
            "imex_continuity_ps_increment_over_u_correction": _float_info(
                continuity_rms / u_corr_rms.detach().clamp_min(1.0e-30)
            ),
            "imex_ps_projection_delta_rms": _float_info(ps_projection_delta_rms),
            **self.correction_budget(
                state,
                explicit,
                out,
                q=ps_new - state.ps,
                u_star=u_explicit,
                u_corr=u_corr,
                mtheta_explicit=mtheta_explicit,
                mtheta_remap_rms=mtheta_remap_rms,
                dt=dt,
            ),
        }
        self.last_solver_info = info
        return out, info


@register_model("dry_core_pressure_projection_si_conservative_midpoint_flux", status="negative_result", component="throughput", description="Midpoint-flux conservative pressure projection; improved short JW-B but failed 15-day gate.", default_safe=False)
class DryCorePressureProjectionSIMidpointFlux(DryCorePressureProjectionSIConservative):
    """Conservative projection using midpoint velocity in continuity."""

    scheme_name = "pressure_projection_si_conservative_midpoint_flux"
    pressure_update_mode = "conservative_midpoint_flux"

    def __init__(
        self,
        dynamics: DryHydrostaticDynamics | None = None,
        solver: PressureCorrectionCG | None = None,
        *,
        mesh: Any | None = None,
        column: Any | None = None,
        remap_mtheta: bool = False,
        centering_alpha: float = 0.5,
    ) -> None:
        super().__init__(
            dynamics=dynamics,
            solver=solver,
            mesh=mesh,
            column=column,
            remap_mtheta=remap_mtheta,
            centering_alpha=centering_alpha,
        )

    def operator_diagnostics(self, state: State, dt: float | torch.Tensor) -> dict[str, Any]:
        state = self.bind_state(state)
        terms = self.dynamics.tendency_terms(state)
        tangent = self._centered_tangent_state(state, terms, dt=dt)
        dt_eff = 0.5 * torch.as_tensor(dt, dtype=state.ps.dtype, device=state.ps.device)
        out = self.solver.operator_diagnostics(tangent, dt_eff)
        out["imex_operator_effective_dt_factor"] = 0.5
        out["imex_operator_tangent_state"] = "explicit_midpoint_lagged"
        out["imex_operator_pressure_only"] = True
        out["imex_operator_conservative_projection"] = True
        return out

    def step_with_info(self, state: State, dt: float | torch.Tensor) -> tuple[State, dict[str, Any]]:
        state = self.bind_state(state)
        terms = self.dynamics.tendency_terms(state)
        rhs_explicit = self.dynamics.terms_total(terms, like=state)
        tangent = self._centered_tangent_state(state, terms, dt=dt)
        dt_ps = torch.as_tensor(dt, dtype=state.ps.dtype, device=state.ps.device)
        dt_u = torch.as_tensor(dt, dtype=state.u_e.dtype, device=state.u_e.device)
        dt_scalar = torch.as_tensor(dt, dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)

        f_u = terms["u_e"]["total"]
        u_explicit = state.u_e + dt_u * f_u
        u_mid_explicit = 0.5 * (state.u_e + u_explicit)
        dp_edge_lag = self.solver.edge_mass_weight(tangent).to(dtype=state.u_e.dtype, device=state.u_e.device)
        rhs_q = self._continuity_increment(
            state=state,
            tangent=tangent,
            dp_edge_lag=dp_edge_lag,
            u_flux=u_mid_explicit,
            dt=dt_ps,
        )
        q, info = self.solver.solve(rhs_q, state=tangent, dt=0.5 * dt_ps)

        pressure_grad_q = self.solver.pressure_gradient(q, tangent).to(dtype=state.u_e.dtype, device=state.u_e.device)
        u_corr = -float(self.solver.damping) * dt_u * pressure_grad_q
        u_new = u_explicit + u_corr
        u_flux = 0.5 * (state.u_e + u_new)
        continuity_increment = self._continuity_increment(
            state=state,
            tangent=tangent,
            dp_edge_lag=dp_edge_lag,
            u_flux=u_flux,
            dt=dt_ps,
        )
        ps_new = state.ps + continuity_increment
        ps_projection_delta_rms = _rms(ps_new - (state.ps + q))
        ps_mass_mismatch = (ps_new - state.ps) - continuity_increment
        ps_explicit = state.ps + dt_ps * rhs_explicit.ps

        mtheta_explicit = state.mtheta_i + dt_scalar * rhs_explicit.mtheta_i
        if self.remap_mtheta:
            dm_explicit = interface_mass(state.column.layer_thickness(ps_explicit))
            theta_explicit = mtheta_explicit / dm_explicit.clamp_min(torch.finfo(mtheta_explicit.dtype).tiny)
            dm_new = interface_mass(state.column.layer_thickness(ps_new))
            mtheta_new = dm_new * theta_explicit
            mtheta_remap_rms = _rms(mtheta_new - mtheta_explicit)
        else:
            mtheta_new = mtheta_explicit
            mtheta_remap_rms = torch.zeros((), dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)
        tracer_new = None if state.tracer_m is None else state.tracer_m + dt_scalar * rhs_explicit.tracer_m
        explicit = State(
            u_explicit,
            mtheta_explicit,
            ps_explicit,
            tracer_new,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        out = State(
            u_new,
            mtheta_new,
            ps_new,
            tracer_new,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        tangent_theta_delta = _rms(tangent.theta() - state.theta())
        continuity_rms = _rms(continuity_increment)
        u_corr_rms = _rms(u_corr)
        info = {
            **info,
            "imex_scheme": self.scheme_name,
            "imex_pressure_iterations": info["iterations"],
            "imex_pressure_initial_residual": info["initial_residual"],
            "imex_pressure_final_residual": info["final_residual"],
            "imex_pressure_relative_residual": info["relative_residual"],
            "imex_pressure_converged": info["converged"],
            "imex_pressure_correction_rms": _float_info(_rms(q)),
            "imex_u_correction_rms": _float_info(u_corr_rms),
            "imex_ps_correction_rms": _float_info(_rms(ps_new - state.ps)),
            "imex_raw_q_correction_rms": _float_info(_rms(q)),
            "imex_pressure_response_min": _float_info(self.solver.pressure_response_level(tangent).min()),
            "imex_pressure_response_max": _float_info(self.solver.pressure_response_level(tangent).max()),
            "imex_edge_mass_min": _float_info(dp_edge_lag.min()),
            "imex_edge_mass_max": _float_info(dp_edge_lag.max()),
            "imex_mtheta_pressure_remap_rms": _float_info(mtheta_remap_rms),
            "imex_mtheta_pressure_remap": self.remap_mtheta,
            "imex_tangent_state": "explicit_midpoint_lagged",
            "imex_tangent_ps_delta_rms": _float_info(_rms(tangent.ps - state.ps)),
            "imex_tangent_mtheta_delta_rms": _float_info(_rms(tangent.mtheta_i - state.mtheta_i)),
            "imex_tangent_theta_delta_rms": _float_info(tangent_theta_delta),
            "imex_nonpressure_split": True,
            "imex_pressure_only": True,
            "imex_lagged_thermo": True,
            "imex_centering_alpha": float(self.centering_alpha),
            "imex_pressure_update_mode": self.pressure_update_mode,
            "imex_conservative_projection": True,
            "imex_flux_velocity_mode": "midpoint_old_new",
            "imex_continuity_ps_increment_rms": _float_info(continuity_rms),
            "imex_continuity_ps_increment_max_abs": _float_info(continuity_increment.detach().abs().max()),
            "imex_ps_mass_consistency_mismatch_rms": _float_info(_rms(ps_mass_mismatch)),
            "imex_ps_mass_consistency_mismatch_max_abs": _float_info(ps_mass_mismatch.detach().abs().max()),
            "imex_ps_mass_consistency_mismatch_over_ps_correction": _float_info(
                _rms(ps_mass_mismatch) / _rms(ps_new - state.ps).detach().clamp_min(1.0e-30)
            ),
            "imex_continuity_ps_increment_over_u_correction": _float_info(
                continuity_rms / u_corr_rms.detach().clamp_min(1.0e-30)
            ),
            "imex_ps_projection_delta_rms": _float_info(ps_projection_delta_rms),
            **self.correction_budget(
                state,
                explicit,
                out,
                q=ps_new - state.ps,
                u_star=u_explicit,
                u_corr=u_corr,
                mtheta_explicit=mtheta_explicit,
                mtheta_remap_rms=mtheta_remap_rms,
                dt=dt,
            ),
        }
        self.last_solver_info = info
        return out, info


@register_model("dry_core_pressure_projection_si_conservative_picard1", status="negative_result", component="throughput", description="One-Picard conservative pressure projection variant; neutral diagnostic result.", default_safe=False)
class DryCorePressureProjectionSIPicard1(DryCorePressureProjectionSIConservative):
    """Conservative projection with one residual Picard refinement."""

    scheme_name = "pressure_projection_si_conservative_picard1"
    pressure_update_mode = "conservative_picard1"

    def step_with_info(self, state: State, dt: float | torch.Tensor) -> tuple[State, dict[str, Any]]:
        state = self.bind_state(state)
        terms = self.dynamics.tendency_terms(state)
        rhs_explicit = self.dynamics.terms_total(terms, like=state)
        tangent1 = self._centered_tangent_state(state, terms, dt=dt)
        dt_ps = torch.as_tensor(dt, dtype=state.ps.dtype, device=state.ps.device)
        dt_u = torch.as_tensor(dt, dtype=state.u_e.dtype, device=state.u_e.device)
        dt_scalar = torch.as_tensor(dt, dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)

        f_u = terms["u_e"]["total"]
        u_explicit = state.u_e + dt_u * f_u
        dp_edge1 = self.solver.edge_mass_weight(tangent1).to(dtype=state.u_e.dtype, device=state.u_e.device)
        rhs_q1 = self._continuity_increment(
            state=state,
            tangent=tangent1,
            dp_edge_lag=dp_edge1,
            u_flux=u_explicit,
            dt=dt_ps,
        )
        q1, info1 = self.solver.solve(rhs_q1, state=tangent1, dt=dt_ps)
        grad1 = self.solver.pressure_gradient(q1, tangent1).to(dtype=state.u_e.dtype, device=state.u_e.device)
        u_corr1 = -float(self.solver.damping) * dt_u * grad1
        u_first = u_explicit + u_corr1
        cont1 = self._continuity_increment(
            state=state,
            tangent=tangent1,
            dp_edge_lag=dp_edge1,
            u_flux=u_first,
            dt=dt_ps,
        )
        ps_first = state.ps + cont1

        mtheta_explicit = state.mtheta_i + dt_scalar * rhs_explicit.mtheta_i
        tracer_new = None if state.tracer_m is None else state.tracer_m + dt_scalar * rhs_explicit.tracer_m
        tangent2 = State(
            u_first,
            mtheta_explicit,
            ps_first,
            tracer_new,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        dp_edge2 = self.solver.edge_mass_weight(tangent2).to(dtype=state.u_e.dtype, device=state.u_e.device)
        residual_rhs = self._continuity_increment(
            state=state,
            tangent=tangent2,
            dp_edge_lag=dp_edge2,
            u_flux=u_first,
            dt=dt_ps,
        ) - (ps_first - state.ps)
        q2, info2 = self.solver.solve(residual_rhs, state=tangent2, dt=dt_ps)
        grad2 = self.solver.pressure_gradient(q2, tangent2).to(dtype=state.u_e.dtype, device=state.u_e.device)
        u_corr2 = -float(self.solver.damping) * dt_u * grad2
        u_new = u_first + u_corr2
        continuity_increment = self._continuity_increment(
            state=state,
            tangent=tangent2,
            dp_edge_lag=dp_edge2,
            u_flux=u_new,
            dt=dt_ps,
        )
        ps_new = state.ps + continuity_increment
        ps_projection_delta_rms = _rms(ps_new - (state.ps + q1 + q2))
        ps_mass_mismatch = (ps_new - state.ps) - continuity_increment
        ps_explicit = state.ps + dt_ps * rhs_explicit.ps

        if self.remap_mtheta:
            dm_explicit = interface_mass(state.column.layer_thickness(ps_explicit))
            theta_explicit = mtheta_explicit / dm_explicit.clamp_min(torch.finfo(mtheta_explicit.dtype).tiny)
            dm_new = interface_mass(state.column.layer_thickness(ps_new))
            mtheta_new = dm_new * theta_explicit
            mtheta_remap_rms = _rms(mtheta_new - mtheta_explicit)
        else:
            mtheta_new = mtheta_explicit
            mtheta_remap_rms = torch.zeros((), dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)
        explicit = State(
            u_explicit,
            mtheta_explicit,
            ps_explicit,
            tracer_new,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        out = State(
            u_new,
            mtheta_new,
            ps_new,
            tracer_new,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        tangent_theta_delta = _rms(tangent2.theta() - state.theta())
        continuity_rms = _rms(continuity_increment)
        u_corr = u_new - u_explicit
        u_corr_rms = _rms(u_corr)
        pressure_correction = ps_new - state.ps
        relative_residual = max(float(info1["relative_residual"]), float(info2["relative_residual"]))
        info = {
            **info2,
            "iterations": int(info1["iterations"]) + int(info2["iterations"]),
            "initial_residual": info1["initial_residual"],
            "final_residual": info2["final_residual"],
            "relative_residual": relative_residual,
            "converged": bool(info1["converged"] and info2["converged"]),
            "imex_scheme": self.scheme_name,
            "imex_pressure_iterations": int(info1["iterations"]) + int(info2["iterations"]),
            "imex_pressure_initial_residual": info1["initial_residual"],
            "imex_pressure_final_residual": info2["final_residual"],
            "imex_pressure_relative_residual": relative_residual,
            "imex_pressure_converged": bool(info1["converged"] and info2["converged"]),
            "imex_picard1_first_relative_residual": info1["relative_residual"],
            "imex_picard1_second_relative_residual": info2["relative_residual"],
            "imex_picard1_residual_rhs_rms": _float_info(_rms(residual_rhs)),
            "imex_picard1_q1_rms": _float_info(_rms(q1)),
            "imex_picard1_q2_rms": _float_info(_rms(q2)),
            "imex_pressure_correction_rms": _float_info(_rms(pressure_correction)),
            "imex_u_correction_rms": _float_info(u_corr_rms),
            "imex_ps_correction_rms": _float_info(_rms(pressure_correction)),
            "imex_raw_q_correction_rms": _float_info(_rms(q1 + q2)),
            "imex_pressure_response_min": _float_info(self.solver.pressure_response_level(tangent2).min()),
            "imex_pressure_response_max": _float_info(self.solver.pressure_response_level(tangent2).max()),
            "imex_edge_mass_min": _float_info(dp_edge2.min()),
            "imex_edge_mass_max": _float_info(dp_edge2.max()),
            "imex_mtheta_pressure_remap_rms": _float_info(mtheta_remap_rms),
            "imex_mtheta_pressure_remap": self.remap_mtheta,
            "imex_tangent_state": "picard1_corrected_pressure",
            "imex_tangent_ps_delta_rms": _float_info(_rms(tangent2.ps - state.ps)),
            "imex_tangent_mtheta_delta_rms": _float_info(_rms(tangent2.mtheta_i - state.mtheta_i)),
            "imex_tangent_theta_delta_rms": _float_info(tangent_theta_delta),
            "imex_nonpressure_split": True,
            "imex_pressure_only": True,
            "imex_lagged_thermo": True,
            "imex_centering_alpha": float(self.centering_alpha),
            "imex_pressure_update_mode": self.pressure_update_mode,
            "imex_conservative_projection": True,
            "imex_picard_iterations": 1,
            "imex_continuity_ps_increment_rms": _float_info(continuity_rms),
            "imex_continuity_ps_increment_max_abs": _float_info(continuity_increment.detach().abs().max()),
            "imex_ps_mass_consistency_mismatch_rms": _float_info(_rms(ps_mass_mismatch)),
            "imex_ps_mass_consistency_mismatch_max_abs": _float_info(ps_mass_mismatch.detach().abs().max()),
            "imex_ps_mass_consistency_mismatch_over_ps_correction": _float_info(
                _rms(ps_mass_mismatch) / _rms(pressure_correction).detach().clamp_min(1.0e-30)
            ),
            "imex_continuity_ps_increment_over_u_correction": _float_info(
                continuity_rms / u_corr_rms.detach().clamp_min(1.0e-30)
            ),
            "imex_ps_projection_delta_rms": _float_info(ps_projection_delta_rms),
            **self.correction_budget(
                state,
                explicit,
                out,
                q=pressure_correction,
                u_star=u_explicit,
                u_corr=u_corr,
                mtheta_explicit=mtheta_explicit,
                mtheta_remap_rms=mtheta_remap_rms,
                dt=dt,
            ),
        }
        self.last_solver_info = info
        return out, info


@register_model("dry_core_pressure_projection_si_conservative_theta_flux", status="negative_result", component="throughput", description="Corrected-flux theta transport pressure projection variant; negative diagnostic result.", default_safe=False)
class DryCorePressureProjectionSIThetaFlux(DryCorePressureProjectionSIConservative):
    """Conservative projection with mtheta transported by corrected mass flux."""

    scheme_name = "pressure_projection_si_conservative_theta_flux"
    pressure_update_mode = "conservative_theta_flux"

    def step_with_info(self, state: State, dt: float | torch.Tensor) -> tuple[State, dict[str, Any]]:
        state = self.bind_state(state)
        terms = self.dynamics.tendency_terms(state)
        rhs_explicit = self.dynamics.terms_total(terms, like=state)
        tangent = self._centered_tangent_state(state, terms, dt=dt)
        dt_ps = torch.as_tensor(dt, dtype=state.ps.dtype, device=state.ps.device)
        dt_u = torch.as_tensor(dt, dtype=state.u_e.dtype, device=state.u_e.device)
        dt_scalar = torch.as_tensor(dt, dtype=state.mtheta_i.dtype, device=state.mtheta_i.device)

        f_u = terms["u_e"]["total"]
        u_explicit = state.u_e + dt_u * f_u
        dp_edge_lag = self.solver.edge_mass_weight(tangent).to(dtype=state.u_e.dtype, device=state.u_e.device)
        rhs_q = self._continuity_increment(
            state=state,
            tangent=tangent,
            dp_edge_lag=dp_edge_lag,
            u_flux=u_explicit,
            dt=dt_ps,
        )
        q, info = self.solver.solve(rhs_q, state=tangent, dt=dt_ps)

        pressure_grad_q = self.solver.pressure_gradient(q, tangent).to(dtype=state.u_e.dtype, device=state.u_e.device)
        u_corr = -float(self.solver.damping) * dt_u * pressure_grad_q
        u_new = u_explicit + u_corr
        continuity_increment = self._continuity_increment(
            state=state,
            tangent=tangent,
            dp_edge_lag=dp_edge_lag,
            u_flux=u_new,
            dt=dt_ps,
        )
        ps_new = state.ps + continuity_increment
        ps_projection_delta_rms = _rms(ps_new - (state.ps + q))
        ps_mass_mismatch = (ps_new - state.ps) - continuity_increment
        ps_explicit = state.ps + dt_ps * rhs_explicit.ps
        mtheta_explicit = state.mtheta_i + dt_scalar * rhs_explicit.mtheta_i
        mtheta_new = self._mtheta_from_corrected_flux(
            state=state,
            tangent=tangent,
            theta_i=state.theta(),
            u_flux=u_new,
            dt=dt_scalar,
        )
        mtheta_remap_rms = _rms(mtheta_new - mtheta_explicit)
        tracer_new = None if state.tracer_m is None else state.tracer_m + dt_scalar * rhs_explicit.tracer_m
        explicit = State(
            u_explicit,
            mtheta_explicit,
            ps_explicit,
            tracer_new,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        out = State(
            u_new,
            mtheta_new,
            ps_new,
            tracer_new,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )
        tangent_theta_delta = _rms(tangent.theta() - state.theta())
        continuity_rms = _rms(continuity_increment)
        u_corr_rms = _rms(u_corr)
        info = {
            **info,
            "imex_scheme": self.scheme_name,
            "imex_pressure_iterations": info["iterations"],
            "imex_pressure_initial_residual": info["initial_residual"],
            "imex_pressure_final_residual": info["final_residual"],
            "imex_pressure_relative_residual": info["relative_residual"],
            "imex_pressure_converged": info["converged"],
            "imex_pressure_correction_rms": _float_info(_rms(q)),
            "imex_u_correction_rms": _float_info(u_corr_rms),
            "imex_ps_correction_rms": _float_info(_rms(ps_new - state.ps)),
            "imex_raw_q_correction_rms": _float_info(_rms(q)),
            "imex_pressure_response_min": _float_info(self.solver.pressure_response_level(tangent).min()),
            "imex_pressure_response_max": _float_info(self.solver.pressure_response_level(tangent).max()),
            "imex_edge_mass_min": _float_info(dp_edge_lag.min()),
            "imex_edge_mass_max": _float_info(dp_edge_lag.max()),
            "imex_mtheta_pressure_remap_rms": _float_info(mtheta_remap_rms),
            "imex_mtheta_pressure_remap": False,
            "imex_mtheta_corrected_flux": True,
            "imex_mtheta_corrected_flux_rms": _float_info(mtheta_remap_rms),
            "imex_tangent_state": "explicit_lagged",
            "imex_tangent_ps_delta_rms": _float_info(_rms(tangent.ps - state.ps)),
            "imex_tangent_mtheta_delta_rms": _float_info(_rms(tangent.mtheta_i - state.mtheta_i)),
            "imex_tangent_theta_delta_rms": _float_info(tangent_theta_delta),
            "imex_nonpressure_split": True,
            "imex_pressure_only": True,
            "imex_lagged_thermo": True,
            "imex_centering_alpha": float(self.centering_alpha),
            "imex_pressure_update_mode": self.pressure_update_mode,
            "imex_conservative_projection": True,
            "imex_flux_velocity_mode": "corrected_velocity",
            "imex_continuity_ps_increment_rms": _float_info(continuity_rms),
            "imex_continuity_ps_increment_max_abs": _float_info(continuity_increment.detach().abs().max()),
            "imex_ps_mass_consistency_mismatch_rms": _float_info(_rms(ps_mass_mismatch)),
            "imex_ps_mass_consistency_mismatch_max_abs": _float_info(ps_mass_mismatch.detach().abs().max()),
            "imex_ps_mass_consistency_mismatch_over_ps_correction": _float_info(
                _rms(ps_mass_mismatch) / _rms(ps_new - state.ps).detach().clamp_min(1.0e-30)
            ),
            "imex_continuity_ps_increment_over_u_correction": _float_info(
                continuity_rms / u_corr_rms.detach().clamp_min(1.0e-30)
            ),
            "imex_ps_projection_delta_rms": _float_info(ps_projection_delta_rms),
            **self.correction_budget(
                state,
                explicit,
                out,
                q=ps_new - state.ps,
                u_star=u_explicit,
                u_corr=u_corr,
                mtheta_explicit=mtheta_explicit,
                mtheta_remap_rms=mtheta_remap_rms,
                dt=dt,
            ),
        }
        self.last_solver_info = info
        return out, info


from .ark2 import (
    ARK2_TABLEAU,
    DryPEIMEXARK2,
    DryPELinearOperator,
    IMEXConvergenceError,
    IMEXSolverConfig,
    IMEXStageInfo,
    IMEXStepInfo,
)


__all__ = [
    "ARK2_TABLEAU",
    "BlockBiCGSTAB",
    "DryCorePressureIMEXEuler",
    "DryCorePressureProjectionSIConservative",
    "DryCorePressureProjectionSIMidpointFlux",
    "DryCorePressureProjectionSIPicard1",
    "DryCorePressureProjectionSIThetaFlux",
    "DryCorePressureScalarSIPressureOnlyCentered",
    "DryCorePressureScalarSIPressureOnlyLaggedThermo",
    "DryCorePressureScalarSIProjectionProbe",
    "DryCorePressureScalarSIThermoMidpoint",
    "DryCorePressureSIMidpoint",
    "DryCorePressureSIThermoMidpoint",
    "DryPEIMEXARK2",
    "DryPELinearOperator",
    "IMEXConvergenceError",
    "IMEXSolverConfig",
    "IMEXStageInfo",
    "IMEXStepInfo",
    "PressureCorrectionCG",
]
