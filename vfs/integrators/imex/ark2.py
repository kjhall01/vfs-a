"""Matrix-free ARK2 stepping for the Field-native dry hydrostatic state.

This module ports the solver architecture of the ``vfs-imex`` research tree
onto VFS's richer ``State``/``VoronoiMesh`` API and its historical
``[..., horizontal, vertical]`` storage order.  The nonlinear tendency stays
owned by ``DryHydrostaticDynamics``; the stepper freezes its exact JVP at the
start of a step and treats that block implicitly with restarted GMRES.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, NamedTuple

import torch

from ...core.state import State
from ...geometry.mesh.multilevel import VoronoiMeshHierarchy


class ARKTableau(NamedTuple):
    explicit: tuple[tuple[float, ...], ...]
    implicit: tuple[tuple[float, ...], ...]
    weights: tuple[float, ...]
    order: int
    name: str

    @property
    def n_stages(self) -> int:
        return len(self.weights)


_SQRT2 = math.sqrt(2.0)
_GAMMA = 1.0 - 1.0 / _SQRT2
_A32 = (3.0 + 2.0 * _SQRT2) / 6.0
_B12 = 1.0 / (2.0 * _SQRT2)

# Giraldo, Kelly, and Constantinescu's three-stage, second-order ARK pair.
ARK2_TABLEAU = ARKTableau(
    explicit=(
        (0.0, 0.0, 0.0),
        (2.0 - _SQRT2, 0.0, 0.0),
        (1.0 - _A32, _A32, 0.0),
    ),
    implicit=(
        (0.0, 0.0, 0.0),
        (_GAMMA, _GAMMA, 0.0),
        (_B12, _B12, _GAMMA),
    ),
    weights=(_B12, _B12, _GAMMA),
    order=2,
    name="ark2-giraldo-3stage",
)


@dataclass(frozen=True)
class IMEXSolverConfig:
    """Controls for the matrix-free Field-native ARK solve."""

    rtol: float | None = None
    atol: float = 0.0
    max_iterations: int = 40
    restart: int = 20
    reorthogonalize: bool = True
    implicit_operator: str = "coupled"
    preconditioner: str = "none"
    schur_tolerance_safety: float = 0.8
    multilevel_coarsening: int = 3
    multilevel_max_levels: int = 4
    multilevel_min_frequency: int = 3
    multilevel_stencil_size: int = 3
    multilevel_smoothing_iterations: int = 1
    multilevel_jacobi_weight: float = 0.85
    compile_kernels: bool = False
    compile_backend: str | None = None
    compile_mode: str | None = None

    def resolved_rtol(self, dtype: torch.dtype) -> float:
        return float(self.rtol) if self.rtol is not None else (1.0e-8 if dtype == torch.float64 else 1.0e-5)

    def validate(self) -> None:
        if self.rtol is not None and self.rtol <= 0.0:
            raise ValueError("rtol must be positive")
        if self.atol < 0.0:
            raise ValueError("atol must be nonnegative")
        if self.max_iterations < 1 or self.restart < 1:
            raise ValueError("max_iterations and restart must be positive")
        if self.implicit_operator not in {"coupled", "fast_wave", "gravity_wave"}:
            raise ValueError("implicit_operator must be 'coupled', 'fast_wave', or 'gravity_wave'")
        if self.preconditioner not in {"none", "hydrostatic_schur", "multilevel_hydrostatic_schur"}:
            raise ValueError(
                "preconditioner must be 'none', 'hydrostatic_schur', or "
                "'multilevel_hydrostatic_schur'"
            )
        if not 0.0 < self.schur_tolerance_safety <= 1.0:
            raise ValueError("schur_tolerance_safety must be in (0, 1]")
        if self.multilevel_coarsening < 2:
            raise ValueError("multilevel_coarsening must be at least two")
        if self.multilevel_max_levels < 2:
            raise ValueError("multilevel_max_levels must be at least two")
        if self.multilevel_min_frequency < 1:
            raise ValueError("multilevel_min_frequency must be positive")
        if self.multilevel_stencil_size < 1:
            raise ValueError("multilevel_stencil_size must be positive")
        if self.multilevel_smoothing_iterations < 1:
            raise ValueError("multilevel_smoothing_iterations must be positive")
        if not 0.0 < self.multilevel_jacobi_weight < 1.0:
            raise ValueError("multilevel_jacobi_weight must be in (0, 1)")
        if self.compile_backend is not None and not str(self.compile_backend).strip():
            raise ValueError("compile_backend must be non-empty when provided")
        if self.compile_mode is not None and not str(self.compile_mode).strip():
            raise ValueError("compile_mode must be non-empty when provided")


@dataclass(frozen=True)
class IMEXStageInfo:
    stage: int
    implicit_coefficient: float
    iterations: int
    matvecs: int
    preconditioner_applications: int
    recycled_dimension: int
    initial_residual: float
    final_residual: float
    relative_residual: float
    converged: bool


@dataclass(frozen=True)
class IMEXStepInfo:
    method: str
    linear_operator: str
    preconditioner: str
    dt: float
    stages: tuple[IMEXStageInfo, ...]
    converged: bool
    total_iterations: int
    total_matvecs: int
    total_preconditioner_applications: int


class IMEXConvergenceError(RuntimeError):
    """Raised when an adaptive ARK stage does not meet its residual target."""

    def __init__(self, stage_info: IMEXStageInfo) -> None:
        super().__init__(
            f"IMEX stage {stage_info.stage} failed after {stage_info.iterations} GMRES iterations "
            f"with relative residual {stage_info.relative_residual:.3e}"
        )
        self.stage_info = stage_info


def _values(state: State) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return state.u_e.values, state.mtheta_i.values, state.ps.values


class _StateInnerProduct:
    """Fixed, field-aware scaling for Krylov convergence decisions."""

    def __init__(self, anchor: State) -> None:
        u, mtheta, ps = _values(anchor)
        tiny = torch.finfo(u.dtype).tiny
        self.u_scale = u.detach().square().mean().sqrt().clamp_min(torch.as_tensor(10.0, dtype=u.dtype, device=u.device))
        mtheta_floor = torch.as_tensor(1.0, dtype=mtheta.dtype, device=mtheta.device)
        self.mtheta_scale = mtheta.detach().square().mean().sqrt().clamp_min(mtheta_floor)
        self.ps_scale = ps.detach().square().mean().sqrt().clamp_min(torch.as_tensor(1.0e5, dtype=ps.dtype, device=ps.device))
        self.tiny = tiny

    def dot(self, left: State, right: State) -> torch.Tensor:
        lu, lm, lp = _values(left)
        ru, rm, rp = _values(right)
        return (
            (lu * ru).mean() / self.u_scale.square()
            + (lm * rm).mean() / self.mtheta_scale.square()
            + (lp * rp).mean() / self.ps_scale.square()
        )

    def norm(self, value: State) -> torch.Tensor:
        return self.dot(value, value).clamp_min(0.0).sqrt()


@dataclass(frozen=True)
class _KrylovRecycle:
    """One restarted-GMRES subspace for the next ARK stage of the same block."""

    residual_basis: tuple[State, ...]
    solution_basis: tuple[State, ...]
    hessenberg: torch.Tensor


@dataclass(frozen=True)
class _GMRESInfo:
    solution: State
    iterations: int
    matvecs: int
    preconditioner_applications: int
    recycled_dimension: int
    initial_residual: torch.Tensor
    final_residual: torch.Tensor
    converged: bool
    recycle: _KrylovRecycle | None


class DryPELinearOperator:
    """Step-frozen full JVP or analytic pressure/transport wave block."""

    def __init__(
        self,
        dynamics,
        anchor: State,
        *,
        implicit_operator: str = "coupled",
        tendency: Callable[[State], State] | None = None,
        jvp_tendency: Callable[[State], State] | None = None,
    ) -> None:
        if implicit_operator not in {"coupled", "gravity_wave"}:
            raise NotImplementedError(
                "VFS 1.0 exposes the coupled exact-JVP and dry analytic "
                "gravity-wave operators; fast-wave reduction is retained for "
                "a later port increment"
            )
        self.dynamics = dynamics
        self.anchor = anchor
        self.implicit_operator = implicit_operator
        self.tendency = dynamics.tendency if tendency is None else tendency
        # Dynamo cannot currently trace the Field/column views wrapped by a
        # forward-mode dual tensor. Keep the exact JVP eager while allowing
        # ordinary nonlinear evaluations to use compiled kernels.
        self.jvp_tendency = dynamics.tendency if jvp_tendency is None else jvp_tendency
        self.base_tendency = self.tendency(anchor)
        self.primitives = None
        if implicit_operator == "gravity_wave":
            self._validate_gravity_wave_support()
            self.primitives = dynamics._transport_primitives(anchor)
        self.conservative_wave = None
        if implicit_operator == "gravity_wave":
            from ...atmosphere.conservative import DryHydrostaticConservativeDynamics
            if isinstance(dynamics, DryHydrostaticConservativeDynamics):
                from .conservative_wave import ConservativePressureWave
                self.conservative_wave = ConservativePressureWave(dynamics, anchor)
        self.applications = 0
        self._compiled_gravity_wave_wind: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
        self._compiled_gravity_wave_thermodynamic: (
            Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None
        ) = None

    def compile_analytic_gravity_wave(
        self,
        *,
        backend: str | None = None,
        mode: str | None = None,
    ) -> None:
        """Compile the gravity-wave half blocks while adaptive GMRES stays eager."""

        if self.implicit_operator != "gravity_wave":
            return
        kwargs: dict[str, str | bool] = {"fullgraph": True}
        if backend is not None:
            kwargs["backend"] = backend
        if mode is not None:
            kwargs["mode"] = mode
        self._compiled_gravity_wave_wind = torch.compile(self._gravity_wave_wind, **kwargs)
        self._compiled_gravity_wave_thermodynamic = torch.compile(
            self._gravity_wave_thermodynamic,
            **kwargs,
        )

    def _validate_gravity_wave_support(self) -> None:
        """Reject recipe combinations outside the ported analytic block.

        The donor Schur path differentiates the GL2 hydrostatic pressure
        gradient and a centered transport block analytically.  For the
        virtual-temperature recipe, its vapor factor is frozen at the ARK
        anchor: vapor transport and vapor-induced factor changes stay in the
        explicit residual, while hydrostatics sees the anchor virtual
        temperature. The conservative diagonal recipe instead uses its own
        matched thermal-energy pressure derivative. Other PGF recipes remain
        outside this analytic split.
        """

        if getattr(self.anchor.column, "phi_full_mode", None) != "direct_u_gl2_adjoint":
            raise ValueError(
                "implicit_operator='gravity_wave' requires "
                "column.phi_full_mode='direct_u_gl2_adjoint'"
            )
        from ...atmosphere.conservative import DryHydrostaticConservativeDynamics
        if isinstance(self.dynamics, DryHydrostaticConservativeDynamics):
            self.dynamics.check_state_compatible(self.anchor)
            return
        thermodynamics_name = getattr(getattr(self.dynamics, "thermodynamics", None), "name", None)
        if thermodynamics_name not in {"dry", "virtual_temperature"}:
            raise ValueError(
                "implicit_operator='gravity_wave' requires dry or frozen-coefficient "
                "virtual-temperature thermodynamics"
            )
        if getattr(getattr(self.dynamics, "pressure_gradient", None), "name", None) not in {
            "terrain_aware_log_pressure_sigma",
            "log_pressure_sigma",
        }:
            raise ValueError(
                "implicit_operator='gravity_wave' requires the canonical dry "
                "terrain-aware/log-pressure pressure-gradient strategy"
            )

    @staticmethod
    def _interface_mass(dp: torch.Tensor) -> torch.Tensor:
        half = 0.5 * dp
        return torch.cat([half[..., :1], half[..., :-1] + half[..., 1:], half[..., -1:]], dim=-1)

    def _transport_tangent(self, direction: State) -> dict[str, torch.Tensor]:
        """Analytic dry transport tangent in main-VFS axis order.

        This is the Field-native adaptation of the donor mass-continuity
        tangent.  It deliberately works on tensors inside the validated
        operator, while State/Field metadata stays at the public boundary.
        """

        assert self.primitives is not None
        anchor = self.anchor
        base = self.primitives
        u, mtheta, ps = _values(direction)
        d_dp = (anchor.column.B_interface[1:] - anchor.column.B_interface[:-1]).to(
            dtype=ps.dtype, device=ps.device
        ) * ps
        d_dm = self._interface_mass(d_dp)
        d_dp_e = anchor.cell_to_edge(d_dp)
        d_mass_flux = u * base["dp_e"] + anchor.u_e.values * d_dp_e
        d_layer_div = anchor.div(d_mass_flux)
        d_dps_dt = -d_layer_div.sum(dim=-1, keepdim=True)
        d_mu = self.dynamics._vertical_mass_flux(anchor, d_layer_div, d_dps_dt)
        d_theta = (mtheta - base["theta"] * d_dm) / base["dm"].clamp_min(torch.finfo(mtheta.dtype).tiny)
        return {
            "theta": d_theta,
            "mass_flux": d_mass_flux,
            "dps_dt": d_dps_dt,
            "omega": 0.5 * (d_mu[..., :-1] + d_mu[..., 1:]),
        }

    def _centered_mtheta_tangent(self, tangent: dict[str, torch.Tensor]) -> torch.Tensor:
        """Analytic centered CP thermodynamic response to a wind perturbation."""

        assert self.primitives is not None
        anchor = self.anchor
        theta = self.primitives["theta"]
        mass_flux = self.primitives["mass_flux"]
        omega = self.primitives["omega"]
        d_theta = tangent["theta"]
        d_mass_flux = tangent["mass_flux"]
        d_omega = tangent["omega"]
        cells = anchor.mesh.cellsOnEdge.to(device=d_theta.device)
        theta_edge = 0.5 * (theta[..., cells[:, 0], :] + theta[..., cells[:, 1], :])
        d_theta_edge = 0.5 * (d_theta[..., cells[:, 0], :] + d_theta[..., cells[:, 1], :])
        d_upper_flux = 0.5 * (
            d_mass_flux * theta_edge[..., :-1] + mass_flux * d_theta_edge[..., :-1]
        )
        d_lower_flux = 0.5 * (
            d_mass_flux * theta_edge[..., 1:] + mass_flux * d_theta_edge[..., 1:]
        )
        horizontal = torch.zeros_like(d_theta)
        horizontal[..., :-1] -= anchor.div(d_upper_flux)
        horizontal[..., 1:] -= anchor.div(d_lower_flux)

        theta_vertical = 0.5 * (theta[..., :-1] + theta[..., 1:])
        d_theta_vertical = 0.5 * (d_theta[..., :-1] + d_theta[..., 1:])
        d_vertical_flux = d_omega * theta_vertical + omega * d_theta_vertical
        vertical = torch.zeros_like(d_theta)
        vertical[..., :-1] -= d_vertical_flux
        vertical[..., 1:] += d_vertical_flux
        return horizontal + vertical

    @staticmethod
    def _centered_mtheta_tendency(state: State, primitives: dict[str, torch.Tensor]) -> torch.Tensor:
        """Smooth centered CP transport used only to define the gravity block."""

        theta = primitives["theta"]
        mass_flux = primitives["mass_flux"]
        omega = primitives["omega"]
        cells = state.mesh.cellsOnEdge.to(device=theta.device)
        theta_edge = 0.5 * (theta[..., cells[:, 0], :] + theta[..., cells[:, 1], :])
        horizontal = torch.zeros_like(theta)
        horizontal[..., :-1] -= state.div(0.5 * mass_flux * theta_edge[..., :-1])
        horizontal[..., 1:] -= state.div(0.5 * mass_flux * theta_edge[..., 1:])
        vertical_flux = omega * 0.5 * (theta[..., :-1] + theta[..., 1:])
        vertical = torch.zeros_like(theta)
        vertical[..., :-1] -= vertical_flux
        vertical[..., 1:] += vertical_flux
        return horizontal + vertical

    @staticmethod
    def _gl2_geopotential_ps_derivative(
        theta: torch.Tensor,
        pi_i: torch.Tensor,
        pi_full: torch.Tensor,
        d_pi_i: torch.Tensor,
        d_pi_full: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``d(phi_full)/d(ps)`` for direct-u GL2 placement.

        This is the analytic pressure derivative of
        :meth:`CharneyPhillipsColumn.geopotential_full_direct_u_gl2` at fixed
        interface theta.  Keeping it local to the experimental IMEX operator
        avoids expanding the public column API for one solver-specific tangent.
        """

        tiny = torch.finfo(theta.dtype).tiny
        theta_top, theta_bottom = theta[..., :-1], theta[..., 1:]
        theta_bar = 0.5 * (theta_top + theta_bottom)
        delta_pi = (pi_i[..., 1:] - pi_i[..., :-1]).clamp_min(tiny)
        d_delta_pi = d_pi_i[..., 1:] - d_pi_i[..., :-1]
        d_layer_integral = theta_bar * d_delta_pi
        d_suffix = torch.flip(torch.cumsum(torch.flip(d_layer_integral, dims=[-1]), dim=-1), dims=[-1])
        d_phi_i = torch.cat([d_suffix, torch.zeros_like(d_suffix[..., :1])], dim=-1)

        pi_top, pi_bottom = pi_i[..., :-1], pi_i[..., 1:]
        d_pi_top, d_pi_bottom = d_pi_i[..., :-1], d_pi_i[..., 1:]
        half = 0.5 * (pi_bottom - pi_full)
        d_half = 0.5 * (d_pi_bottom - d_pi_full)
        mid = 0.5 * (pi_bottom + pi_full)
        d_mid = 0.5 * (d_pi_bottom + d_pi_full)
        node = 1.0 / math.sqrt(3.0)
        q0, q1 = mid - half * node, mid + half * node
        d_q0, d_q1 = d_mid - d_half * node, d_mid + d_half * node
        s0 = (q0 - pi_top) / delta_pi
        s1 = (q1 - pi_top) / delta_pi
        d_s0 = ((d_q0 - d_pi_top) * delta_pi - (q0 - pi_top) * d_delta_pi) / delta_pi.square()
        d_s1 = ((d_q1 - d_pi_top) * delta_pi - (q1 - pi_top) * d_delta_pi) / delta_pi.square()
        theta_q0 = theta_top + s0 * (theta_bottom - theta_top)
        theta_q1 = theta_top + s1 * (theta_bottom - theta_top)
        d_partial = d_half * (theta_q0 + theta_q1) + half * (d_s0 + d_s1) * (theta_bottom - theta_top)
        return d_phi_i[..., 1:] + d_partial

    def _pressure_gradient_tangent(self, direction: State, d_theta: torch.Tensor) -> torch.Tensor:
        """Analytic GL2/log-pressure hydrostatic PGF response."""

        assert self.primitives is not None
        anchor = self.anchor
        column = anchor.column
        theta = self.primitives["theta"]
        ps = anchor.ps.values
        d_ps = direction.ps.values
        tiny = torch.finfo(ps.dtype).tiny

        # Virtual-temperature factors are coefficients of the analytic
        # gravity-wave block.  They are frozen at the ARK anchor so this
        # remains a linear operator in (u, mtheta, ps); tracer transport and
        # changes in vapor remain in the explicit residual.  A virtual recipe
        # with no vapor metadata correctly reduces to the dry factor of one.
        thermodynamics = getattr(self.dynamics, "thermodynamics", None)
        factor_i = None
        factor_full = None
        if getattr(thermodynamics, "name", None) == "virtual_temperature":
            interface_factor = getattr(thermodynamics, "virtual_temperature_factor_interface", None)
            full_factor = getattr(thermodynamics, "virtual_temperature_factor_full", None)
            factor_i = None if not callable(interface_factor) else interface_factor(anchor)
            factor_full = None if not callable(full_factor) else full_factor(anchor)
        theta_hydro = theta if factor_i is None else theta * factor_i
        d_theta_hydro = d_theta if factor_i is None else d_theta * factor_i

        p_i = column.interface_pressure(ps)
        pi_i = column.exner_interface(ps)
        pi_full = column.exner_full_pmid(ps)
        d_pi_i_per_ps = column.kappa * pi_i / p_i.clamp_min(tiny) * column.B_interface
        p_full = column.full_pressure(ps)
        d_pi_full_per_ps = column.kappa * pi_full / p_full.clamp_min(tiny) * column.B_full

        zero_theta = torch.zeros_like(theta_hydro)
        d_phi_theta = column.geopotential_full(d_theta_hydro, ps) - column.geopotential_full(zero_theta, ps)
        d_phi_dps = self._gl2_geopotential_ps_derivative(
            theta_hydro,
            pi_i,
            pi_full,
            d_pi_i_per_ps,
            d_pi_full_per_ps,
        )
        d_pgf = -anchor.grad(d_phi_theta + d_phi_dps * d_ps)

        d_pi_i = d_pi_i_per_ps * d_ps
        d_pi_full = d_pi_full_per_ps * d_ps
        pi_top, pi_bottom = pi_i[..., :-1], pi_i[..., 1:]
        d_pi_top, d_pi_bottom = d_pi_i[..., :-1], d_pi_i[..., 1:]
        delta_pi = (pi_bottom - pi_top).clamp_min(tiny)
        interpolation = (pi_full - pi_top) / delta_pi
        d_interpolation = (
            (d_pi_full - d_pi_top) * delta_pi
            - (pi_full - pi_top) * (d_pi_bottom - d_pi_top)
        ) / delta_pi.square().clamp_min(tiny)
        # The hydrostatic reconstruction above uses interface virtual theta.
        # The sigma-PGF specific-volume term instead uses the full-level dry
        # theta placement multiplied by its own full-level vapor factor, just
        # as ``VirtualTemperatureThermodynamics.pgf_components`` does.
        theta_top, theta_bottom = theta[..., :-1], theta[..., 1:]
        d_theta_top, d_theta_bottom = d_theta[..., :-1], d_theta[..., 1:]
        theta_full = theta_top + interpolation * (theta_bottom - theta_top)
        d_theta_full = (
            d_theta_top
            + d_interpolation * (theta_bottom - theta_top)
            + interpolation * (d_theta_bottom - d_theta_top)
        )
        if factor_full is not None:
            theta_full = theta_full * factor_full
            d_theta_full = d_theta_full * factor_full
        temperature = theta_full * (pi_full / column.c_p)
        d_temperature = (d_theta_full * pi_full + theta_full * d_pi_full) / column.c_p
        temperature_edge = anchor.cell_to_edge(temperature)
        d_temperature_edge = anchor.cell_to_edge(d_temperature)
        log_pressure = torch.log(p_full)
        d_log_pressure = column.B_full * d_ps / p_full.clamp_min(tiny)
        d_pgf -= column.R_d * (
            d_temperature_edge * anchor.grad(log_pressure)
            + temperature_edge * anchor.grad(d_log_pressure)
        )
        return d_pgf

    def _gravity_wave_apply(self, direction: State) -> State:
        """Apply the donor gravity-wave off-diagonal implicit block."""

        wind = self.gravity_wave_wind(direction.mtheta_i.values, direction.ps.values)
        mtheta, ps = self.gravity_wave_thermodynamic(direction.u_e.values)
        return State(
            wind,
            mtheta,
            ps,
            None if direction.tracer_m is None else torch.zeros_like(direction.tracer_m.values),
            mesh=self.anchor.mesh,
            column=self.anchor.column,
            attrs=self.anchor.attrs,
        )

    def gravity_wave_wind(self, mtheta: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
        if self._compiled_gravity_wave_wind is not None:
            return self._compiled_gravity_wave_wind(mtheta, ps)
        return self._gravity_wave_wind(mtheta, ps)

    def gravity_wave_thermodynamic(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._compiled_gravity_wave_thermodynamic is not None:
            return self._compiled_gravity_wave_thermodynamic(u)
        return self._gravity_wave_thermodynamic(u)

    def _gravity_wave_wind(self, mtheta: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
        """Return the analytic thermodynamic-to-wind gravity-wave block."""

        if self.conservative_wave is not None:
            return self.conservative_wave.wind(mtheta, ps)
        thermodynamic = State(
            torch.zeros_like(self.anchor.u_e.values),
            mtheta,
            ps,
            mesh=self.anchor.mesh,
            column=self.anchor.column,
            attrs=self.anchor.attrs,
        )
        tangent = self._transport_tangent(thermodynamic)
        return self._pressure_gradient_tangent(thermodynamic, tangent["theta"])

    def _gravity_wave_thermodynamic(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the analytic wind-to-thermodynamic gravity-wave block."""

        batch_shape = u.shape[:-2]
        wind_only = State(
            u,
            torch.zeros(
                *batch_shape,
                self.anchor.mtheta_i.values.shape[-2],
                self.anchor.mtheta_i.values.shape[-1],
                dtype=u.dtype,
                device=u.device,
            ),
            torch.zeros(
                *batch_shape,
                self.anchor.ps.values.shape[-2],
                self.anchor.ps.values.shape[-1],
                dtype=u.dtype,
                device=u.device,
            ),
            mesh=self.anchor.mesh,
            column=self.anchor.column,
            attrs=self.anchor.attrs,
        )
        wind_tangent = self._transport_tangent(wind_only)
        return self._centered_mtheta_tangent(wind_tangent), wind_tangent["dps_dt"]

    def apply(self, direction: State) -> State:
        self.applications += 1
        # JVP pytree metadata is part of the tree specification.  Tendencies
        # may give diagnostic names to their Field outputs, so rebuild the
        # direction with the anchor's canonical prognostic metadata.
        direction = State(
            direction.u_e.values,
            direction.mtheta_i.values,
            direction.ps.values,
            None if direction.tracer_m is None else direction.tracer_m.values,
            mesh=self.anchor.mesh,
            column=self.anchor.column,
            attrs=self.anchor.attrs,
        )
        if self.implicit_operator == "gravity_wave":
            return self._gravity_wave_apply(direction)
        _value, tangent = torch.func.jvp(self.jvp_tendency, (self.anchor,), (direction,))
        return tangent


def _gmres_fixed_work(
    matvec: Callable[[State], State],
    rhs: State,
    inner: _StateInnerProduct,
    *,
    rtol: float,
    atol: float,
    max_iterations: int,
    restart: int,
    reorthogonalize: bool,
    fixed_iterations: int | None = None,
    preconditioner: Callable[[State], State] | None = None,
) -> tuple[State, int, int, torch.Tensor, torch.Tensor, bool]:
    """Restarted flexible GMRES using State algebra.

    ``preconditioner`` is a right preconditioner.  Retaining the transformed
    solution basis makes fixed-work solves differentiable while allowing a
    nonlinear-free multilevel correction to vary by Krylov vector.
    """

    x = rhs.zeros_like()
    rhs_norm = inner.norm(rhs)
    target = torch.maximum(
        torch.as_tensor(float(atol), dtype=rhs_norm.dtype, device=rhs_norm.device),
        torch.as_tensor(float(rtol), dtype=rhs_norm.dtype, device=rhs_norm.device) * rhs_norm,
    )
    iterations = 0
    matvecs = 0
    residual = rhs
    residual_norm = rhs_norm
    limit = int(fixed_iterations) if fixed_iterations is not None else int(max_iterations)

    while iterations < limit:
        beta = inner.norm(residual)
        if bool((beta == 0.0).detach().cpu()):
            return x, iterations, matvecs, rhs_norm, beta, True
        if fixed_iterations is None and bool((beta <= target).detach().cpu()):
            break
        basis: list[State] = [residual / beta]
        hessenberg = torch.zeros((restart + 1, restart), dtype=beta.dtype, device=beta.device)
        cycle = min(restart, limit - iterations)
        solution_basis: list[State] = []
        y = None
        used = 0
        for j in range(cycle):
            solution_vector = basis[j] if preconditioner is None else preconditioner(basis[j])
            solution_basis.append(solution_vector)
            w = matvec(solution_vector)
            matvecs += 1
            for i in range(j + 1):
                hessenberg[i, j] = inner.dot(basis[i], w)
                w = w - hessenberg[i, j] * basis[i]
            if reorthogonalize:
                for i in range(j + 1):
                    correction = inner.dot(basis[i], w)
                    hessenberg[i, j] = hessenberg[i, j] + correction
                    w = w - correction * basis[i]
            hessenberg[j + 1, j] = inner.norm(w)
            if j + 1 < cycle:
                basis.append(w / hessenberg[j + 1, j].clamp_min(inner.tiny))
            reduced_rhs = torch.zeros(j + 2, dtype=beta.dtype, device=beta.device)
            reduced_rhs[0] = beta
            y = torch.linalg.lstsq(hessenberg[: j + 2, : j + 1], reduced_rhs).solution
            candidate = x
            for i in range(j + 1):
                candidate = candidate + y[i] * solution_basis[i]
            candidate_residual = rhs - matvec(candidate)
            matvecs += 1
            candidate_norm = inner.norm(candidate_residual)
            iterations += 1
            used = j + 1
            if fixed_iterations is None and bool((candidate_norm <= target).detach().cpu()):
                return candidate, iterations, matvecs, rhs_norm, candidate_norm, True
        if y is None:  # pragma: no cover - cycle is always positive.
            break
        for i in range(used):
            x = x + y[i] * solution_basis[i]
        residual = rhs - matvec(x)
        matvecs += 1
        residual_norm = inner.norm(residual)
        if fixed_iterations is None and bool((residual_norm <= target).detach().cpu()):
            return x, iterations, matvecs, rhs_norm, residual_norm, True

    final_residual = rhs - matvec(x)
    matvecs += 1
    final_norm = inner.norm(final_residual)
    return x, iterations, matvecs, rhs_norm, final_norm, bool((final_norm <= target).detach().cpu())


def _gmres(
    matvec: Callable[[State], State],
    rhs: State,
    inner: _StateInnerProduct,
    *,
    rtol: float,
    atol: float,
    max_iterations: int,
    restart: int,
    reorthogonalize: bool,
    fixed_iterations: int | None = None,
    preconditioner: Callable[[State], State] | None = None,
    recycle: _KrylovRecycle | None = None,
) -> _GMRESInfo:
    """Restarted flexible GMRES with incremental Givens residual estimates.

    Adaptive iterations use the donor's incremental least-squares update: a
    true matrix residual is evaluated once per restart cycle rather than once
    per Arnoldi column. The full ARK-stage residual remains the acceptance
    criterion in :func:`_hydrostatic_schur_solve`. Fixed work retains the
    prior unrolled path so it stays a predictable differentiable contract.
    """

    if fixed_iterations is not None:
        solution, iterations, matvecs, initial, final, converged = _gmres_fixed_work(
            matvec,
            rhs,
            inner,
            rtol=rtol,
            atol=atol,
            max_iterations=max_iterations,
            restart=restart,
            reorthogonalize=reorthogonalize,
            fixed_iterations=fixed_iterations,
            preconditioner=preconditioner,
        )
        return _GMRESInfo(
            solution=solution,
            iterations=iterations,
            matvecs=matvecs,
            preconditioner_applications=iterations if preconditioner is not None else 0,
            recycled_dimension=0,
            initial_residual=initial,
            final_residual=final,
            converged=converged,
            recycle=None,
        )

    x = rhs.zeros_like()
    rhs_norm = inner.norm(rhs)
    target = torch.maximum(
        torch.as_tensor(float(atol), dtype=rhs_norm.dtype, device=rhs_norm.device),
        torch.as_tensor(float(rtol), dtype=rhs_norm.dtype, device=rhs_norm.device) * rhs_norm,
    )
    recycled_dimension = 0
    if recycle is None:
        residual = rhs
    else:
        projected_rhs = torch.stack([inner.dot(vector, rhs) for vector in recycle.residual_basis])
        coefficients = torch.linalg.lstsq(recycle.hessenberg, projected_rhs).solution
        for coefficient, vector in zip(coefficients, recycle.solution_basis, strict=True):
            x = x + coefficient * vector
        projected_operator = rhs.zeros_like()
        for coefficient, vector in zip(
            recycle.hessenberg @ coefficients,
            recycle.residual_basis,
            strict=True,
        ):
            projected_operator = projected_operator + coefficient * vector
        residual = rhs - projected_operator
        recycled_dimension = len(recycle.solution_basis)
    iterations = 0
    matvecs = 0
    preconditioner_applications = 0
    recycle_out = recycle

    while iterations < int(max_iterations):
        beta = inner.norm(residual)
        if bool((beta == 0.0).detach().cpu()):
            return _GMRESInfo(
                x,
                iterations,
                matvecs,
                preconditioner_applications,
                recycled_dimension,
                beta,
                beta,
                True,
                recycle_out,
            )
        if bool((beta <= target).detach().cpu()):
            return _GMRESInfo(
                x,
                iterations,
                matvecs,
                preconditioner_applications,
                recycled_dimension,
                beta,
                beta,
                True,
                recycle_out,
            )

        width = min(int(restart), int(max_iterations) - iterations)
        basis: list[State] = [residual / beta]
        solution_basis: list[State] = []
        hessenberg = torch.zeros((width + 1, width), dtype=beta.dtype, device=beta.device)
        zero = beta * 0.0
        rotated_rhs = [beta] + [zero for _ in range(width)]
        rotation_cosines: list[torch.Tensor] = []
        rotation_sines: list[torch.Tensor] = []
        triangular_columns: list[list[torch.Tensor]] = []
        active_columns = 0

        for column in range(width):
            solution_vector = basis[column] if preconditioner is None else preconditioner(basis[column])
            if preconditioner is not None:
                preconditioner_applications += 1
            solution_basis.append(solution_vector)
            work = matvec(solution_vector)
            matvecs += 1
            for row, vector in enumerate(basis):
                coefficient = inner.dot(vector, work)
                hessenberg[row, column] = coefficient
                work = work - coefficient * vector
            if reorthogonalize:
                for row, vector in enumerate(basis):
                    coefficient = inner.dot(vector, work)
                    hessenberg[row, column] = hessenberg[row, column] + coefficient
                    work = work - coefficient * vector

            next_norm = inner.norm(work)
            hessenberg[column + 1, column] = next_norm
            arnoldi_breakdown = bool((next_norm <= torch.finfo(beta.dtype).eps).detach().cpu())
            if not arnoldi_breakdown:
                basis.append(work / next_norm)

            column_values = [hessenberg[row, column] for row in range(column + 2)]
            for row in range(column):
                upper, lower = column_values[row], column_values[row + 1]
                cosine, sine = rotation_cosines[row], rotation_sines[row]
                column_values[row] = cosine * upper + sine * lower
                column_values[row + 1] = -sine * upper + cosine * lower
            diagonal, subdiagonal = column_values[column], column_values[column + 1]
            radius = torch.hypot(diagonal, subdiagonal)
            if bool((radius <= torch.finfo(beta.dtype).eps).detach().cpu()):
                cosine, sine = torch.ones_like(radius), torch.zeros_like(radius)
            else:
                cosine, sine = diagonal / radius, subdiagonal / radius
            rotation_cosines.append(cosine)
            rotation_sines.append(sine)
            column_values[column] = cosine * diagonal + sine * subdiagonal
            column_values[column + 1] = zero
            triangular_columns.append(column_values[: column + 1])
            rotated_value = rotated_rhs[column]
            rotated_rhs[column] = cosine * rotated_value
            rotated_rhs[column + 1] = -sine * rotated_value

            iterations += 1
            active_columns = column + 1
            if bool((rotated_rhs[active_columns].abs() <= target).detach().cpu()) or arnoldi_breakdown:
                break

        coefficients = [zero for _ in range(active_columns)]
        for row in range(active_columns - 1, -1, -1):
            value = rotated_rhs[row]
            for column in range(row + 1, active_columns):
                value = value - triangular_columns[column][row] * coefficients[column]
            diagonal = triangular_columns[row][row]
            coefficients[row] = (
                zero
                if bool((diagonal.abs() <= torch.finfo(beta.dtype).eps).detach().cpu())
                else value / diagonal
            )
        for coefficient, vector in zip(coefficients, solution_basis, strict=True):
            x = x + coefficient * vector
        recycle_out = _KrylovRecycle(
            residual_basis=tuple(basis),
            solution_basis=tuple(solution_basis),
            hessenberg=hessenberg[: len(basis), :active_columns].clone(),
        )
        residual = rhs - matvec(x)
        matvecs += 1
        final = inner.norm(residual)
        if bool((final <= target).detach().cpu()):
            return _GMRESInfo(
                x,
                iterations,
                matvecs,
                preconditioner_applications,
                recycled_dimension,
                rhs_norm,
                final,
                True,
                recycle_out,
            )

    final = inner.norm(residual)
    return _GMRESInfo(
        x,
        iterations,
        matvecs,
        preconditioner_applications,
        recycled_dimension,
        rhs_norm,
        final,
        bool((final <= target).detach().cpu()),
        recycle_out,
    )


class _HydrostaticSchurSystem:
    """Exact reduced factorization of the analytic gravity-wave stage block."""

    def __init__(self, operator: DryPELinearOperator, factor: float) -> None:
        if operator.implicit_operator != "gravity_wave":
            raise ValueError("hydrostatic_schur requires implicit_operator='gravity_wave'")
        self.operator = operator
        self.factor = float(factor)
        self.anchor = operator.anchor
        self.zero = self.anchor.zeros_like()

    def make_state(self, u: torch.Tensor, mtheta: torch.Tensor, ps: torch.Tensor) -> State:
        return State(
            u,
            mtheta,
            ps,
            None if self.anchor.tracer_m is None else torch.zeros_like(self.anchor.tracer_m.values),
            mesh=self.anchor.mesh,
            column=self.anchor.column,
            attrs=self.anchor.attrs,
        )

    def thermodynamic(self, mtheta: torch.Tensor, ps: torch.Tensor) -> State:
        return self.make_state(self.zero.u_e.values, mtheta, ps)

    def wind_response(self, value: State) -> torch.Tensor:
        return self.operator.gravity_wave_wind(value.mtheta_i.values, value.ps.values)

    def thermodynamic_response(self, u: torch.Tensor) -> State:
        mtheta, ps = self.operator.gravity_wave_thermodynamic(u)
        return self.thermodynamic(mtheta, ps)

    def rhs(self, value: State) -> State:
        """Eliminate wind from the full stage right-hand side."""

        wind_rhs = self.thermodynamic_response(value.u_e.values)
        return self.thermodynamic(
            value.mtheta_i.values + self.factor * wind_rhs.mtheta_i.values,
            value.ps.values + self.factor * wind_rhs.ps.values,
        )

    def matvec(self, value: State) -> State:
        """Apply ``I - factor**2 H G`` to a thermodynamic perturbation."""

        feedback = self.thermodynamic_response(self.wind_response(value))
        return self.thermodynamic(
            value.mtheta_i.values - self.factor * self.factor * feedback.mtheta_i.values,
            value.ps.values - self.factor * self.factor * feedback.ps.values,
        )

    def back_substitute(self, rhs: State, thermodynamic: State) -> State:
        """Recover wind after solving the reduced thermodynamic system."""

        return State(
            rhs.u_e.values + self.factor * self.wind_response(thermodynamic),
            thermodynamic.mtheta_i.values,
            thermodynamic.ps.values,
            # The analytic gravity-wave block has zero tracer tendency.
            # Its exact stage inverse is therefore the identity on tracers.
            None if rhs.tracer_m is None else rhs.tracer_m.values,
            mesh=self.anchor.mesh,
            column=self.anchor.column,
            attrs=self.anchor.attrs,
        )


@dataclass(frozen=True)
class _ModalHelmholtzSetup:
    """Cached vertical-mode coefficients for a geometric Schur V-cycle."""

    transform: torch.Tensor
    reconstruct: torch.Tensor
    diagonal_eigenvalues: torch.Tensor
    offdiagonal_eigenvalues: torch.Tensor
    partner_indices: torch.Tensor
    laplacian_diagonals: tuple[torch.Tensor, ...]
    coarse_horizontal_eigenvalues: torch.Tensor
    coarse_horizontal_eigenvectors: torch.Tensor
    coarse_sqrt_area: torch.Tensor


class _MultilevelHydrostaticSchurPreconditioner:
    """Modal geometric V-cycle for the reduced analytic Schur system.

    A coarse VFS state supplies the horizontal/vertical coupling fit; the
    finest reduced system remains the exact frozen analytic Schur operator.
    This is therefore a right preconditioner, not a replacement equation or a
    new dynamics tendency.
    """

    def __init__(
        self,
        system: _HydrostaticSchurSystem,
        hierarchy: VoronoiMeshHierarchy,
        inner: _StateInnerProduct,
        *,
        smoothing_iterations: int,
        jacobi_weight: float,
        modal_setup: _ModalHelmholtzSetup | None = None,
    ) -> None:
        if hierarchy.meshes[0] is not system.anchor.mesh:
            raise ValueError("multilevel hierarchy does not match the Schur operator mesh")
        self.system = system
        self.hierarchy = hierarchy
        self.inner = inner
        self.smoothing_iterations = int(smoothing_iterations)
        self.jacobi_weight = float(jacobi_weight)
        self.mtheta_scale = inner.mtheta_scale
        self.ps_scale = inner.ps_scale
        if modal_setup is None:
            self._initialize_modal_helmholtz()
        else:
            self._load_modal_setup(modal_setup)

    def modal_setup(self) -> _ModalHelmholtzSetup:
        return _ModalHelmholtzSetup(
            self.modal_transform,
            self.modal_reconstruct,
            self.modal_diagonal_eigenvalues,
            self.modal_offdiagonal_eigenvalues,
            self.modal_partner_indices,
            self.modal_laplacian_diagonals,
            self.coarse_horizontal_eigenvalues,
            self.coarse_horizontal_eigenvectors,
            self.coarse_sqrt_area,
        )

    def _load_modal_setup(self, setup: _ModalHelmholtzSetup) -> None:
        self.modal_transform = setup.transform
        self.modal_reconstruct = setup.reconstruct
        self.modal_diagonal_eigenvalues = setup.diagonal_eigenvalues
        self.modal_offdiagonal_eigenvalues = setup.offdiagonal_eigenvalues
        self.modal_partner_indices = setup.partner_indices
        self.modal_laplacian_diagonals = setup.laplacian_diagonals
        self.coarse_horizontal_eigenvalues = setup.coarse_horizontal_eigenvalues
        self.coarse_horizontal_eigenvectors = setup.coarse_horizontal_eigenvectors
        self.coarse_sqrt_area = setup.coarse_sqrt_area

    @staticmethod
    def _scalar_laplacian(mesh, value: torch.Tensor) -> torch.Tensor:
        return -(mesh.radius * mesh.radius) * mesh.div(mesh.grad(value))

    @staticmethod
    def _scalar_laplacian_diagonal(mesh, reference: torch.Tensor) -> torch.Tensor:
        edge_coefficient = (mesh.dv_edge / mesh.dc_edge)[mesh.safe_edges_on_cell]
        edge_coefficient = torch.where(mesh.valid_edges_on_cell, edge_coefficient, torch.zeros_like(edge_coefficient))
        return (
            mesh.radius
            * mesh.radius
            * edge_coefficient.sum(dim=-1)
            / mesh.cell_area
        ).to(dtype=reference.dtype, device=reference.device)

    def _pack_normalized(self, value: State) -> torch.Tensor:
        return torch.cat(
            (
                value.mtheta_i.values / self.mtheta_scale,
                value.ps.values / self.ps_scale,
            ),
            dim=-1,
        )

    def _unpack_normalized(self, value: torch.Tensor) -> State:
        return self.system.thermodynamic(
            value[..., :-1] * self.mtheta_scale,
            value[..., -1:] * self.ps_scale,
        )

    def _coarse_system(self) -> _HydrostaticSchurSystem:
        anchor = self.system.anchor
        for transfer, mesh in zip(self.hierarchy.transfers, self.hierarchy.meshes[1:]):
            mtheta = transfer.restrict(anchor.mtheta_i.values)
            ps = transfer.restrict(anchor.ps.values)
            tracer = (
                None
                if anchor.tracer_m is None
                else transfer.restrict(
                    anchor.tracer_m.values,
                    horizontal_axis=anchor.tracer_m.horizontal_axis,
                )
            )
            assert isinstance(mtheta, torch.Tensor) and isinstance(ps, torch.Tensor)
            assert tracer is None or isinstance(tracer, torch.Tensor)
            column = anchor.column
            if getattr(column.phi_s, "ndim", 0) > 0:
                phi_s = transfer.restrict(column.phi_s)
                assert isinstance(phi_s, torch.Tensor)
                column = type(column)(
                    column.n_levels,
                    column.A_interface,
                    column.B_interface,
                    phi_s=phi_s,
                    phi_full_mode=column.phi_full_mode,
                    R_d=column.R_d,
                    c_p=column.c_p,
                    p0=column.p0,
                    g=column.g,
                ).to(device=mtheta.device)
            u_shape = (*anchor.u_e.values.shape[:-2], mesh.nEdges, anchor.u_e.values.shape[-1])
            anchor = State(
                torch.zeros(u_shape, dtype=anchor.u_e.values.dtype, device=anchor.u_e.values.device),
                mtheta,
                ps,
                tracer,
                mesh=mesh,
                column=column,
                attrs=self.system.anchor.attrs,
            )
        operator = DryPELinearOperator(
            self.system.operator.dynamics,
            anchor,
            implicit_operator="gravity_wave",
            tendency=self.system.operator.tendency,
            jvp_tendency=self.system.operator.jvp_tendency,
        )
        return _HydrostaticSchurSystem(operator, self.system.factor)

    def _initialize_modal_helmholtz(self) -> None:
        coarse_system = self._coarse_system()
        grid = self.hierarchy.meshes[-1]
        anchor = coarse_system.anchor
        n_interfaces = anchor.mtheta_i.values.shape[-1]
        n_modes = n_interfaces + 1
        phi = grid.cell_xyz[:, 0].to(dtype=anchor.ps.values.dtype, device=anchor.ps.values.device).unsqueeze(-1)
        laplacian_phi = self._scalar_laplacian(grid, phi)
        area = grid.cell_area.to(dtype=phi.dtype, device=phi.device).unsqueeze(-1)
        area = area / area.sum().clamp_min(torch.finfo(phi.dtype).tiny)
        denominator = (laplacian_phi.square() * area).sum().clamp_min(torch.finfo(phi.dtype).tiny)
        columns: list[torch.Tensor] = []
        for mode in range(n_modes):
            mtheta = torch.zeros_like(anchor.mtheta_i.values)
            ps = torch.zeros_like(anchor.ps.values)
            if mode < n_interfaces:
                mtheta[..., :, mode] = phi[..., 0] * self.mtheta_scale
            else:
                ps[..., :, 0] = phi[..., 0] * self.ps_scale
            value = coarse_system.thermodynamic(mtheta, ps)
            response = self._pack_normalized(coarse_system.matvec(value) - value)
            coefficient = (response * laplacian_phi * area).sum(dim=-2) / denominator
            while coefficient.ndim > 1:
                coefficient = coefficient.mean(dim=0)
            columns.append(coefficient)
        coupling = torch.stack(columns, dim=1).detach()
        eigenvalues, eigenvectors = torch.linalg.eig(coupling.to(device="cpu"))
        eigenvectors_inverse = torch.linalg.inv(eigenvectors)
        tolerance = 128.0 * torch.finfo(coupling.dtype).eps * eigenvalues.abs().max().clamp_min(1.0)
        transform_rows: list[torch.Tensor] = []
        reconstruct_columns: list[torch.Tensor] = []
        diagonal_values: list[torch.Tensor] = []
        offdiagonal_values: list[torch.Tensor] = []
        partner_indices: list[int] = []
        for mode in range(n_modes):
            imaginary = eigenvalues[mode].imag
            if float(imaginary.abs()) <= float(tolerance):
                channel = len(transform_rows)
                transform_rows.append(eigenvectors_inverse[mode].real)
                reconstruct_columns.append(eigenvectors[:, mode].real)
                diagonal_values.append(eigenvalues[mode].real)
                offdiagonal_values.append(torch.zeros_like(imaginary))
                partner_indices.append(channel)
            elif float(imaginary) > 0.0:
                channel = len(transform_rows)
                transform_rows.extend((eigenvectors_inverse[mode].real, eigenvectors_inverse[mode].imag))
                reconstruct_columns.extend((2.0 * eigenvectors[:, mode].real, -2.0 * eigenvectors[:, mode].imag))
                diagonal_values.extend((eigenvalues[mode].real, eigenvalues[mode].real))
                offdiagonal_values.extend((-imaginary, imaginary))
                partner_indices.extend((channel + 1, channel))
        if len(transform_rows) != n_modes:
            raise RuntimeError("vertical Schur modes did not form real conjugate blocks")
        self.modal_transform = torch.stack(transform_rows).to(dtype=phi.dtype, device=phi.device)
        self.modal_reconstruct = torch.stack(reconstruct_columns, dim=1).to(dtype=phi.dtype, device=phi.device)
        self.modal_diagonal_eigenvalues = torch.stack(diagonal_values).to(dtype=phi.dtype, device=phi.device)
        self.modal_offdiagonal_eigenvalues = torch.stack(offdiagonal_values).to(dtype=phi.dtype, device=phi.device)
        self.modal_partner_indices = torch.as_tensor(partner_indices, dtype=torch.long, device=phi.device)
        self.modal_laplacian_diagonals = tuple(
            self._scalar_laplacian_diagonal(mesh, phi) for mesh in self.hierarchy.meshes
        )

        coarse_count = grid.nCells
        identity = torch.eye(coarse_count, dtype=phi.dtype, device=phi.device)
        laplacian_matrix = self._scalar_laplacian(grid, identity).mT
        sqrt_area = grid.cell_area.to(dtype=phi.dtype, device=phi.device).sqrt()
        symmetric_laplacian = sqrt_area.unsqueeze(-1) * laplacian_matrix / sqrt_area.unsqueeze(0).clamp_min(
            torch.finfo(phi.dtype).tiny
        )
        symmetric_laplacian = 0.5 * (symmetric_laplacian + symmetric_laplacian.mT)
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_laplacian.to(device="cpu"))
        self.coarse_horizontal_eigenvalues = eigenvalues.clamp_min(0.0).to(dtype=phi.dtype, device=phi.device)
        self.coarse_horizontal_eigenvectors = eigenvectors.to(dtype=phi.dtype, device=phi.device)
        self.coarse_sqrt_area = sqrt_area

    def _modal_matvec(self, level: int, value: torch.Tensor) -> torch.Tensor:
        laplacian = self._scalar_laplacian(self.hierarchy.meshes[level], value)
        diagonal = self.modal_diagonal_eigenvalues.to(dtype=value.dtype, device=value.device)
        offdiagonal = self.modal_offdiagonal_eigenvalues.to(dtype=value.dtype, device=value.device)
        partners = self.modal_partner_indices.to(device=value.device)
        return value + diagonal * laplacian + offdiagonal * laplacian[..., partners]

    def _modal_coarse_solve(self, rhs: torch.Tensor) -> torch.Tensor:
        sqrt_area = self.coarse_sqrt_area.to(dtype=rhs.dtype, device=rhs.device)
        area_view = sqrt_area.reshape((1,) * (rhs.ndim - 2) + (-1, 1))
        weighted = rhs * area_view
        eigenvectors = self.coarse_horizontal_eigenvectors.to(dtype=rhs.dtype, device=rhs.device)
        coefficients = torch.einsum("ch,...cm->...mh", eigenvectors, weighted)
        horizontal = self.coarse_horizontal_eigenvalues.to(dtype=rhs.dtype, device=rhs.device)
        diagonal = 1.0 + self.modal_diagonal_eigenvalues.to(dtype=rhs.dtype, device=rhs.device).unsqueeze(-1) * horizontal
        offdiagonal = self.modal_offdiagonal_eigenvalues.to(dtype=rhs.dtype, device=rhs.device).unsqueeze(-1) * horizontal
        partners = self.modal_partner_indices.to(device=rhs.device)
        partner_diagonal = diagonal[partners]
        partner_offdiagonal = offdiagonal[partners]
        scale = diagonal * partner_diagonal - offdiagonal * partner_offdiagonal
        coefficients = (partner_diagonal * coefficients - offdiagonal * coefficients[..., partners, :]) / scale
        weighted_solution = torch.einsum("ch,...mh->...cm", eigenvectors, coefficients)
        return weighted_solution / area_view

    def _modal_v_cycle(self, level: int, rhs: torch.Tensor) -> torch.Tensor:
        if level == len(self.hierarchy.meshes) - 1:
            return self._modal_coarse_solve(rhs)
        laplacian_diagonal = self.modal_laplacian_diagonals[level].to(dtype=rhs.dtype, device=rhs.device)
        diagonal = 1.0 + laplacian_diagonal.unsqueeze(-1) * self.modal_diagonal_eigenvalues.to(
            dtype=rhs.dtype, device=rhs.device
        )
        offdiagonal = laplacian_diagonal.unsqueeze(-1) * self.modal_offdiagonal_eigenvalues.to(
            dtype=rhs.dtype, device=rhs.device
        )
        partners = self.modal_partner_indices.to(device=rhs.device)
        partner_diagonal = diagonal[..., partners]
        partner_offdiagonal = offdiagonal[..., partners]
        diagonal_scale = diagonal * partner_diagonal - offdiagonal * partner_offdiagonal

        def smooth(current: torch.Tensor) -> torch.Tensor:
            return self.jacobi_weight * (
                partner_diagonal * current - offdiagonal * current[..., partners]
            ) / diagonal_scale

        correction = torch.zeros_like(rhs)
        residual = rhs
        for _ in range(self.smoothing_iterations):
            increment = smooth(residual)
            correction = correction + increment
            residual = residual - self._modal_matvec(level, increment)
        coarse_rhs = self.hierarchy.transfers[level].restrict(residual)
        assert isinstance(coarse_rhs, torch.Tensor)
        coarse_correction = self._modal_v_cycle(level + 1, coarse_rhs)
        prolonged = self.hierarchy.transfers[level].prolong(coarse_correction)
        assert isinstance(prolonged, torch.Tensor)
        correction = correction + prolonged
        residual = residual - self._modal_matvec(level, prolonged)
        for _ in range(self.smoothing_iterations):
            increment = smooth(residual)
            correction = correction + increment
            residual = residual - self._modal_matvec(level, increment)
        return correction

    def __call__(self, value: State) -> State:
        normalized = self._pack_normalized(value)
        modal = torch.einsum(
            "ij,...cj->...ci",
            self.modal_transform.to(dtype=normalized.dtype, device=normalized.device),
            normalized,
        )
        modal_correction = self._modal_v_cycle(0, modal)
        normalized_correction = torch.einsum(
            "ij,...cj->...ci",
            self.modal_reconstruct.to(dtype=modal_correction.dtype, device=modal_correction.device),
            modal_correction,
        )
        return self._unpack_normalized(normalized_correction)


def _hydrostatic_schur_solve(
    operator: DryPELinearOperator,
    rhs: State,
    inner: _StateInnerProduct,
    *,
    factor: float,
    rtol: float,
    atol: float,
    max_iterations: int,
    restart: int,
    reorthogonalize: bool,
    tolerance_safety: float,
    fixed_iterations: int | None,
    preconditioner: Callable[[State], State] | None = None,
    recycle: _KrylovRecycle | None = None,
) -> _GMRESInfo:
    """Solve the analytic gravity-wave stage by block Schur elimination.

    For the validated gravity-wave block, ``L = [[0, G], [H, 0]]`` where
    ``G`` maps thermodynamic perturbations to wind and ``H`` maps wind to the
    centered thermodynamic/continuity response.  Eliminating wind yields the
    lower-dimensional exact system ``(I - factor**2 H G) z = rhs_z + factor H
    rhs_u``.  This is an exact factorization of the *ported analytic block*;
    only the iterative reduced solve is approximate.
    """

    system = _HydrostaticSchurSystem(operator, factor)
    reduced_rhs = system.rhs(rhs)

    full_rhs_norm = inner.norm(rhs)
    full_target = torch.maximum(
        torch.as_tensor(float(atol), dtype=full_rhs_norm.dtype, device=full_rhs_norm.device),
        torch.as_tensor(float(rtol), dtype=full_rhs_norm.dtype, device=full_rhs_norm.device) * full_rhs_norm,
    )
    reduced_rhs_norm = inner.norm(reduced_rhs)
    reduced_rtol = float(
        (
            torch.as_tensor(float(tolerance_safety), dtype=full_target.dtype, device=full_target.device)
            * full_target
            / reduced_rhs_norm.clamp_min(inner.tiny)
        )
        .clamp(min=torch.finfo(full_target.dtype).eps, max=1.0)
        .detach()
        .cpu()
    )
    reduced = _gmres(
        system.matvec,
        reduced_rhs,
        inner,
        rtol=reduced_rtol,
        atol=0.0,
        max_iterations=max_iterations,
        restart=restart,
        reorthogonalize=reorthogonalize,
        fixed_iterations=fixed_iterations,
        preconditioner=preconditioner,
        recycle=recycle,
    )
    correction = system.back_substitute(rhs, reduced.solution)
    full_residual = rhs - (correction - factor * operator.apply(correction))
    final_norm = inner.norm(full_residual)
    return _GMRESInfo(
        solution=correction,
        iterations=reduced.iterations,
        matvecs=reduced.matvecs + 1,
        preconditioner_applications=reduced.preconditioner_applications,
        recycled_dimension=reduced.recycled_dimension,
        initial_residual=full_rhs_norm,
        final_residual=final_norm,
        converged=bool((final_norm <= full_target).detach().cpu()),
        recycle=reduced.recycle,
    )


class DryPEIMEXARK2:
    """Second-order frozen-JVP ARK2 stepper for Field-native VFS states.

    ``explicit_tendency_providers`` add source terms to the nonlinear,
    explicit ARK residual at every stage.  This is the appropriate path for
    stateless (or externally managed) atmospheric physics such as a
    tendency-form :class:`SCMPhysics`: the hydrostatic gravity-wave operator
    remains owned solely by ``dynamics``.  Stateful split-process components
    deliberately remain outside this interface because an ARK stage has no
    unambiguous process-memory update contract.
    """

    def __init__(
        self,
        dynamics,
        solver_config: IMEXSolverConfig | None = None,
        *,
        explicit_tendency_providers=None,
    ) -> None:
        self.dynamics = dynamics
        self.solver_config = solver_config or IMEXSolverConfig()
        self.solver_config.validate()
        self.tableau = ARK2_TABLEAU
        if explicit_tendency_providers is None:
            self.explicit_tendency_providers = ()
        elif isinstance(explicit_tendency_providers, (list, tuple)):
            self.explicit_tendency_providers = tuple(explicit_tendency_providers)
        else:
            self.explicit_tendency_providers = (explicit_tendency_providers,)
        self.last_step_info: IMEXStepInfo | None = None
        self.last_guard_used_fallback = False
        self.guarded_fixed_steps = 0
        self.guarded_fallback_steps = 0
        self._compiled_dynamics = None
        self._tendency: Callable[[State], State] = dynamics.tendency
        self._multilevel_hierarchies: dict[int, VoronoiMeshHierarchy] = {}
        self._multilevel_modal_setups: dict[tuple[object, ...], _ModalHelmholtzSetup] = {}
        if self.solver_config.compile_kernels:
            self._tendency = self._compile_tendency()

    @staticmethod
    def _provider_tendency(provider, state: State) -> State:
        tendency = getattr(provider, "tendency", None)
        value = tendency(state) if callable(tendency) else provider(state) if callable(provider) else None
        if not isinstance(value, State):
            name = type(provider).__name__
            raise TypeError(f"explicit tendency provider {name} must return a State")
        return value

    def _explicit_tendency(self, state: State) -> State:
        """Evaluate dynamics plus ARK-explicit source tendencies.

        The frozen linear operator is intentionally *not* differentiated
        through these providers.  They are part of ``F`` in the ARK split;
        only the dry/virtual hydrostatic gravity-wave block is ``L``.
        """

        total = self._tendency(state)
        for provider in self.explicit_tendency_providers:
            total = total + self._provider_tendency(provider, state)
        return total

    def _multilevel_hierarchy(self, state: State) -> VoronoiMeshHierarchy:
        """Return the static geometric hierarchy associated with ``state.mesh``."""

        if state.mesh is None:
            raise ValueError("multilevel_hydrostatic_schur requires a State-bound mesh")
        key = id(state.mesh)
        cached = self._multilevel_hierarchies.get(key)
        if cached is not None and cached.meshes[0] is state.mesh:
            return cached
        hierarchy = VoronoiMeshHierarchy.build(
            state.mesh,
            coarsening=self.solver_config.multilevel_coarsening,
            max_levels=self.solver_config.multilevel_max_levels,
            minimum_frequency=self.solver_config.multilevel_min_frequency,
            stencil_size=self.solver_config.multilevel_stencil_size,
        )
        self._multilevel_hierarchies[key] = hierarchy
        return hierarchy

    def _multilevel_preconditioner(
        self,
        operator: DryPELinearOperator,
        inner: _StateInnerProduct,
        *,
        factor: float,
        hierarchy: VoronoiMeshHierarchy,
    ) -> _MultilevelHydrostaticSchurPreconditioner:
        """Build or reuse the frozen modal fit for one ARK diagonal factor."""

        key = (
            id(operator.anchor.mesh),
            id(operator.anchor.column),
            id(operator.dynamics),
            float(factor),
            operator.anchor.mtheta_i.values.shape[-1],
            operator.anchor.mtheta_i.values.dtype,
            operator.anchor.mtheta_i.values.device.type,
            operator.anchor.mtheta_i.values.device.index,
        )
        setup = self._multilevel_modal_setups.get(key)
        preconditioner = _MultilevelHydrostaticSchurPreconditioner(
            _HydrostaticSchurSystem(operator, factor),
            hierarchy,
            inner,
            smoothing_iterations=self.solver_config.multilevel_smoothing_iterations,
            jacobi_weight=self.solver_config.multilevel_jacobi_weight,
            modal_setup=setup,
        )
        if setup is None:
            self._multilevel_modal_setups[key] = preconditioner.modal_setup()
        return preconditioner

    def _compile_tendency(self) -> Callable[[State], State]:
        """Compile a validation-free clone of a stateless dry tendency.

        The canonical dynamics performs host-side physical-state checks, which
        are deliberately kept in eager kernels and cannot be captured by a
        full graph.  A deep-copied, parameter-free dynamics module with those
        checks disabled gives ordinary IMEX tendency evaluations an equivalent
        tensor kernel while preserving the caller's validated dynamics object.
        The exact forward-mode JVP remains eager because current Dynamo cannot
        trace the dual-tensor column views it receives. This is an experimental
        inference optimization, not a compiled training path for learned
        dynamics.
        """

        if not isinstance(self.dynamics, torch.nn.Module):
            raise TypeError("compile_kernels requires a torch.nn.Module dynamics object")
        if any(parameter.requires_grad for parameter in self.dynamics.parameters()):
            raise ValueError(
                "compile_kernels supports parameter-free dynamics only; use eager IMEX for learned dynamics"
            )
        if not hasattr(self.dynamics, "validate_state"):
            raise ValueError(
                "compile_kernels requires dynamics with a validate_state flag so physical checks stay eager"
            )
        compiled_dynamics = copy.deepcopy(self.dynamics)
        compiled_dynamics.validate_state = False
        self._compiled_dynamics = compiled_dynamics
        kwargs: dict[str, str | bool] = {"fullgraph": True}
        if self.solver_config.compile_backend is not None:
            kwargs["backend"] = self.solver_config.compile_backend
        if self.solver_config.compile_mode is not None:
            kwargs["mode"] = self.solver_config.compile_mode
        return torch.compile(compiled_dynamics.tendency, **kwargs)

    def _step(self, state: State, dt: float, *, fixed_iterations: int | None) -> tuple[State, IMEXStepInfo]:
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and positive, got {dt!r}")
        if self.solver_config.preconditioner in {
            "hydrostatic_schur",
            "multilevel_hydrostatic_schur",
        } and self.solver_config.implicit_operator != "gravity_wave":
            raise ValueError(
                f"preconditioner={self.solver_config.preconditioner!r} requires "
                "implicit_operator='gravity_wave'"
            )
        operator = DryPELinearOperator(
            self.dynamics,
            state,
            implicit_operator=self.solver_config.implicit_operator,
            tendency=self._tendency,
            jvp_tendency=self.dynamics.tendency,
        )
        if self.solver_config.compile_kernels:
            operator.compile_analytic_gravity_wave(
                backend=self.solver_config.compile_backend,
                mode=self.solver_config.compile_mode,
            )
        inner = _StateInnerProduct(state)
        stages: list[State] = []
        explicit_terms: list[State] = []
        implicit_terms: list[State] = []
        infos: list[IMEXStageInfo] = []
        stage_preconditioners: dict[float, _MultilevelHydrostaticSchurPreconditioner] = {}
        krylov_recycle: _KrylovRecycle | None = None
        recycle_factor: float | None = None
        hierarchy = (
            self._multilevel_hierarchy(state)
            if self.solver_config.preconditioner == "multilevel_hydrostatic_schur"
            else None
        )
        for i in range(self.tableau.n_stages):
            known = state.zeros_like()
            for j in range(i):
                known = known + dt * (
                    self.tableau.explicit[i][j] * explicit_terms[j]
                    + self.tableau.implicit[i][j] * implicit_terms[j]
                )
            diagonal = self.tableau.implicit[i][i]
            # Keep the frozen base tendency in the explicit component.  The
            # donor's qualified ARK path treats only ``L_n (y - y_n)`` as the
            # implicit tendency, not the affine ``F_n + L_n (y - y_n)``.
            # Both are legitimate splits, but they have materially different
            # long-wave phase error at the one-hour production step.
            rhs = known
            if diagonal == 0.0:
                correction = rhs
                zero_norm = inner.norm(rhs)
                solve = _GMRESInfo(
                    solution=correction,
                    iterations=0,
                    matvecs=0,
                    preconditioner_applications=0,
                    recycled_dimension=0,
                    initial_residual=zero_norm,
                    final_residual=zero_norm,
                    converged=True,
                    recycle=None,
                )
            else:
                factor = dt * diagonal
                stage_recycle = krylov_recycle if recycle_factor == factor else None
                if self.solver_config.preconditioner == "hydrostatic_schur":
                    solve = _hydrostatic_schur_solve(
                        operator,
                        rhs,
                        inner,
                        factor=factor,
                        rtol=self.solver_config.resolved_rtol(state.u_e.values.dtype),
                        atol=self.solver_config.atol,
                        max_iterations=self.solver_config.max_iterations,
                        restart=self.solver_config.restart,
                        reorthogonalize=self.solver_config.reorthogonalize,
                        tolerance_safety=self.solver_config.schur_tolerance_safety,
                        fixed_iterations=fixed_iterations,
                        recycle=stage_recycle,
                    )
                elif self.solver_config.preconditioner == "multilevel_hydrostatic_schur":
                    assert hierarchy is not None
                    if factor not in stage_preconditioners:
                        stage_preconditioners[factor] = self._multilevel_preconditioner(
                            operator,
                            inner,
                            factor=factor,
                            hierarchy=hierarchy,
                        )
                    solve = _hydrostatic_schur_solve(
                        operator,
                        rhs,
                        inner,
                        factor=factor,
                        rtol=self.solver_config.resolved_rtol(state.u_e.values.dtype),
                        atol=self.solver_config.atol,
                        max_iterations=self.solver_config.max_iterations,
                        restart=self.solver_config.restart,
                        reorthogonalize=self.solver_config.reorthogonalize,
                        tolerance_safety=self.solver_config.schur_tolerance_safety,
                        fixed_iterations=fixed_iterations,
                        preconditioner=stage_preconditioners[factor],
                        recycle=stage_recycle,
                    )
                else:
                    solve = _gmres(
                        lambda value: value - factor * operator.apply(value),
                        rhs,
                        inner,
                        rtol=self.solver_config.resolved_rtol(state.u_e.values.dtype),
                        atol=self.solver_config.atol,
                        max_iterations=self.solver_config.max_iterations,
                        restart=self.solver_config.restart,
                        reorthogonalize=self.solver_config.reorthogonalize,
                        fixed_iterations=fixed_iterations,
                        recycle=stage_recycle,
                    )
                correction = solve.solution
                if fixed_iterations is None:
                    krylov_recycle = solve.recycle
                    recycle_factor = factor
                else:
                    krylov_recycle = None
                    recycle_factor = None
            stage = state + correction
            linear = operator.apply(correction)
            nonlinear = self._explicit_tendency(stage) - linear
            info = IMEXStageInfo(
                stage=i,
                implicit_coefficient=float(diagonal),
                iterations=solve.iterations,
                matvecs=solve.matvecs,
                preconditioner_applications=solve.preconditioner_applications,
                recycled_dimension=solve.recycled_dimension,
                initial_residual=float(solve.initial_residual.detach().cpu()),
                final_residual=float(solve.final_residual.detach().cpu()),
                relative_residual=float(
                    (solve.final_residual / solve.initial_residual.clamp_min(inner.tiny)).detach().cpu()
                ),
                converged=solve.converged,
            )
            if fixed_iterations is None and not solve.converged:
                raise IMEXConvergenceError(info)
            stages.append(stage)
            explicit_terms.append(nonlinear)
            implicit_terms.append(linear)
            infos.append(info)
        next_state = state
        for weight, explicit, implicit in zip(self.tableau.weights, explicit_terms, implicit_terms, strict=True):
            next_state = next_state + dt * weight * (explicit + implicit)
        step_info = IMEXStepInfo(
            method=self.tableau.name if fixed_iterations is None else f"{self.tableau.name}-fixed-{fixed_iterations}",
            linear_operator=(
                ("frozen_analytic_conservative_gravity_wave"
                 if operator.conservative_wave is not None else "frozen_analytic_gravity_wave")
                if self.solver_config.implicit_operator == "gravity_wave"
                else "frozen_exact_jvp_coupled_pe"
            ),
            preconditioner=self.solver_config.preconditioner,
            dt=dt,
            stages=tuple(infos),
            converged=all(info.converged for info in infos),
            total_iterations=sum(info.iterations for info in infos),
            total_matvecs=sum(info.matvecs for info in infos) + operator.applications,
            total_preconditioner_applications=sum(info.preconditioner_applications for info in infos),
        )
        return next_state, step_info

    def step_with_info(self, state: State, dt: float) -> tuple[State, IMEXStepInfo]:
        next_state, info = self._step(state, dt, fixed_iterations=None)
        self.last_step_info = info
        return next_state, info

    def step(self, state: State, dt: float) -> State:
        return self.step_with_info(state, dt)[0]

    def step_fixed_with_info(self, state: State, dt: float, *, iterations: int = 4) -> tuple[State, IMEXStepInfo]:
        if iterations < 1:
            raise ValueError("iterations must be at least one")
        next_state, info = self._step(state, dt, fixed_iterations=int(iterations))
        self.last_step_info = info
        return next_state, info

    def step_fixed(self, state: State, dt: float, *, iterations: int = 4) -> State:
        return self.step_fixed_with_info(state, dt, iterations=iterations)[0]

    def step_guarded(self, state: State, dt: float, *, iterations: int = 4) -> State:
        """Accept a residual-verified fixed step or fall back to adaptive GMRES.

        Fixed work is useful for repeatable inference and future static graph
        capture, but it must never silently under-solve an implicit stage.
        This eager guard retains the adaptive ARK path as the science-safe
        fallback whenever the requested fixed iteration budget misses its
        configured residual tolerance.
        """

        fixed, info = self.step_fixed_with_info(state, dt, iterations=iterations)
        if info.converged:
            self.last_guard_used_fallback = False
            self.guarded_fixed_steps += 1
            return fixed
        self.last_guard_used_fallback = True
        self.guarded_fallback_steps += 1
        return self.step(state, dt)


__all__ = [
    "ARK2_TABLEAU",
    "DryPEIMEXARK2",
    "DryPELinearOperator",
    "IMEXConvergenceError",
    "IMEXSolverConfig",
    "IMEXStageInfo",
    "IMEXStepInfo",
]
