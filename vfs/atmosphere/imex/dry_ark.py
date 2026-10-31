"""Conservative dry wave linearization and six-stage ARK4 advancement.

At each outer step, freeze the analytic pressure/transport response L at y0.
Solve for stage increments with L(y-y0), leaving F(y)-L(y-y0) explicit. The
common weighted sum advances the pressure-weighted dry prognostic contents.
"""
from __future__ import annotations
from typing import Callable
import math
import torch
from vfs.atmosphere.state import State, interface_mass
from vfs.geometry.mesh.multilevel import VoronoiMeshHierarchy
from vfs.integrators.tableau import ARK4_TABLEAU
from vfs.atmosphere.imex.solvers import (
    IMEXSolverConfig, IMEXStageInfo, IMEXStepInfo, IMEXConvergenceError,
    _StateInnerProduct, _GMRESInfo, _KrylovRecycle, _ModalHelmholtzSetup,
    _HydrostaticSchurSystem, _MultilevelHydrostaticSchurPreconditioner,
    _values, _gmres, _hydrostatic_schur_solve,
)

class DryPELinearOperator:
    """Step-frozen analytic conservative pressure/transport wave block."""

    def __init__(
        self,
        dynamics,
        anchor: State,
        *,
        tendency: Callable[[State], State] | None = None,
    ) -> None:
        self.dynamics = dynamics
        self.anchor = anchor
        self.tendency = dynamics.tendency if tendency is None else tendency
        self.base_tendency = self.tendency(anchor)
        self._validate_gravity_wave_support()
        self.primitives = dynamics._transport_primitives(anchor)
        from vfs.atmosphere.imex.conservative_wave import ConservativePressureWave
        self.conservative_wave = ConservativePressureWave(dynamics, anchor)
        self.applications = 0

    def _validate_gravity_wave_support(self):
        from vfs.atmosphere.conservative import DryHydrostaticConservativeDynamics
        if type(self.dynamics) is not DryHydrostaticConservativeDynamics:
            raise TypeError("dry ARK4 retains the conservative dry core only")
        self.dynamics.check_state_compatible(self.anchor)


    def _transport_tangent(self, direction: State) -> dict[str, torch.Tensor]:
        """Analytic dry transport tangent in main-VFS axis order.

        Differentiate pressure mass continuity and centered scalar transport.
        Wind is edge/full (m/s), pressure is cell/1 (Pa), and potential-
        temperature content is cell/interface (Pa K).
        """

        assert self.primitives is not None
        anchor = self.anchor
        base = self.primitives
        u, mtheta, ps = _values(direction)
        d_dp = (anchor.column.B_interface[1:] - anchor.column.B_interface[:-1]).to(
            dtype=ps.dtype, device=ps.device
        ) * ps
        d_dm = interface_mass(d_dp)
        d_dp_e = anchor.cell_to_edge(d_dp)
        d_mass_flux = u * base["dp_e"] + anchor.u_e * d_dp_e
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


    def _gravity_wave_apply(self, direction: State) -> State:
        """Apply the conservative gravity-wave off-diagonal implicit block."""

        wind = self.gravity_wave_wind(direction.mtheta_i, direction.ps)
        mtheta, ps = self.gravity_wave_thermodynamic(direction.u_e)
        return State(
            wind,
            mtheta,
            ps,
            None if direction.tracer_m is None else torch.zeros_like(direction.tracer_m),
            mesh=self.anchor.mesh,
            column=self.anchor.column,
            attrs=self.anchor.attrs,
        )

    def gravity_wave_wind(self, mtheta, ps):
        """Analytic pressure-force response at fixed potential-temperature content."""
        return self.conservative_wave.wind(mtheta, ps)

    def gravity_wave_thermodynamic(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the analytic wind-to-thermodynamic gravity-wave block."""

        batch_shape = u.shape[:-2]
        wind_only = State(
            u,
            torch.zeros(
                *batch_shape,
                self.anchor.mtheta_i.shape[-2],
                self.anchor.mtheta_i.shape[-1],
                dtype=u.dtype,
                device=u.device,
            ),
            torch.zeros(
                *batch_shape,
                self.anchor.ps.shape[-2],
                self.anchor.ps.shape[-1],
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
        # The perturbation shares the frozen anchor's geometry and attributes;
        # its tensor payload is borrowed without copying or changing gradients.
        direction = State(
            direction.u_e,
            direction.mtheta_i,
            direction.ps,
            None if direction.tracer_m is None else direction.tracer_m,
            mesh=self.anchor.mesh,
            column=self.anchor.column,
            attrs=self.anchor.attrs,
        )
        return self._gravity_wave_apply(direction)


class DryPEIMEXARK:
    """Six-stage ARK4 for the conservative dry core.

    The nonlinear F is explicit except for the analytic pressure/transport
    response L(y-y0) frozen at the outer anchor. Adaptive GMRES must meet its
    true full-stage residual; no compiled/fixed-work or physics-provider path.
    """

    def __init__(
        self,
        dynamics,
        solver_config: IMEXSolverConfig | None = None,
    ) -> None:
        self.dynamics = dynamics
        self.solver_config = solver_config or IMEXSolverConfig()
        self.solver_config.validate()
        self.tableau = ARK4_TABLEAU
        self.last_step_info: IMEXStepInfo | None = None
        self._tendency: Callable[[State], State] = dynamics.tendency
        self._multilevel_hierarchies: dict[int, VoronoiMeshHierarchy] = {}
        self._multilevel_modal_setups: dict[tuple[object, ...], _ModalHelmholtzSetup] = {}

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
            operator.anchor.mtheta_i.shape[-1],
            operator.anchor.mtheta_i.dtype,
            operator.anchor.mtheta_i.device.type,
            operator.anchor.mtheta_i.device.index,
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


    def _step(self, state: State, dt: float) -> tuple[State, IMEXStepInfo]:
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and positive, got {dt!r}")
        operator = DryPELinearOperator(
            self.dynamics,
            state,
            tendency=self._tendency,
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
                        rtol=self.solver_config.resolved_rtol(state.u_e.dtype),
                        atol=self.solver_config.atol,
                        max_iterations=self.solver_config.max_iterations,
                        restart=self.solver_config.restart,
                        reorthogonalize=self.solver_config.reorthogonalize,
                        tolerance_safety=self.solver_config.schur_tolerance_safety,
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
                        rtol=self.solver_config.resolved_rtol(state.u_e.dtype),
                        atol=self.solver_config.atol,
                        max_iterations=self.solver_config.max_iterations,
                        restart=self.solver_config.restart,
                        reorthogonalize=self.solver_config.reorthogonalize,
                        tolerance_safety=self.solver_config.schur_tolerance_safety,
                        preconditioner=stage_preconditioners[factor],
                        recycle=stage_recycle,
                    )
                else:
                    solve = _gmres(
                        lambda value: value - factor * operator.apply(value),
                        rhs,
                        inner,
                        rtol=self.solver_config.resolved_rtol(state.u_e.dtype),
                        atol=self.solver_config.atol,
                        max_iterations=self.solver_config.max_iterations,
                        restart=self.solver_config.restart,
                        reorthogonalize=self.solver_config.reorthogonalize,
                        recycle=stage_recycle,
                    )
                correction = solve.solution
                krylov_recycle = solve.recycle
                recycle_factor = factor
            stage = state + correction
            linear = operator.apply(correction)
            nonlinear = self._tendency(stage) - linear
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
            if not solve.converged:
                raise IMEXConvergenceError(info)
            stages.append(stage)
            explicit_terms.append(nonlinear)
            implicit_terms.append(linear)
            infos.append(info)
        next_state = state
        for weight, explicit, implicit in zip(self.tableau.weights, explicit_terms, implicit_terms, strict=True):
            next_state = next_state + dt * weight * (explicit + implicit)
        step_info = IMEXStepInfo(
            method=self.tableau.name,
            linear_operator="frozen_analytic_conservative_gravity_wave",
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
        next_state, info = self._step(state, dt)
        self.last_step_info = info
        return next_state, info

    def step(self, state: State, dt: float) -> State:
        return self.step_with_info(state, dt)[0]
