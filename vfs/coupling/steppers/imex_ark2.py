"""Coupled atmosphere--ocean ARK2 with an atmospheric gravity-wave block.

The explicit ARK residual is the *full* ``CoupledAtmosphereOceanModel``
tendency, so atmosphere--ocean momentum/heat/freshwater exchange and ocean
dynamics are evaluated at every ARK stage.  Only the atmospheric hydrostatic
gravity-wave operator is implicit.  Stateful process components are excluded:
their restart memory needs a separate ARK-stage contract.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

import torch

from ...integrators.imex.ark2 import (
    ARK2_TABLEAU,
    DryPELinearOperator,
    IMEXConvergenceError,
    IMEXSolverConfig,
    IMEXStageInfo,
    _StateInnerProduct,
    _gmres,
    _hydrostatic_schur_solve,
)
from ...process import PhysicsState
from ..ark_stage_sources import add_scaled_ark_memory, evaluate_ark_stage_source, initial_ark_source_memory
from ..state import CoupledState, CoupledStepResult


@dataclass(frozen=True)
class CoupledIMEXStepInfo:
    """Per-step convergence data for :class:`CoupledAtmosphereOceanARK2`."""

    dt: float
    stages: tuple[IMEXStageInfo, ...]
    converged: bool
    total_iterations: int
    total_matvecs: int
    total_preconditioner_applications: int
    stage_source_names: tuple[str, ...] = ()
    stage_source_diagnostics: tuple[dict[str, dict], ...] = ()


class CoupledAtmosphereOceanARK2:
    """ARK2 split with a coupled explicit residual and atmospheric implicit L.

    This is a coupled-block method, not an atmosphere-only step routed through
    a process list: every nonlinear stage calls ``model.tendency`` on a full
    :class:`CoupledState`.  The ocean correction is an exact identity solve
    because the present implicit operator has no ocean block.
    """

    recipe_name = "coupled_atmosphere_ocean_ark2_gravity_wave"

    def __init__(
        self,
        model,
        solver_config: IMEXSolverConfig | None = None,
        *,
        stage_sources: Mapping[str, object] | list[object] | tuple[object, ...] | None = None,
    ) -> None:
        self.model = model
        self.solver_config = solver_config or IMEXSolverConfig(
            implicit_operator="gravity_wave", preconditioner="hydrostatic_schur"
        )
        self.solver_config.validate()
        if self.solver_config.implicit_operator != "gravity_wave":
            raise ValueError("CoupledAtmosphereOceanARK2 requires implicit_operator='gravity_wave'")
        if self.solver_config.preconditioner not in {"none", "hydrostatic_schur"}:
            raise ValueError(
                "CoupledAtmosphereOceanARK2 currently supports preconditioner='none' or 'hydrostatic_schur'"
            )
        if getattr(model, "atmosphere_processes", None) or getattr(model, "coupled_processes", None):
            raise ValueError(
                "CoupledAtmosphereOceanARK2 accepts a coupled tendency block only; "
                "stateful process components require an ARK-stage memory contract"
            )
        atmosphere_model = getattr(model, "atmosphere_model", None)
        dynamics = getattr(atmosphere_model, "dry", None)
        if dynamics is None:
            dynamics = getattr(atmosphere_model, "dynamics", None)
        if dynamics is None and callable(getattr(atmosphere_model, "tendency", None)):
            # ``CoupledAtmosphereOceanModel`` also accepts a dynamics object
            # directly, as used by its lightweight dycore driver.
            dynamics = atmosphere_model
        if dynamics is None:
            raise TypeError("coupled atmosphere model must expose dry hydrostatic dynamics as .dry or .dynamics")
        self.dynamics = dynamics
        if stage_sources is None:
            self.stage_sources: dict[str, object] = {}
        elif isinstance(stage_sources, Mapping):
            self.stage_sources = {str(name): source for name, source in stage_sources.items()}
        else:
            self.stage_sources = {f"source_{i}": source for i, source in enumerate(stage_sources)}
        for name, source in self.stage_sources.items():
            if not callable(getattr(source, "initial_ark_physics_state", None)) or not callable(
                getattr(source, "ark_stage_tendency", None)
            ):
                raise TypeError(
                    f"stage source {name!r} must implement initial_ark_physics_state and ark_stage_tendency"
                )
            validator = getattr(source, "validate_ark_model", None)
            if callable(validator):
                validator(model)
        self.last_step_info: CoupledIMEXStepInfo | None = None
        # Kept for map-based source controllers that need the source-free
        # coupled ARK trajectory.  Ordinary callers should use last_step_info.
        self.last_stage_states: tuple[CoupledState, ...] = ()
        self._compiled_dynamics = None
        self._atmosphere_tendency: Callable = dynamics.tendency
        if self.solver_config.compile_kernels:
            self._atmosphere_tendency = self._compile_atmosphere_tendency()

    @staticmethod
    def _zero_ocean(state: CoupledState):
        return state.ocean.zeros_like()

    def _linear_operator(self, state: CoupledState) -> DryPELinearOperator:
        operator = DryPELinearOperator(
            self.dynamics,
            state.atmosphere,
            implicit_operator="gravity_wave",
            tendency=self._atmosphere_tendency,
            jvp_tendency=self.dynamics.tendency,
        )
        if self.solver_config.compile_kernels:
            operator.compile_analytic_gravity_wave(
                backend=self.solver_config.compile_backend,
                mode=self.solver_config.compile_mode,
            )
        return operator

    def _compile_atmosphere_tendency(self) -> Callable:
        """Compile only the stateless atmospheric dynamics kernel.

        The coupled residual still evaluates ocean dynamics and air--sea
        exchange eagerly at each ARK stage.  That mirrors ``DryPEIMEXARK2``:
        model-owned validation remains eager while the parameter-free
        atmospheric tensor kernel may be compiled for inference throughput.
        """

        if not isinstance(self.dynamics, torch.nn.Module):
            raise TypeError("compile_kernels requires a torch.nn.Module atmospheric dynamics object")
        if any(parameter.requires_grad for parameter in self.dynamics.parameters()):
            raise ValueError(
                "compile_kernels supports parameter-free atmospheric dynamics only; use eager coupled ARK2 otherwise"
            )
        if not hasattr(self.dynamics, "validate_state"):
            raise ValueError(
                "compile_kernels requires atmospheric dynamics with a validate_state flag"
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

    def _source_memory_root(
        self,
        state: CoupledState,
        physics_state: PhysicsState | Mapping[str, object] | None,
    ) -> PhysicsState:
        if physics_state is None:
            root = PhysicsState()
        elif isinstance(physics_state, PhysicsState):
            root = physics_state
        elif isinstance(physics_state, Mapping):
            root = PhysicsState(dict(physics_state))
        else:
            raise TypeError("physics_state must be PhysicsState, a mapping, or None")
        data = dict(root.data)
        for name, source in self.stage_sources.items():
            if name not in data:
                data[name] = initial_ark_source_memory(source, state)
        return PhysicsState(data=data, attrs=dict(root.attrs))

    def _step(
        self,
        state: CoupledState,
        dt: float,
        *,
        physics_state: PhysicsState | Mapping[str, object] | None = None,
        time_seconds: float = 0.0,
        stage_state_corrections: tuple[CoupledState, ...] | None = None,
    ) -> tuple[CoupledState, CoupledIMEXStepInfo, PhysicsState | None]:
        dt = float(dt)
        if not torch.isfinite(torch.tensor(dt)) or dt <= 0.0:
            raise ValueError(f"dt must be finite and positive, got {dt!r}")
        base = self.model.bind_state(state)
        if stage_state_corrections is None:
            finite_stage_corrections = tuple(base.zeros_like() for _ in range(ARK2_TABLEAU.n_stages))
        else:
            finite_stage_corrections = tuple(stage_state_corrections)
            if len(finite_stage_corrections) != ARK2_TABLEAU.n_stages:
                raise ValueError("stage_state_corrections must provide one CoupledState per ARK stage")
            if not all(isinstance(correction, CoupledState) for correction in finite_stage_corrections):
                raise TypeError("stage_state_corrections must contain CoupledState values")
        source_root = self._source_memory_root(base, physics_state) if self.stage_sources else None
        if source_root is not None:
            for source in self.stage_sources.values():
                context = getattr(source, "set_ark_step_context", None)
                if callable(context):
                    context(dt)
        operator = self._linear_operator(base)
        inner = _StateInnerProduct(base.atmosphere)
        stages: list[CoupledState] = []
        explicit_terms: list[CoupledState] = []
        implicit_terms: list[CoupledState] = []
        source_memory_tendencies: list[dict[str, object]] = []
        source_stage_diagnostics: list[dict[str, dict]] = []
        infos: list[IMEXStageInfo] = []
        for i in range(ARK2_TABLEAU.n_stages):
            known = base.zeros_like()
            for j in range(i):
                known = known + dt * (
                    ARK2_TABLEAU.explicit[i][j] * explicit_terms[j]
                    + ARK2_TABLEAU.implicit[i][j] * implicit_terms[j]
                )
            # Finite-map source controllers use a Picard-provided physical
            # state increment at each ARK abscissa.  Add it before the
            # implicit solve so both the nonlinear coupled residual and the
            # gravity-wave response see the source-corrected stage state.
            known = known + finite_stage_corrections[i]
            diagonal = ARK2_TABLEAU.implicit[i][i]
            if diagonal == 0.0:
                correction_atmosphere = known.atmosphere
                iterations = matvecs = applications = 0
                initial = final = inner.norm(correction_atmosphere)
                converged = True
            else:
                factor = dt * diagonal
                if self.solver_config.preconditioner == "hydrostatic_schur":
                    solve = _hydrostatic_schur_solve(
                        operator,
                        known.atmosphere,
                        inner,
                        factor=factor,
                        rtol=self.solver_config.resolved_rtol(base.atmosphere.u_e.values.dtype),
                        atol=self.solver_config.atol,
                        max_iterations=self.solver_config.max_iterations,
                        restart=self.solver_config.restart,
                        reorthogonalize=self.solver_config.reorthogonalize,
                        tolerance_safety=self.solver_config.schur_tolerance_safety,
                        fixed_iterations=None,
                    )
                else:
                    solve = _gmres(
                        lambda value: value - factor * operator.apply(value),
                        known.atmosphere,
                        inner,
                        rtol=self.solver_config.resolved_rtol(base.atmosphere.u_e.values.dtype),
                        atol=self.solver_config.atol,
                        max_iterations=self.solver_config.max_iterations,
                        restart=self.solver_config.restart,
                        reorthogonalize=self.solver_config.reorthogonalize,
                        fixed_iterations=None,
                    )
                correction_atmosphere = solve.solution
                iterations, matvecs = solve.iterations, solve.matvecs
                applications = solve.preconditioner_applications
                initial, final, converged = solve.initial_residual, solve.final_residual, solve.converged
            correction = CoupledState(correction_atmosphere, known.ocean, attrs=base._attrs_copy())
            stage = base + correction
            atmospheric_linear = operator.apply(correction_atmosphere)
            linear = CoupledState(atmospheric_linear, self._zero_ocean(base), attrs=base._attrs_copy())
            nonlinear = self.model.tendency(stage, atmosphere_tendency=self._atmosphere_tendency) - linear
            source_explicit = base.zeros_like()
            stage_memory_tendencies: dict[str, object] = {}
            stage_diagnostics: dict[str, dict] = {}
            if source_root is not None:
                stage_time = float(time_seconds) + dt * sum(ARK2_TABLEAU.explicit[i])
                for name, source in self.stage_sources.items():
                    source_memory = source_root.component(name)
                    for j in range(i):
                        source_memory = add_scaled_ark_memory(
                            source_memory,
                            source_memory_tendencies[j][name],
                            dt * ARK2_TABLEAU.explicit[i][j],
                        )
                    projector = getattr(source, "project_ark_memory", None)
                    if callable(projector):
                        source_memory = projector(source_memory)
                    source_result = evaluate_ark_stage_source(
                        source,
                        stage,
                        source_memory,
                        time_seconds=stage_time,
                    )
                    source_explicit = source_explicit + source_result.state_tendency
                    stage_memory_tendencies[name] = source_result.memory_tendency
                    stage_diagnostics[name] = dict(source_result.diagnostics or {})
            relative = final / initial.clamp_min(inner.tiny)
            info = IMEXStageInfo(
                stage=i,
                implicit_coefficient=float(diagonal),
                iterations=int(iterations),
                matvecs=int(matvecs),
                preconditioner_applications=int(applications),
                recycled_dimension=0,
                initial_residual=float(initial.detach().cpu()),
                final_residual=float(final.detach().cpu()),
                relative_residual=float(relative.detach().cpu()),
                converged=bool(converged),
            )
            if not info.converged:
                raise IMEXConvergenceError(info)
            stages.append(stage)
            explicit_terms.append(nonlinear + source_explicit)
            implicit_terms.append(linear)
            source_memory_tendencies.append(stage_memory_tendencies)
            source_stage_diagnostics.append(stage_diagnostics)
            infos.append(info)
        next_state = base
        for weight, explicit, implicit in zip(ARK2_TABLEAU.weights, explicit_terms, implicit_terms, strict=True):
            next_state = next_state + dt * weight * (explicit + implicit)
        step_info = CoupledIMEXStepInfo(
            dt=dt,
            stages=tuple(infos),
            converged=True,
            total_iterations=sum(info.iterations for info in infos),
            total_matvecs=sum(info.matvecs for info in infos) + operator.applications,
            total_preconditioner_applications=sum(info.preconditioner_applications for info in infos),
            stage_source_names=tuple(self.stage_sources),
            stage_source_diagnostics=tuple(source_stage_diagnostics),
        )
        self.last_stage_states = tuple(stages)
        next_physics_state = None
        if source_root is not None:
            data = dict(source_root.data)
            for name in self.stage_sources:
                memory = source_root.component(name)
                for weight, stage_tendency in zip(ARK2_TABLEAU.weights, source_memory_tendencies, strict=True):
                    memory = add_scaled_ark_memory(memory, stage_tendency[name], dt * weight)
                projector = getattr(self.stage_sources[name], "project_ark_memory", None)
                if callable(projector):
                    memory = projector(memory)
                data[name] = memory
            next_physics_state = PhysicsState(data=data, attrs=dict(source_root.attrs))
        self.last_step_info = step_info
        return next_state, step_info, next_physics_state

    def step_with_info(
        self,
        state: CoupledState,
        dt: float,
        *,
        stage_state_corrections: tuple[CoupledState, ...] | None = None,
    ) -> tuple[CoupledState, CoupledIMEXStepInfo]:
        if self.stage_sources:
            raise ValueError("stage_sources own restart memory; use step_with_physics_state(...) instead")
        next_state, info, _physics_state = self._step(
            state,
            dt,
            stage_state_corrections=stage_state_corrections,
        )
        return next_state, info

    def step(self, state: CoupledState, dt: float) -> CoupledState:
        return self.step_with_info(state, dt)[0]

    def step_with_physics_state(
        self,
        state: CoupledState,
        dt: float,
        *,
        physics_state: PhysicsState | Mapping[str, object] | None = None,
        time_seconds: float = 0.0,
    ) -> CoupledStepResult:
        """Advance coupled ARK2 and its explicit stage-source restart memory.

        Each source memory derivative is combined with ARK's explicit tableau;
        this is the required path for restart-bearing physics.  Ordinary
        split-process components deliberately cannot enter this method.
        """

        next_state, info, next_physics_state = self._step(
            state,
            dt,
            physics_state=physics_state,
            time_seconds=time_seconds,
        )
        exchange = self.model.surface_exchange(state, include_audit=True)
        diagnostics = {
            "imex_method": self.recipe_name,
            "imex_converged": info.converged,
            "imex_total_iterations": info.total_iterations,
            "imex_total_matvecs": info.total_matvecs,
            "imex_total_preconditioner_applications": info.total_preconditioner_applications,
            "imex_max_stage_relative_residual": max(stage.relative_residual for stage in info.stages),
            "imex_stage_source_names": info.stage_source_names,
            "imex_stage_source_diagnostics": info.stage_source_diagnostics,
        }
        diagnostics.update(self.model._diagnostics_from_exchange(exchange, next_state))
        return CoupledStepResult(
            next_state,
            exchange=exchange,
            diagnostics=diagnostics,
            physics_state=next_physics_state,
        )

    def step_with_diagnostics(self, state: CoupledState, dt: float) -> CoupledStepResult:
        if self.stage_sources:
            raise ValueError("stage_sources own restart memory; use step_with_physics_state(...) instead")
        next_state, info = self.step_with_info(state, dt)
        exchange = self.model.surface_exchange(state, include_audit=True)
        diagnostics = {
            "imex_method": self.recipe_name,
            "imex_converged": info.converged,
            "imex_total_iterations": info.total_iterations,
            "imex_total_matvecs": info.total_matvecs,
            "imex_total_preconditioner_applications": info.total_preconditioner_applications,
            "imex_max_stage_relative_residual": max(stage.relative_residual for stage in info.stages),
        }
        diagnostics.update(self.model._diagnostics_from_exchange(exchange, next_state))
        return CoupledStepResult(next_state, exchange=exchange, diagnostics=diagnostics)


__all__ = ["CoupledAtmosphereOceanARK2", "CoupledIMEXStepInfo"]
