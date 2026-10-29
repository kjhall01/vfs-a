"""Experimental stage-trajectory physical-map surface coupling.

This controller is intentionally separate from both the additive ARK-memory
source and the qualified Strang composition.  A source-free coupled ARK2 core
first supplies a trajectory, then each Picard pass feeds the prior physical
surface-map increment back into the next ARK stage equations.  The surface
restart tree is the endpoint of positive physical substeps, never an RK sum of
finite-map rates.

The method is a bounded Picard coupling experiment, not a promoted ARK order
claim.  Its only purpose is to test whether stage-aware state sampling reduces
the source/core splitting error that currently limits the full EVP case.  It
must pass the same external gate as Strang before it is used operationally.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ...integrators.imex.ark2 import ARK2_TABLEAU
from ...process import PhysicsState
from ..steppers.imex_ark2 import CoupledAtmosphereOceanARK2
from ..state import CoupledState, CoupledStepResult
from .finite_map import CoupledSurfaceSystemStageSource


class CoupledSurfaceStageMapARK2:
    """Picard-couple a bounded surface map to ARK2 stage equations.

    The first Picard path is the source-free ARK stage trajectory.  Subsequent
    core passes put the previous physical-map increment at each ARK abscissa
    on the implicit-solve right hand side, so the coupled dynamics residual
    sees the source-corrected stage state. Each map pass restarts the finite
    surface memory at the step input; only the final pass contributes a state
    increment and memory endpoint. This prevents repeated maps from
    double-applying packets.  Within a map pass, the state supplied to each
    physical substep also retains the *new* accumulated map increment.  The
    prior Picard correction is subtracted from the ARK trajectory before that
    new increment is applied, so an already-predicted source contribution is
    not counted twice.
    """

    recipe_name = "coupled_surface_stage_map_ark2_experimental"

    def __init__(
        self,
        core_stepper: CoupledAtmosphereOceanARK2,
        surface_source: CoupledSurfaceSystemStageSource,
        *,
        surface_subcycle_seconds: float = 1.0,
        picard_iterations: int = 2,
        source_name: str = "surface",
    ) -> None:
        if not isinstance(core_stepper, CoupledAtmosphereOceanARK2):
            raise TypeError("core_stepper must be CoupledAtmosphereOceanARK2")
        if core_stepper.stage_sources:
            raise ValueError("CoupledSurfaceStageMapARK2 requires a source-free ARK2 core")
        if not isinstance(surface_source, CoupledSurfaceSystemStageSource):
            raise TypeError("surface_source must be CoupledSurfaceSystemStageSource")
        if not torch.isfinite(torch.as_tensor(float(surface_subcycle_seconds))) or float(surface_subcycle_seconds) <= 0.0:
            raise ValueError("surface_subcycle_seconds must be finite and positive")
        if int(picard_iterations) < 1:
            raise ValueError("picard_iterations must be at least one")
        if not str(source_name):
            raise ValueError("source_name must be non-empty")
        surface_source.validate_ark_model(core_stepper.model)
        self.core_stepper = core_stepper
        self.surface_source = surface_source
        self.surface_subcycle_seconds = float(surface_subcycle_seconds)
        self.picard_iterations = int(picard_iterations)
        self.source_name = str(source_name)
        self.last_step_info = None

    def initial_physics_state(self, state: CoupledState) -> PhysicsState:
        return PhysicsState({self.source_name: self.surface_source.initial_ark_physics_state(state)})

    def _physics_root(
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
        if self.source_name in root.data:
            return root
        return root.with_component(self.source_name, self.surface_source.initial_ark_physics_state(state))

    @staticmethod
    def _merge_diagnostics(target: dict[str, Any], source: Mapping[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, bool):
                target[key] = bool(target.get(key, False) or value)
            elif isinstance(value, (int, float)):
                target[key] = max(float(target.get(key, 0.0)), abs(float(value)))

    @staticmethod
    def _stage_nodes(base: CoupledState, stages: tuple[CoupledState, ...]) -> tuple[tuple[float, ...], tuple[CoupledState, ...]]:
        abscissas = tuple(float(sum(row)) for row in ARK2_TABLEAU.explicit)
        if len(stages) != len(abscissas):
            raise RuntimeError("ARK core did not retain all stage states")
        if abs(abscissas[0]) > 1.0e-14 or abs(abscissas[-1] - 1.0) > 1.0e-14:
            raise RuntimeError("stage-map controller requires explicit ARK nodes at 0 and 1")
        if any(right < left for left, right in zip(abscissas, abscissas[1:])):
            raise RuntimeError("stage-map controller requires nondecreasing explicit ARK nodes")
        # Stage zero is expected to equal base, but retain the exact base state
        # as the source-map t=0 owner to avoid a roundoff-dependent contract.
        return abscissas, (base, *stages[1:])

    @staticmethod
    def _interpolate(nodes_t: tuple[float, ...], nodes: tuple[CoupledState, ...], fraction: float) -> CoupledState:
        """Return the quadratic ARK-stage interpolant at ``fraction``.

        The three Giraldo ARK2 abscissas are distinct and include zero and
        one.  Lagrange interpolation through their coupled states provides
        the required second-order stage trajectory for physical map samples;
        a piecewise-linear segment would reduce the source/core composition
        to first order between stage times.
        """

        if len(nodes_t) != len(nodes) or len(nodes_t) != ARK2_TABLEAU.n_stages:
            raise ValueError("stage interpolation requires every ARK2 node")
        f = min(1.0, max(0.0, float(fraction)))
        out = nodes[0].zeros_like()
        for index, node in enumerate(nodes):
            weight = 1.0
            for other_index, other_time in enumerate(nodes_t):
                if other_index == index:
                    continue
                denominator = nodes_t[index] - other_time
                if abs(denominator) <= 1.0e-14:
                    raise RuntimeError("stage-map controller requires distinct ARK nodes")
                weight *= (f - other_time) / denominator
            out = out + weight * node
        return out

    def _map_along_stage_path(
        self,
        base: CoupledState,
        memory: Mapping[str, Any],
        *,
        dt: float,
        time_seconds: float,
        nodes_t: tuple[float, ...],
        nodes: tuple[CoupledState, ...],
        predictor_stage_increments: tuple[CoupledState, ...],
    ) -> tuple[tuple[CoupledState, ...], dict[str, Any], dict[str, Any], int]:
        if len(predictor_stage_increments) != len(nodes_t):
            raise ValueError("predictor_stage_increments must provide every ARK2 node")
        current_memory: Mapping[str, Any] = memory
        increment = base.zeros_like()
        stage_increments: list[CoupledState] = [increment]
        diagnostics: dict[str, Any] = {}
        remaining = float(dt)
        elapsed = 0.0
        steps = 0
        tolerance = 1.0e-12 * max(1.0, abs(float(dt)))
        next_stage = 1
        while remaining > tolerance:
            next_boundary = float(dt) if next_stage >= len(nodes_t) else float(dt) * nodes_t[next_stage]
            until_boundary = max(0.0, next_boundary - elapsed)
            sub_dt = min(float(self.surface_subcycle_seconds), remaining, until_boundary)
            if sub_dt <= tolerance:
                while next_stage < len(nodes_t) and elapsed >= float(dt) * nodes_t[next_stage] - tolerance:
                    stage_increments.append(increment)
                    next_stage += 1
                continue
            fraction = (elapsed + 0.5 * sub_dt) / float(dt)
            stage_sample = self._interpolate(nodes_t, nodes, fraction)
            predicted_increment = self._interpolate(nodes_t, predictor_stage_increments, fraction)
            # ``nodes`` is the previous Picard coupled trajectory.  Follow
            # its non-surface component while replacing its interpolated
            # surface-map contribution with the one accumulated by this map
            # pass.  This makes successive physical substeps stateful in the
            # coupled fields as well as in the surface restart tree.
            sample = stage_sample + (increment - predicted_increment)
            result = self.surface_source.advance_explicit_substep(
                sample,
                current_memory,
                sub_dt,
                time_seconds=float(time_seconds) + elapsed,
            )
            increment = increment + (result.state - sample)
            current_memory = result.memory
            self._merge_diagnostics(diagnostics, result.diagnostics)
            elapsed += sub_dt
            remaining -= sub_dt
            steps += 1
            while next_stage < len(nodes_t) and elapsed >= float(dt) * nodes_t[next_stage] - tolerance:
                stage_increments.append(increment)
                next_stage += 1
        while len(stage_increments) < len(nodes_t):
            stage_increments.append(increment)
        return tuple(stage_increments), dict(current_memory), diagnostics, steps

    def step_with_physics_state(
        self,
        state: CoupledState,
        dt: float,
        *,
        physics_state: PhysicsState | Mapping[str, object] | None = None,
        time_seconds: float = 0.0,
    ) -> CoupledStepResult:
        dt = float(dt)
        if not torch.isfinite(torch.as_tensor(dt)) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        root = self._physics_root(state, physics_state)
        stage_corrections = tuple(state.zeros_like() for _ in range(ARK2_TABLEAU.n_stages))
        core_state = state
        info = None
        memory: dict[str, Any] = dict(root.component(self.source_name))
        diagnostics: dict[str, Any] = {}
        subcycles = 0
        total_iterations = 0
        total_matvecs = 0
        for _iteration in range(int(self.picard_iterations)):
            core_state, info = self.core_stepper.step_with_info(
                state,
                dt,
                stage_state_corrections=stage_corrections,
            )
            nodes_t, nodes = self._stage_nodes(state, self.core_stepper.last_stage_states)
            stage_corrections, memory, map_diagnostics, map_steps = self._map_along_stage_path(
                state,
                root.component(self.source_name),
                dt=dt,
                time_seconds=float(time_seconds),
                nodes_t=nodes_t,
                nodes=nodes,
                predictor_stage_increments=stage_corrections,
            )
            self._merge_diagnostics(diagnostics, map_diagnostics)
            subcycles += map_steps
            total_iterations += int(info.total_iterations)
            total_matvecs += int(info.total_matvecs)
        assert info is not None
        next_state = core_state + stage_corrections[-1]
        diagnostics.update(
            {
                "surface_integration": "experimental_stage_map_picard",
                "surface_stage_map_picard_iterations": int(self.picard_iterations),
                "surface_subcycle_seconds": self.surface_subcycle_seconds,
                "surface_subcycles": subcycles,
                "imex_core_picard_solves": int(self.picard_iterations),
                "imex_core_total_iterations": total_iterations,
                "imex_core_total_matvecs": total_matvecs,
                "imex_core_converged": bool(info.converged),
            }
        )
        self.last_step_info = info
        return CoupledStepResult(
            next_state,
            diagnostics=diagnostics,
            physics_state=root.with_component(self.source_name, memory),
        )

    def step(self, state: CoupledState, dt: float) -> CoupledState:
        raise ValueError("CoupledSurfaceStageMapARK2 owns restart memory; use step_with_physics_state(...)")


__all__ = ["CoupledSurfaceStageMapARK2"]
