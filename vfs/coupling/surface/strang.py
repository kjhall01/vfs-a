"""Physical surface subcycling around the coupled ARK2 gravity-wave core.

The restart-bearing land/ice surface model is a finite, conservative map with
capacity and complete-melt events.  Those events cannot in general be carried
through an additive ARK memory tableau by clipping an estimated derivative.
This module provides the deliberately separate, correctness-first route:

``surface(dt / 2) -> coupled ARK2 core(dt) -> surface(dt / 2)``.

Each surface half-step is itself partitioned into short calls to
``SurfaceSystem.step``.  It is a symmetric Strang composition with an ARK2
gravity-wave core, not a claim that the full surface source belongs to the
additive ARK tableau.  Throughput must therefore be measured independently
before this route can be promoted beyond a numerical qualification.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ...process import PhysicsState
from ..steppers.imex_ark2 import CoupledAtmosphereOceanARK2
from ..state import CoupledState, CoupledStepResult
from .finite_map import CoupledSurfaceSystemStageSource


class CoupledSurfaceStrangARK2:
    """Subcycle a physical surface map symmetrically around an ARK2 core.

    ``core_stepper`` must be a source-free
    :class:`CoupledAtmosphereOceanARK2`.  Its model retains coupled
    atmosphere--ocean momentum and dynamics, while ``surface_source`` owns
    fraction-aware heat, freshwater, and all restart-bearing surface memory.
    This separation makes the finite-map ownership explicit and avoids
    double-adding a source through both routes.
    """

    recipe_name = "coupled_surface_strang_ark2_gravity_wave"

    def __init__(
        self,
        core_stepper: CoupledAtmosphereOceanARK2,
        surface_source: CoupledSurfaceSystemStageSource,
        *,
        surface_subcycle_seconds: float = 1.0,
        source_name: str = "surface",
    ) -> None:
        if not isinstance(core_stepper, CoupledAtmosphereOceanARK2):
            raise TypeError("core_stepper must be a CoupledAtmosphereOceanARK2")
        if core_stepper.stage_sources:
            raise ValueError("CoupledSurfaceStrangARK2 requires a source-free ARK2 core stepper")
        if not isinstance(surface_source, CoupledSurfaceSystemStageSource):
            raise TypeError("surface_source must be a CoupledSurfaceSystemStageSource")
        if not torch.isfinite(torch.as_tensor(float(surface_subcycle_seconds))) or float(surface_subcycle_seconds) <= 0.0:
            raise ValueError("surface_subcycle_seconds must be finite and positive")
        if not str(source_name):
            raise ValueError("source_name must be non-empty")
        surface_source.validate_ark_model(core_stepper.model)
        self.core_stepper = core_stepper
        self.surface_source = surface_source
        self.surface_subcycle_seconds = float(surface_subcycle_seconds)
        self.source_name = str(source_name)
        self.last_step_info = None

    def initial_physics_state(self, state: CoupledState) -> PhysicsState:
        """Return the restart tree owned by the physical surface source."""

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
        """Accumulate subcycle diagnostics without retaining tensor graphs."""

        for key, value in source.items():
            if isinstance(value, bool):
                target[key] = bool(target.get(key, False) or value)
            elif isinstance(value, (int, float)):
                target[key] = max(float(target.get(key, 0.0)), abs(float(value)))

    def _advance_surface(
        self,
        state: CoupledState,
        memory: Mapping[str, Any],
        duration: float,
        *,
        time_seconds: float,
    ) -> tuple[CoupledState, dict[str, Any], dict[str, Any], int]:
        """Apply the physical source map over ``duration`` in bounded pieces."""

        current = state
        current_memory: Mapping[str, Any] = memory
        diagnostics: dict[str, Any] = {}
        remaining = float(duration)
        elapsed = 0.0
        steps = 0
        tolerance = 1.0e-12 * max(1.0, abs(remaining))
        while remaining > tolerance:
            sub_dt = min(float(self.surface_subcycle_seconds), remaining)
            result = self.surface_source.advance_explicit_substep(
                current,
                current_memory,
                sub_dt,
                time_seconds=float(time_seconds) + elapsed,
            )
            current = result.state
            current_memory = result.memory
            self._merge_diagnostics(diagnostics, result.diagnostics)
            elapsed += sub_dt
            remaining -= sub_dt
            steps += 1
        return current, dict(current_memory), diagnostics, steps

    def step_with_physics_state(
        self,
        state: CoupledState,
        dt: float,
        *,
        physics_state: PhysicsState | Mapping[str, object] | None = None,
        time_seconds: float = 0.0,
    ) -> CoupledStepResult:
        """Advance a symmetric physical-surface / ARK2-core composition."""

        dt = float(dt)
        if not torch.isfinite(torch.as_tensor(dt)) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        root = self._physics_root(state, physics_state)
        memory = root.component(self.source_name)
        first, memory, diagnostics, first_steps = self._advance_surface(
            state,
            memory,
            0.5 * dt,
            time_seconds=float(time_seconds),
        )
        core, info = self.core_stepper.step_with_info(first, dt)
        final, memory, second_diagnostics, second_steps = self._advance_surface(
            core,
            memory,
            0.5 * dt,
            time_seconds=float(time_seconds) + 0.5 * dt,
        )
        self._merge_diagnostics(diagnostics, second_diagnostics)
        diagnostics.update(
            {
                "surface_integration": "physical_subcycled_strang",
                "surface_subcycle_seconds": self.surface_subcycle_seconds,
                "surface_subcycles": first_steps + second_steps,
                "imex_stage_source_names": (self.source_name,),
                "imex_core_total_iterations": int(info.total_iterations),
                "imex_core_total_matvecs": int(info.total_matvecs),
                "imex_core_converged": bool(info.converged),
            }
        )
        self.last_step_info = info
        return CoupledStepResult(
            final,
            diagnostics=diagnostics,
            physics_state=root.with_component(self.source_name, memory),
        )

    def step(self, state: CoupledState, dt: float) -> CoupledState:
        """Reject dropping restart memory from a stateful surface integration."""

        raise ValueError(
            "CoupledSurfaceStrangARK2 owns restart-bearing surface memory; "
            "use step_with_physics_state(...)"
        )


__all__ = ["CoupledSurfaceStrangARK2"]
