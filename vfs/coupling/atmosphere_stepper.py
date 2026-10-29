"""Coupled-process adapters for atmosphere-only state steppers."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from ..core.registry import register_component
from ..process import PhysicsStepResult
from .state import CoupledState


@register_component("coupled_atmosphere_stepper_process")
class CoupledAtmosphereStepperProcess(nn.Module):
    """Apply an atmosphere-only stepper inside coupled split-step ordering.

    This is intentionally just a routing adapter.  It lets experimental
    atmosphere steppers, including pressure-IMEX recipes, participate in the
    coupled process sequence without taking ownership of ocean state.
    """

    recipe_name = "coupled_atmosphere_stepper_process"

    def __init__(self, stepper: Any) -> None:
        super().__init__()
        self.stepper = stepper

    def initial_physics_state(self, state: CoupledState) -> dict[str, Any]:
        return {}

    def step(self, state: CoupledState, dt: float, physics_state=None) -> PhysicsStepResult:
        step_with_info = getattr(self.stepper, "step_with_info", None)
        if callable(step_with_info):
            atmosphere, diagnostics = step_with_info(state.atmosphere, dt)
        else:
            step = getattr(self.stepper, "step", None)
            if callable(step):
                atmosphere = step(state.atmosphere, dt)
            elif callable(self.stepper):
                atmosphere = self.stepper(state.atmosphere, dt)
            else:
                raise TypeError("stepper must expose step_with_info(state, dt), step(state, dt), or be callable")
            diagnostics = {}
        next_state = CoupledState(atmosphere, state.ocean, attrs=state._attrs_copy())
        return PhysicsStepResult(next_state, {} if physics_state is None else physics_state, dict(diagnostics or {}))


__all__ = ["CoupledAtmosphereStepperProcess"]
