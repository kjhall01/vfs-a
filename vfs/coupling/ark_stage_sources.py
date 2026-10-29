"""Strict explicit-source contract for coupled ARK stage integration.

Split-process ``step`` methods cannot safely be inserted into an ARK stage:
their restart memory would be advanced in an ambiguous order.  A source that
wants to participate in coupled ARK must instead expose a state tendency and a
time-derivative for its *own* restart memory.  This module holds that compact,
testable contract and the tensor-tree arithmetic used by the ARK tableau.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Number
from typing import Any, Protocol, runtime_checkable

import torch

from .state import CoupledState


@dataclass(frozen=True)
class CoupledARKStageSourceResult:
    """One explicit source evaluation on an ARK stage.

    ``memory_tendency`` has the same tensor/numeric mapping structure as the
    source's ``initial_ark_physics_state`` result.  It is a derivative, not a
    replacement restart state; ARK combines it with the explicit tableau in
    exactly the same way as its state tendency.
    """

    state_tendency: CoupledState
    memory_tendency: Any = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CoupledARKStageSource(Protocol):
    """Protocol required for a restart-bearing coupled ARK source."""

    def initial_ark_physics_state(self, state: CoupledState) -> Any:
        """Return tensor/numeric source memory at the beginning of a step."""

    def ark_stage_tendency(
        self,
        state: CoupledState,
        physics_state: Any,
        *,
        time_seconds: float,
    ) -> CoupledARKStageSourceResult:
        """Return state and memory derivatives at one explicit ARK stage."""


def initial_ark_source_memory(source: Any, state: CoupledState) -> Any:
    initializer = getattr(source, "initial_ark_physics_state", None)
    if not callable(initializer):
        raise TypeError(
            f"ARK stage source {type(source).__name__} must define initial_ark_physics_state(state)"
        )
    return initializer(state)


def evaluate_ark_stage_source(
    source: Any,
    state: CoupledState,
    physics_state: Any,
    *,
    time_seconds: float,
) -> CoupledARKStageSourceResult:
    method = getattr(source, "ark_stage_tendency", None)
    if not callable(method):
        raise TypeError(
            f"ARK stage source {type(source).__name__} must define "
            "ark_stage_tendency(state, physics_state, *, time_seconds)"
        )
    raw = method(state, physics_state, time_seconds=float(time_seconds))
    if isinstance(raw, CoupledARKStageSourceResult):
        result = raw
    elif isinstance(raw, tuple):
        if len(raw) == 2:
            result = CoupledARKStageSourceResult(raw[0], raw[1])
        elif len(raw) == 3:
            result = CoupledARKStageSourceResult(raw[0], raw[1], dict(raw[2] or {}))
        else:
            raise TypeError("ARK stage source tuple results must be (state_tendency, memory_tendency[, diagnostics])")
    elif isinstance(raw, Mapping) and "state_tendency" in raw:
        result = CoupledARKStageSourceResult(
            raw["state_tendency"],
            raw.get("memory_tendency"),
            dict(raw.get("diagnostics", {})),
        )
    else:
        raise TypeError(
            "ARK stage source must return CoupledARKStageSourceResult, "
            "(state_tendency, memory_tendency), or a mapping with state_tendency"
        )
    if not isinstance(result.state_tendency, CoupledState):
        raise TypeError("ARK stage source state_tendency must be a CoupledState")
    return result


def add_scaled_ark_memory(base: Any, tendency: Any, scale: float) -> Any:
    """Return ``base + scale * tendency`` for a strict restart-memory tree."""

    if tendency is None:
        return base
    if torch.is_tensor(base) and torch.is_tensor(tendency):
        if base.shape != tendency.shape:
            raise ValueError(
                "ARK source memory tendency shape must match its memory state: "
                f"{tuple(tendency.shape)} != {tuple(base.shape)}"
            )
        return base + float(scale) * tendency.to(dtype=base.dtype, device=base.device)
    if isinstance(base, Number) and isinstance(tendency, Number):
        return type(base)(base + float(scale) * tendency)
    if isinstance(base, Mapping) and isinstance(tendency, Mapping):
        extra = set(tendency) - set(base)
        if extra:
            raise ValueError(f"ARK source memory tendency has unknown keys: {sorted(map(str, extra))}")
        return {
            str(name): add_scaled_ark_memory(value, tendency.get(name), scale)
            for name, value in base.items()
        }
    raise TypeError(
        "ARK source memory must be a matching tensor/numeric/mapping tree; "
        f"got base={type(base).__name__}, tendency={type(tendency).__name__}"
    )


def weighted_ark_memory(*terms: tuple[float, Any]) -> Any:
    """Return a strict weighted sum of matching restart-memory trees."""

    if not terms:
        raise ValueError("weighted_ark_memory requires at least one term")
    values = [value for _weight, value in terms]
    if any(value is None for value in values):
        if all(value is None for value in values):
            return None
        raise TypeError("cannot combine None and non-None ARK memory")
    first = values[0]
    if torch.is_tensor(first):
        total = torch.zeros_like(first)
        for weight, value in terms:
            if not torch.is_tensor(value) or value.shape != first.shape:
                raise TypeError("weighted ARK tensor memory must have matching tensor leaves")
            total = total + float(weight) * value.to(dtype=first.dtype, device=first.device)
        return total
    if isinstance(first, Number):
        if not all(isinstance(value, Number) for value in values):
            raise TypeError("weighted ARK numeric memory must have matching numeric leaves")
        return type(first)(sum(float(weight) * value for weight, value in terms))
    if isinstance(first, Mapping):
        keys = set(first)
        if not all(isinstance(value, Mapping) and set(value) == keys for value in values):
            raise TypeError("weighted ARK mapping memory must have matching keys")
        return {
            str(name): weighted_ark_memory(*[(weight, value[name]) for weight, value in terms])
            for name in first
        }
    raise TypeError(f"unsupported ARK memory leaf {type(first).__name__}")


__all__ = [
    "CoupledARKStageSource",
    "CoupledARKStageSourceResult",
    "weighted_ark_memory",
]
