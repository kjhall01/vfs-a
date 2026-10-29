"""State containers for coupled atmosphere/ocean integrations."""

from __future__ import annotations

from typing import Any

from ..core.state import State
from ..process import PhysicsState


class CoupledState:
    """A compact atmosphere/ocean state pair for explicit coupled stepping."""

    def __init__(
        self,
        atmosphere: State,
        ocean: Any,
        *,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self.atmosphere = atmosphere
        self.ocean = ocean
        self.attrs = dict(attrs or {})

    def _attrs_copy(self) -> dict[str, Any]:
        return dict(self.attrs or {})

    def with_states(
        self,
        atmosphere: State,
        ocean: Any,
        *,
        attrs: dict[str, Any] | None = None,
    ) -> "CoupledState":
        return CoupledState(
            atmosphere,
            ocean,
            attrs=self._attrs_copy() if attrs is None else attrs,
        )

    def to(self, *args, **kwargs) -> "CoupledState":
        return self.with_states(self.atmosphere.to(*args, **kwargs), self.ocean.to(*args, **kwargs))

    def clone(self) -> "CoupledState":
        return self.with_states(self.atmosphere.clone(), self.ocean.clone())

    def detach(self) -> "CoupledState":
        return self.with_states(self.atmosphere.detach(), self.ocean.detach())

    def requires_grad_(self, requires_grad: bool = True) -> "CoupledState":
        self.atmosphere.requires_grad_(requires_grad)
        self.ocean.requires_grad_(requires_grad)
        return self

    def zeros_like(self) -> "CoupledState":
        return self.with_states(self.atmosphere.zeros_like(), self.ocean.zeros_like())

    def _combined_attrs(self, other: "CoupledState") -> dict[str, Any]:
        attrs = dict(other.attrs or {})
        attrs.update(self.attrs or {})
        return attrs

    def __add__(self, other: "CoupledState") -> "CoupledState":
        return CoupledState(
            self.atmosphere + other.atmosphere,
            self.ocean + other.ocean,
            attrs=self._combined_attrs(other),
        )

    def __sub__(self, other: "CoupledState") -> "CoupledState":
        return CoupledState(
            self.atmosphere - other.atmosphere,
            self.ocean - other.ocean,
            attrs=self._combined_attrs(other),
        )

    def __mul__(self, scalar) -> "CoupledState":
        return self.with_states(self.atmosphere * scalar, self.ocean * scalar)

    def __rmul__(self, scalar) -> "CoupledState":
        return self.__mul__(scalar)

    def __truediv__(self, scalar) -> "CoupledState":
        return self.__mul__(1.0 / scalar)


class CoupledStepResult:
    """Result object for coupled steps, interface diagnostics, and process state."""

    def __init__(
        self,
        state: CoupledState,
        *,
        exchange: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
        physics_state: PhysicsState | None = None,
    ) -> None:
        self.state = state
        self.exchange = dict(exchange or {})
        self.diagnostics = dict(diagnostics or {})
        self.physics_state = physics_state


def _register_pytree_nodes() -> None:
    try:
        from torch.utils import _pytree
    except Exception:
        return

    def flatten_state(state: CoupledState):
        return (state.atmosphere, state.ocean), state._attrs_copy()

    def unflatten_state(children, attrs):
        atmosphere, ocean = children
        return CoupledState(atmosphere, ocean, attrs=attrs)

    try:
        _pytree.register_pytree_node(CoupledState, flatten_state, unflatten_state)
    except ValueError:
        pass



_register_pytree_nodes()


__all__ = ["CoupledState", "CoupledStepResult"]
