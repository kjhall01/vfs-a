"""Small graph-state and component contracts used by the graph runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn


class GraphState:
    """Named collection of component-owned states."""

    def __init__(self, states: Mapping[str, Any]) -> None:
        self._states = {str(name): state for name, state in dict(states).items()}
        if not self._states:
            raise ValueError("GraphState requires at least one named state")

    def __contains__(self, name: str) -> bool:
        return str(name) in self._states

    def __getitem__(self, name: str) -> Any:
        return self._states[str(name)]

    def get(self, name: str, default: Any = None) -> Any:
        return self._states.get(str(name), default)

    def keys(self):
        return self._states.keys()

    def values(self):
        return self._states.values()

    def items(self):
        return self._states.items()

    def to_dict(self) -> dict[str, Any]:
        return dict(self._states)

    def with_state(self, name: str, state: Any) -> "GraphState":
        states = dict(self._states)
        states[str(name)] = state
        return GraphState(states)

    def to(self, *args, **kwargs) -> "GraphState":
        return GraphState(
            {
                name: state.to(*args, **kwargs) if hasattr(state, "to") else state
                for name, state in self._states.items()
            }
        )

    def clone(self) -> "GraphState":
        return GraphState(
            {
                name: state.clone() if hasattr(state, "clone") else state
                for name, state in self._states.items()
            }
        )

    def detach(self) -> "GraphState":
        return GraphState(
            {
                name: state.detach() if hasattr(state, "detach") else state
                for name, state in self._states.items()
            }
        )


class GraphComponentAPI(nn.Module):
    """Base contract for runtime graph components."""

    def bind_state(self, state: Any, *, mesh=None, imports: Mapping[str, Any] | None = None) -> Any:
        del mesh, imports
        return state

    def exports(self, state: Any | None = None, *, mesh=None, imports: Mapping[str, Any] | None = None) -> dict[str, Any]:
        del state, mesh, imports
        return {}

    def provide_terms(self, state: Any, *, mesh=None, imports: Mapping[str, Any] | None = None):
        del state, mesh, imports
        raise NotImplementedError(f"{type(self).__name__} does not provide tendency terms")

    def update_state(self, state: Any, *, dt: float, mesh=None, imports: Mapping[str, Any] | None = None) -> Any:
        del state, dt, mesh, imports
        raise NotImplementedError(f"{type(self).__name__} does not update state directly")

    def evaluate(self, model: "ComponentGraphModel", state: Any) -> Any:
        """Optionally return a read-only payload at an evaluate checkpoint."""

        del model, state
        return None


class GraphInitializerAPI(nn.Module):
    """Base contract for graph initialization components."""

    def build(self, model: "ComponentGraphModel", target: str, *, dtype=None, device=None) -> Any:
        del model, target, dtype, device
        raise NotImplementedError(f"{type(self).__name__} does not build initial states")


class GraphDiagnosticAPI(nn.Module):
    """Base contract for read-only graph diagnostics."""

    def row(self, model: "ComponentGraphModel", state0: Any | None, state: Any, **kwargs) -> dict[str, Any]:
        del model, state0, state, kwargs
        raise NotImplementedError(f"{type(self).__name__} does not produce diagnostic rows")


class GraphObserverAPI(nn.Module):
    """Base contract for read-only graph observers."""

    def observe(self, model: "ComponentGraphModel", state: Any) -> dict[str, Any]:
        del model, state
        raise NotImplementedError(f"{type(self).__name__} does not observe graph state")
