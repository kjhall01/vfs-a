"""Passive tracer transport hooks for dry atmospheric dynamics."""

from __future__ import annotations

import torch

from ..core.state import State
from .dry_strategies import UpwindTracerTransport


# Numerical functions bound explicitly by DryHydrostaticDynamics.
def _tracer_tendency_upwind(
    self,
    state: State,
    u_e: torch.Tensor,
    q: torch.Tensor,
    dp: torch.Tensor,
    mu: torch.Tensor,
) -> torch.Tensor:
    return UpwindTracerTransport().tendency(self, state, u_e, q, dp, mu)

def _tracer_tendency(
    self,
    state: State,
    u_e: torch.Tensor,
    q: torch.Tensor,
    dp: torch.Tensor,
    mu: torch.Tensor,
) -> torch.Tensor:
    return self.tracer_transport.tendency(self, state, u_e, q, dp, mu)

def tracer_tendency_terms(
    self,
    state: State,
    *,
    primitives: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor] | None:
    """Return passive-tracer terms owned by the tracer transport scheme."""

    if state.tracer_m is None:
        return None
    ctx = self._transport_primitives(state) if primitives is None else primitives
    q = state.tracers()
    assert q is not None
    transport = self._tracer_tendency(state, state.u_e, q, ctx["dp"], ctx["mu"])
    return {
        "transport": transport,
        "total": transport,
    }

__all__ = ['_tracer_tendency_upwind', '_tracer_tendency', 'tracer_tendency_terms']
