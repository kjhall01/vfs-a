"""Small explicit time integrators for State-shaped tendencies."""

from __future__ import annotations

from typing import TypeVar

State = TypeVar("State")


def _call_tendency(provider, state):
    """A single RHS provider; bounded moist Euler uses its separate dt ledger."""
    return provider.tendency(state) if hasattr(provider, "tendency") else provider(state)


def forward_euler(provider_or_providers, state: State, dt: float) -> State:
    """Forward Euler step. Useful for tests and tiny smoke checks."""

    dt = float(dt)
    return state + dt * _call_tendency(provider_or_providers, state)


def rk4(provider_or_providers, state: State, dt: float) -> State:
    """Classic fourth-order Runge-Kutta."""

    dt = float(dt)
    k1 = _call_tendency(provider_or_providers, state)
    k2 = _call_tendency(provider_or_providers, state + (dt / 2.0) * k1)
    k3 = _call_tendency(provider_or_providers, state + (dt / 2.0) * k2)
    k4 = _call_tendency(provider_or_providers, state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def ssprk3(provider_or_providers, state: State, dt: float) -> State:
    """Optimal third-order SSPRK/Shu-Osher step."""

    dt = float(dt)
    L0 = _call_tendency(provider_or_providers, state)
    s1 = state + dt * L0

    L1 = _call_tendency(provider_or_providers, s1)
    s2_arg = s1 + dt * L1
    s2 = state + (1.0 / 4.0) * (s2_arg - state)

    L2 = _call_tendency(provider_or_providers, s2)
    s3_arg = s2 + dt * L2
    return state + (2.0 / 3.0) * (s3_arg - state)
