"""Small explicit time integrators for State-shaped tendencies."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Callable

import torch

from .state import State


_MISSING = object()


def _iter_tendency_providers(provider_or_providers):
    if isinstance(provider_or_providers, (list, tuple)):
        for provider in provider_or_providers:
            yield from _iter_tendency_providers(provider)
    else:
        yield provider_or_providers


@contextmanager
def _tendency_dt_context(provider_or_providers, dt: float):
    """Expose the current explicit stage size to dt-aware optional schemes.

    The canonical public tendency API remains ``provider.tendency(state)``.
    A few bounded transport limiters need the actual stage length to compute
    admissible antidiffusive fluxes.  The integrators provide that information
    as transient private context so ordinary providers stay unchanged.
    """

    providers = [
        provider
        for provider in _iter_tendency_providers(provider_or_providers)
        if hasattr(provider, "__dict__")
    ]
    old_values = [(provider, getattr(provider, "_current_tendency_dt", _MISSING)) for provider in providers]
    for provider in providers:
        setattr(provider, "_current_tendency_dt", float(dt))
    try:
        yield
    finally:
        for provider, value in old_values:
            if value is _MISSING:
                try:
                    delattr(provider, "_current_tendency_dt")
                except AttributeError:
                    pass
            else:
                setattr(provider, "_current_tendency_dt", value)


def _call_tendency(provider_or_providers, state: State) -> State:
    """Call one tendency provider or add a short list of providers."""

    if isinstance(provider_or_providers, (list, tuple)):
        if not provider_or_providers:
            raise ValueError("at least one tendency provider is required")
        total = _call_tendency(provider_or_providers[0], state)
        for provider in provider_or_providers[1:]:
            total = total + _call_tendency(provider, state)
        return total
    if hasattr(provider_or_providers, "tendency"):
        return provider_or_providers.tendency(state)
    if isinstance(provider_or_providers, Callable):
        return provider_or_providers(state)
    raise TypeError("tendency provider must have .tendency(state) or be callable")


def forward_euler(provider_or_providers, state: State, dt: float) -> State:
    """Forward Euler step. Useful for tests and tiny smoke checks."""

    dt = float(dt)
    with _tendency_dt_context(provider_or_providers, dt):
        return state + dt * _call_tendency(provider_or_providers, state)


def rk4(provider_or_providers, state: State, dt: float) -> State:
    """Classic fourth-order Runge-Kutta."""

    dt = float(dt)
    with _tendency_dt_context(provider_or_providers, dt):
        k1 = _call_tendency(provider_or_providers, state)
        k2 = _call_tendency(provider_or_providers, state + (dt / 2.0) * k1)
        k3 = _call_tendency(provider_or_providers, state + (dt / 2.0) * k2)
        k4 = _call_tendency(provider_or_providers, state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def ssprk3(provider_or_providers, state: State, dt: float) -> State:
    """Optimal third-order SSPRK/Shu-Osher step."""

    dt = float(dt)
    with _tendency_dt_context(provider_or_providers, dt):
        L0 = _call_tendency(provider_or_providers, state)
        s1 = state + dt * L0

        L1 = _call_tendency(provider_or_providers, s1)
        s2_arg = s1 + dt * L1
        s2 = state + (1.0 / 4.0) * (s2_arg - state)

        L2 = _call_tendency(provider_or_providers, s2)
        s3_arg = s2 + dt * L2
        return state + (2.0 / 3.0) * (s3_arg - state)


def compile_tendency_provider(
    provider_or_callable,
    *,
    backend: str | None = None,
    mode: str | None = None,
    fullgraph: bool = False,
    dynamic: bool | None = None,
):
    """Compile one State -> State tendency callable.

    ``State`` is registered as a PyTorch pytree, so graph capture can see the
    prognostic tensor leaves while mesh/column references remain static context.
    """

    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable in this PyTorch build")
    fn = provider_or_callable.tendency if hasattr(provider_or_callable, "tendency") else provider_or_callable
    if not isinstance(fn, Callable):
        raise TypeError("compile_tendency_provider expects a callable or an object with .tendency(state)")
    kwargs = {"fullgraph": bool(fullgraph)}
    if backend:
        kwargs["backend"] = backend
    if mode:
        kwargs["mode"] = mode
    if dynamic is not None:
        kwargs["dynamic"] = bool(dynamic)
    return torch.compile(fn, **kwargs)
