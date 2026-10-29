"""Lightweight structural contracts for dynamics diagnostics."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable


StateT = TypeVar("StateT")
TermDict = dict[str, dict[str, Any]]


@runtime_checkable
class TendencyProvider(Protocol[StateT]):
    """Object that can produce a state-shaped tendency."""

    def tendency(self, state: StateT) -> StateT:
        ...


@runtime_checkable
class TendencyTermsProvider(TendencyProvider[StateT], Protocol[StateT]):
    """Object that can decompose and reconstruct its tendency."""

    def tendency_terms(self, state: StateT) -> TermDict:
        ...

    def terms_total(self, terms: TermDict, *, like: StateT | None = None) -> StateT:
        ...


@runtime_checkable
class DiagnosticsProvider(TendencyProvider[StateT], Protocol[StateT]):
    """Object with a detached reporting facade."""

    @property
    def diagnostics(self) -> Any:
        ...


__all__ = [
    "DiagnosticsProvider",
    "StateT",
    "TendencyProvider",
    "TendencyTermsProvider",
    "TermDict",
]
