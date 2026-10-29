"""Tiny build-time registries for optional config construction."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any, Callable


_COMPONENTS: dict[str, Callable[..., Any]] = {}
_INTEGRATORS: dict[str, Callable[..., Any]] = {}
_MODELS: dict[str, Callable[..., Any]] = {}
_MODEL_METADATA: dict[str, "RecipeMetadata"] = {}
_REGRID_OPERATORS: dict[str, Callable[..., Any]] = {}


_VALID_MODEL_STATUSES = {"stable", "provisional", "experimental", "negative_result", "deprecated"}


@dataclass(frozen=True)
class RecipeMetadata:
    """Public status metadata for a registered recipe/model constructor."""

    name: str
    status: str = "experimental"
    component: str = "unknown"
    description: str = ""
    default_safe: bool = False


def _register(registry: dict[str, Callable[..., Any]], name: str, constructor: Callable[..., Any] | None):
    def decorator(obj: Callable[..., Any]):
        registry[str(name)] = obj
        return obj

    if constructor is None:
        return decorator
    return decorator(constructor)


def register_component(name: str, constructor: Callable[..., Any] | None = None):
    return _register(_COMPONENTS, name, constructor)


def register_integrator(name: str, integrator: Callable[..., Any] | None = None):
    return _register(_INTEGRATORS, name, integrator)


def _normalize_model_status(status: str) -> str:
    value = str(status)
    if value not in _VALID_MODEL_STATUSES:
        expected = ", ".join(sorted(_VALID_MODEL_STATUSES))
        raise ValueError(f"unknown model recipe status {value!r}; expected one of: {expected}")
    return value


def register_model_metadata(
    name: str,
    *,
    status: str = "experimental",
    component: str = "unknown",
    description: str = "",
    default_safe: bool = False,
) -> RecipeMetadata:
    metadata = RecipeMetadata(
        name=str(name),
        status=_normalize_model_status(status),
        component=str(component),
        description=str(description),
        default_safe=bool(default_safe),
    )
    _MODEL_METADATA[str(name)] = metadata
    return metadata


def register_model(
    name: str,
    model: Callable[..., Any] | None = None,
    *,
    status: str = "experimental",
    component: str = "unknown",
    description: str = "",
    default_safe: bool = False,
):
    register_model_metadata(
        name,
        status=status,
        component=component,
        description=description,
        default_safe=default_safe,
    )
    return _register(_MODELS, name, model)


def register_regrid_operator(name: str, operator: Callable[..., Any] | None = None):
    return _register(_REGRID_OPERATORS, name, operator)


def get_component(name: str) -> Callable[..., Any] | None:
    return _COMPONENTS.get(str(name))


def get_integrator(name: str) -> Callable[..., Any] | None:
    return _INTEGRATORS.get(str(name))


def get_model(name: str) -> Callable[..., Any] | None:
    return _MODELS.get(str(name))


def get_model_metadata(name: str) -> RecipeMetadata | None:
    return _MODEL_METADATA.get(str(name))


def get_regrid_operator(name: str) -> Callable[..., Any] | None:
    return _REGRID_OPERATORS.get(str(name))


def _status_filter(status: str | Iterable[str] | None) -> set[str] | None:
    if status is None:
        return None
    if isinstance(status, str):
        statuses = {status}
    else:
        statuses = {str(item) for item in status}
    return {_normalize_model_status(item) for item in statuses}


def registered_models(
    *,
    status: str | Iterable[str] | None = None,
    include_experimental: bool = True,
    include_negative_results: bool = True,
    include_deprecated: bool = True,
) -> tuple[str, ...]:
    statuses = _status_filter(status)
    names: list[str] = []
    for name in sorted(_MODELS):
        metadata = _MODEL_METADATA.get(name, RecipeMetadata(name=name))
        if statuses is not None and metadata.status not in statuses:
            continue
        if not include_experimental and metadata.status == "experimental":
            continue
        if not include_negative_results and metadata.status == "negative_result":
            continue
        if not include_deprecated and metadata.status == "deprecated":
            continue
        names.append(name)
    return tuple(names)


def registered_model_metadata(
    *,
    status: str | Iterable[str] | None = None,
    include_experimental: bool = True,
    include_negative_results: bool = True,
    include_deprecated: bool = True,
) -> tuple[RecipeMetadata, ...]:
    return tuple(
        _MODEL_METADATA.get(name, RecipeMetadata(name=name))
        for name in registered_models(
            status=status,
            include_experimental=include_experimental,
            include_negative_results=include_negative_results,
            include_deprecated=include_deprecated,
        )
    )


def registered_regrid_operators() -> tuple[str, ...]:
    return tuple(sorted(_REGRID_OPERATORS))
