"""Metadata registry for the compact component-graph configuration language."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn


_GRAPH_KINDS = {"component", "diagnostic", "initializer", "observer"}
_GRAPH_STATUSES = {"stable", "provisional", "experimental", "negative_result", "deprecated"}


@dataclass(frozen=True)
class GraphComponentMetadata:
    """Construction and validation facts for one registered graph module."""

    name: str
    cls: type[nn.Module]
    kind: str = "component"
    status: str = "provisional"
    state_type: str | None = None
    default_owns: tuple[str, ...] = ()
    default_exports: tuple[str, ...] = ()
    default_provides: dict[str, tuple[str, ...]] | None = None
    requires_mesh: bool = False
    requires_column: bool = False
    description: str = ""

    @property
    def class_path(self) -> str:
        return f"{self.cls.__module__}.{self.cls.__qualname__}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "class_path": self.class_path,
            "state_type": self.state_type,
            "default_owns": list(self.default_owns),
            "default_exports": list(self.default_exports),
            "default_provides": {
                target: list(variables)
                for target, variables in dict(self.default_provides or {}).items()
            },
            "requires_mesh": self.requires_mesh,
            "requires_column": self.requires_column,
            "description": self.description,
        }


_GRAPH_REGISTRY: dict[str, dict[str, GraphComponentMetadata]] = {
    kind: {} for kind in sorted(_GRAPH_KINDS)
}


def _register_graph_metadata(metadata: GraphComponentMetadata) -> None:
    kind = str(metadata.kind)
    status = str(metadata.status)
    if kind not in _GRAPH_KINDS:
        raise ValueError(f"unknown graph registry kind {kind!r}")
    if status not in _GRAPH_STATUSES:
        raise ValueError(f"unknown graph registry status {status!r}")
    if metadata.name in _GRAPH_REGISTRY[kind]:
        raise ValueError(f"duplicate graph {kind} registration {metadata.name!r}")
    _GRAPH_REGISTRY[kind][metadata.name] = metadata


def graph_component_metadata(kind: str | None = None) -> dict[str, GraphComponentMetadata]:
    """Return registered metadata, optionally restricted to one graph kind."""

    if kind is not None:
        selected = str(kind)
        if selected not in _GRAPH_KINDS:
            raise ValueError(f"unknown graph registry kind {selected!r}")
        return dict(_GRAPH_REGISTRY[selected])
    return {
        f"{selected}:{name}": metadata
        for selected in sorted(_GRAPH_REGISTRY)
        for name, metadata in _GRAPH_REGISTRY[selected].items()
    }


def graph_component_catalog(kind: str | None = None) -> dict[str, dict[str, Any]]:
    """Return JSON/Markdown-friendly graph registry metadata."""

    return {name: metadata.to_dict() for name, metadata in graph_component_metadata(kind).items()}
