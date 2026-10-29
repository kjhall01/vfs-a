"""Split-step process physics state and driver utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from .core.integrators import forward_euler, ssprk3


@dataclass
class PhysicsState(Mapping[str, Any]):
    """Restartable state owned by process-physics components.

    This container is deliberately separate from the atmospheric ``State`` and
    ocean ``OceanState`` tensor payloads.  It is for slow or process-owned
    quantities such as slab/surface reservoirs, caches, and column-model memory.
    Tensor leaves stay as tensors for differentiable split-step workflows;
    NetCDF I/O detaches only at the serialization boundary.
    """

    data: dict[str, Any] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = dict(self.data or {})
        self.attrs = dict(self.attrs or {})

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def component(self, name: str, default: Any = None) -> Any:
        return self.data.get(str(name), default)

    def with_component(self, name: str, value: Any) -> "PhysicsState":
        data = dict(self.data)
        data[str(name)] = value
        return PhysicsState(data=data, attrs=dict(self.attrs))

    def to(self, *args, **kwargs) -> "PhysicsState":
        return PhysicsState(_map_tree(self.data, lambda x: x.to(*args, **kwargs)), attrs=dict(self.attrs))

    def clone(self) -> "PhysicsState":
        return PhysicsState(_map_tree(self.data, _clone_leaf), attrs=dict(self.attrs))

    def detach(self) -> "PhysicsState":
        return PhysicsState(_map_tree(self.data, _detach_leaf), attrs=dict(self.attrs))

    def to_netcdf(self, path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> None:
        """Write tensor and JSON scalar leaves to a compact NetCDF file."""

        import xarray as xr

        tensor_entries: list[tuple[tuple[str, ...], torch.Tensor]] = []
        scalar_entries: list[dict[str, Any]] = []
        _flatten_for_netcdf(self.data, (), tensor_entries, scalar_entries)

        data_vars = {}
        paths: dict[str, list[str]] = {}
        used: set[str] = set()
        for path_tuple, tensor in tensor_entries:
            name = _safe_var_name(path_tuple)
            base = name
            i = 1
            while name in used:
                name = f"{base}_{i}"
                i += 1
            used.add(name)
            arr = tensor.detach().cpu().numpy()
            dims = tuple(f"{name}_dim{i}" for i in range(arr.ndim))
            data_vars[name] = (dims, arr)
            paths[name] = list(path_tuple)

        attrs = {
            "physics_state_format": "vfs_physics_state_v1",
            "physics_state_tensor_paths_json": json.dumps(paths, sort_keys=True),
            "physics_state_scalar_paths_json": json.dumps(scalar_entries, sort_keys=True),
            "physics_state_attrs_json": json.dumps(self.attrs, default=str, sort_keys=True),
        }
        if metadata:
            attrs["metadata_json"] = json.dumps(dict(metadata), default=str, sort_keys=True)
        ds = xr.Dataset(data_vars=data_vars, attrs=attrs)
        ds.to_netcdf(path)

    @classmethod
    def from_netcdf(cls, path: str | Path, *, dtype=None, device=None) -> "PhysicsState":
        """Load a ``PhysicsState`` written by :meth:`to_netcdf`."""

        import xarray as xr

        with xr.open_dataset(path) as ds:
            tensor_paths = json.loads(str(ds.attrs.get("physics_state_tensor_paths_json", "{}")))
            scalar_entries = json.loads(str(ds.attrs.get("physics_state_scalar_paths_json", "[]")))
            attrs = json.loads(str(ds.attrs.get("physics_state_attrs_json", "{}")))
            data: dict[str, Any] = {}
            for var_name, path_list in tensor_paths.items():
                tensor = torch.as_tensor(ds[var_name].values)
                if dtype is not None or device is not None:
                    tensor = tensor.to(dtype=dtype if dtype is not None else tensor.dtype, device=device)
                _assign_tree_value(data, tuple(str(part) for part in path_list), tensor)
            for entry in scalar_entries:
                _assign_tree_value(data, tuple(str(part) for part in entry["path"]), entry["value"])
        return cls(data=data, attrs=attrs)

    def load_netcdf(self, path: str | Path, *, dtype=None, device=None) -> "PhysicsState":
        """Instance-style loader matching the geometry-first ``State`` API."""

        loaded = type(self).from_netcdf(path, dtype=dtype, device=device)
        self.data = loaded.data
        self.attrs = loaded.attrs
        return self


@dataclass
class PhysicsStepResult:
    """Result from one split process-physics step."""

    state: Any
    physics_state: PhysicsState | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ProcessPhysicsDriver:
    """Reusable split driver for dynamics tendencies plus process components.

    Dynamics/source tendencies are integrated additively with the configured RK
    integrator, then process components are applied as split steps.  A process
    component may expose ``step(state, dt, physics_state=...)`` or be directly
    callable with the same signature and return a
    ``PhysicsStepResult``/``State``/tuple. Components with only
    ``tendency(state)`` are advanced by a forward-Euler source update.
    """

    def __init__(
        self,
        *,
        tendency_providers: Any | None = None,
        process_components: Mapping[str, Any] | list[Any] | tuple[Any, ...] | None = None,
        integrator=ssprk3,
        process_substeps: int = 1,
    ) -> None:
        self.tendency_providers = tendency_providers
        self.process_components = _named_components(process_components)
        self.integrator = integrator
        self.process_substeps = max(1, int(process_substeps))

    def initial_physics_state(self, state: Any) -> PhysicsState:
        return initial_physics_state_for_components(self.process_components, state)

    def step(
        self,
        state: Any,
        dt: float,
        physics_state: PhysicsState | Mapping[str, Any] | None = None,
    ) -> PhysicsStepResult:
        current = state
        if self.tendency_providers is not None:
            current = self.integrator(self.tendency_providers, current, dt)
        return step_process_components(
            self.process_components,
            current,
            dt,
            physics_state=physics_state,
            substeps=self.process_substeps,
        )


def initial_physics_state_for_components(components: Mapping[str, Any], state: Any) -> PhysicsState:
    data = {}
    for name, component in components.items():
        component_state = _component_initial_physics_state(component, state)
        if component_state is not None:
            data[str(name)] = component_state
    return PhysicsState(data)


def step_process_components(
    components: Mapping[str, Any],
    state: Any,
    dt: float,
    *,
    physics_state: PhysicsState | Mapping[str, Any] | None = None,
    substeps: int | Mapping[str, int] = 1,
) -> PhysicsStepResult:
    root = _as_physics_state(physics_state)
    current = state
    diagnostics: dict[str, Any] = {}

    if isinstance(substeps, Mapping):
        for name, component in components.items():
            n_substeps = _component_substeps(substeps, name)
            sub_dt = float(dt) / n_substeps
            last_diag: dict[str, Any] = {}
            for _ in range(n_substeps):
                current, root, last_diag = _step_one_process_component(
                    name,
                    component,
                    current,
                    sub_dt,
                    root,
                )
            if last_diag:
                last_diag = dict(last_diag)
                last_diag.setdefault("process_substeps", n_substeps)
                last_diag.setdefault("process_substep_dt", sub_dt)
                diagnostics[str(name)] = last_diag
        return PhysicsStepResult(current, root, diagnostics)

    n_global_substeps = max(1, int(substeps))
    sub_dt = float(dt) / n_global_substeps
    for _ in range(n_global_substeps):
        for name, component in components.items():
            current, root, last_diag = _step_one_process_component(
                name,
                component,
                current,
                sub_dt,
                root,
            )
            if last_diag:
                last_diag = dict(last_diag)
                last_diag.setdefault("process_substeps", n_global_substeps)
                last_diag.setdefault("process_substep_dt", sub_dt)
                diagnostics[str(name)] = last_diag
    return PhysicsStepResult(current, root, diagnostics)


def _named_components(components: Mapping[str, Any] | list[Any] | tuple[Any, ...] | None) -> dict[str, Any]:
    if components is None:
        return {}
    if isinstance(components, Mapping):
        return {str(name): component for name, component in components.items()}
    return {f"process_{i}": component for i, component in enumerate(components)}


def _as_physics_state(value: PhysicsState | Mapping[str, Any] | None) -> PhysicsState:
    if value is None:
        return PhysicsState()
    if isinstance(value, PhysicsState):
        return value
    if isinstance(value, Mapping):
        return PhysicsState(dict(value))
    raise TypeError("physics_state must be PhysicsState, a mapping, or None")


def _component_substeps(substeps: Mapping[str, int], name: str) -> int:
    raw = substeps.get(str(name), substeps.get("*", 1))
    out = int(raw)
    if out <= 0:
        raise ValueError(f"substeps for component {name!r} must be positive")
    return out


def _step_one_process_component(
    name: str,
    component: Any,
    state: Any,
    dt: float,
    root: PhysicsState,
) -> tuple[Any, PhysicsState, dict[str, Any]]:
    component_state = root.component(name, None)
    if component_state is None:
        component_state = _component_initial_physics_state(component, state)
    result = _call_process_component(component, state, dt, component_state)
    normalized = _normalize_process_result(
        result,
        fallback_state=state,
        fallback_component_state=component_state,
    )
    next_root = root
    if normalized.physics_state is not None:
        next_root = root.with_component(name, normalized.physics_state)
    return normalized.state, next_root, dict(normalized.diagnostics or {})


def _component_initial_physics_state(component: Any, state: Any) -> Any:
    for name in ("initial_physics_state", "initialize_physics_state"):
        method = getattr(component, name, None)
        if callable(method):
            return method(state)
    return None


def _call_process_component(component: Any, state: Any, dt: float, component_state: Any) -> Any:
    step = getattr(component, "step", None)
    if callable(step):
        return _call_process_step_callable(step, state, dt, component_state)
    if hasattr(component, "tendency"):
        return forward_euler(component, state, dt)
    if callable(component):
        return _call_process_step_callable(component, state, dt, component_state)
    raise TypeError("process component must expose step(state, dt, ...) or tendency(state)")


def _call_process_step_callable(fn, state: Any, dt: float, component_state: Any) -> Any:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(state, dt, physics_state=component_state)
    params = signature.parameters
    if "physics_state" in params:
        return fn(state, dt, physics_state=component_state)
    if "process_state" in params:
        return fn(state, dt, process_state=component_state)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return fn(state, dt, physics_state=component_state)
    positional = [
        param
        for param in params.values()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 3:
        return fn(state, dt, component_state)
    return fn(state, dt)


def _normalize_process_result(result: Any, *, fallback_state: Any, fallback_component_state: Any) -> PhysicsStepResult:
    if isinstance(result, PhysicsStepResult):
        return result
    if isinstance(result, tuple):
        if len(result) == 2:
            return PhysicsStepResult(result[0], result[1], {})
        if len(result) == 3:
            return PhysicsStepResult(result[0], result[1], dict(result[2] or {}))
    if isinstance(result, Mapping) and "state" in result:
        return PhysicsStepResult(
            result["state"],
            result.get("physics_state", fallback_component_state),
            dict(result.get("diagnostics", {})),
        )
    if result is None:
        return PhysicsStepResult(fallback_state, fallback_component_state, {})
    return PhysicsStepResult(result, fallback_component_state, {})


def _map_tree(value: Any, fn) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _map_tree(child, fn) for key, child in value.items()}
    if torch.is_tensor(value):
        return fn(value)
    return value


def _clone_leaf(value: torch.Tensor) -> torch.Tensor:
    return value.clone()


def _detach_leaf(value: torch.Tensor) -> torch.Tensor:
    return value.detach()


def _flatten_for_netcdf(
    value: Any,
    path: tuple[str, ...],
    tensor_entries: list[tuple[tuple[str, ...], torch.Tensor]],
    scalar_entries: list[dict[str, Any]],
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _flatten_for_netcdf(child, path + (str(key),), tensor_entries, scalar_entries)
        return
    if torch.is_tensor(value):
        tensor_entries.append((path or ("value",), value))
        return
    scalar = _json_scalar(value)
    if scalar is not _UNSUPPORTED:
        scalar_entries.append({"path": list(path or ("value",)), "value": scalar})
        return
    raise TypeError(
        "PhysicsState NetCDF only supports tensor and JSON scalar leaves; "
        f"got {type(value).__name__} at {path}"
    )


def _assign_tree_value(root: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    if not path:
        raise ValueError("physics-state NetCDF path cannot be empty")
    current = root
    for part in path[:-1]:
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"conflicting physics-state path at {path}")
        current = child
    current[path[-1]] = value


def _safe_var_name(path: tuple[str, ...]) -> str:
    raw = "__".join(path or ("value",))
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw)
    if not safe or safe[0].isdigit():
        safe = f"v_{safe}"
    return safe


class _Unsupported:
    pass


_UNSUPPORTED = _Unsupported()


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return _UNSUPPORTED


__all__ = [
    "PhysicsState",
    "PhysicsStepResult",
    "ProcessPhysicsDriver",
    "initial_physics_state_for_components",
    "step_process_components",
]
