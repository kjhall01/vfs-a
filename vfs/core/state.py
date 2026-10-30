"""Self-describing PyTorch-native model state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import torch

from .config import TensorFieldRequirement
from vfs.field import Field


DEFAULT_GRAVITY = 9.80665
STATE_SPEC_ATTR = "state_spec_json"
STATE_SPEC_VERSION = "state_spec_v1"


def interface_mass(dp: torch.Tensor) -> torch.Tensor:
    """Charney-Phillips interface mass from full-level pressure thickness."""

    half = 0.5 * dp
    interior = half[..., :-1] + half[..., 1:]
    return torch.cat([half[..., :1], interior, half[..., -1:]], dim=-1)


def _as_prognostic_tensor(
    value: Field | torch.Tensor,
    *,
    location: str,
    vertical_location: str,
    name: str,
    component_axes: tuple[str, ...] = (),
) -> torch.Tensor:
    if isinstance(value, Field):
        if (
            value.location != location
            or value.vertical_location != vertical_location
            or value.component_axes != component_axes
        ):
            raise ValueError(
                f"{name} must be a {location}/{vertical_location} Field with "
                f"component_axes={component_axes!r}"
            )
        value = value.values
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or a compatible Field")
    if value.ndim < 2 + len(component_axes):
        raise ValueError(f"{name} needs at least {2 + len(component_axes)} axes")
    return value


@dataclass
class PrognosticFields:
    """Tensor-only dry payload; geometry is borrowed by State.

    Compatible legacy Field inputs are unwrapped without copying. Assignments
    obey the same rule, so public payload attributes always expose tensors.
    """

    u_e: torch.Tensor
    mtheta_i: torch.Tensor
    ps: torch.Tensor
    tracer_m: torch.Tensor | None = None

    def __setattr__(self, name, value):
        if name == "u_e":
            value = _as_prognostic_tensor(
                value, location="edge", vertical_location="full", name=name
            )
        elif name == "mtheta_i":
            value = _as_prognostic_tensor(
                value, location="cell", vertical_location="interface", name=name
            )
        elif name == "ps":
            value = _as_prognostic_tensor(
                value, location="cell", vertical_location="surface", name=name
            )
        elif name == "tracer_m" and value is not None:
            value = _as_prognostic_tensor(
                value, location="cell", vertical_location="full", name=name,
                component_axes=("tracer",),
            )
        super().__setattr__(name, value)

    def to(self, *args, **kwargs) -> "PrognosticFields":
        return PrognosticFields(
            self.u_e.to(*args, **kwargs),
            self.mtheta_i.to(*args, **kwargs),
            self.ps.to(*args, **kwargs),
            None if self.tracer_m is None else self.tracer_m.to(*args, **kwargs),
        )

    def clone(self) -> "PrognosticFields":
        return PrognosticFields(
            self.u_e.clone(),
            self.mtheta_i.clone(),
            self.ps.clone(),
            None if self.tracer_m is None else self.tracer_m.clone(),
        )

    def detach(self) -> "PrognosticFields":
        return PrognosticFields(
            self.u_e.detach(),
            self.mtheta_i.detach(),
            self.ps.detach(),
            None if self.tracer_m is None else self.tracer_m.detach(),
        )

    def requires_grad_(self, requires_grad: bool = True) -> "PrognosticFields":
        self.u_e.requires_grad_(requires_grad)
        self.mtheta_i.requires_grad_(requires_grad)
        self.ps.requires_grad_(requires_grad)
        if self.tracer_m is not None and self.tracer_m.is_floating_point():
            self.tracer_m.requires_grad_(requires_grad)
        return self

    def zeros_like(self) -> "PrognosticFields":
        return PrognosticFields(
            torch.zeros_like(self.u_e),
            torch.zeros_like(self.mtheta_i),
            torch.zeros_like(self.ps),
            None if self.tracer_m is None else torch.zeros_like(self.tracer_m),
        )

    def __add__(self, other: "PrognosticFields") -> "PrognosticFields":
        return PrognosticFields(
            self.u_e + other.u_e,
            self.mtheta_i + other.mtheta_i,
            self.ps + other.ps,
            _combine_tracer(self.tracer_m, other.tracer_m, torch.add),
        )

    def __sub__(self, other: "PrognosticFields") -> "PrognosticFields":
        return PrognosticFields(
            self.u_e - other.u_e,
            self.mtheta_i - other.mtheta_i,
            self.ps - other.ps,
            _combine_tracer(self.tracer_m, other.tracer_m, torch.sub),
        )

    def __mul__(self, scalar) -> "PrognosticFields":
        return PrognosticFields(
            self.u_e * scalar,
            self.mtheta_i * scalar,
            self.ps * scalar,
            None if self.tracer_m is None else self.tracer_m * scalar,
        )

    def __rmul__(self, scalar) -> "PrognosticFields":
        return self.__mul__(scalar)

    def __truediv__(self, scalar) -> "PrognosticFields":
        return self.__mul__(1.0 / scalar)


@dataclass(frozen=True)
class TracerSpec:
    """Semantic metadata for one packed tracer in ``tracer_m``."""

    name: str
    index: int
    role: str = "passive"
    quantity: str = "dry_air_mixing_ratio"
    units: str = ""
    horizontal_location: str = "cell"
    vertical_location: str = "full"
    storage: str = "tracer_m"

    def __post_init__(self) -> None:
        if int(self.index) < 0:
            raise ValueError("TracerSpec.index must be non-negative")
        if not str(self.name):
            raise ValueError("TracerSpec.name must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "index": int(self.index),
            "role": str(self.role),
            "quantity": str(self.quantity),
            "units": str(self.units),
            "horizontal_location": str(self.horizontal_location),
            "vertical_location": str(self.vertical_location),
            "storage": str(self.storage),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TracerSpec":
        return cls(
            name=str(data["name"]),
            index=int(data["index"]),
            role=str(data.get("role", "passive")),
            quantity=str(data.get("quantity", "dry_air_mixing_ratio")),
            units=str(data.get("units", "")),
            horizontal_location=str(data.get("horizontal_location", "cell")),
            vertical_location=str(data.get("vertical_location", "full")),
            storage=str(data.get("storage", "tracer_m")),
        )


@dataclass(frozen=True)
class StateSpec:
    """Lightweight schema for State field layout and named tracers."""

    tracers: tuple[TracerSpec, ...] = ()
    version: str = STATE_SPEC_VERSION

    def __init__(self, tracers=(), version: str = STATE_SPEC_VERSION) -> None:
        object.__setattr__(self, "tracers", tuple(_as_tracer_spec(tracer) for tracer in tracers))
        object.__setattr__(self, "version", str(version))
        self._validate_tracer_specs()

    def _validate_tracer_specs(self) -> None:
        indices = [tracer.index for tracer in self.tracers]
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate tracer indices in StateSpec: {indices}")
        names = [tracer.name for tracer in self.tracers]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate tracer names in StateSpec: {names}")
        for tracer in self.tracers:
            if tracer.storage != "tracer_m":
                raise ValueError(f"unsupported tracer storage {tracer.storage!r}")
            if tracer.horizontal_location != "cell" or tracer.vertical_location != "full":
                raise ValueError(
                    "packed tracers must live on cell/full levels; "
                    f"got {tracer.horizontal_location!r}/{tracer.vertical_location!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "fields": {
                "u_e": {"horizontal_location": "edge", "vertical_location": "full"},
                "mtheta_i": {"horizontal_location": "cell", "vertical_location": "interface"},
                "ps": {"horizontal_location": "cell", "vertical_location": "surface"},
                "tracer_m": {"horizontal_location": "cell", "vertical_location": "full", "optional": True},
            },
            "tracers": [tracer.to_dict() for tracer in self.tracers],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_attrs(self) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            STATE_SPEC_ATTR: self.to_json(),
            "state_spec_version": self.version,
            "n_named_tracers": int(len(self.tracers)),
        }
        if self.tracers:
            attrs["tracer_names"] = ",".join(tracer.name for tracer in self.tracers)
        return attrs

    def tracer(self, name: str) -> TracerSpec | None:
        for tracer in self.tracers:
            if tracer.name == str(name):
                return tracer
        return None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateSpec":
        return cls(
            tracers=tuple(TracerSpec.from_dict(item) for item in data.get("tracers", ())),
            version=str(data.get("version", STATE_SPEC_VERSION)),
        )

    @classmethod
    def from_attrs(cls, attrs: dict[str, Any] | None) -> "StateSpec":
        attrs = dict(attrs or {})
        raw = attrs.get(STATE_SPEC_ATTR)
        if raw:
            try:
                return cls.from_dict(json.loads(str(raw)))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid {STATE_SPEC_ATTR!r} metadata") from exc
        return cls(tracers=_legacy_tracer_specs_from_attrs(attrs))

    def validate(self, state: "State") -> None:
        mesh = state._require_mesh()
        column = state._require_column()
        fields = state._require_fields()
        n_cells = int(mesh.nCells)
        n_edges = int(mesh.nEdges)
        n_levels = int(column.n_levels)
        _require_trailing_shape("u_e", fields.u_e, (n_edges, n_levels))
        _require_trailing_shape("mtheta_i", fields.mtheta_i, (n_cells, n_levels + 1))
        _require_trailing_shape("ps", fields.ps, (n_cells, 1))
        if fields.tracer_m is None:
            if self.tracers:
                raise ValueError("StateSpec declares tracers but State.tracer_m is absent")
            return
        tracer_m = _normalize_tracer_tensor(fields.tracer_m, n_cells, n_levels)
        if self.tracers:
            n_tracers = int(tracer_m.shape[-1])
            max_index = max(tracer.index for tracer in self.tracers)
            if max_index >= n_tracers:
                raise ValueError(
                    f"StateSpec references tracer index {max_index}, "
                    f"but State.tracer_m has only {n_tracers} tracers"
                )


@dataclass(frozen=True)
class TracerRequirement:
    """Semantic requirement for a named packed tracer."""

    name: str
    quantity: str | None = None
    role: str | None = None
    index: int | None = None
    units: str | None = None

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("TracerRequirement.name must be non-empty")
        if self.index is not None and int(self.index) < 0:
            raise ValueError("TracerRequirement.index must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "quantity": self.quantity,
            "role": self.role,
            "index": None if self.index is None else int(self.index),
            "units": self.units,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TracerRequirement":
        index = data.get("index")
        return cls(
            name=str(data["name"]),
            quantity=None if data.get("quantity") is None else str(data.get("quantity")),
            role=None if data.get("role") is None else str(data.get("role")),
            index=None if index is None else int(index),
            units=None if data.get("units") is None else str(data.get("units")),
        )

    def merged(self, other: "TracerRequirement") -> "TracerRequirement":
        if self.name != other.name:
            raise ValueError(f"cannot merge requirements for different tracers {self.name!r} and {other.name!r}")
        return TracerRequirement(
            name=self.name,
            quantity=_merge_optional_requirement(self.quantity, other.quantity, f"{self.name}.quantity"),
            role=_merge_optional_requirement(self.role, other.role, f"{self.name}.role"),
            index=_merge_optional_requirement(self.index, other.index, f"{self.name}.index"),
            units=_merge_optional_requirement(self.units, other.units, f"{self.name}.units"),
        )


@dataclass(frozen=True)
class StateRequirement:
    """Component contract for the State fields and named tracers it needs."""

    fields: tuple[TensorFieldRequirement, ...] = ()
    tracers: tuple[TracerRequirement, ...] = ()
    require_mesh: bool = True
    require_column: bool = True
    require_fields: bool = True

    def __init__(
        self,
        fields=(),
        tracers=(),
        *,
        require_mesh: bool = True,
        require_column: bool = True,
        require_fields: bool = True,
    ) -> None:
        object.__setattr__(self, "fields", tuple(_as_tensor_field_requirement(req) for req in fields))
        object.__setattr__(self, "tracers", tuple(_as_tracer_requirement(req) for req in tracers))
        object.__setattr__(self, "require_mesh", bool(require_mesh))
        object.__setattr__(self, "require_column", bool(require_column))
        object.__setattr__(self, "require_fields", bool(require_fields))
        field_names = [req.name for req in self.fields]
        if len(field_names) != len(set(field_names)):
            raise ValueError(f"duplicate state field requirements: {field_names}")
        names = [req.name for req in self.tracers]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate tracer requirements: {names}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": [req.to_dict() for req in self.fields],
            "tracers": [req.to_dict() for req in self.tracers],
            "require_mesh": self.require_mesh,
            "require_column": self.require_column,
            "require_fields": self.require_fields,
        }

    @classmethod
    def merge(cls, *requirements: "StateRequirement | None") -> "StateRequirement":
        by_field: dict[str, TensorFieldRequirement] = {}
        by_name: dict[str, TracerRequirement] = {}
        require_mesh = False
        require_column = False
        require_fields = False
        for requirement in requirements:
            if requirement is None:
                continue
            requirement = _as_state_requirement(requirement)
            require_mesh = require_mesh or requirement.require_mesh
            require_column = require_column or requirement.require_column
            require_fields = require_fields or requirement.require_fields
            for field_req in requirement.fields:
                if field_req.name in by_field and by_field[field_req.name] != field_req:
                    raise ValueError(f"conflicting state field requirements for field {field_req.name!r}")
                by_field[field_req.name] = field_req
            for tracer in requirement.tracers:
                by_name[tracer.name] = by_name[tracer.name].merged(tracer) if tracer.name in by_name else tracer
        return cls(
            fields=tuple(by_field.values()),
            tracers=tuple(by_name.values()),
            require_mesh=require_mesh,
            require_column=require_column,
            require_fields=require_fields,
        )

    def validate(self, state: "State", *, owner: str = "component") -> None:
        if self.require_mesh:
            state._require_mesh()
        if self.require_column:
            state._require_column()
        fields = None
        if self.require_fields or self.fields:
            fields = state._require_fields()
        if self.require_fields:
            state._require_fields()
        if self.require_fields or self.tracers:
            state.validate_schema()
        if self.fields:
            context = _state_requirement_context(state)
            for field_req in self.fields:
                field_req.validate(fields, owner=owner, context=context)
        spec = state.spec
        for requirement in self.tracers:
            tracer = spec.tracer(requirement.name)
            if tracer is None:
                hint = _legacy_tracer_attr_hint(requirement.name)
                raise ValueError(f"{owner} requires named tracer {requirement.name!r} ({hint})")
            if requirement.index is not None and int(tracer.index) != int(requirement.index):
                raise ValueError(
                    f"{owner} requires tracer {requirement.name!r} at index {int(requirement.index)}, "
                    f"got {int(tracer.index)}"
                )
            if requirement.quantity is not None and str(tracer.quantity) != str(requirement.quantity):
                raise ValueError(
                    f"{owner} requires tracer {requirement.name!r} quantity={requirement.quantity!r}, "
                    f"got {tracer.quantity!r}"
                )
            if requirement.role is not None and str(tracer.role) != str(requirement.role):
                raise ValueError(
                    f"{owner} requires tracer {requirement.name!r} role={requirement.role!r}, "
                    f"got {tracer.role!r}"
                )
            if requirement.units is not None and str(tracer.units) != str(requirement.units):
                raise ValueError(
                    f"{owner} requires tracer {requirement.name!r} units={requirement.units!r}, "
                    f"got {tracer.units!r}"
                )


@dataclass(init=False)
class State:
    """Prognostic fields bound to mesh/column interpretation.

    ``State`` is intentionally not an ``nn.Module``. Recipe/model classes own
    long-lived mesh and column modules; State carries references to them so the
    tensor fields can describe themselves. Derived methods remain uncached legacy
    compatibility helpers; prefer model.physical_fields(state) in new workflows.
    Construction shares tensors without copying, detaching or converting them.
    """

    fields: PrognosticFields | None
    mesh: Any | None = None
    column: Any | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        u_e: torch.Tensor | PrognosticFields | None = None,
        mtheta_i: torch.Tensor | None = None,
        ps: torch.Tensor | None = None,
        tracer_m: torch.Tensor | None = None,
        *,
        fields: PrognosticFields | None = None,
        mesh: Any | None = None,
        column: Any | None = None,
        attrs: dict[str, Any] | None = None,
        spec: StateSpec | None = None,
    ) -> None:
        if fields is not None:
            if any(value is not None for value in (u_e, mtheta_i, ps, tracer_m)):
                raise TypeError("pass either fields=... or raw tensors, not both")
            payload = fields
        elif isinstance(u_e, PrognosticFields) and mtheta_i is None and ps is None and tracer_m is None:
            payload = u_e
        elif u_e is None and mtheta_i is None and ps is None and tracer_m is None:
            if mesh is None and column is None:
                raise TypeError("State requires prognostic tensors or mesh/column binding for an unloaded restart shell")
            payload = None
        else:
            if u_e is None or mtheta_i is None or ps is None:
                raise TypeError("State requires u_e, mtheta_i, and ps tensors")
            payload = PrognosticFields(u_e, mtheta_i, ps, tracer_m)
        self.fields = payload
        self.mesh = mesh
        self.column = column
        self.attrs = dict(attrs or {})
        if spec is not None:
            self.attrs.update(spec.to_attrs())

    def _require_fields(self) -> PrognosticFields:
        if self.fields is None:
            raise ValueError("State has no prognostic fields; load or construct fields before using tensors")
        return self.fields

    @property
    def u_e(self) -> torch.Tensor:
        return self._require_fields().u_e

    @u_e.setter
    def u_e(self, value: Field | torch.Tensor) -> None:
        fields = self._require_fields()
        self.fields = PrognosticFields(value, fields.mtheta_i, fields.ps, fields.tracer_m)

    @property
    def mtheta_i(self) -> torch.Tensor:
        return self._require_fields().mtheta_i

    @mtheta_i.setter
    def mtheta_i(self, value: Field | torch.Tensor) -> None:
        fields = self._require_fields()
        self.fields = PrognosticFields(fields.u_e, value, fields.ps, fields.tracer_m)

    @property
    def ps(self) -> torch.Tensor:
        return self._require_fields().ps

    @ps.setter
    def ps(self, value: Field | torch.Tensor) -> None:
        fields = self._require_fields()
        self.fields = PrognosticFields(fields.u_e, fields.mtheta_i, value, fields.tracer_m)

    @property
    def tracer_m(self) -> torch.Tensor | None:
        return self._require_fields().tracer_m

    @tracer_m.setter
    def tracer_m(self, value: Field | torch.Tensor | None) -> None:
        fields = self._require_fields()
        self.fields = PrognosticFields(fields.u_e, fields.mtheta_i, fields.ps, value)

    @classmethod
    def from_theta(
        cls,
        u_e: torch.Tensor,
        theta_i: torch.Tensor,
        ps: torch.Tensor,
        *,
        mesh,
        column,
        tracer_m: torch.Tensor | None = None,
        q_tracers: torch.Tensor | None = None,
        attrs: dict[str, Any] | None = None,
        spec: StateSpec | None = None,
    ) -> "State":
        """Construct from interface theta (K) and surface pressure (Pa).

        Tracers must already match pressure dtype/device; conversion is explicit.
        The stored mtheta_i is pressure-weighted theta, not moist entropy.
        """

        if tracer_m is not None and q_tracers is not None:
            raise ValueError("pass only one of tracer_m or q_tracers")
        ps_values = ps.values if isinstance(ps, Field) else ps
        dp = column.layer_thickness(ps_values)
        dm = interface_mass(dp)
        tracer_state = None
        for name, tensor in (("q_tracers", q_tracers), ("tracer_m", tracer_m)):
            if tensor is not None and (tensor.dtype != dp.dtype or tensor.device != dp.device):
                raise ValueError(f"{name} must match pressure dtype/device; convert explicitly with .to()")
        if q_tracers is not None:
            q = _normalize_tracer_tensor(q_tracers, mesh.nCells, column.n_levels)
            tracer_state = q * dp[..., None]
        elif tracer_m is not None:
            tracer_state = _normalize_tracer_tensor(tracer_m, mesh.nCells, column.n_levels)
        out_attrs = dict(attrs or {})
        out_attrs.setdefault("gravity", DEFAULT_GRAVITY)
        theta_values = theta_i.values if isinstance(theta_i, Field) else theta_i
        return cls(u_e, dm * theta_values, ps, tracer_state, mesh=mesh, column=column, attrs=out_attrs, spec=spec)

    def _attrs_copy(self) -> dict[str, Any]:
        return dict(self.attrs or {})

    @property
    def spec(self) -> StateSpec:
        return StateSpec.from_attrs(self.attrs)

    def validate_schema(self, spec: StateSpec | None = None) -> None:
        """Validate tensor layout and named tracer indices against a schema."""

        (spec or self.spec).validate(self)

    def _with_tensors(
        self,
        u_e: torch.Tensor,
        mtheta_i: torch.Tensor,
        ps: torch.Tensor,
        tracer_m: torch.Tensor | None,
    ) -> "State":
        return self.with_fields(PrognosticFields(u_e, mtheta_i, ps, tracer_m))

    def with_fields(self, fields: PrognosticFields) -> "State":
        """Return a new State with the same mesh/column binding."""

        return State(fields=fields, mesh=self.mesh, column=self.column, attrs=self._attrs_copy())

    def to_netcdf(self, path, *, metadata: dict[str, Any] | None = None) -> str:
        """Write this native State snapshot as NetCDF.

        This is authoritative state I/O: it writes prognostic fields plus
        geometry/column-derived fields that can be recovered from State alone.
        Dynamics-specific enrichment belongs in diagnostics products.
        """

        from vfs.core.io import _save_native_state_netcdf

        return _save_native_state_netcdf(path, self, metadata=metadata)

    def load_netcdf(self, path, *, dtype=None, device=None) -> "State":
        """Return a State loaded from a native NetCDF snapshot.

        The receiver supplies mesh/column binding and is not mutated.
        """

        mesh = self.mesh
        column = self.column
        if mesh is None or column is None:
            raise ValueError("State.load_netcdf requires a receiver with mesh=... and column=...")
        try:
            import xarray as xr
        except Exception as exc:  # pragma: no cover - optional backend dependent.
            raise RuntimeError("Loading NetCDF snapshots requires xarray") from exc

        ds = xr.open_dataset(path)
        try:
            required = ("u_edge", "mtheta_interface", "ps")
            missing = [name for name in required if name not in ds.variables]
            if missing:
                raise ValueError(f"NetCDF snapshot is missing required State variables: {missing}")
            u_e = torch.as_tensor(ds["u_edge"].values)
            mtheta_i = torch.as_tensor(ds["mtheta_interface"].values)
            ps = torch.as_tensor(ds["ps"].values)
            if ps.ndim == 1:
                ps = ps[:, None]
            tracer_m = torch.as_tensor(ds["tracer_m"].values) if "tracer_m" in ds.variables else None
        finally:
            attrs = dict(ds.attrs)
            ds.close()
        if dtype is not None or device is not None:
            u_e = u_e.to(dtype=dtype if dtype is not None else u_e.dtype, device=device if device is not None else u_e.device)
            mtheta_i = mtheta_i.to(
                dtype=dtype if dtype is not None else mtheta_i.dtype,
                device=device if device is not None else mtheta_i.device,
            )
            ps = ps.to(dtype=dtype if dtype is not None else ps.dtype, device=device if device is not None else ps.device)
            if tracer_m is not None:
                tracer_m = tracer_m.to(
                    dtype=dtype if dtype is not None else tracer_m.dtype,
                    device=device if device is not None else tracer_m.device,
                )
        out = State(u_e, mtheta_i, ps, tracer_m, mesh=mesh, column=column, attrs=attrs)
        out._validate_restart_metadata()
        out.validate_schema()
        return out

    def _validate_restart_metadata(self) -> None:
        mesh = self._require_mesh()
        column = self._require_column()
        attrs = dict(self.attrs or {})
        expected = {
            "n_cells": int(mesh.nCells),
            "n_edges": int(mesh.nEdges),
            "n_levels": int(column.n_levels),
        }
        for key, value in expected.items():
            if key in attrs and str(attrs[key]) not in ("", str(value)):
                raise ValueError(f"NetCDF restart metadata {key}={attrs[key]!r} does not match expected {value}")

    def _combined_references(self, other: "State") -> tuple[Any | None, Any | None, dict[str, Any]]:
        mesh = self.mesh if self.mesh is not None else other.mesh
        column = self.column if self.column is not None else other.column
        if self.mesh is not None and other.mesh is not None and self.mesh is not other.mesh:
            raise ValueError("cannot combine State objects with different mesh references")
        if self.column is not None and other.column is not None and self.column is not other.column:
            raise ValueError("cannot combine State objects with different column references")
        attrs = dict(other.attrs or {})
        attrs.update(self.attrs or {})
        return mesh, column, attrs

    def _require_mesh(self):
        if self.mesh is None:
            raise ValueError("State has no mesh; construct it with mesh=...")
        return self.mesh

    def _require_column(self):
        if self.column is None:
            raise ValueError("State has no column; construct it with column=...")
        return self.column

    def _gravity(self) -> float:
        return float((self.attrs or {}).get("gravity", DEFAULT_GRAVITY))

    def to(self, *args, **kwargs) -> "State":
        """Move prognostic tensors; mesh/column references are preserved."""

        return self.with_fields(self._require_fields().to(*args, **kwargs))

    def clone(self) -> "State":
        return self.with_fields(self._require_fields().clone())

    def detach(self) -> "State":
        return self.with_fields(self._require_fields().detach())

    def requires_grad_(self, requires_grad: bool = True) -> "State":
        self._require_fields().requires_grad_(requires_grad)
        return self

    def zeros_like(self) -> "State":
        return self.with_fields(self._require_fields().zeros_like())

    def __add__(self, other: "State") -> "State":
        mesh, column, attrs = self._combined_references(other)
        return State(fields=self.fields + other.fields, mesh=mesh, column=column, attrs=attrs)

    def __sub__(self, other: "State") -> "State":
        mesh, column, attrs = self._combined_references(other)
        return State(fields=self.fields - other.fields, mesh=mesh, column=column, attrs=attrs)

    def __mul__(self, scalar) -> "State":
        return self.with_fields(self.fields * scalar)

    def __rmul__(self, scalar) -> "State":
        return self.__mul__(scalar)

    def __truediv__(self, scalar) -> "State":
        return self.__mul__(1.0 / scalar)

    def grad(self, cell_field: torch.Tensor) -> torch.Tensor:
        return self._require_mesh().grad(cell_field)

    def div(self, edge_flux: torch.Tensor) -> torch.Tensor:
        return self._require_mesh().div(edge_flux)

    def curl(self, edge_field: torch.Tensor) -> torch.Tensor:
        return self._require_mesh().curl(edge_field)

    def curl_adjoint(self, vertex_field: torch.Tensor) -> torch.Tensor:
        return self._require_mesh().curl_adjoint(vertex_field)

    def vector_laplacian_edge(self, edge_field: torch.Tensor | None = None) -> torch.Tensor:
        return self._require_mesh().vector_laplacian_edge(self.u_e if edge_field is None else edge_field)

    def cell_to_edge(self, cell_field: torch.Tensor) -> torch.Tensor:
        return self._require_mesh().cell_to_edge(cell_field)

    def edge_to_cell(self, edge_field: torch.Tensor) -> torch.Tensor:
        return self._require_mesh().edge_to_cell(edge_field)

    def tangential_velocity(self, edge_field: torch.Tensor) -> torch.Tensor:
        return self._require_mesh().tangential_velocity(edge_field)

    def tangential_velocity_adjoint(self, edge_field: torch.Tensor) -> torch.Tensor:
        return self._require_mesh().tangential_velocity_adjoint(edge_field)

    def ke_cell(self, edge_wind: torch.Tensor | None = None) -> torch.Tensor:
        return self._require_mesh().ke_cell(self.u_e if edge_wind is None else edge_wind)

    def cell_wind_vector(self, edge_wind: torch.Tensor | None = None) -> torch.Tensor:
        return self._require_mesh().cell_wind_vector(self.u_e if edge_wind is None else edge_wind)

    def layer_dp(self) -> torch.Tensor:
        return self._require_column().layer_thickness(self.ps)

    def interface_pressure(self) -> torch.Tensor:
        return self._require_column().interface_pressure(self.ps)

    def full_pressure(self) -> torch.Tensor:
        return self._require_column().full_pressure(self.ps)

    def exner_interface(self) -> torch.Tensor:
        return self._require_column().exner_interface(self.ps)

    def exner_full(self) -> torch.Tensor:
        return self._require_column().exner_full(self.ps)

    def theta(self) -> torch.Tensor:
        return self.mtheta_i / interface_mass(self.layer_dp())

    def temperature(self) -> torch.Tensor:
        return self._require_column().temperature_full(self.theta(), self.ps)

    def geopotential(self) -> torch.Tensor:
        return self._require_column().geopotential_full(self.theta(), self.ps)

    def specific_volume(self) -> torch.Tensor:
        return self._require_column().specific_volume_full(self.theta(), self.ps)

    def tracers(self) -> torch.Tensor | None:
        if self.tracer_m is None:
            return None
        tracer_m = _normalize_tracer_tensor(self.tracer_m, self._require_mesh().nCells, self._require_column().n_levels)
        dp = self.layer_dp()
        return tracer_m / dp[..., None].clamp_min(torch.finfo(dp.dtype).tiny)

    def total_mass(self) -> torch.Tensor:
        dp = self.layer_dp()
        return self._require_mesh().integrate_cell(dp.sum(dim=-1, keepdim=True)) / self._gravity()

    def total_tracer_mass(self) -> torch.Tensor | None:
        if self.tracer_m is None:
            return None
        tracer_m = _normalize_tracer_tensor(self.tracer_m, self._require_mesh().nCells, self._require_column().n_levels)
        return self._require_mesh().integrate_cell(tracer_m.sum(dim=-2)) / self._gravity()

    def static_stability(self) -> torch.Tensor:
        theta = self.theta()
        return theta[..., :-1] - theta[..., 1:]

    def static_instability_penalty(self, margin: float = 0.0) -> torch.Tensor:
        return torch.relu(float(margin) - self.static_stability()).square().mean()

    def to_latlon(self, regridder, *, fields="default") -> dict[str, Any]:
        """Return common State fields on a latitude/longitude grid.

        The regridder owns cached weights.  State only prepares canonical
        variables and delegates the remapping.
        """

        if not hasattr(regridder, "state_to_latlon"):
            raise TypeError("State.to_latlon expects a regridder with state_to_latlon(state, fields=...)")
        return regridder.state_to_latlon(self, fields=fields)


def _combine_tracer(left: torch.Tensor | None, right: torch.Tensor | None, op) -> torch.Tensor | None:
    if left is None and right is None:
        return None
    if left is not None and right is not None:
        return op(left, right)
    raise ValueError("cannot combine State objects with inconsistent tracer_m presence")


def _as_tracer_spec(value: TracerSpec | dict[str, Any]) -> TracerSpec:
    if isinstance(value, TracerSpec):
        return value
    return TracerSpec.from_dict(dict(value))


def _as_tracer_requirement(value: TracerRequirement | dict[str, Any]) -> TracerRequirement:
    if isinstance(value, TracerRequirement):
        return value
    return TracerRequirement.from_dict(dict(value))


def _as_tensor_field_requirement(value: TensorFieldRequirement | dict[str, Any]) -> TensorFieldRequirement:
    if isinstance(value, TensorFieldRequirement):
        return value
    return TensorFieldRequirement.from_dict(dict(value))


def _as_state_requirement(value: StateRequirement | dict[str, Any]) -> StateRequirement:
    if isinstance(value, StateRequirement):
        return value
    return StateRequirement(**dict(value))


def _merge_optional_requirement(left, right, label: str):
    if left is None:
        return right
    if right is None:
        return left
    if left != right:
        raise ValueError(f"conflicting state requirements for {label}: {left!r} vs {right!r}")
    return left


def _legacy_tracer_attr_hint(name: str) -> str:
    if name == "vapor":
        return "vapor_tracer_index"
    if name == "cloud_liquid":
        return "cloud_liquid_tracer_index"
    return f"StateSpec.tracers[{name}]"


def _legacy_tracer_specs_from_attrs(attrs: dict[str, Any]) -> tuple[TracerSpec, ...]:
    specs: list[TracerSpec] = []
    if attrs.get("vapor_tracer_index") not in (None, ""):
        specs.append(
            TracerSpec(
                name="vapor",
                index=int(attrs["vapor_tracer_index"]),
                role=str(attrs.get("vapor_role", "passive")),
                quantity=str(attrs.get("vapor_quantity", "dry_air_mixing_ratio")),
                units=str(attrs.get("vapor_units", "kg kg-1 dry_air")),
            )
        )
    if attrs.get("cloud_liquid_tracer_index") not in (None, ""):
        specs.append(
            TracerSpec(
                name="cloud_liquid",
                index=int(attrs["cloud_liquid_tracer_index"]),
                role=str(attrs.get("cloud_liquid_role", "active")),
                quantity=str(attrs.get("cloud_liquid_quantity", "dry_air_mixing_ratio")),
                units=str(attrs.get("cloud_liquid_units", "kg kg-1 dry_air")),
            )
        )
    return tuple(specs)


def _state_requirement_context(state: State) -> dict[str, int]:
    mesh = state._require_mesh()
    column = state._require_column()
    context = {
        "nCells": int(mesh.nCells),
        "nEdges": int(mesh.nEdges),
        "nLevels": int(column.n_levels),
        "nInterfaces": int(column.n_levels) + 1,
    }
    if state.fields is not None and state.fields.tracer_m is not None:
        tracer_m = _normalize_tracer_tensor(state.fields.tracer_m, mesh.nCells, column.n_levels)
        context["nTracers"] = int(tracer_m.shape[-1])
    return context


def _require_trailing_shape(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if value.ndim < len(shape) or tuple(int(dim) for dim in value.shape[-len(shape) :]) != tuple(shape):
        raise ValueError(f"{name} must have trailing shape {shape}; got {tuple(value.shape)}")


def _normalize_tracer_tensor(value: torch.Tensor, n_cells: int, n_levels: int) -> torch.Tensor:
    if value.ndim >= 2 and value.shape[-2] == n_cells and value.shape[-1] == n_levels:
        value = value.unsqueeze(-1)
    if value.ndim < 3 or value.shape[-3] != n_cells or value.shape[-2] != n_levels:
        raise ValueError(
            "tracer tensor must have shape [..., nCells, N] or "
            f"[..., nCells, N, nTracers]; got {tuple(value.shape)}, "
            f"expected nCells={n_cells}, N={n_levels}"
        )
    return value


def _register_pytree_nodes() -> None:
    try:
        from torch.utils import _pytree
    except Exception:
        return

    def flatten_field(value: Field):
        return (value.values,), (
            value.location,
            value.vertical_location,
            value.name,
            value.units,
            value.component_axes,
        )

    def unflatten_field(children, context):
        (values,) = children
        location, vertical_location, name, units, component_axes = context
        return Field(
            values,
            location=location,
            vertical_location=vertical_location,
            name=name,
            units=units,
            component_axes=component_axes,
        )

    def flatten_fields(fields: PrognosticFields):
        if fields.tracer_m is None:
            return (fields.u_e, fields.mtheta_i, fields.ps), False
        return (fields.u_e, fields.mtheta_i, fields.ps, fields.tracer_m), True

    def unflatten_fields(children, has_tracer: bool):
        if has_tracer:
            u_e, mtheta_i, ps, tracer_m = children
        else:
            u_e, mtheta_i, ps = children
            tracer_m = None
        return PrognosticFields(u_e, mtheta_i, ps, tracer_m)

    def flatten_state(state: State):
        return (state._require_fields(),), (state.mesh, state.column, state._attrs_copy())

    def unflatten_state(children, context):
        (fields,) = children
        mesh, column, attrs = context
        return State(fields=fields, mesh=mesh, column=column, attrs=attrs)

    for cls, flatten, unflatten in (
        (Field, flatten_field, unflatten_field),
        (PrognosticFields, flatten_fields, unflatten_fields),
        (State, flatten_state, unflatten_state),
    ):
        try:
            _pytree.register_pytree_node(cls, flatten, unflatten)
        except ValueError:
            pass


_register_pytree_nodes()
