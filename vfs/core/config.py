"""Typed build-time configuration helpers for VFS models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TensorFieldRequirement:
    """Requirement for one tensor field on a state-like object or mapping."""

    name: str
    aliases: tuple[str, ...] = ()
    trailing_shape: tuple[Any, ...] | None = None
    optional: bool = False
    finite: bool = False

    def __init__(
        self,
        name: str,
        *,
        aliases=(),
        trailing_shape: tuple[Any, ...] | None = None,
        optional: bool = False,
        finite: bool = False,
    ) -> None:
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "aliases", tuple(str(alias) for alias in aliases))
        object.__setattr__(self, "trailing_shape", None if trailing_shape is None else tuple(trailing_shape))
        object.__setattr__(self, "optional", bool(optional))
        object.__setattr__(self, "finite", bool(finite))
        if not self.name:
            raise ValueError("TensorFieldRequirement.name must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "trailing_shape": None if self.trailing_shape is None else list(self.trailing_shape),
            "optional": self.optional,
            "finite": self.finite,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorFieldRequirement":
        return cls(
            str(data["name"]),
            aliases=tuple(data.get("aliases", ())),
            trailing_shape=None if data.get("trailing_shape") is None else tuple(data.get("trailing_shape", ())),
            optional=bool(data.get("optional", False)),
            finite=bool(data.get("finite", False)),
        )

    def resolve(self, obj: Any) -> tuple[str | None, Any | None]:
        for name in (self.name,) + self.aliases:
            if isinstance(obj, Mapping) and name in obj and obj[name] is not None:
                return name, obj[name]
            if hasattr(obj, name):
                value = getattr(obj, name)
                if value is not None:
                    return name, value
        return None, None

    def validate_value(
        self,
        value: Any,
        *,
        key: str | None = None,
        owner: str = "component",
        context: Mapping[str, int] | None = None,
    ) -> None:
        name = self.name if key is None else str(key)
        if self.trailing_shape is not None:
            shape = _value_shape(value)
            expected = tuple(_resolve_shape_dim(dim, context or {}) for dim in self.trailing_shape)
            _validate_trailing_shape(owner, name, shape, expected)
        if self.finite:
            try:
                import torch

                tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(_value_payload(value))
                if not torch.isfinite(tensor).all().item():
                    raise FloatingPointError(f"{owner} requires finite tensor field {name!r}")
            except FloatingPointError:
                raise
            except Exception as exc:
                raise ValueError(f"{owner} cannot check finiteness for tensor field {name!r}") from exc

    def validate(
        self,
        obj: Any,
        *,
        owner: str = "component",
        context: Mapping[str, int] | None = None,
    ) -> None:
        key, value = self.resolve(obj)
        if key is None:
            if self.optional:
                return
            choices = (self.name,) + self.aliases
            raise ValueError(f"{owner} requires tensor field {choices}")
        self.validate_value(value, key=key, owner=owner, context=context)


@dataclass(frozen=True)
class DataFieldRequirement:
    """Requirement for one field in an external data mapping."""

    name: str
    aliases: tuple[str, ...] = ()
    trailing_shape: tuple[Any, ...] | None = None
    units: str | None = None
    quantity: str | None = None
    optional: bool = False

    def __init__(
        self,
        name: str,
        *,
        aliases=(),
        trailing_shape: tuple[Any, ...] | None = None,
        units: str | None = None,
        quantity: str | None = None,
        optional: bool = False,
    ) -> None:
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "aliases", tuple(str(alias) for alias in aliases))
        object.__setattr__(self, "trailing_shape", None if trailing_shape is None else tuple(trailing_shape))
        object.__setattr__(self, "units", None if units is None else str(units))
        object.__setattr__(self, "quantity", None if quantity is None else str(quantity))
        object.__setattr__(self, "optional", bool(optional))
        if not self.name:
            raise ValueError("DataFieldRequirement.name must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "trailing_shape": None if self.trailing_shape is None else list(self.trailing_shape),
            "units": self.units,
            "quantity": self.quantity,
            "optional": self.optional,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataFieldRequirement":
        return cls(
            str(data["name"]),
            aliases=tuple(data.get("aliases", ())),
            trailing_shape=None if data.get("trailing_shape") is None else tuple(data.get("trailing_shape", ())),
            units=None if data.get("units") is None else str(data.get("units")),
            quantity=None if data.get("quantity") is None else str(data.get("quantity")),
            optional=bool(data.get("optional", False)),
        )


@dataclass(frozen=True)
class DataRequirement:
    """Semantic contract for external grid/boundary data mappings."""

    fields: tuple[DataFieldRequirement, ...] = ()

    def __init__(self, fields=()) -> None:
        object.__setattr__(self, "fields", tuple(_as_data_field_requirement(field) for field in fields))
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate data field requirements: {names}")

    def to_dict(self) -> dict[str, Any]:
        return {"fields": [field.to_dict() for field in self.fields]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataRequirement":
        return cls(fields=tuple(data.get("fields", ())))

    @classmethod
    def merge(cls, *requirements: "DataRequirement | Mapping[str, Any] | None") -> "DataRequirement":
        by_name: dict[str, DataFieldRequirement] = {}
        for requirement in requirements:
            if requirement is None:
                continue
            requirement = _as_data_requirement(requirement)
            for field_req in requirement.fields:
                if field_req.name in by_name and by_name[field_req.name] != field_req:
                    raise ValueError(f"conflicting data requirements for field {field_req.name!r}")
                by_name[field_req.name] = field_req
        return cls(tuple(by_name.values()))

    def validate(self, data: Mapping[str, Any], *, owner: str = "component", context: Mapping[str, int] | None = None) -> None:
        context = dict(context or {})
        for field_req in self.fields:
            key = _resolve_data_key(data, field_req)
            if key is None:
                if field_req.optional:
                    continue
                choices = (field_req.name,) + field_req.aliases
                raise ValueError(f"{owner} requires data field {choices}")
            value = data[key]
            attrs = _data_value_attrs(value)
            if field_req.units is not None and str(attrs.get("units", "")) != field_req.units:
                raise ValueError(
                    f"{owner} requires data field {key!r} units={field_req.units!r}, "
                    f"got {attrs.get('units')!r}"
                )
            if field_req.quantity is not None and str(attrs.get("quantity", "")) != field_req.quantity:
                raise ValueError(
                    f"{owner} requires data field {key!r} quantity={field_req.quantity!r}, "
                    f"got {attrs.get('quantity')!r}"
                )
            if field_req.trailing_shape is not None:
                shape = _value_shape(value)
                expected = tuple(_resolve_shape_dim(dim, context) for dim in field_req.trailing_shape)
                _validate_trailing_shape(owner, f"data field {key!r}", shape, expected)


def _as_data_field_requirement(value: DataFieldRequirement | Mapping[str, Any]) -> DataFieldRequirement:
    if isinstance(value, DataFieldRequirement):
        return value
    return DataFieldRequirement.from_dict(value)


def _as_data_requirement(value: DataRequirement | Mapping[str, Any]) -> DataRequirement:
    if isinstance(value, DataRequirement):
        return value
    return DataRequirement.from_dict(value)


def _resolve_data_key(data: Mapping[str, Any], requirement: DataFieldRequirement) -> str | None:
    for name in (requirement.name,) + requirement.aliases:
        if name in data:
            return name
    return None


def _data_value_attrs(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        attrs = value.get("attrs", {})
        return attrs if isinstance(attrs, Mapping) else {}
    attrs = getattr(value, "attrs", {})
    return attrs if isinstance(attrs, Mapping) else {}


def _value_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in ("values", "data", "value"):
            if key in value:
                return value[key]
    if hasattr(value, "shape"):
        return value
    for attr in ("values", "data"):
        payload = getattr(value, attr, None)
        if payload is not None and not callable(payload):
            return payload
    return value


def _value_shape(value: Any) -> tuple[int, ...]:
    payload = _value_payload(value)
    shape = getattr(payload, "shape", None)
    if shape is None:
        try:
            import torch

            shape = torch.as_tensor(payload).shape
        except Exception as exc:
            raise ValueError(f"data field has no inspectable shape: {type(value).__name__}") from exc
    return tuple(int(dim) for dim in shape)


def _resolve_shape_dim(dim: Any, context: Mapping[str, int]) -> int:
    if isinstance(dim, str):
        if dim not in context:
            raise ValueError(f"shape token {dim!r} is not available in validation context")
        return int(context[dim])
    return int(dim)


def _validate_trailing_shape(owner: str, name: str, shape: tuple[int, ...], expected: tuple[int, ...]) -> None:
    actual = tuple(shape[-len(expected) :]) if expected else ()
    if len(shape) < len(expected) or actual != expected:
        raise ValueError(
            f"{owner} requires {name} trailing shape {expected}; "
            f"got {shape}"
        )


__all__ = ["TensorFieldRequirement", "DataFieldRequirement", "DataRequirement"]
