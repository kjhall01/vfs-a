"""Location-aware tensor values used by the VFS 1.0 public API.

``Field`` deliberately keeps numerical storage in an ordinary
``torch.Tensor``.  The wrapper adds the geometric interpretation that used to
be implicit in a caller's choice of mesh operator.

VFS uses the historical main-repository axis convention: the horizontal axis
comes before every vertical or component payload axis.  Thus an atmospheric
edge/full-level wind is ``[..., n_edges, n_levels]`` and a packed tracer is
``[..., n_cells, n_levels, n_tracers]``.  Leading axes are batch axes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from typing import Literal

import torch


HorizontalLocation = Literal["cell", "edge", "vertex", "latlon"]
VerticalLocation = Literal["full", "interface", "surface"]


@dataclass(frozen=True)
class Field:
    """A tensor together with its horizontal and vertical placement.

    ``component_axes`` names trailing axes after the optional vertical axis.
    It is primarily needed for packed tracers, for example
    ``("tracer",)``.  Native mesh locations have one horizontal axis; a
    ``"latlon"`` Field has adjacent latitude/longitude axes before every
    vertical/component payload axis.  The metadata makes placement
    unambiguous even when a field has vertical levels and components.
    """

    values: torch.Tensor
    location: HorizontalLocation
    vertical_location: VerticalLocation | None = None
    name: str | None = None
    units: str | None = None
    component_axes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("Field.values must be a torch.Tensor")
        if self.location not in {"cell", "edge", "vertex", "latlon"}:
            raise ValueError(
                "Field location must be one of 'cell', 'edge', 'vertex', or 'latlon'; "
                f"got {self.location!r}."
            )
        if self.vertical_location not in {None, "full", "interface", "surface"}:
            raise ValueError(
                "Field vertical_location must be None, 'full', 'interface', "
                f"or 'surface'; got {self.vertical_location!r}."
            )
        if any(not isinstance(axis, str) or not axis for axis in self.component_axes):
            raise ValueError("Field component_axes must contain non-empty strings")
        required_axes = self.horizontal_ndim + len(self.component_axes) + (self.vertical_location is not None)
        if self.values.ndim < required_axes:
            raise ValueError(
                f"Field {self.name or '<unnamed>'!r} needs at least {required_axes} "
                f"axes for its placement metadata; got shape {tuple(self.values.shape)}"
            )

    @property
    def horizontal_ndim(self) -> int:
        """Number of contiguous horizontal axes represented by this Field."""

        return 2 if self.location == "latlon" else 1

    @property
    def horizontal_axis(self) -> int:
        """First horizontal axis (cell/edge/vertex or latitude)."""

        return -self.horizontal_ndim - len(self.component_axes) - (self.vertical_location is not None)

    @property
    def horizontal_axes(self) -> tuple[int, ...]:
        """Contiguous horizontal axes in storage order."""

        first = self.horizontal_axis
        return tuple(range(first, first + self.horizontal_ndim))

    @property
    def horizontal_shape(self) -> tuple[int, ...]:
        """Native horizontal shape, e.g. ``(nCells,)`` or ``(nlat, nlon)``."""

        ndim = self.values.ndim
        return tuple(int(self.values.shape[axis % ndim]) for axis in self.horizontal_axes)

    @property
    def horizontal_size(self) -> int:
        return int(prod(self.horizontal_shape))

    @property
    def vertical_axis(self) -> int | None:
        if self.vertical_location is None:
            return None
        return -1 - len(self.component_axes)

    def with_values(self, values: torch.Tensor, *, name: str | None = None) -> "Field":
        """Preserve placement metadata while replacing numerical values."""

        return replace(self, values=values, name=self.name if name is None else name)

    def to(self, *args, **kwargs) -> "Field":
        return self.with_values(self.values.to(*args, **kwargs))

    def clone(self) -> "Field":
        return self.with_values(self.values.clone())

    def detach(self) -> "Field":
        return self.with_values(self.values.detach())

    def zeros_like(self) -> "Field":
        return self.with_values(torch.zeros_like(self.values))

    def requires_grad_(self, requires_grad: bool = True) -> "Field":
        self.values.requires_grad_(requires_grad)
        return self

    # These small forwarding hooks make the Field migration practical for
    # owner modules while they are converted from tensor algebra to explicit
    # ``.values`` kernels.  They intentionally expose only tensor *results*;
    # callers cannot lose placement metadata by passing a Field through a mesh
    # operator, which is the public VFS boundary.
    def __getattr__(self, name):
        # ``torch.load`` creates the object before restoring dataclass fields.
        # Reject the missing storage attribute directly, rather than trying to
        # proxy it, so deserialization does not recurse.  The ordinary
        # ``self.values`` path also stays legible to ``torch.compile``.
        if name == "values":
            raise AttributeError(name)
        return getattr(self.values, name)

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value) -> None:
        """Mutate numerical storage without changing Field placement.

        A few conservative process kernels perform indexed updates on cloned
        prognostic fields.  Supporting that operation avoids stripping the
        placement wrapper during the VFS 1.0 migration.
        """

        self.values[key] = value.values if isinstance(value, Field) else value

    def __len__(self) -> int:
        return len(self.values)

    @classmethod
    def __torch_function__(cls, function, types, args=(), kwargs=None):
        """Let tensor-only transitional kernels consume Field values.

        The result is deliberately a tensor: placement-changing operations
        must be expressed through mesh/regridding APIs, which construct a new
        Field with validated metadata.
        """

        if kwargs is None:
            kwargs = {}

        def unwrap(value):
            if isinstance(value, Field):
                return value.values
            if isinstance(value, tuple):
                return tuple(unwrap(item) for item in value)
            if isinstance(value, list):
                return [unwrap(item) for item in value]
            if isinstance(value, dict):
                return {key: unwrap(item) for key, item in value.items()}
            return value

        return function(*unwrap(args), **unwrap(kwargs))

    def _binary(self, other, operation) -> "Field":
        other_values = other.values if isinstance(other, Field) else other
        if isinstance(other, Field) and (
            self.location != other.location
            or self.vertical_location != other.vertical_location
            or self.component_axes != other.component_axes
        ):
            raise ValueError("cannot combine Field values with different placement metadata")
        return self.with_values(operation(self.values, other_values))

    def __add__(self, other) -> "Field":
        return self._binary(other, torch.add)

    def __radd__(self, other) -> "Field":
        return self + other

    def __sub__(self, other) -> "Field":
        return self._binary(other, torch.sub)

    def __rsub__(self, other) -> "Field":
        return self.with_values(torch.sub(other, self.values))

    def __mul__(self, other) -> "Field":
        return self._binary(other, torch.mul)

    def __rmul__(self, other) -> "Field":
        return self * other

    def __truediv__(self, other) -> "Field":
        return self._binary(other, torch.div)

    def __rtruediv__(self, other) -> "Field":
        return self.with_values(torch.div(other, self.values))

    def __pow__(self, other) -> "Field":
        return self._binary(other, torch.pow)

    def __neg__(self) -> "Field":
        return self.with_values(torch.neg(self.values))

    def __abs__(self) -> "Field":
        return self.with_values(torch.abs(self.values))

    # Comparisons describe numerical masks rather than spatial quantities, so
    # they intentionally return tensors.
    def __lt__(self, other) -> torch.Tensor:
        return self.values < (other.values if isinstance(other, Field) else other)

    def __le__(self, other) -> torch.Tensor:
        return self.values <= (other.values if isinstance(other, Field) else other)

    def __gt__(self, other) -> torch.Tensor:
        return self.values > (other.values if isinstance(other, Field) else other)

    def __ge__(self, other) -> torch.Tensor:
        return self.values >= (other.values if isinstance(other, Field) else other)

    def __eq__(self, other) -> torch.Tensor:  # type: ignore[override]
        return self.values == (other.values if isinstance(other, Field) else other)

    def __ne__(self, other) -> torch.Tensor:  # type: ignore[override]
        return self.values != (other.values if isinstance(other, Field) else other)


def require_field(field: Field, *, location: HorizontalLocation, name: str = "field") -> Field:
    """Validate a public mesh-operator argument and return it unchanged."""

    if not isinstance(field, Field):
        raise TypeError(f"{name} must be a vfs.Field object")
    if field.location != location:
        raise ValueError(f"{name} must be located on {location}s, got {field.location!r}")
    return field


# Field is a VFS-owned, data-only checkpoint type.  Registering it lets the
# PyTorch 2.6+ safe ``weights_only`` loader restore State snapshots without
# forcing downstream diagnostics to opt out of safe deserialization.
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([Field])


__all__ = ["Field", "HorizontalLocation", "VerticalLocation", "require_field"]
