"""Ocean prognostic tensors. Geometry is borrowed; I/O is explicit in ocean.io.

u_e: edge/layer normal velocity (m/s); h: cell/layer thickness (m).
temp_h=h*T (m degC), salt_h=h*S, tracer_h=h*q. Layers run top to bottom.
Derived concentrations retain the archive's tiny-denominator convention; this
is not a positivity limiter. Constructing a state never casts or detaches it.
"""
from __future__ import annotations
from typing import Any
import torch
from vfs.field import Field
from .column import OceanColumn
from .constants import OCEAN_ALPHA_T, OCEAN_BETA_S, OCEAN_RHO0, OCEAN_S0, OCEAN_T0
from .eos import linear_seawater_density
from .helpers import _combine_optional, _normalize_tracer_tensor, _require_tracer_shape, _require_trailing_shape


def _as_ocean_tensor(value, *, location, name, component_axes=()):
    if isinstance(value, Field):
        if (value.location != location or value.vertical_location != "full"
                or value.component_axes != component_axes):
            raise ValueError(f"{name} must be a {location}/full Field with component_axes={component_axes!r}")
        value = value.values
    if value is not None and not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor; convert external arrays explicitly")
    return value


class OceanState:
    """Self-describing ocean state with conservative layer tracers."""

    def __init__(
        self,
        u_e: Field | torch.Tensor | None = None,
        h: Field | torch.Tensor | None = None,
        temp_h: Field | torch.Tensor | None = None,
        salt_h: Field | torch.Tensor | None = None,
        tracer_h: Field | torch.Tensor | None = None,
        *,
        mesh=None,
        column: OceanColumn | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self.u_e = _as_ocean_tensor(u_e, location="edge", name="u_e")
        self.h = _as_ocean_tensor(h, location="cell", name="h")
        self.temp_h = _as_ocean_tensor(temp_h, location="cell", name="temp_h")
        self.salt_h = _as_ocean_tensor(salt_h, location="cell", name="salt_h")
        self.tracer_h = _as_ocean_tensor(
            tracer_h,
            location="cell",
            name="tracer_h",
            component_axes=("tracer",),
        )
        self.mesh = mesh
        self.column = column
        self.attrs = dict(attrs or {})

    @classmethod
    def from_temperature(
        cls,
        u_e: torch.Tensor,
        h: torch.Tensor,
        temperature: torch.Tensor,
        salinity: torch.Tensor,
        tracer: torch.Tensor | None = None,
        *,
        mesh=None,
        column: OceanColumn | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> "OceanState":
        tracer_h = None if tracer is None else h[..., None] * tracer
        return cls(u_e, h, h * temperature, h * salinity, tracer_h, mesh=mesh, column=column, attrs=attrs)

    def _attrs_copy(self) -> dict[str, Any]:
        return dict(self.attrs or {})

    def _require_mesh(self):
        if self.mesh is None:
            raise ValueError("OceanState has no mesh; construct it with mesh=...")
        return self.mesh

    def _require_column(self) -> OceanColumn:
        if self.column is None:
            raise ValueError("OceanState has no column; construct it with column=...")
        return self.column

    def _require_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.u_e is None or self.h is None or self.temp_h is None or self.salt_h is None:
            raise ValueError("OceanState prognostic tensors are absent")
        return self.u_e, self.h, self.temp_h, self.salt_h

    def validate(self) -> None:
        mesh = self._require_mesh()
        column = self._require_column()
        u_e, h, temp_h, salt_h = self._require_tensors()
        n_cells = int(mesh.nCells)
        n_edges = int(mesh.nEdges)
        n_layers = int(column.n_layers)
        _require_trailing_shape("u_e", u_e, (n_edges, n_layers))
        _require_trailing_shape("h", h, (n_cells, n_layers))
        _require_trailing_shape("temp_h", temp_h, (n_cells, n_layers))
        _require_trailing_shape("salt_h", salt_h, (n_cells, n_layers))
        if self.tracer_h is not None:
            _require_tracer_shape(self.tracer_h, n_cells, n_layers)

    def with_tensors(
        self,
        u_e: torch.Tensor,
        h: torch.Tensor,
        temp_h: torch.Tensor,
        salt_h: torch.Tensor,
        tracer_h: torch.Tensor | None,
    ) -> "OceanState":
        return OceanState(u_e, h, temp_h, salt_h, tracer_h, mesh=self.mesh, column=self.column, attrs=self._attrs_copy())

    def to(self, *args, **kwargs) -> "OceanState":
        u_e, h, temp_h, salt_h = self._require_tensors()
        return self.with_tensors(
            u_e.to(*args, **kwargs),
            h.to(*args, **kwargs),
            temp_h.to(*args, **kwargs),
            salt_h.to(*args, **kwargs),
            None if self.tracer_h is None else self.tracer_h.to(*args, **kwargs),
        )

    def clone(self) -> "OceanState":
        u_e, h, temp_h, salt_h = self._require_tensors()
        return self.with_tensors(
            u_e.clone(),
            h.clone(),
            temp_h.clone(),
            salt_h.clone(),
            None if self.tracer_h is None else self.tracer_h.clone(),
        )

    def detach(self) -> "OceanState":
        u_e, h, temp_h, salt_h = self._require_tensors()
        return self.with_tensors(
            u_e.detach(),
            h.detach(),
            temp_h.detach(),
            salt_h.detach(),
            None if self.tracer_h is None else self.tracer_h.detach(),
        )

    def requires_grad_(self, requires_grad: bool = True) -> "OceanState":
        u_e, h, temp_h, salt_h = self._require_tensors()
        u_e.requires_grad_(requires_grad)
        h.requires_grad_(requires_grad)
        temp_h.requires_grad_(requires_grad)
        salt_h.requires_grad_(requires_grad)
        if self.tracer_h is not None and self.tracer_h.is_floating_point():
            self.tracer_h.requires_grad_(requires_grad)
        return self

    def zeros_like(self) -> "OceanState":
        u_e, h, temp_h, salt_h = self._require_tensors()
        return self.with_tensors(
            torch.zeros_like(u_e),
            torch.zeros_like(h),
            torch.zeros_like(temp_h),
            torch.zeros_like(salt_h),
            None if self.tracer_h is None else torch.zeros_like(self.tracer_h),
        )

    def _combined_references(self, other: "OceanState") -> tuple[Any | None, OceanColumn | None, dict[str, Any]]:
        mesh = self.mesh if self.mesh is not None else other.mesh
        column = self.column if self.column is not None else other.column
        if self.mesh is not None and other.mesh is not None and self.mesh is not other.mesh:
            raise ValueError("cannot combine OceanState objects with different mesh references")
        if self.column is not None and other.column is not None and self.column is not other.column:
            raise ValueError("cannot combine OceanState objects with different column references")
        attrs = dict(other.attrs or {})
        attrs.update(self.attrs or {})
        return mesh, column, attrs

    def __add__(self, other: "OceanState") -> "OceanState":
        mesh, column, attrs = self._combined_references(other)
        return OceanState(
            self.u_e + other.u_e,
            self.h + other.h,
            self.temp_h + other.temp_h,
            self.salt_h + other.salt_h,
            _combine_optional(self.tracer_h, other.tracer_h, torch.add),
            mesh=mesh,
            column=column,
            attrs=attrs,
        )

    def __sub__(self, other: "OceanState") -> "OceanState":
        mesh, column, attrs = self._combined_references(other)
        return OceanState(
            self.u_e - other.u_e,
            self.h - other.h,
            self.temp_h - other.temp_h,
            self.salt_h - other.salt_h,
            _combine_optional(self.tracer_h, other.tracer_h, torch.sub),
            mesh=mesh,
            column=column,
            attrs=attrs,
        )

    def __mul__(self, scalar) -> "OceanState":
        return OceanState(
            self.u_e * scalar,
            self.h * scalar,
            self.temp_h * scalar,
            self.salt_h * scalar,
            None if self.tracer_h is None else self.tracer_h * scalar,
            mesh=self.mesh,
            column=self.column,
            attrs=self._attrs_copy(),
        )

    def __rmul__(self, scalar) -> "OceanState":
        return self.__mul__(scalar)

    def __truediv__(self, scalar) -> "OceanState":
        return self.__mul__(1.0 / scalar)










    def temperature(self) -> torch.Tensor:
        return self.temp_h / self.h.clamp_min(torch.finfo(self.h.dtype).tiny)

    def salinity(self) -> torch.Tensor:
        return self.salt_h / self.h.clamp_min(torch.finfo(self.h.dtype).tiny)

    def density(
        self,
        *,
        rho0: float = OCEAN_RHO0,
        alpha: float = OCEAN_ALPHA_T,
        beta: float = OCEAN_BETA_S,
        T0: float = OCEAN_T0,
        S0: float = OCEAN_S0,
    ) -> torch.Tensor:
        return linear_seawater_density(
            self.temperature(),
            self.salinity(),
            rho0=rho0,
            alpha=alpha,
            beta=beta,
            T0=T0,
            S0=S0,
        )

    def eta(self) -> torch.Tensor:
        return self._require_column().eta(self.h)

    def total_volume(self) -> torch.Tensor:
        return self._require_mesh().integrate_cell(self.h.sum(dim=-1, keepdim=True))

    def total_heat_content(self) -> torch.Tensor:
        return self._require_mesh().integrate_cell(self.temp_h.sum(dim=-1, keepdim=True))

    def total_salt_content(self) -> torch.Tensor:
        return self._require_mesh().integrate_cell(self.salt_h.sum(dim=-1, keepdim=True))

    def total_tracer_content(self) -> torch.Tensor | None:
        if self.tracer_h is None:
            return None
        tracer_h = _normalize_tracer_tensor(self.tracer_h, self._require_mesh().nCells, self._require_column().n_layers)
        return self._require_mesh().integrate_cell(tracer_h.sum(dim=-2))

    def tracers(self) -> torch.Tensor | None:
        if self.tracer_h is None:
            return None
        tracer_h = _normalize_tracer_tensor(self.tracer_h, self._require_mesh().nCells, self._require_column().n_layers)
        return tracer_h / self.h[..., None].clamp_min(torch.finfo(self.h.dtype).tiny)






def _register_pytree_nodes() -> None:
    try:
        from torch.utils import _pytree
    except Exception:
        return

    def flatten_state(state: OceanState):
        tensors = (state.u_e, state.h, state.temp_h, state.salt_h)
        has_tracer = state.tracer_h is not None
        if has_tracer:
            tensors = tensors + (state.tracer_h,)
        return tensors, (has_tracer, state.mesh, state.column, state._attrs_copy())

    def unflatten_state(children, context):
        has_tracer, mesh, column, attrs = context
        if has_tracer:
            u_e, h, temp_h, salt_h, tracer_h = children
        else:
            u_e, h, temp_h, salt_h = children
            tracer_h = None
        return OceanState(u_e, h, temp_h, salt_h, tracer_h, mesh=mesh, column=column, attrs=attrs)

    try:
        _pytree.register_pytree_node(OceanState, flatten_state, unflatten_state)
    except ValueError:
        pass


_register_pytree_nodes()
