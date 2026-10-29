"""Self-describing ocean state and state requirements."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from vfs.core.config import TensorFieldRequirement
from vfs.field import Field

from .column import OceanColumn
from .constants import OCEAN_ALPHA_T, OCEAN_BETA_S, OCEAN_RHO0, OCEAN_S0, OCEAN_T0
from .eos import linear_seawater_density
from .helpers import (
    _combine_optional,
    _netcdf_attrs,
    _normalize_tracer_tensor,
    _require_tracer_shape,
    _require_trailing_shape,
)


_OCEAN_FIELD_TRAILING_SHAPES = {
    "u_e": ("nEdges", "nLayers"),
    "h": ("nCells", "nLayers"),
    "temp_h": ("nCells", "nLayers"),
    "salt_h": ("nCells", "nLayers"),
    "tracer_h": ("nCells", "nLayers", "nTracers"),
}


def _as_ocean_field(
    value: Field | torch.Tensor | None,
    *,
    location: str,
    name: str,
    component_axes: tuple[str, ...] = (),
) -> Field | None:
    if value is None:
        return None
    if isinstance(value, Field):
        if (
            value.location != location
            or value.vertical_location != "full"
            or value.component_axes != component_axes
        ):
            raise ValueError(
                f"{name} must be a {location}/full Field with component_axes={component_axes!r}"
            )
        return value
    return Field(
        value,
        location=location,
        vertical_location="full",
        name=name,
        component_axes=component_axes,
    )


def _default_ocean_field_requirement(name: str) -> TensorFieldRequirement:
    name = str(name)
    shape = _OCEAN_FIELD_TRAILING_SHAPES.get(name)
    return TensorFieldRequirement(name, trailing_shape=shape)


def _as_ocean_field_requirement(value: TensorFieldRequirement | dict[str, Any]) -> TensorFieldRequirement:
    if isinstance(value, TensorFieldRequirement):
        return value
    return TensorFieldRequirement.from_dict(dict(value))


def _ocean_state_requirement_context(state: "OceanState") -> dict[str, int]:
    mesh = state._require_mesh()
    column = state._require_column()
    context = {
        "nCells": int(mesh.nCells),
        "nEdges": int(mesh.nEdges),
        "nLayers": int(column.n_layers),
        "nInterfaces": int(column.n_layers) + 1,
    }
    if state.tracer_h is not None:
        tracer_h = _normalize_tracer_tensor(state.tracer_h, mesh.nCells, column.n_layers)
        context["nTracers"] = int(tracer_h.shape[-1])
    return context


class OceanStateRequirement:
    """Semantic state contract for VFS-O components."""

    def __init__(
        self,
        *,
        fields: tuple[TensorFieldRequirement | dict[str, Any], ...] | list[TensorFieldRequirement | dict[str, Any]] | None = None,
        required_fields: tuple[str, ...] | list[str] = ("u_e", "h", "temp_h", "salt_h"),
        min_tracers: int = 0,
    ) -> None:
        self.fields = tuple(
            _as_ocean_field_requirement(field)
            for field in (
                tuple(_default_ocean_field_requirement(name) for name in required_fields)
                if fields is None
                else fields
            )
        )
        self.required_fields = tuple(field.name for field in self.fields if not field.optional)
        self.min_tracers = int(min_tracers)
        if self.min_tracers < 0:
            raise ValueError("min_tracers must be nonnegative")
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate ocean state field requirements: {names}")

    def validate(self, state: "OceanState", *, owner: str = "component") -> None:
        if not isinstance(state, OceanState):
            raise TypeError(f"{owner} requires OceanState, got {type(state).__name__}")
        if state.mesh is None:
            raise ValueError(f"{owner} requires OceanState.mesh")
        if state.column is None:
            raise ValueError(f"{owner} requires OceanState.column")
        context = _ocean_state_requirement_context(state)
        for field in self.fields:
            field.validate(state, owner=owner, context=context)
        state.validate()
        if self.min_tracers:
            if state.tracer_h is None:
                raise ValueError(f"{owner} requires at least {self.min_tracers} ocean tracer(s)")
            tracer_h = _normalize_tracer_tensor(state.tracer_h, state.mesh.nCells, state.column.n_layers)
            if int(tracer_h.shape[-1]) < self.min_tracers:
                raise ValueError(
                    f"{owner} requires at least {self.min_tracers} ocean tracer(s), "
                    f"got {int(tracer_h.shape[-1])}"
                )


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
        self.u_e = _as_ocean_field(u_e, location="edge", name="u_e")
        self.h = _as_ocean_field(h, location="cell", name="h")
        self.temp_h = _as_ocean_field(temp_h, location="cell", name="temp_h")
        self.salt_h = _as_ocean_field(salt_h, location="cell", name="salt_h")
        self.tracer_h = _as_ocean_field(
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

    def ke_cell(self, edge_wind: torch.Tensor | None = None) -> torch.Tensor:
        return self._require_mesh().ke_cell(self.u_e if edge_wind is None else edge_wind)

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

    def to_netcdf(self, path, *, metadata: dict[str, Any] | None = None) -> str:
        try:
            import xarray as xr
        except Exception as exc:  # pragma: no cover - optional backend dependent.
            raise RuntimeError("OceanState NetCDF writing requires xarray and a NetCDF backend") from exc

        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh = self._require_mesh()
        column = self._require_column()
        attrs = _netcdf_attrs(self.attrs)
        attrs.update(_netcdf_attrs(metadata or {}))
        attrs.update(
            {
                "created_by": "vfs.ocean.OceanState.to_netcdf",
                "state_kind": "ocean",
                "n_cells": int(mesh.nCells),
                "n_edges": int(mesh.nEdges),
                "n_layers": int(column.n_layers),
                "gravity": float(column.gravity),
                "rho0": float(column.rho0),
            }
        )
        data_vars = {
            "u_edge": (("edge", "layer"), self.u_e.detach().cpu().numpy()),
            "h": (("cell", "layer"), self.h.detach().cpu().numpy()),
            "temperature": (("cell", "layer"), self.temperature().detach().cpu().numpy()),
            "salinity": (("cell", "layer"), self.salinity().detach().cpu().numpy()),
            "temp_h": (("cell", "layer"), self.temp_h.detach().cpu().numpy()),
            "salt_h": (("cell", "layer"), self.salt_h.detach().cpu().numpy()),
            "eta": (("cell",), self.eta().detach().cpu().reshape(-1).numpy()),
            "rest_thickness": (("layer",), column.rest_thickness.detach().cpu().numpy()),
        }
        if self.tracer_h is not None:
            tracer_h = _normalize_tracer_tensor(self.tracer_h, mesh.nCells, column.n_layers)
            data_vars["tracer_h"] = (("cell", "layer", "tracer"), tracer_h.detach().cpu().numpy())
        lat = torch.rad2deg(torch.asin(mesh.cell_xyz[:, 2].detach().cpu().clamp(-1.0, 1.0)))
        lon = torch.remainder(torch.rad2deg(torch.atan2(mesh.cell_xyz[:, 1].detach().cpu(), mesh.cell_xyz[:, 0].detach().cpu())), 360.0)
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "cell": list(range(int(mesh.nCells))),
                "edge": list(range(int(mesh.nEdges))),
                "layer": list(range(int(column.n_layers))),
                "lat_cell_deg": (("cell",), lat.numpy()),
                "lon_cell_deg": (("cell",), lon.numpy()),
            },
            attrs=attrs,
        )
        ds.to_netcdf(path, format="NETCDF4", engine="netcdf4")
        ds.close()
        return str(path)

    def load_netcdf(self, path, *, dtype=None, device=None) -> "OceanState":
        mesh = self.mesh
        column = self.column
        if mesh is None or column is None:
            raise ValueError("OceanState.load_netcdf requires a receiver with mesh=... and column=...")
        try:
            import xarray as xr
        except Exception as exc:  # pragma: no cover - optional backend dependent.
            raise RuntimeError("OceanState NetCDF loading requires xarray") from exc

        ds = xr.open_dataset(path)
        try:
            required = ("u_edge", "h", "temp_h", "salt_h")
            missing = [name for name in required if name not in ds.variables]
            if missing:
                raise ValueError(f"NetCDF snapshot is missing required OceanState variables: {missing}")
            u_e = torch.as_tensor(ds["u_edge"].values)
            h = torch.as_tensor(ds["h"].values)
            temp_h = torch.as_tensor(ds["temp_h"].values)
            salt_h = torch.as_tensor(ds["salt_h"].values)
            tracer_h = torch.as_tensor(ds["tracer_h"].values) if "tracer_h" in ds.variables else None
            attrs = dict(ds.attrs)
        finally:
            ds.close()
        if dtype is not None or device is not None:
            target_dtype = dtype if dtype is not None else u_e.dtype
            target_device = device if device is not None else u_e.device
            u_e = u_e.to(dtype=target_dtype, device=target_device)
            h = h.to(dtype=target_dtype, device=target_device)
            temp_h = temp_h.to(dtype=target_dtype, device=target_device)
            salt_h = salt_h.to(dtype=target_dtype, device=target_device)
            if tracer_h is not None:
                tracer_h = tracer_h.to(dtype=target_dtype, device=target_device)
        out = OceanState(u_e, h, temp_h, salt_h, tracer_h, mesh=mesh, column=column, attrs=attrs)
        out._validate_restart_metadata()
        out.validate()
        return out

    def _validate_restart_metadata(self) -> None:
        mesh = self._require_mesh()
        column = self._require_column()
        attrs = dict(self.attrs or {})
        expected = {"n_cells": int(mesh.nCells), "n_edges": int(mesh.nEdges), "n_layers": int(column.n_layers)}
        for key, value in expected.items():
            if key in attrs and str(attrs[key]) not in ("", str(value)):
                raise ValueError(f"OceanState restart metadata {key}={attrs[key]!r} does not match expected {value}")

    def to_latlon(self, regridder, *, fields="default") -> dict[str, Any]:
        """Return common OceanState fields on a latitude/longitude grid."""

        if not hasattr(regridder, "state_to_latlon"):
            raise TypeError("OceanState.to_latlon expects a regridder with state_to_latlon(state, fields=...)")
        return regridder.state_to_latlon(self, fields=fields)


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
