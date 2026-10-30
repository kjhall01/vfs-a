"""Lat/lon <-> native mesh remapping operators.

The native operators here are ordinary PyTorch modules with cached gather
indices and weights.  They are differentiable with respect to field values, but
not with respect to mesh/grid geometry.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
import torch.nn as nn

from vfs.field import Field, require_field


def _as_float_tensor(value, *, dtype: torch.dtype = torch.float64, device=None) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


def _xyz_from_lat_lon(lat_rad: torch.Tensor, lon_rad: torch.Tensor) -> torch.Tensor:
    cos_lat = torch.cos(lat_rad)
    return torch.stack(
        [
            cos_lat * torch.cos(lon_rad),
            cos_lat * torch.sin(lon_rad),
            torch.sin(lat_rad),
        ],
        dim=-1,
    )


def _cell_lat_lon_rad(mesh) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = mesh.cell_xyz
    lat = torch.asin(xyz[:, 2].clamp(-1.0, 1.0))
    lon = torch.atan2(xyz[:, 1], xyz[:, 0])
    return lat, lon


def _local_east_north_from_xyz(xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lat = torch.asin(xyz[..., 2].clamp(-1.0, 1.0))
    lon = torch.atan2(xyz[..., 1], xyz[..., 0])
    east = torch.stack([-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)], dim=-1)
    north = torch.stack(
        [
            -torch.sin(lat) * torch.cos(lon),
            -torch.sin(lat) * torch.sin(lon),
            torch.cos(lat),
        ],
        dim=-1,
    )
    return east, north


def _squeeze_last_singleton(value: Field | torch.Tensor) -> Field | torch.Tensor:
    if isinstance(value, Field):
        if (
            value.vertical_location == "surface"
            and not value.component_axes
            and int(value.values.shape[-1]) == 1
        ):
            return Field(
                value.values.squeeze(-1),
                location=value.location,
                name=value.name,
                units=value.units,
            )
        return value
    if value.ndim >= 1 and value.shape[-1] == 1:
        return value.squeeze(-1)
    return value


def _native_field_to_source_layout(field: Field) -> tuple[torch.Tensor, int]:
    """Move a native Field's horizontal axis before batch/payload axes."""

    if field.location != "cell":
        raise ValueError(
            "native point regridders accept cell Fields; "
            f"got location {field.location!r}"
        )
    axis = field.horizontal_axis % field.values.ndim
    return field.values.movedim(axis, 0), axis


def _latlon_field_from_source_layout(template: Field, values: torch.Tensor, *, batch_ndim: int) -> Field:
    """Restore ``[..., lat, lon, payload]`` from ``[lat, lon, ..., payload]``."""

    if batch_ndim:
        payload_start = 2 + int(batch_ndim)
        permutation = (
            *range(2, payload_start),
            0,
            1,
            *range(payload_start, values.ndim),
        )
        values = values.permute(permutation)
    return Field(
        values,
        location="latlon",
        vertical_location=template.vertical_location,
        name=template.name,
        units=template.units,
        component_axes=template.component_axes,
    )


def _latlon_field_to_source_layout(field: Field) -> tuple[torch.Tensor, int]:
    """Move a lat/lon Field's two horizontal axes to the front."""

    require_field(field, location="latlon", name="grid_field")
    axis = field.horizontal_axis % field.values.ndim
    return field.values.movedim((axis, axis + 1), (0, 1)), axis


def _native_field_from_latlon_layout(template: Field, values: torch.Tensor, *, batch_ndim: int) -> Field:
    """Restore ``[..., cell, payload]`` from ``[cell, ..., payload]``."""

    if batch_ndim:
        payload_start = 1 + int(batch_ndim)
        permutation = (
            *range(1, payload_start),
            0,
            *range(payload_start, values.ndim),
        )
        values = values.permute(permutation)
    return Field(
        values,
        location="cell",
        vertical_location=template.vertical_location,
        name=template.name,
        units=template.units,
        component_axes=template.component_axes,
    )


def _cell_field(
    values: Field | torch.Tensor,
    *,
    vertical_location: str | None = None,
    name: str | None = None,
    units: str | None = None,
    component_axes: tuple[str, ...] = (),
) -> Field:
    """Tag a derived native-cell quantity before regridding it."""

    if isinstance(values, Field):
        return values
    return Field(
        values,
        location="cell",
        vertical_location=vertical_location,
        name=name,
        units=units,
        component_axes=component_axes,
    )


class LatLonGrid(nn.Module):
    """Regular equiangular latitude/longitude grid.

    Latitudes and longitudes are cell centers in degrees.  Longitude is
    periodic and defaults to the [0, 360) convention.
    """

    def __init__(
        self,
        lat_deg: torch.Tensor | Iterable[float],
        lon_deg: torch.Tensor | Iterable[float],
        *,
        radius: float = 6.371e6,
        degrees: bool = True,
    ) -> None:
        super().__init__()
        lat = _as_float_tensor(lat_deg)
        lon = _as_float_tensor(lon_deg)
        if not degrees:
            lat = torch.rad2deg(lat)
            lon = torch.rad2deg(lon)
        if lat.ndim != 1 or lon.ndim != 1:
            raise ValueError("LatLonGrid expects 1D latitude and longitude center arrays")
        if lat.numel() < 2 or lon.numel() < 2:
            raise ValueError("LatLonGrid requires at least two latitudes and two longitudes")
        self.nlat = int(lat.numel())
        self.nlon = int(lon.numel())
        self.radius = float(radius)
        self.register_buffer("lat_deg", lat)
        self.register_buffer("lon_deg", lon)

        lat_edges = _regular_edges_from_centers(lat, clamp=(-90.0, 90.0))
        lon_edges = _regular_edges_from_centers(lon)
        lat0 = torch.deg2rad(lat_edges[:-1])
        lat1 = torch.deg2rad(lat_edges[1:])
        dlon = torch.deg2rad((lon_edges[1:] - lon_edges[:-1]).abs())
        area = (float(radius) ** 2) * (torch.sin(lat1) - torch.sin(lat0)).abs()[:, None] * dlon[None, :]
        self.register_buffer("cell_area", area)

    @classmethod
    def regular(
        cls,
        nlat: int,
        nlon: int,
        *,
        lat_min: float = -90.0,
        lat_max: float = 90.0,
        lon_min: float = 0.0,
        lon_max: float = 360.0,
        radius: float = 6.371e6,
    ) -> "LatLonGrid":
        """Create an equiangular cell-center grid."""

        if nlat < 2 or nlon < 2:
            raise ValueError("regular lat/lon grids require nlat >= 2 and nlon >= 2")
        lat_edges = torch.linspace(float(lat_min), float(lat_max), int(nlat) + 1, dtype=torch.float64)
        lon_edges = torch.linspace(float(lon_min), float(lon_max), int(nlon) + 1, dtype=torch.float64)
        lat = 0.5 * (lat_edges[:-1] + lat_edges[1:])
        lon = 0.5 * (lon_edges[:-1] + lon_edges[1:])
        return cls(lat, lon, radius=radius)

    @property
    def shape(self) -> tuple[int, int]:
        return self.nlat, self.nlon

    def meshgrid(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.meshgrid(self.lat_deg, self.lon_deg, indexing="ij")

    def xyz(self) -> torch.Tensor:
        lat_deg, lon_deg = self.meshgrid()
        return _xyz_from_lat_lon(torch.deg2rad(lat_deg), torch.deg2rad(lon_deg))


class MeshToLatLonKNN(nn.Module):
    """Differentiable diagnostic remap from native cell centers to lat/lon."""

    is_differentiable = True
    is_conservative = False
    preserves_constants = True

    def __init__(
        self,
        grid: LatLonGrid,
        index: torch.Tensor,
        weight: torch.Tensor,
        *,
        mesh_area: torch.Tensor | None = None,
        method: str = "knn",
    ) -> None:
        super().__init__()
        self.grid = grid
        self.nlat = grid.nlat
        self.nlon = grid.nlon
        self.k = int(index.shape[-1])
        self.method = str(method)
        if mesh_area is None:
            n_cells = int(index.max().item()) + 1
            mesh_area = torch.ones(n_cells, dtype=torch.float64)
        self.nCells = int(mesh_area.numel())
        self.register_buffer("index", index.to(dtype=torch.int64))
        self.register_buffer("weight", weight.to(dtype=torch.float64))
        self.register_buffer("mesh_area", mesh_area.detach().to(dtype=torch.float64))
        self.register_buffer("grid_area", grid.cell_area.detach().reshape(-1).to(dtype=torch.float64))
        self.register_buffer("target_xyz", grid.xyz().reshape(-1, 3).to(dtype=torch.float64))

    @classmethod
    def from_mesh(
        cls,
        mesh,
        grid: LatLonGrid,
        *,
        k: int = 8,
        power: float = 2.0,
        eps: float = 1.0e-12,
    ) -> "MeshToLatLonKNN":
        """Build inverse-distance weights from mesh cell centers to grid cells."""

        if k < 1:
            raise ValueError("k must be >= 1")
        k = min(int(k), int(mesh.nCells))
        src = mesh.cell_xyz.detach().to(dtype=torch.float64, device="cpu")
        dst = grid.xyz().reshape(-1, 3).detach().to(dtype=torch.float64, device="cpu")
        dot = (dst @ src.T).clamp(-1.0, 1.0)
        values, index = torch.topk(dot, k=k, dim=-1, largest=True)
        distance = torch.acos(values).clamp_min(float(eps))
        weight = distance.pow(-float(power))
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weight.dtype).tiny)
        return cls(grid, index, weight, mesh_area=mesh.cell_area.detach())

    @classmethod
    def from_cell_lat_lon(
        cls,
        lat_deg: torch.Tensor | Iterable[float],
        lon_deg: torch.Tensor | Iterable[float],
        grid: LatLonGrid,
        *,
        cell_area: torch.Tensor | Iterable[float] | None = None,
        k: int = 8,
        power: float = 2.0,
        eps: float = 1.0e-12,
    ) -> "MeshToLatLonKNN":
        """Build KNN weights from source cell-center latitude/longitude arrays."""

        lat = _as_float_tensor(lat_deg, dtype=torch.float64, device="cpu")
        lon = _as_float_tensor(lon_deg, dtype=torch.float64, device="cpu")
        if lat.ndim != 1 or lon.ndim != 1 or lat.shape != lon.shape:
            raise ValueError("source latitude and longitude arrays must be 1D with matching shape")
        if k < 1:
            raise ValueError("k must be >= 1")
        k = min(int(k), int(lat.numel()))
        src = _xyz_from_lat_lon(torch.deg2rad(lat), torch.deg2rad(lon))
        dst = grid.xyz().reshape(-1, 3).detach().to(dtype=torch.float64, device="cpu")
        dot = (dst @ src.T).clamp(-1.0, 1.0)
        values, index = torch.topk(dot, k=k, dim=-1, largest=True)
        distance = torch.acos(values).clamp_min(float(eps))
        weight = distance.pow(-float(power))
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weight.dtype).tiny)
        if cell_area is None:
            area = torch.ones(lat.numel(), dtype=torch.float64)
        else:
            area = _as_float_tensor(cell_area, dtype=torch.float64, device="cpu")
            if area.shape != lat.shape:
                raise ValueError("cell_area must have the same shape as source latitude/longitude arrays")
        return cls(grid, index, weight, mesh_area=area)

    def _forward_values(self, cell_field: torch.Tensor) -> torch.Tensor:
        """Remap tensor values with leading source-point dimension."""

        if cell_field.ndim < 1:
            raise ValueError("cell_field must have a leading nCells dimension")
        index = self.index.to(device=cell_field.device)
        if int(cell_field.shape[0]) <= int(index.max().item()):
            raise ValueError("cell_field leading dimension is smaller than this regridder expects")
        weight = self.weight.to(device=cell_field.device, dtype=cell_field.dtype)
        gathered = cell_field[index]
        view = (weight.shape[0], weight.shape[1]) + (1,) * (gathered.ndim - 2)
        out = (gathered * weight.reshape(view)).sum(dim=1)
        return out.reshape((self.nlat, self.nlon) + tuple(cell_field.shape[1:]))

    def forward(self, cell_field: Field | torch.Tensor) -> Field | torch.Tensor:
        """Remap a native-cell Field to a regular lat/lon Field.

        Tensor input remains available for offline compatibility. Field input
        preserves batch, vertical, component, name, and units metadata while
        changing its horizontal location to ``"latlon"``.
        """

        if not isinstance(cell_field, Field):
            return self._forward_values(cell_field)
        values, batch_ndim = _native_field_to_source_layout(cell_field)
        mapped = self._forward_values(values)
        return _latlon_field_from_source_layout(cell_field, mapped, batch_ndim=batch_ndim)

    def coordinates(self) -> dict[str, torch.Tensor]:
        return {
            "lat_deg": self.grid.lat_deg,
            "lon_deg": self.grid.lon_deg,
            "cell_area": self.grid.cell_area,
        }

    def adjoint(self) -> "MeshToLatLonAdjoint":
        """Return the area-weighted adjoint/backprojection operator.

        If this remapper is ``W``, the returned operator ``W_adj`` satisfies
        ``<W x, y>_grid = <x, W_adj y>_mesh`` using grid and mesh cell-area
        inner products.  This is not an inverse interpolation.
        """

        return MeshToLatLonAdjoint(
            self.grid,
            self.index.detach(),
            self.weight.detach(),
            self.mesh_area.detach(),
            method=f"{self.method}_adjoint",
        )

    def state_to_latlon(self, state, *, fields: Iterable[str] | str = "default") -> dict[str, Any]:
        """Return common state fields on the lat/lon grid."""

        is_ocean = hasattr(state, "h") and hasattr(state, "temp_h") and not hasattr(state, "ps")
        requested = _normalise_field_request(fields, state_kind="ocean" if is_ocean else "atmosphere")
        base = state.h if is_ocean else state.ps
        out: dict[str, Any] = {
            "lat_deg": self.grid.lat_deg.to(device=base.device, dtype=base.dtype),
            "lon_deg": self.grid.lon_deg.to(device=base.device, dtype=base.dtype),
            "attrs": {
                "regrid_method": self.method,
                "conservative": False,
                "differentiable_wrt_fields": True,
            },
        }

        if is_ocean:
            if "h" in requested:
                out["h"] = self(state.h)
            if "eta" in requested:
                out["eta"] = _squeeze_last_singleton(
                    self(_cell_field(state.eta(), vertical_location="surface", name="eta", units="m"))
                )
            if "temperature" in requested:
                out["temperature"] = self(
                    _cell_field(state.temperature(), vertical_location="full", name="temperature", units="K")
                )
            if "salinity" in requested:
                out["salinity"] = self(_cell_field(state.salinity(), vertical_location="full", name="salinity"))
            if "density" in requested:
                out["density"] = self(_cell_field(state.density(), vertical_location="full", name="density", units="kg m-3"))
            if "tracers" in requested:
                q = state.tracers()
                if q is not None:
                    out["q_tracer"] = self(
                        _cell_field(
                            q,
                            vertical_location="full",
                            name="q_tracer",
                            component_axes=("tracer",),
                        )
                    )
            if "wind" in requested:
                wind_vec = state.mesh.cell_wind_vector(state.u_e)
                wind_ll = self(
                    _cell_field(
                        wind_vec,
                        vertical_location="full",
                        name="wind_vector",
                        units="m s-1",
                        component_axes=("cartesian",),
                    )
                )
                east, north = _local_east_north_from_xyz(
                    self.target_xyz.to(device=wind_ll.values.device, dtype=wind_ll.values.dtype)
                )
                east = east.reshape(self.nlat, self.nlon, 1, 3)
                north = north.reshape(self.nlat, self.nlon, 1, 3)
                u_east = (wind_ll.values * east).sum(dim=-1)
                v_north = (wind_ll.values * north).sum(dim=-1)
                out["wind_u_east"] = Field(
                    u_east,
                    location="latlon",
                    vertical_location="full",
                    name="wind_u_east",
                    units="m s-1",
                )
                out["wind_v_north"] = Field(
                    v_north,
                    location="latlon",
                    vertical_location="full",
                    name="wind_v_north",
                    units="m s-1",
                )
                out["wind_speed"] = Field(
                    (u_east.square() + v_north.square()).sqrt(),
                    location="latlon",
                    vertical_location="full",
                    name="wind_speed",
                    units="m s-1",
                )
            return out

        if "ps" in requested:
            out["ps"] = _squeeze_last_singleton(self(state.ps))
        if "layer_dp" in requested or "dp" in requested:
            out["layer_dp"] = self(_cell_field(state.layer_dp(), vertical_location="full", name="layer_dp", units="Pa"))
        if "theta" in requested:
            out["theta_interface"] = self(
                _cell_field(state.theta(), vertical_location="interface", name="theta_interface", units="K")
            )
        if "temperature" in requested:
            out["temperature_full"] = self(
                _cell_field(state.temperature(), vertical_location="full", name="temperature_full", units="K")
            )
        if "pressure" in requested:
            out["pressure_full"] = self(
                _cell_field(state.full_pressure(), vertical_location="full", name="pressure_full", units="Pa")
            )
            out["pressure_interface"] = self(
                _cell_field(state.interface_pressure(), vertical_location="interface", name="pressure_interface", units="Pa")
            )
        if "geopotential" in requested:
            out["geopotential_full"] = self(
                _cell_field(state.geopotential(), vertical_location="full", name="geopotential_full", units="m2 s-2")
            )
        if "tracers" in requested:
            q = state.tracers()
            if q is not None:
                out["q_tracer"] = self(
                    _cell_field(q, vertical_location="full", name="q_tracer", component_axes=("tracer",))
                )
        if "wind" in requested:
            wind_vec = state.cell_wind_vector()
            wind_ll = self(
                _cell_field(
                    wind_vec,
                    vertical_location="full",
                    name="wind_vector",
                    units="m s-1",
                    component_axes=("cartesian",),
                )
            )
            east, north = _local_east_north_from_xyz(
                self.target_xyz.to(device=wind_ll.values.device, dtype=wind_ll.values.dtype)
            )
            east = east.reshape(self.nlat, self.nlon, 1, 3)
            north = north.reshape(self.nlat, self.nlon, 1, 3)
            u_east = (wind_ll.values * east).sum(dim=-1)
            v_north = (wind_ll.values * north).sum(dim=-1)
            out["wind_u_east"] = Field(
                u_east,
                location="latlon",
                vertical_location="full",
                name="wind_u_east",
                units="m s-1",
            )
            out["wind_v_north"] = Field(
                v_north,
                location="latlon",
                vertical_location="full",
                name="wind_v_north",
                units="m s-1",
            )
            out["wind_speed"] = Field(
                (u_east.square() + v_north.square()).sqrt(),
                location="latlon",
                vertical_location="full",
                name="wind_speed",
                units="m s-1",
            )
        return out


class MeshToLatLonAdjoint(nn.Module):
    """Area-weighted adjoint of :class:`MeshToLatLonKNN`.

    This is a mathematical backprojection for gradients and variational
    diagnostics.  It is not a lat/lon-to-mesh interpolation recipe.
    """

    is_differentiable = True
    is_conservative = False
    preserves_constants = False
    is_adjoint = True

    def __init__(
        self,
        grid: LatLonGrid,
        index: torch.Tensor,
        weight: torch.Tensor,
        mesh_area: torch.Tensor,
        *,
        method: str = "knn_adjoint",
    ) -> None:
        super().__init__()
        self.grid = grid
        self.nlat = grid.nlat
        self.nlon = grid.nlon
        self.nCells = int(mesh_area.numel())
        self.k = int(index.shape[-1])
        self.method = str(method)
        self.register_buffer("index", index.to(dtype=torch.int64))
        self.register_buffer("weight", weight.to(dtype=torch.float64))
        self.register_buffer("mesh_area", mesh_area.detach().to(dtype=torch.float64))
        self.register_buffer("grid_area", grid.cell_area.detach().reshape(-1).to(dtype=torch.float64))

    def _forward_values(self, grid_field: torch.Tensor) -> torch.Tensor:
        """Backproject tensor values with leading [nlat, nlon] dimensions."""

        if grid_field.ndim < 2:
            raise ValueError("grid_field must have leading [nlat, nlon] dimensions")
        if int(grid_field.shape[0]) != self.nlat or int(grid_field.shape[1]) != self.nlon:
            raise ValueError(
                f"grid_field leading shape must be {(self.nlat, self.nlon)}, "
                f"got {tuple(grid_field.shape[:2])}"
            )
        trailing = tuple(grid_field.shape[2:])
        flat = grid_field.reshape(self.nlat * self.nlon, -1)
        index = self.index.to(device=grid_field.device)
        weight = self.weight.to(device=grid_field.device, dtype=grid_field.dtype)
        grid_area = self.grid_area.to(device=grid_field.device, dtype=grid_field.dtype)
        mesh_area = self.mesh_area.to(device=grid_field.device, dtype=grid_field.dtype)

        contrib = flat[:, None, :] * weight[:, :, None] * grid_area[:, None, None]
        out = torch.zeros(
            self.nCells,
            flat.shape[-1],
            dtype=grid_field.dtype,
            device=grid_field.device,
        )
        out.scatter_add_(0, index[:, :, None].expand(-1, -1, flat.shape[-1]).reshape(-1, flat.shape[-1]), contrib.reshape(-1, flat.shape[-1]))
        out = out / mesh_area[:, None].clamp_min(torch.finfo(grid_field.dtype).tiny)
        return out.reshape((self.nCells,) + trailing)

    def forward(self, grid_field: Field | torch.Tensor) -> Field | torch.Tensor:
        """Backproject a lat/lon Field to the native cell Field space."""

        if not isinstance(grid_field, Field):
            return self._forward_values(grid_field)
        values, batch_ndim = _latlon_field_to_source_layout(grid_field)
        mapped = self._forward_values(values)
        return _native_field_from_latlon_layout(grid_field, mapped, batch_ndim=batch_ndim)


class LatLonToMeshBilinear(nn.Module):
    """Differentiable bilinear remap from a regular lat/lon grid to mesh cells."""

    is_differentiable = True
    is_conservative = False
    preserves_constants = True

    def __init__(self, grid: LatLonGrid, index: torch.Tensor, weight: torch.Tensor) -> None:
        super().__init__()
        self.grid = grid
        self.nlat = grid.nlat
        self.nlon = grid.nlon
        self.nCells = int(index.shape[0])
        self.register_buffer("index", index.to(dtype=torch.int64))
        self.register_buffer("weight", weight.to(dtype=torch.float64))

    @classmethod
    def from_mesh(cls, grid: LatLonGrid, mesh) -> "LatLonToMeshBilinear":
        """Build bilinear weights from a regular lat/lon grid to mesh cells."""

        lat, lon = _cell_lat_lon_rad(mesh)
        lat_deg = torch.rad2deg(lat.detach().cpu().to(dtype=torch.float64))
        lon_deg = torch.rad2deg(lon.detach().cpu().to(dtype=torch.float64))
        index, weight = _regular_bilinear_weights(grid, lat_deg, lon_deg)
        return cls(grid, index, weight)

    def _forward_values(self, grid_field: torch.Tensor) -> torch.Tensor:
        """Remap tensor values with leading [nlat, nlon] dimensions."""

        if grid_field.ndim < 2:
            raise ValueError("grid_field must have leading [nlat, nlon] dimensions")
        if int(grid_field.shape[0]) != self.nlat or int(grid_field.shape[1]) != self.nlon:
            raise ValueError(
                f"grid_field leading shape must be {(self.nlat, self.nlon)}, "
                f"got {tuple(grid_field.shape[:2])}"
            )
        flat = grid_field.reshape(self.nlat * self.nlon, *grid_field.shape[2:])
        index = self.index.to(device=grid_field.device)
        weight = self.weight.to(device=grid_field.device, dtype=grid_field.dtype)
        gathered = flat[index]
        view = (weight.shape[0], weight.shape[1]) + (1,) * (gathered.ndim - 2)
        return (gathered * weight.reshape(view)).sum(dim=1)

    def forward(self, grid_field: Field | torch.Tensor) -> Field | torch.Tensor:
        """Remap a regular lat/lon Field to a native-cell Field."""

        if not isinstance(grid_field, Field):
            return self._forward_values(grid_field)
        values, batch_ndim = _latlon_field_to_source_layout(grid_field)
        mapped = self._forward_values(values)
        return _native_field_from_latlon_layout(grid_field, mapped, batch_ndim=batch_ndim)


class ESMFGridToMesh:
    """Optional xESMF regular-grid to mesh-cell remapper.

    This wrapper is for offline analysis and is not differentiable.  It imports
    xESMF lazily so the core VFS package does not depend on ESMF/ESMPy.
    """

    is_differentiable = False
    is_conservative = False

    def __init__(self, grid: LatLonGrid, mesh, *, method: str = "bilinear", periodic: bool = True) -> None:
        xe, xr = _import_xesmf()
        lat, lon = _cell_lat_lon_rad(mesh)
        ds_in = xr.Dataset(coords={"lat": grid.lat_deg.detach().cpu().numpy(), "lon": grid.lon_deg.detach().cpu().numpy()})
        ds_out = xr.Dataset(
            coords={
                "lat": ("cell", torch.rad2deg(lat).detach().cpu().numpy()),
                "lon": ("cell", torch.rad2deg(lon).detach().cpu().numpy()),
            }
        )
        self.grid = grid
        self.mesh = mesh
        self.method = str(method)
        self.regridder = xe.Regridder(ds_in, ds_out, method, locstream_out=True, periodic=periodic)

    def _call_values(self, grid_field: torch.Tensor) -> torch.Tensor:
        import xarray as xr

        device = grid_field.device
        dtype = grid_field.dtype
        arr = grid_field.detach().cpu().numpy()
        extra_shape = arr.shape[2:]
        da = xr.DataArray(arr.reshape(self.grid.nlat, self.grid.nlon, -1), dims=("lat", "lon", "component"))
        out = self.regridder(da).values.reshape(int(self.mesh.nCells), *extra_shape)
        return torch.as_tensor(out, dtype=dtype, device=device)

    def __call__(self, grid_field: Field | torch.Tensor) -> Field | torch.Tensor:
        """Offline xESMF remap with the same Field placement contract."""

        if not isinstance(grid_field, Field):
            return self._call_values(grid_field)
        values, batch_ndim = _latlon_field_to_source_layout(grid_field)
        mapped = self._call_values(values)
        return _native_field_from_latlon_layout(grid_field, mapped, batch_ndim=batch_ndim)


class ESMFMeshToGrid:
    """Optional xESMF mesh-cell to regular-grid remapper.

    This first wrapper uses mesh cells as an ESMF LocStream source.  It is useful
    for offline nearest-style products; conservative polygon remapping is a
    larger follow-up.
    """

    is_differentiable = False
    is_conservative = False

    def __init__(self, mesh, grid: LatLonGrid, *, method: str = "nearest_s2d", periodic: bool = True) -> None:
        xe, xr = _import_xesmf()
        lat, lon = _cell_lat_lon_rad(mesh)
        ds_in = xr.Dataset(
            coords={
                "lat": ("cell", torch.rad2deg(lat).detach().cpu().numpy()),
                "lon": ("cell", torch.rad2deg(lon).detach().cpu().numpy()),
            }
        )
        ds_out = xr.Dataset(coords={"lat": grid.lat_deg.detach().cpu().numpy(), "lon": grid.lon_deg.detach().cpu().numpy()})
        self.mesh = mesh
        self.grid = grid
        self.method = str(method)
        self.regridder = xe.Regridder(ds_in, ds_out, method, locstream_in=True, periodic=periodic)

    def _call_values(self, cell_field: torch.Tensor) -> torch.Tensor:
        import xarray as xr

        device = cell_field.device
        dtype = cell_field.dtype
        arr = cell_field.detach().cpu().numpy()
        extra_shape = arr.shape[1:]
        da = xr.DataArray(arr.reshape(int(self.mesh.nCells), -1), dims=("cell", "component"))
        out = self.regridder(da).values.reshape(self.grid.nlat, self.grid.nlon, *extra_shape)
        return torch.as_tensor(out, dtype=dtype, device=device)

    def __call__(self, cell_field: Field | torch.Tensor) -> Field | torch.Tensor:
        """Offline xESMF remap with the same Field placement contract."""

        if not isinstance(cell_field, Field):
            return self._call_values(cell_field)
        values, batch_ndim = _native_field_to_source_layout(cell_field)
        mapped = self._call_values(values)
        return _latlon_field_from_source_layout(cell_field, mapped, batch_ndim=batch_ndim)


def _regular_edges_from_centers(centers: torch.Tensor, *, clamp: tuple[float, float] | None = None) -> torch.Tensor:
    diffs = centers[1:] - centers[:-1]
    if not torch.allclose(diffs, torch.full_like(diffs, diffs[0]), rtol=1.0e-6, atol=1.0e-10):
        raise ValueError("LatLonGrid currently expects regularly spaced centers")
    dx = diffs[0]
    edges = torch.empty(centers.numel() + 1, dtype=centers.dtype, device=centers.device)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * dx
    edges[-1] = centers[-1] + 0.5 * dx
    if clamp is not None:
        edges = edges.clamp(float(clamp[0]), float(clamp[1]))
    return edges


def _regular_bilinear_weights(grid: LatLonGrid, lat_deg: torch.Tensor, lon_deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lat_centers = grid.lat_deg.detach().cpu().to(dtype=torch.float64)
    lon_centers = grid.lon_deg.detach().cpu().to(dtype=torch.float64)
    dlat = lat_centers[1] - lat_centers[0]
    dlon = lon_centers[1] - lon_centers[0]
    lat_coord = (lat_deg - lat_centers[0]) / dlat
    i0 = torch.floor(lat_coord).to(torch.int64).clamp(0, grid.nlat - 1)
    i1 = (i0 + 1).clamp(0, grid.nlat - 1)
    wi = (lat_coord - i0.to(lat_coord.dtype)).clamp(0.0, 1.0)
    wi = torch.where(i0 == i1, torch.zeros_like(wi), wi)

    period = dlon.abs() * grid.nlon
    lon_wrapped = torch.remainder(lon_deg - lon_centers[0], period) + lon_centers[0]
    lon_coord = (lon_wrapped - lon_centers[0]) / dlon
    j0 = torch.floor(lon_coord).to(torch.int64) % grid.nlon
    j1 = (j0 + 1) % grid.nlon
    wj = (lon_coord - torch.floor(lon_coord)).clamp(0.0, 1.0)

    idx00 = i0 * grid.nlon + j0
    idx01 = i0 * grid.nlon + j1
    idx10 = i1 * grid.nlon + j0
    idx11 = i1 * grid.nlon + j1
    weight = torch.stack(
        [
            (1.0 - wi) * (1.0 - wj),
            (1.0 - wi) * wj,
            wi * (1.0 - wj),
            wi * wj,
        ],
        dim=-1,
    )
    index = torch.stack([idx00, idx01, idx10, idx11], dim=-1)
    return index, weight


def _normalise_field_request(fields: Iterable[str] | str, *, state_kind: str = "atmosphere") -> set[str]:
    if fields == "default":
        if state_kind == "ocean":
            return {"eta", "h", "temperature", "salinity", "wind"}
        return {"ps", "theta", "temperature", "pressure", "wind", "tracers"}
    if fields == "all":
        if state_kind == "ocean":
            return {"eta", "h", "temperature", "salinity", "density", "wind", "tracers"}
        return {"ps", "theta", "temperature", "pressure", "layer_dp", "geopotential", "wind", "tracers"}
    if isinstance(fields, str):
        return {fields}
    return {str(field) for field in fields}


def _import_xesmf():
    try:
        import xarray as xr
        import xesmf as xe
    except Exception as exc:  # pragma: no cover - optional dependency dependent.
        raise ImportError(
            "ESMF/xESMF regridding requires optional analysis dependencies. "
            "Install xesmf and esmpy from conda-forge to use ESMFGridToMesh or ESMFMeshToGrid."
        ) from exc
    return xe, xr


__all__ = [
    "ESMFGridToMesh",
    "ESMFMeshToGrid",
    "LatLonGrid",
    "LatLonToMeshBilinear",
    "MeshToLatLonAdjoint",
    "MeshToLatLonKNN",
]
