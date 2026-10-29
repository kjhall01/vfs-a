"""Restartable sea-ice state containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vfs.field import Field

from ..surface.conservation import ConservativeTransfer
from ..surface.fractions import SurfaceFractions
from ..surface.utils import as_cell_column, cell_area_column, field_values


def _ice_field(values: Field | torch.Tensor, *, name: str) -> Field:
    return Field(field_values(values), location="cell", vertical_location="surface", name=name)


def _ice_edge_field(values: Field | torch.Tensor, *, name: str) -> Field:
    """Normalize one edge-normal sea-ice momentum field."""

    tensor = field_values(values)
    if tensor.ndim == 1:
        tensor = tensor[:, None]
    if tensor.ndim != 2 or int(tensor.shape[-1]) != 1:
        raise ValueError(f"{name} must have shape [nEdges, 1] or [nEdges], got {tuple(tensor.shape)}")
    return Field(tensor, location="edge", vertical_location="surface", name=name)


@dataclass(frozen=True)
class SeaIceState:
    sea_ice_fraction: Field | torch.Tensor
    sea_ice_thickness: Field | torch.Tensor
    snow_on_sea_ice: Field | torch.Tensor
    sea_ice_surface_temperature: Field | torch.Tensor
    sea_ice_u_e: Field | torch.Tensor | None = None
    sea_ice_stress_xx: Field | torch.Tensor | None = None
    sea_ice_stress_xy: Field | torch.Tensor | None = None
    sea_ice_stress_yy: Field | torch.Tensor | None = None

    @classmethod
    def uniform(
        cls,
        n_cells: int,
        *,
        sea_ice_fraction: float = 0.0,
        sea_ice_thickness: float = 0.0,
        snow_on_sea_ice: float = 0.0,
        sea_ice_surface_temperature: float = 268.0,
        n_edges: int | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> "SeaIceState":
        return cls(
            as_cell_column(sea_ice_fraction, n_cells=n_cells, dtype=dtype, device=device, name="sea_ice_fraction"),
            as_cell_column(sea_ice_thickness, n_cells=n_cells, dtype=dtype, device=device, name="sea_ice_thickness"),
            as_cell_column(snow_on_sea_ice, n_cells=n_cells, dtype=dtype, device=device, name="snow_on_sea_ice"),
            as_cell_column(
                sea_ice_surface_temperature,
                n_cells=n_cells,
                dtype=dtype,
                device=device,
                name="sea_ice_surface_temperature",
            ),
            None if n_edges is None else torch.zeros(int(n_edges), 1, dtype=dtype, device=device),
            None if n_edges is None else torch.zeros(int(n_cells), 1, dtype=dtype, device=device),
            None if n_edges is None else torch.zeros(int(n_cells), 1, dtype=dtype, device=device),
            None if n_edges is None else torch.zeros(int(n_cells), 1, dtype=dtype, device=device),
        )

    def __post_init__(self) -> None:
        like = as_cell_column(
            self.sea_ice_fraction,
            n_cells=int(field_values(self.sea_ice_fraction).numel()),
            name="sea_ice_fraction",
        )
        object.__setattr__(self, "sea_ice_fraction", _ice_field(like.clamp(0.0, 1.0), name="sea_ice_fraction"))
        for name in ("sea_ice_thickness", "snow_on_sea_ice", "sea_ice_surface_temperature"):
            object.__setattr__(self, name, _ice_field(as_cell_column(getattr(self, name), like=like, name=name), name=name))
        if self.sea_ice_u_e is not None:
            object.__setattr__(self, "sea_ice_u_e", _ice_edge_field(self.sea_ice_u_e, name="sea_ice_u_e"))
        for name in ("sea_ice_stress_xx", "sea_ice_stress_xy", "sea_ice_stress_yy"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _ice_field(as_cell_column(value, like=like, name=name), name=name))

    def to(self, *args, **kwargs) -> "SeaIceState":
        return SeaIceState(
            self.sea_ice_fraction.to(*args, **kwargs),
            self.sea_ice_thickness.to(*args, **kwargs),
            self.snow_on_sea_ice.to(*args, **kwargs),
            self.sea_ice_surface_temperature.to(*args, **kwargs),
            None if self.sea_ice_u_e is None else self.sea_ice_u_e.to(*args, **kwargs),
            None if self.sea_ice_stress_xx is None else self.sea_ice_stress_xx.to(*args, **kwargs),
            None if self.sea_ice_stress_xy is None else self.sea_ice_stress_xy.to(*args, **kwargs),
            None if self.sea_ice_stress_yy is None else self.sea_ice_stress_yy.to(*args, **kwargs),
        )

    def clone(self) -> "SeaIceState":
        return SeaIceState(
            self.sea_ice_fraction.clone(),
            self.sea_ice_thickness.clone(),
            self.snow_on_sea_ice.clone(),
            self.sea_ice_surface_temperature.clone(),
            None if self.sea_ice_u_e is None else self.sea_ice_u_e.clone(),
            None if self.sea_ice_stress_xx is None else self.sea_ice_stress_xx.clone(),
            None if self.sea_ice_stress_xy is None else self.sea_ice_stress_xy.clone(),
            None if self.sea_ice_stress_yy is None else self.sea_ice_stress_yy.clone(),
        )

    def detach(self) -> "SeaIceState":
        return SeaIceState(
            self.sea_ice_fraction.detach(),
            self.sea_ice_thickness.detach(),
            self.snow_on_sea_ice.detach(),
            self.sea_ice_surface_temperature.detach(),
            None if self.sea_ice_u_e is None else self.sea_ice_u_e.detach(),
            None if self.sea_ice_stress_xx is None else self.sea_ice_stress_xx.detach(),
            None if self.sea_ice_stress_xy is None else self.sea_ice_stress_xy.detach(),
            None if self.sea_ice_stress_yy is None else self.sea_ice_stress_yy.detach(),
        )

    def total_water(self, *, mesh: Any | None = None, ice_density: float = 917.0) -> torch.Tensor:
        area = cell_area_column(mesh, self.sea_ice_thickness)
        ice = self.sea_ice_thickness * self.sea_ice_fraction * float(ice_density)
        snow = self.snow_on_sea_ice * self.sea_ice_fraction
        return ((ice + snow) * area).sum()

    def momentum_storage(self, *, mesh: Any, ice_density: float = 917.0) -> torch.Tensor:
        """Return Cartesian horizontal ice momentum from edge-normal velocity.

        The sea-ice momentum closure stores velocity at native C-grid edges,
        matching atmosphere and ocean velocities.  ``edge_side_area`` is the
        mesh-owned edge control area used for the corresponding stress
        exchange, so the returned vector is suitable for conservation audits.
        """

        if self.sea_ice_u_e is None:
            return torch.zeros(3, dtype=self.sea_ice_thickness.dtype, device=self.sea_ice_thickness.device)
        u_e = field_values(self.sea_ice_u_e)
        mass_cell = float(ice_density) * field_values(self.sea_ice_thickness).clamp_min(0.0) * field_values(
            self.sea_ice_fraction
        ).clamp(0.0, 1.0)
        mass_edge = mesh.cell_to_edge(mass_cell).to(dtype=u_e.dtype, device=u_e.device)
        area = mesh.edge_side_area.sum(dim=-1, keepdim=True).to(dtype=u_e.dtype, device=u_e.device)
        normal = mesh.edge_normal.to(dtype=u_e.dtype, device=u_e.device)
        return (mass_edge * area * u_e * normal).sum(dim=0)


@dataclass(frozen=True)
class SeaIceStepResult:
    state: SeaIceState
    fractions: SurfaceFractions
    ocean_heat_flux: torch.Tensor
    ocean_freshwater_flux: torch.Tensor
    ocean_salt_flux: torch.Tensor
    transfers: tuple[ConservativeTransfer, ...]
    diagnostics: dict[str, torch.Tensor]
