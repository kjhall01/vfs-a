"""Restartable land-state containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vfs.field import Field

from ..surface.conservation import ConservativeTransfer
from ..surface.fractions import SurfaceFractions
from ..surface.utils import as_cell_column, cell_area_column, field_values


def _land_field(
    values: Field | torch.Tensor,
    *,
    name: str,
    vertical_location: str = "surface",
    component_axes: tuple[str, ...] = (),
) -> Field:
    tensor = field_values(values)
    return Field(
        tensor,
        location="cell",
        vertical_location=vertical_location,
        name=name,
        component_axes=component_axes,
    )


@dataclass(frozen=True)
class LandSurfaceState:
    """Dense land-surface prognostic tensors on all cells.

    Soil fields are shaped ``[n_cells, n_soil_layers]`` so hydrology and
    thermal diffusion can stay vectorized on GPU. Cell-level stores remain
    shaped ``[n_cells, 1]``. Water stores are kg m-2 and temperatures are K.
    """

    soil_liquid_water: Field | torch.Tensor
    soil_temperature: Field | torch.Tensor
    snow_water_equivalent: Field | torch.Tensor
    skin_temperature: Field | torch.Tensor
    soil_ice: Field | torch.Tensor | None = None
    canopy_water: Field | torch.Tensor | None = None
    land_use_type: Field | torch.Tensor | None = None

    @classmethod
    def uniform(
        cls,
        n_cells: int,
        *,
        soil_water: float = 50.0,
        soil_temperature: float = 280.0,
        snow_water_equivalent: float = 0.0,
        skin_temperature: float = 280.0,
        soil_ice: float = 0.0,
        canopy_water: float = 0.0,
        land_use_type: Any = 0,
        n_soil_layers: int | None = None,
        soil_layer_thickness: tuple[float, ...] | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> "LandSurfaceState":
        if soil_layer_thickness is not None:
            weights = torch.as_tensor(soil_layer_thickness, dtype=dtype, device=device)
            if weights.ndim != 1 or int(weights.numel()) == 0:
                raise ValueError("soil_layer_thickness must be a non-empty 1D sequence")
            if bool((weights <= 0.0).any().detach().cpu().item()):
                raise ValueError("soil_layer_thickness values must be positive")
            weights = weights / weights.sum()
        else:
            n_layers = int(n_soil_layers or 1)
            if n_layers <= 0:
                raise ValueError("n_soil_layers must be positive")
            weights = torch.full((n_layers,), 1.0 / n_layers, dtype=dtype, device=device)
        soil_total = as_cell_column(soil_water, n_cells=n_cells, dtype=dtype, device=device, name="soil_water")
        ice_total = as_cell_column(soil_ice, n_cells=n_cells, dtype=dtype, device=device, name="soil_ice")
        return cls(
            soil_total * weights.reshape(1, -1),
            as_cell_column(soil_temperature, n_cells=n_cells, dtype=dtype, device=device, name="soil_temperature")
            .expand(n_cells, int(weights.numel()))
            .clone(),
            as_cell_column(
                snow_water_equivalent,
                n_cells=n_cells,
                dtype=dtype,
                device=device,
                name="snow_water_equivalent",
            ),
            as_cell_column(skin_temperature, n_cells=n_cells, dtype=dtype, device=device, name="skin_temperature"),
            soil_ice=ice_total * weights.reshape(1, -1),
            canopy_water=as_cell_column(canopy_water, n_cells=n_cells, dtype=dtype, device=device, name="canopy_water"),
            land_use_type=as_cell_column(land_use_type, n_cells=n_cells, dtype=torch.long, device=device, name="land_use_type"),
        )

    def __post_init__(self) -> None:
        liquid = _as_soil_layers(self.soil_liquid_water, name="soil_liquid_water")
        object.__setattr__(
            self,
            "soil_liquid_water",
            _land_field(liquid, name="soil_liquid_water", vertical_location="full"),
        )
        object.__setattr__(
            self,
            "soil_temperature",
            _land_field(
                _as_soil_layers(self.soil_temperature, like=liquid, name="soil_temperature"),
                name="soil_temperature",
                vertical_location="full",
            ),
        )
        ice = torch.zeros_like(liquid) if self.soil_ice is None else _as_soil_layers(self.soil_ice, like=liquid, name="soil_ice")
        object.__setattr__(self, "soil_ice", _land_field(ice, name="soil_ice", vertical_location="full"))
        n_cells = int(liquid.shape[0])
        object.__setattr__(
            self,
            "snow_water_equivalent",
            _land_field(
                as_cell_column(self.snow_water_equivalent, n_cells=n_cells, dtype=liquid.dtype, device=liquid.device, name="snow_water_equivalent"),
                name="snow_water_equivalent",
            ),
        )
        object.__setattr__(
            self,
            "skin_temperature",
            _land_field(
                as_cell_column(self.skin_temperature, n_cells=n_cells, dtype=liquid.dtype, device=liquid.device, name="skin_temperature"),
                name="skin_temperature",
            ),
        )
        canopy = torch.zeros((n_cells, 1), dtype=liquid.dtype, device=liquid.device) if self.canopy_water is None else as_cell_column(
            self.canopy_water,
            n_cells=n_cells,
            dtype=liquid.dtype,
            device=liquid.device,
            name="canopy_water",
        )
        object.__setattr__(self, "canopy_water", _land_field(canopy, name="canopy_water"))
        land_use = torch.zeros((n_cells, 1), dtype=torch.long, device=liquid.device) if self.land_use_type is None else as_cell_column(
            self.land_use_type,
            n_cells=n_cells,
            dtype=torch.long,
            device=liquid.device,
            name="land_use_type",
        )
        object.__setattr__(self, "land_use_type", _land_field(land_use, name="land_use_type"))

    @property
    def n_soil_layers(self) -> int:
        return int(self.soil_liquid_water.values.shape[1])

    @property
    def soil_water(self) -> Field:
        return _land_field(self.soil_liquid_water.values.sum(dim=1, keepdim=True), name="soil_water")

    @property
    def top_soil_temperature(self) -> Field:
        return _land_field(self.soil_temperature.values[:, :1], name="top_soil_temperature")

    def to(self, *args, **kwargs) -> "LandSurfaceState":
        return LandSurfaceState(
            self.soil_liquid_water.to(*args, **kwargs),
            self.soil_temperature.to(*args, **kwargs),
            self.snow_water_equivalent.to(*args, **kwargs),
            self.skin_temperature.to(*args, **kwargs),
            self.soil_ice.to(*args, **kwargs),
            self.canopy_water.to(*args, **kwargs),
            self.land_use_type.to(*args, **kwargs),
        )

    def clone(self) -> "LandSurfaceState":
        return LandSurfaceState(
            self.soil_liquid_water.clone(),
            self.soil_temperature.clone(),
            self.snow_water_equivalent.clone(),
            self.skin_temperature.clone(),
            self.soil_ice.clone(),
            self.canopy_water.clone(),
            self.land_use_type.clone(),
        )

    def detach(self) -> "LandSurfaceState":
        return LandSurfaceState(
            self.soil_liquid_water.detach(),
            self.soil_temperature.detach(),
            self.snow_water_equivalent.detach(),
            self.skin_temperature.detach(),
            self.soil_ice.detach(),
            self.canopy_water.detach(),
            self.land_use_type.detach(),
        )

    def total_water(self, fractions: SurfaceFractions, *, mesh: Any | None = None) -> torch.Tensor:
        active = fractions.land_fraction.to(dtype=self.soil_water.dtype, device=self.soil_water.device)
        area = cell_area_column(mesh, self.soil_water)
        soil_total = self.soil_liquid_water.sum(dim=1, keepdim=True) + self.soil_ice.sum(dim=1, keepdim=True)
        storage = soil_total + self.snow_water_equivalent + self.canopy_water
        return (storage * active * area).sum()


@dataclass(frozen=True)
class LandSurfaceStepResult:
    state: LandSurfaceState
    runoff_mass: torch.Tensor
    transfers: tuple[ConservativeTransfer, ...]
    diagnostics: dict[str, torch.Tensor]


def _as_soil_layers(value: torch.Tensor, *, like: torch.Tensor | None = None, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(field_values(value))
    if like is not None:
        tensor = tensor.to(dtype=like.dtype, device=like.device)
    if tensor.ndim == 0:
        if like is None:
            raise ValueError(f"{name} scalar input requires a reference field")
        return tensor.expand_as(like).clone()
    if tensor.ndim == 1:
        tensor = tensor.reshape(-1, 1)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [n_cells] or [n_cells, n_soil_layers]")
    if like is None:
        return tensor
    if int(tensor.shape[0]) != int(like.shape[0]):
        raise ValueError(f"{name} n_cells={int(tensor.shape[0])} does not match {int(like.shape[0])}")
    if int(tensor.shape[1]) == int(like.shape[1]):
        return tensor
    if int(tensor.shape[1]) == 1:
        return tensor.expand_as(like).clone()
    raise ValueError(f"{name} n_soil_layers={int(tensor.shape[1])} does not match {int(like.shape[1])}")
