"""Tensor-native fractional surface ownership masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vfs.field import Field

from .utils import as_cell_column, require_finite


def _surface_field(values: Field | torch.Tensor, *, name: str) -> Field:
    tensor = values.values if isinstance(values, Field) else values
    return Field(tensor, location="cell", vertical_location="surface", name=name)


@dataclass(frozen=True)
class SurfaceFractions:
    """Global fractional ownership fields on the native cell mesh.

    Fractions are not object-per-cell metadata. Every field is a dense tensor
    with shape ``[nCells, 1]`` after construction. Sea ice and glacier are
    overlays constrained by the base ocean/land fractions.
    """

    land_fraction: Field | torch.Tensor
    ocean_fraction: Field | torch.Tensor
    sea_ice_fraction: Field | torch.Tensor
    glacier_fraction: Field | torch.Tensor
    snow_fraction: Field | torch.Tensor

    def __post_init__(self) -> None:
        land_values = self.land_fraction.values if isinstance(self.land_fraction, Field) else self.land_fraction
        n_cells = int(land_values.shape[-2]) if land_values.ndim >= 2 else int(land_values.numel())
        dtype = land_values.dtype
        device = land_values.device
        land = as_cell_column(self.land_fraction, n_cells=n_cells, dtype=dtype, device=device, name="land_fraction")
        object.__setattr__(
            self,
            "land_fraction",
            _surface_field(land, name="land_fraction"),
        )
        object.__setattr__(
            self,
            "ocean_fraction",
            _surface_field(as_cell_column(self.ocean_fraction, like=land, name="ocean_fraction"), name="ocean_fraction"),
        )
        object.__setattr__(
            self,
            "sea_ice_fraction",
            _surface_field(as_cell_column(self.sea_ice_fraction, like=land, name="sea_ice_fraction"), name="sea_ice_fraction"),
        )
        object.__setattr__(
            self,
            "glacier_fraction",
            _surface_field(as_cell_column(self.glacier_fraction, like=land, name="glacier_fraction"), name="glacier_fraction"),
        )
        object.__setattr__(
            self,
            "snow_fraction",
            _surface_field(as_cell_column(self.snow_fraction, like=land, name="snow_fraction"), name="snow_fraction"),
        )
        self.validate()

    @classmethod
    def from_land_ocean(
        cls,
        land_fraction: Any,
        ocean_fraction: Any | None = None,
        *,
        sea_ice_fraction: Any = 0.0,
        glacier_fraction: Any = 0.0,
        snow_fraction: Any = 0.0,
        n_cells: int | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> "SurfaceFractions":
        land = as_cell_column(land_fraction, n_cells=n_cells, dtype=dtype, device=device, name="land_fraction")
        ocean = 1.0 - land if ocean_fraction is None else as_cell_column(
            ocean_fraction,
            like=land,
            name="ocean_fraction",
        )
        return cls(
            land,
            ocean,
            as_cell_column(sea_ice_fraction, like=land, name="sea_ice_fraction"),
            as_cell_column(glacier_fraction, like=land, name="glacier_fraction"),
            as_cell_column(snow_fraction, like=land, name="snow_fraction"),
        )

    @classmethod
    def global_ocean(
        cls,
        n_cells: int,
        *,
        sea_ice_fraction: Any = 0.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> "SurfaceFractions":
        return cls.from_land_ocean(
            0.0,
            1.0,
            sea_ice_fraction=sea_ice_fraction,
            n_cells=n_cells,
            dtype=dtype,
            device=device,
        )

    def validate(self, *, atol: float = 1.0e-10) -> None:
        fields = {
            "land_fraction": self.land_fraction,
            "ocean_fraction": self.ocean_fraction,
            "sea_ice_fraction": self.sea_ice_fraction,
            "glacier_fraction": self.glacier_fraction,
            "snow_fraction": self.snow_fraction,
        }
        shape = tuple(self.land_fraction.shape)
        for name, value in fields.items():
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} shape {tuple(value.shape)} does not match land_fraction {shape}")
            require_finite(name, value)
            if bool((value < -atol).any().detach().cpu().item()) or bool((value > 1.0 + atol).any().detach().cpu().item()):
                raise ValueError(f"{name} must be in [0, 1]")
        if not bool(torch.allclose(self.land_fraction + self.ocean_fraction, torch.ones_like(self.land_fraction), atol=atol, rtol=0.0)):
            raise ValueError("land_fraction + ocean_fraction must equal 1")
        if bool((self.sea_ice_fraction - self.ocean_fraction > atol).any().detach().cpu().item()):
            raise ValueError("sea_ice_fraction must not exceed ocean_fraction")
        if bool((self.glacier_fraction - self.land_fraction > atol).any().detach().cpu().item()):
            raise ValueError("glacier_fraction must not exceed land_fraction")
        if bool((self.snow_fraction - (self.land_fraction + self.sea_ice_fraction) > atol).any().detach().cpu().item()):
            raise ValueError("snow_fraction must not exceed exposed land plus sea ice")

    def with_overlays(
        self,
        *,
        sea_ice_fraction: torch.Tensor | None = None,
        glacier_fraction: torch.Tensor | None = None,
        snow_fraction: torch.Tensor | None = None,
    ) -> "SurfaceFractions":
        return SurfaceFractions(
            self.land_fraction,
            self.ocean_fraction,
            self.sea_ice_fraction if sea_ice_fraction is None else sea_ice_fraction.clamp(0.0, 1.0),
            self.glacier_fraction if glacier_fraction is None else glacier_fraction.clamp(0.0, 1.0),
            self.snow_fraction if snow_fraction is None else snow_fraction.clamp(0.0, 1.0),
        )

    def to(self, *args, **kwargs) -> "SurfaceFractions":
        return SurfaceFractions(
            self.land_fraction.to(*args, **kwargs),
            self.ocean_fraction.to(*args, **kwargs),
            self.sea_ice_fraction.to(*args, **kwargs),
            self.glacier_fraction.to(*args, **kwargs),
            self.snow_fraction.to(*args, **kwargs),
        )

    def clone(self) -> "SurfaceFractions":
        return SurfaceFractions(
            self.land_fraction.clone(),
            self.ocean_fraction.clone(),
            self.sea_ice_fraction.clone(),
            self.glacier_fraction.clone(),
            self.snow_fraction.clone(),
        )

    def detach(self) -> "SurfaceFractions":
        return SurfaceFractions(
            self.land_fraction.detach(),
            self.ocean_fraction.detach(),
            self.sea_ice_fraction.detach(),
            self.glacier_fraction.detach(),
            self.snow_fraction.detach(),
        )


__all__ = ["SurfaceFractions"]
