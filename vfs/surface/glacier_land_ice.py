"""Minimal glacier/land-ice reservoir skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..core.registry import register_component
from vfs.field import Field
from .conservation import ConservativeTransfer
from .fractions import SurfaceFractions
from .utils import as_cell_column, cell_area_column, field_values


def _glacier_field(values: Field | torch.Tensor, *, name: str) -> Field:
    return Field(field_values(values), location="cell", vertical_location="surface", name=name)


@dataclass(frozen=True)
class GlacierState:
    glacier_fraction: Field | torch.Tensor
    land_ice_thickness: Field | torch.Tensor
    snow_on_glacier: Field | torch.Tensor
    glacier_surface_temperature: Field | torch.Tensor

    @classmethod
    def uniform(
        cls,
        n_cells: int,
        *,
        glacier_fraction: float = 0.0,
        land_ice_thickness: float = 0.0,
        snow_on_glacier: float = 0.0,
        glacier_surface_temperature: float = 268.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> "GlacierState":
        return cls(
            as_cell_column(glacier_fraction, n_cells=n_cells, dtype=dtype, device=device, name="glacier_fraction"),
            as_cell_column(land_ice_thickness, n_cells=n_cells, dtype=dtype, device=device, name="land_ice_thickness"),
            as_cell_column(snow_on_glacier, n_cells=n_cells, dtype=dtype, device=device, name="snow_on_glacier"),
            as_cell_column(glacier_surface_temperature, n_cells=n_cells, dtype=dtype, device=device, name="glacier_surface_temperature"),
        )

    def __post_init__(self) -> None:
        like = as_cell_column(self.glacier_fraction, n_cells=int(field_values(self.glacier_fraction).numel()), name="glacier_fraction")
        object.__setattr__(self, "glacier_fraction", _glacier_field(like.clamp(0.0, 1.0), name="glacier_fraction"))
        for name in ("land_ice_thickness", "snow_on_glacier", "glacier_surface_temperature"):
            object.__setattr__(self, name, _glacier_field(as_cell_column(getattr(self, name), like=like, name=name), name=name))

    def to(self, *args, **kwargs) -> "GlacierState":
        return GlacierState(
            self.glacier_fraction.to(*args, **kwargs),
            self.land_ice_thickness.to(*args, **kwargs),
            self.snow_on_glacier.to(*args, **kwargs),
            self.glacier_surface_temperature.to(*args, **kwargs),
        )

    def clone(self) -> "GlacierState":
        return GlacierState(
            self.glacier_fraction.clone(),
            self.land_ice_thickness.clone(),
            self.snow_on_glacier.clone(),
            self.glacier_surface_temperature.clone(),
        )

    def detach(self) -> "GlacierState":
        return GlacierState(
            self.glacier_fraction.detach(),
            self.land_ice_thickness.detach(),
            self.snow_on_glacier.detach(),
            self.glacier_surface_temperature.detach(),
        )

    def total_water(self, *, mesh: Any | None = None, ice_density: float = 917.0) -> torch.Tensor:
        area = cell_area_column(mesh, self.land_ice_thickness)
        return (self.land_ice_thickness * self.glacier_fraction * float(ice_density) * area + self.snow_on_glacier * self.glacier_fraction * area).sum()


@dataclass(frozen=True)
class GlacierStepResult:
    state: GlacierState
    fractions: SurfaceFractions
    iceberg_source_mass: torch.Tensor
    runoff_mass: torch.Tensor
    transfers: tuple[ConservativeTransfer, ...]
    diagnostics: dict[str, torch.Tensor]


class GlacierLandIce(nn.Module):
    """Surface accumulation/melt/calving placeholder for land ice."""

    recipe_name = "glacier_land_ice"

    def __init__(
        self,
        *,
        ice_density: float = 917.0,
        melt_temperature: float = 273.15,
        melt_factor: float = 2.0e-5,
        evolve_fraction: bool = True,
    ) -> None:
        super().__init__()
        self.ice_density = float(ice_density)
        self.melt_temperature = float(melt_temperature)
        self.melt_factor = float(melt_factor)
        self.evolve_fraction = bool(evolve_fraction)
        if self.ice_density <= 0.0:
            raise ValueError("ice_density must be positive")

    def step(
        self,
        state: GlacierState,
        fractions: SurfaceFractions,
        dt: float,
        *,
        snowfall: torch.Tensor | float = 0.0,
        calving_rate: torch.Tensor | float = 0.0,
        mesh: Any | None = None,
    ) -> GlacierStepResult:
        like = state.land_ice_thickness
        area = cell_area_column(mesh, like)
        glacier = state.glacier_fraction.to(dtype=like.dtype, device=like.device)
        dt_t = torch.as_tensor(float(dt), dtype=like.dtype, device=like.device)
        snow = as_cell_column(snowfall, like=like, name="snowfall").clamp_min(0.0) * glacier
        calving = as_cell_column(calving_rate, like=like, name="calving_rate").clamp_min(0.0) * glacier
        melt_rate = float(self.melt_factor) * (state.glacier_surface_temperature - float(self.melt_temperature)).clamp_min(0.0) * glacier
        melt_mass = torch.minimum(melt_rate * dt_t * area, state.land_ice_thickness * glacier * float(self.ice_density) * area)
        calving_mass = torch.minimum(calving * dt_t * area, state.land_ice_thickness * glacier * float(self.ice_density) * area - melt_mass)
        calving_mass = calving_mass.clamp_min(0.0)

        dh = (melt_mass + calving_mass) / (float(self.ice_density) * area * glacier.clamp_min(torch.finfo(like.dtype).tiny))
        thickness_next = (state.land_ice_thickness - dh).clamp_min(0.0)
        snow_next = state.snow_on_glacier + snow * dt_t
        fraction_next = glacier
        if self.evolve_fraction:
            fraction_next = torch.where(thickness_next <= 1.0e-12, torch.zeros_like(glacier), glacier)
            fraction_next = torch.minimum(fraction_next, fractions.land_fraction.to(dtype=like.dtype, device=like.device))
        fractions_next = fractions.with_overlays(glacier_fraction=fraction_next)
        next_state = GlacierState(fraction_next, thickness_next, snow_next, state.glacier_surface_temperature)
        transfers = (
            ConservativeTransfer("snow_to_glacier", "atmosphere", "glacier", water_mass=snow * dt_t * area),
            ConservativeTransfer("glacier_surface_melt", "glacier", "runoff", water_mass=melt_mass),
            ConservativeTransfer("glacier_calving", "glacier", "iceberg", water_mass=calving_mass),
        )
        diagnostics = {
            "glacier_melt_mass": melt_mass.sum(),
            "glacier_calving_mass": calving_mass.sum(),
            "glacier_water_storage": next_state.total_water(mesh=mesh, ice_density=self.ice_density),
        }
        return GlacierStepResult(next_state, fractions_next, calving_mass, melt_mass, transfers, diagnostics)


register_component("glacier_land_ice", GlacierLandIce)


__all__ = ["GlacierLandIce", "GlacierState", "GlacierStepResult"]
