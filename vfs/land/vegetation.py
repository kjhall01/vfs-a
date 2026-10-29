"""Vegetation and land-use hooks for land-atmosphere coupling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .parameters import LandSurfaceParameters
from .state import LandSurfaceState


@dataclass(frozen=True)
class CanopyInterceptionResult:
    canopy_water: torch.Tensor
    throughfall_rain_rate: torch.Tensor
    throughfall_snow_rate: torch.Tensor
    canopy_evaporation_rate: torch.Tensor
    remaining_evaporation_mass: torch.Tensor
    rain_interception_rate: torch.Tensor
    snow_interception_rate: torch.Tensor


@dataclass(frozen=True)
class LandUseFields:
    albedo: torch.Tensor
    roughness_length: torch.Tensor
    evaporation_factor: torch.Tensor
    root_depth: torch.Tensor
    emissions_factor: torch.Tensor
    canopy_capacity_factor: torch.Tensor


@dataclass(frozen=True)
class LandUseParameterTable:
    """Dense lookup table for static land-use classes."""

    names: tuple[str, ...] = (
        "generic_land",
        "bare_soil",
        "grassland",
        "shrubland",
        "forest",
        "cropland",
        "urban",
    )
    albedo: tuple[float, ...] = (0.25, 0.28, 0.20, 0.22, 0.13, 0.18, 0.15)
    roughness_length: tuple[float, ...] = (0.05, 0.01, 0.10, 0.20, 1.00, 0.08, 1.00)
    evaporation_factor: tuple[float, ...] = (1.00, 0.35, 1.00, 0.90, 0.85, 1.10, 0.15)
    root_depth: tuple[float, ...] = (1000.0, 0.10, 0.60, 1.00, 2.00, 0.80, 0.10)
    emissions_factor: tuple[float, ...] = (1.00, 0.20, 0.80, 1.00, 1.30, 1.00, 0.30)
    canopy_capacity_factor: tuple[float, ...] = (1.00, 0.00, 0.50, 0.80, 2.00, 0.80, 0.10)

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(str(name) for name in self.names))
        for field in (
            "albedo",
            "roughness_length",
            "evaporation_factor",
            "root_depth",
            "emissions_factor",
            "canopy_capacity_factor",
        ):
            object.__setattr__(self, field, _as_tuple(getattr(self, field), field))
        self.validate()

    def validate(self) -> None:
        n = len(self.names)
        if n == 0:
            raise ValueError("land-use table must include at least one class")
        if len(set(self.names)) != n:
            raise ValueError("land-use class names must be unique")
        for field in (
            "albedo",
            "roughness_length",
            "evaporation_factor",
            "root_depth",
            "emissions_factor",
            "canopy_capacity_factor",
        ):
            if len(getattr(self, field)) != n:
                raise ValueError(f"{field} must have one value per land-use class")
        if any(value < 0.0 or value > 1.0 for value in self.albedo):
            raise ValueError("land-use albedo values must be in [0, 1]")
        if any(value < 0.0 for value in self.roughness_length):
            raise ValueError("land-use roughness_length values must be non-negative")
        if any(value < 0.0 for value in self.evaporation_factor):
            raise ValueError("land-use evaporation_factor values must be non-negative")
        if any(value <= 0.0 for value in self.root_depth):
            raise ValueError("land-use root_depth values must be positive")
        if any(value < 0.0 for value in self.emissions_factor):
            raise ValueError("land-use emissions_factor values must be non-negative")
        if any(value < 0.0 for value in self.canopy_capacity_factor):
            raise ValueError("land-use canopy_capacity_factor values must be non-negative")

    def encode(
        self,
        land_use_type: Any,
        *,
        n_cells: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if isinstance(land_use_type, str):
            if n_cells is None:
                raise ValueError("n_cells is required for scalar land-use names")
            index = self._index(land_use_type)
            return torch.full((int(n_cells), 1), index, dtype=torch.long, device=device)
        if _is_string_sequence(land_use_type):
            indices = [self._index(str(name)) for name in land_use_type]
            tensor = torch.as_tensor(indices, dtype=torch.long, device=device).reshape(-1, 1)
            if n_cells is not None and int(tensor.shape[0]) != int(n_cells):
                raise ValueError(f"land_use_type has {int(tensor.shape[0])} cells, expected {int(n_cells)}")
            return tensor
        tensor = torch.as_tensor(land_use_type, dtype=torch.long, device=device)
        if tensor.ndim == 0:
            if n_cells is None:
                raise ValueError("n_cells is required for scalar land-use indices")
            tensor = torch.full((int(n_cells), 1), int(tensor.item()), dtype=torch.long, device=device)
        elif tensor.ndim == 1:
            tensor = tensor[:, None]
        if tensor.ndim != 2 or int(tensor.shape[1]) != 1:
            raise ValueError("land_use_type must have shape [n_cells] or [n_cells, 1]")
        if n_cells is not None and int(tensor.shape[0]) != int(n_cells):
            raise ValueError(f"land_use_type has {int(tensor.shape[0])} cells, expected {int(n_cells)}")
        if bool((tensor < 0).any().detach().cpu().item()) or bool((tensor >= len(self.names)).any().detach().cpu().item()):
            raise ValueError("land_use_type contains an index outside the land-use table")
        return tensor

    def fields(self, land_use_type: Any, *, like: torch.Tensor) -> LandUseFields:
        if isinstance(land_use_type, str) or _is_string_sequence(land_use_type):
            indices = self.encode(land_use_type, n_cells=int(like.shape[0]), device=like.device).reshape(-1)
        else:
            indices = _lookup_indices(land_use_type, like=like)
        return LandUseFields(
            self._lookup("albedo", indices, like),
            self._lookup("roughness_length", indices, like),
            self._lookup("evaporation_factor", indices, like),
            self._lookup("root_depth", indices, like),
            self._lookup("emissions_factor", indices, like),
            self._lookup("canopy_capacity_factor", indices, like),
        )

    def _index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise ValueError(f"unknown land-use class {name!r}; expected one of {self.names}") from exc

    def _lookup(self, field: str, indices: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(getattr(self, field), dtype=like.dtype, device=like.device)
        return values.gather(0, indices).reshape(-1, 1)


def land_use_root_fraction(
    land_use_fields: LandUseFields,
    land_params: LandSurfaceParameters,
    ref: torch.Tensor,
) -> torch.Tensor:
    if land_params.root_fraction is not None:
        root = land_params.root_fraction_tensor(ref).reshape(1, -1)
        return root.expand_as(ref)
    dz = land_params.soil_layer_thickness_tensor(ref)
    bottoms = dz.cumsum(dim=0)
    tops = bottoms - dz
    root_depth = land_use_fields.root_depth.to(dtype=ref.dtype, device=ref.device).clamp_min(torch.finfo(ref.dtype).tiny)
    density = torch.exp(-tops.reshape(1, -1) / root_depth) - torch.exp(-bottoms.reshape(1, -1) / root_depth)
    return density / density.sum(dim=1, keepdim=True).clamp_min(torch.finfo(ref.dtype).tiny)


def canopy_interception(
    state: LandSurfaceState,
    *,
    land_fraction: torch.Tensor,
    rain_rate: torch.Tensor,
    snow_rate: torch.Tensor,
    latent_heat_flux: torch.Tensor,
    dt: torch.Tensor,
    params: LandSurfaceParameters,
    land_use_fields: LandUseFields,
) -> CanopyInterceptionResult:
    ref = state.canopy_water
    land = land_fraction.to(dtype=ref.dtype, device=ref.device)
    active = land > 0.0
    dt = dt.to(dtype=ref.dtype, device=ref.device).clamp_min(torch.finfo(ref.dtype).tiny)
    rain = rain_rate.to(dtype=ref.dtype, device=ref.device).clamp_min(0.0)
    snow = snow_rate.to(dtype=ref.dtype, device=ref.device).clamp_min(0.0)
    incoming = (rain + snow) * dt

    capacity = float(params.canopy_water_capacity) * land_use_fields.canopy_capacity_factor.to(
        dtype=ref.dtype,
        device=ref.device,
    )
    capacity = torch.where(active, capacity, torch.zeros_like(capacity))
    available_capacity = (capacity - ref).clamp_min(0.0)
    intercepted = torch.minimum(incoming, available_capacity)
    incoming_safe = incoming.clamp_min(torch.finfo(ref.dtype).tiny)
    rain_intercepted = torch.where(incoming > 0.0, intercepted * (rain * dt) / incoming_safe, torch.zeros_like(ref))
    snow_intercepted = intercepted - rain_intercepted

    canopy_after_interception = ref + intercepted
    potential_evaporation_mass = (
        latent_heat_flux.to(dtype=ref.dtype, device=ref.device).clamp_min(0.0)
        / float(params.latent_heat_vaporization)
        * dt
        * land_use_fields.evaporation_factor.to(dtype=ref.dtype, device=ref.device).clamp_min(0.0)
    )
    potential_evaporation_mass = torch.where(active, potential_evaporation_mass, torch.zeros_like(potential_evaporation_mass))
    canopy_evaporation = torch.minimum(canopy_after_interception, potential_evaporation_mass)
    canopy_next = canopy_after_interception - canopy_evaporation
    remaining_evaporation = (potential_evaporation_mass - canopy_evaporation).clamp_min(0.0)

    throughfall_rain = (rain - rain_intercepted / dt).clamp_min(0.0)
    throughfall_snow = (snow - snow_intercepted / dt).clamp_min(0.0)
    canopy_next = torch.where(active, canopy_next, ref)
    throughfall_rain = torch.where(active, throughfall_rain, torch.zeros_like(throughfall_rain))
    throughfall_snow = torch.where(active, throughfall_snow, torch.zeros_like(throughfall_snow))
    rain_intercepted = torch.where(active, rain_intercepted, torch.zeros_like(rain_intercepted))
    snow_intercepted = torch.where(active, snow_intercepted, torch.zeros_like(snow_intercepted))
    canopy_evaporation = torch.where(active, canopy_evaporation, torch.zeros_like(canopy_evaporation))
    return CanopyInterceptionResult(
        canopy_next,
        throughfall_rain,
        throughfall_snow,
        canopy_evaporation / dt,
        remaining_evaporation,
        rain_intercepted / dt,
        snow_intercepted / dt,
    )


def _as_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of numbers")
    return tuple(float(value) for value in values)


def _is_string_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and all(isinstance(item, str) for item in value)


def _lookup_indices(land_use_type: Any, *, like: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(land_use_type, dtype=torch.long, device=like.device)
    if tensor.ndim == 0:
        tensor = tensor.expand(int(like.shape[0]))
    elif tensor.ndim == 2 and int(tensor.shape[1]) == 1:
        tensor = tensor[:, 0]
    elif tensor.ndim != 1:
        raise ValueError("land_use_type must have shape [n_cells] or [n_cells, 1]")
    if int(tensor.shape[0]) != int(like.shape[0]):
        raise ValueError(f"land_use_type has {int(tensor.shape[0])} cells, expected {int(like.shape[0])}")
    return tensor
