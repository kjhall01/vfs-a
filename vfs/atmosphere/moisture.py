"""Core tensor helpers for named water tracers."""

from __future__ import annotations

import torch

from vfs.atmosphere.state import State


VAPOR_TRACER_INDEX_ATTR = "vapor_tracer_index"
VAPOR_QUANTITY_ATTR = "vapor_quantity"
VAPOR_QUANTITY_DRY_MIXING_RATIO = "dry_air_mixing_ratio"
VAPOR_QUANTITY_SPECIFIC_HUMIDITY = "specific_humidity"
CLOUD_LIQUID_TRACER_INDEX_ATTR = "cloud_liquid_tracer_index"
CLOUD_LIQUID_QUANTITY_ATTR = "cloud_liquid_quantity"
CLOUD_LIQUID_QUANTITY_DRY_MIXING_RATIO = "dry_air_mixing_ratio"
EPSILON_WATER_DRY_AIR = 0.622


def vapor_tracer_index(state: State, default: int | None = None) -> int | None:
    """Return the named vapor tracer index when present and valid."""

    return _named_tracer_index(state, VAPOR_TRACER_INDEX_ATTR, default=default)


def cloud_liquid_tracer_index(state: State, default: int | None = None) -> int | None:
    """Return the named cloud-liquid tracer index when present and valid."""

    return _named_tracer_index(state, CLOUD_LIQUID_TRACER_INDEX_ATTR, default=default)


def _named_tracer_index(state: State, attr_name: str, *, default: int | None = None) -> int | None:
    value = (state.attrs or {}).get(attr_name, default)
    if value is None:
        return None
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    q = state.tracers()
    if q is None or index < 0 or index >= int(q.shape[-1]):
        return None
    return index


def mixing_ratio_from_specific_humidity(q_v: torch.Tensor) -> torch.Tensor:
    """Convert moist-air specific humidity to dry-air mixing ratio."""

    return q_v / (1.0 - q_v).clamp_min(torch.finfo(q_v.dtype).tiny)


def specific_humidity_from_mixing_ratio(r_v: torch.Tensor) -> torch.Tensor:
    """Convert dry-air vapor mixing ratio to moist-air specific humidity."""

    return r_v / (1.0 + r_v).clamp_min(torch.finfo(r_v.dtype).tiny)


def vapor_mixing_ratio(state: State, *, index: int | None = None) -> torch.Tensor | None:
    """Return named vapor as dry-air mixing ratio."""

    q = state.tracers()
    if q is None:
        return None
    resolved = vapor_tracer_index(state, default=index)
    if resolved is None:
        return None
    vapor = q[..., int(resolved)]
    quantity = str((state.attrs or {}).get(VAPOR_QUANTITY_ATTR, VAPOR_QUANTITY_DRY_MIXING_RATIO))
    if quantity == VAPOR_QUANTITY_SPECIFIC_HUMIDITY:
        return mixing_ratio_from_specific_humidity(vapor)
    return vapor


def cloud_liquid_mixing_ratio(state: State, *, index: int | None = None) -> torch.Tensor | None:
    """Return named cloud liquid as dry-air mixing ratio."""

    q = state.tracers()
    if q is None:
        return None
    resolved = cloud_liquid_tracer_index(state, default=index)
    if resolved is None:
        return None
    quantity = str((state.attrs or {}).get(CLOUD_LIQUID_QUANTITY_ATTR, CLOUD_LIQUID_QUANTITY_DRY_MIXING_RATIO))
    if quantity != CLOUD_LIQUID_QUANTITY_DRY_MIXING_RATIO:
        raise ValueError(
            "cloud_liquid_quantity currently supports only 'dry_air_mixing_ratio'; "
            f"got {quantity!r}"
        )
    return q[..., int(resolved)]


def virtual_temperature_factor_from_mixing_ratio(r_v: torch.Tensor) -> torch.Tensor:
    """Return T_v / T for dry-air water-vapor mixing ratio."""

    denom = (1.0 + r_v).clamp_min(torch.finfo(r_v.dtype).tiny)
    return (1.0 + r_v / float(EPSILON_WATER_DRY_AIR)) / denom


def saturation_vapor_pressure_liquid(temperature: torch.Tensor) -> torch.Tensor:
    """Bolton-style saturation vapor pressure over liquid water in Pa."""

    t = temperature.clamp(150.0, 350.0)
    t_c = t - 273.15
    return 611.2 * torch.exp(17.67 * t_c / (t_c + 243.5).clamp_min(1.0))


def saturation_mixing_ratio(temperature: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
    """Saturation dry-air mixing ratio from dry temperature and pressure."""

    e_s = saturation_vapor_pressure_liquid(temperature)
    e_s = torch.minimum(e_s, 0.99 * pressure)
    return float(EPSILON_WATER_DRY_AIR) * e_s / (pressure - e_s).clamp_min(
        torch.finfo(temperature.dtype).tiny
    )


def relative_humidity_from_mixing_ratio(
    r_v: torch.Tensor,
    temperature: torch.Tensor,
    pressure: torch.Tensor,
) -> torch.Tensor:
    """Return relative humidity against liquid-water saturation."""

    r_sat = saturation_mixing_ratio(temperature, pressure)
    return r_v / r_sat.clamp_min(torch.finfo(r_v.dtype).tiny)


__all__ = [
    "CLOUD_LIQUID_QUANTITY_ATTR",
    "CLOUD_LIQUID_QUANTITY_DRY_MIXING_RATIO",
    "CLOUD_LIQUID_TRACER_INDEX_ATTR",
    "EPSILON_WATER_DRY_AIR",
    "VAPOR_QUANTITY_ATTR",
    "VAPOR_QUANTITY_DRY_MIXING_RATIO",
    "VAPOR_QUANTITY_SPECIFIC_HUMIDITY",
    "VAPOR_TRACER_INDEX_ATTR",
    "cloud_liquid_mixing_ratio",
    "cloud_liquid_tracer_index",
    "mixing_ratio_from_specific_humidity",
    "relative_humidity_from_mixing_ratio",
    "saturation_mixing_ratio",
    "saturation_vapor_pressure_liquid",
    "specific_humidity_from_mixing_ratio",
    "vapor_mixing_ratio",
    "vapor_tracer_index",
    "virtual_temperature_factor_from_mixing_ratio",
]
