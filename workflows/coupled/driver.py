"""Atomic atmosphere--ocean--ice exchange with explicit surface temperatures."""

import math
from workflows.exchange import Exchange
from workflows.pyam21.surface import prepare_surface


def exchange_step(
    adapter, atmosphere, receiver, surface, dt, *, ice_temperature=None, omega=None
):
    if (
        isinstance(dt, bool)
        or not isinstance(dt, (int, float))
        or not math.isfinite(dt)
        or dt <= 0
        or not float(dt).is_integer()
    ):
        raise ValueError("positive integer exchange interval required")
    dt = int(dt)
    if adapter.energy_mode != "conservative":
        raise ValueError("shared exchange requires conservative atmosphere")
    adapter.check(atmosphere)
    receiver.check_geometry(adapter, atmosphere, surface)
    tiles = receiver.tiles(surface, ice_temperature)
    start = prepare_surface(atmosphere, tiles)
    columns, geometry = adapter.mapping.columns(
        start.atmosphere, start.cloud_fraction, omega
    )
    out, surf, mem, rad = adapter.column_call(
        start, dt, columns, geometry, surface_tiles=tiles
    )
    exchange = Exchange.from_am2(adapter, start, dt, columns, out, tiles)
    candidate, exchange, impulses = receiver.prepare(surface, exchange, tiles)
    out["exchange"] = exchange
    air, report = adapter.apply_result(
        start, dt, columns, geometry, out, surf, mem, rad
    )
    water, surface_report = receiver.finish(surface, candidate, exchange, impulses)
    return air, water, exchange, report, surface_report
