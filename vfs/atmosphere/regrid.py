"""Dry atmospheric state-to-physical-field remapping (no cached diagnostics)."""
from typing import Any, Iterable
from vfs.field import Field
from vfs.geometry.remap import _cell_field, _squeeze_last_singleton, _local_east_north_from_xyz

def _normalise_field_request(fields: Iterable[str] | str) -> set[str]:
    if fields == "default":
        return {"ps", "theta", "temperature", "pressure", "wind", "tracers"}
    if fields == "all":
        return {"ps", "theta", "temperature", "pressure", "layer_dp", "geopotential", "wind", "tracers"}
    if isinstance(fields, str):
        return {fields}
    return {str(field) for field in fields}


def state_to_latlon(remapper, state, *, fields: Iterable[str] | str = "default") -> dict[str, Any]:
    """Return common state fields on the lat/lon grid."""

    requested = _normalise_field_request(fields)
    base = state.ps
    out: dict[str, Any] = {
        "lat_deg": remapper.grid.lat_deg.to(device=base.device, dtype=base.dtype),
        "lon_deg": remapper.grid.lon_deg.to(device=base.device, dtype=base.dtype),
        "attrs": {
            "regrid_method": remapper.method,
            "conservative": False,
            "differentiable_wrt_fields": True,
        },
    }

    if "ps" in requested:
        out["ps"] = _squeeze_last_singleton(remapper(state.ps))
    if "layer_dp" in requested or "dp" in requested:
        out["layer_dp"] = remapper(_cell_field(state.layer_dp(), vertical_location="full", name="layer_dp", units="Pa"))
    if "theta" in requested:
        out["theta_interface"] = remapper(
            _cell_field(state.theta(), vertical_location="interface", name="theta_interface", units="K")
        )
    if "temperature" in requested:
        out["temperature_full"] = remapper(
            _cell_field(state.temperature(), vertical_location="full", name="temperature_full", units="K")
        )
    if "pressure" in requested:
        out["pressure_full"] = remapper(
            _cell_field(state.full_pressure(), vertical_location="full", name="pressure_full", units="Pa")
        )
        out["pressure_interface"] = remapper(
            _cell_field(state.interface_pressure(), vertical_location="interface", name="pressure_interface", units="Pa")
        )
    if "geopotential" in requested:
        out["geopotential_full"] = remapper(
            _cell_field(state.geopotential(), vertical_location="full", name="geopotential_full", units="m2 s-2")
        )
    if "tracers" in requested:
        q = state.tracers()
        if q is not None:
            out["q_tracer"] = remapper(
                _cell_field(q, vertical_location="full", name="q_tracer", component_axes=("tracer",))
            )
    if "wind" in requested:
        wind_vec = state.cell_wind_vector()
        wind_ll = remapper(
            _cell_field(
                wind_vec,
                vertical_location="full",
                name="wind_vector",
                units="m s-1",
                component_axes=("cartesian",),
            )
        )
        east, north = _local_east_north_from_xyz(
            remapper.target_xyz.to(device=wind_ll.values.device, dtype=wind_ll.values.dtype)
        )
        east = east.reshape(remapper.nlat, remapper.nlon, 1, 3)
        north = north.reshape(remapper.nlat, remapper.nlon, 1, 3)
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
