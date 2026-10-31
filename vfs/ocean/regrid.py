"""Explicit ocean output boundary; temperatures use the ocean's degC convention."""
from typing import Any, Iterable
from vfs.field import Field
from vfs.geometry.remap import _cell_field, _squeeze_last_singleton, _local_east_north_from_xyz

def _normalise_field_request(fields: Iterable[str] | str) -> set[str]:
    if fields == "default":
        return {"eta", "h", "temperature", "salinity", "wind"}
    if fields == "all":
        return {"eta", "h", "temperature", "salinity", "density", "wind", "tracers"}
    if isinstance(fields, str):
        return {fields}
    return {str(field) for field in fields}


def state_to_latlon(remapper, state, *, fields: Iterable[str] | str = "default",
                    density_provider=None) -> dict[str, Any]:
    """Remap physical fields, using an explicit density provider when supplied.

    Prefer Ocean.to_latlon for model output: it supplies the selected dynamics'
    density. State-only calls retain the legacy linear EOS, explicitly labelled
    in output metadata; a state alone cannot identify the selected equations.
    """

    requested = _normalise_field_request(fields)
    base = state.h
    out: dict[str, Any] = {
        "lat_deg": remapper.grid.lat_deg.to(device=base.device, dtype=base.dtype),
        "lon_deg": remapper.grid.lon_deg.to(device=base.device, dtype=base.dtype),
        "attrs": {
            "regrid_method": remapper.method,
            "conservative": False,
            "differentiable_wrt_fields": True,
        },
    }

    if "h" in requested:
        out["h"] = remapper(state.h)
    if "eta" in requested:
        out["eta"] = _squeeze_last_singleton(
            remapper(_cell_field(state.eta(), vertical_location="surface", name="eta", units="m"))
        )
    if "temperature" in requested:
        out["temperature"] = remapper(
            _cell_field(state.temperature(), vertical_location="full", name="temperature", units="degC")
        )
    if "salinity" in requested:
        out["salinity"] = remapper(_cell_field(state.salinity(), vertical_location="full", name="salinity"))
    if "density" in requested:
        density = state.density() if density_provider is None else density_provider(state)
        out['attrs']['density_convention'] = (
            'legacy_state_linear_eos' if density_provider is None else 'explicit_provider'
        )
        out["density"] = remapper(_cell_field(density, vertical_location="full", name="density", units="kg m-3"))
    if "tracers" in requested:
        q = state.tracers()
        if q is not None:
            out["q_tracer"] = remapper(
                _cell_field(
                    q,
                    vertical_location="full",
                    name="q_tracer",
                    component_axes=("tracer",),
                )
            )
    if "wind" in requested:
        wind_vec = state.mesh.cell_wind_vector(state.u_e)
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
