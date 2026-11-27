"""Cell-based surface fields from accepted coupling intervals, on the native grid."""
import numpy as np
import torch
import xarray as xr

from .grid_io import write_netcdf
from .state_io import array, attach_mesh_attributes

FLUX_ATTRS = {
    "precipitation": {"long_name": "Total surface precipitation (rain and snow)",
                      "standard_name": "precipitation_flux", "units": "kg m-2 s-1", "positive": "down"},
    "evaporation": {"long_name": "Net upward surface water vapor flux (includes sublimation and dew)",
                    "standard_name": "water_evaporation_flux", "units": "kg m-2 s-1", "positive": "up"},
}
SST_ATTRS = {"long_name": "Sea surface Conservative Temperature (top ocean layer, including under ice)",
             "units": "degC"}


def sea_surface_temperature(sea):
    ocean = sea.coupled.ocean.ocean
    h = ocean.h[:, 0]
    return ocean.temp_h[:, 0] / h.clamp_min(torch.finfo(h.dtype).tiny)


def snapshot_fields(ds, sea, last, clock):
    sst = array(sea_surface_temperature(sea))
    ds["sst"] = (("time", "cell"), np.where(ds.land_mask.values, np.nan, sst)[None], SST_ATTRS)
    for name, attrs in FLUX_ATTRS.items():
        rate = array(last[name]) / last["dt"] if last is not None else np.full_like(sst, np.nan)
        ds[name] = (("time", "cell"), rate[None], {**attrs, "cell_methods": "time: mean",
                    "comment": "Mean over the preceding coupling interval; see surface_flux_time_bounds."})
    end = int(ds.time.values[0])
    bounds = [end - last["dt"], end] if last is not None else [np.nan, np.nan]
    ds["surface_flux_time_bounds"] = (("time", "bounds"), [bounds], clock)


class SurfaceAverages:
    """Integrate accepted water transfers and linearly interpolated endpoint SST."""

    def __init__(self, period, start, saved=None):
        self.period = period
        self.start = start
        self.sums = {}
        self.last = None
        if saved:
            self.last = saved["last"]
            self.start = saved["start"]
            self.sums = saved["sums"]

    def pack(self):
        return dict(period=self.period, start=self.start, sums=self.sums, last=self.last)

    def accumulate(self, start, end, amounts, sst0, sst1, emit):
        dt = end - start
        pos = start
        window_start, sums = self.start, self.sums.copy()
        while self.period and pos < end:
            boundary = (pos // self.period + 1) * self.period
            stop = min(end, boundary)
            span = stop - pos
            midpoint = ((pos - start) + (stop - start)) / (2 * dt)
            increments = {name: value * (span / dt) for name, value in amounts.items()}
            increments["sst"] = (sst0 + (sst1 - sst0) * midpoint) * span
            for name, value in increments.items():
                sums[name] = sums.get(name, 0) + value
            if stop == boundary:
                emit(window_start, stop, sums)
                window_start, sums = stop, {}
            pos = stop
        self.start, self.sums = window_start, sums
        self.last = dict(dt=dt, **amounts)

    def flush(self, end, emit):
        if self.sums:
            emit(self.start, end, self.sums)
        self.start = end
        self.sums = {}


def write_surface_mean(directory, grid, clock, start, end, sums):
    # Surface files need native horizontal topology but no subsurface columns.
    horizontal = {"cell", "edge", "vertex", "xyz", "pair", "max_vertices"}
    coords = {name: field for name, field in grid.coords.items() if set(field.dims) <= horizontal}
    ds = xr.Dataset(coords=coords, attrs={**grid.attrs, "vfs_format": "coupled_surface_mean_v1"})
    ds = ds.assign_coords(time=("time", [end], {**clock, "bounds": "time_bounds"}))
    ds["time_bounds"] = (("time", "bounds"), [[start, end]], clock)
    for name, attrs in {**FLUX_ATTRS, "sst": SST_ATTRS}.items():
        mean = array(sums[name]) / (end - start)
        if name == "sst":
            mean = np.where(ds.land_mask.values, np.nan, mean)
        ds[name] = (("time", "cell"), mean[None], {**attrs, "cell_methods": "time: mean"})
    attach_mesh_attributes(ds)
    directory.mkdir(exist_ok=True)
    write_netcdf(directory / f"surface_{end:08d}.nc", ds)
