"""Interactive climate plots. Supply ax to draw in your own Matplotlib layout."""
import numpy as np


def plot_section(summary, contours="zonal_wind", *, ax=None, levels=None, pressure_scale="log",
                 cmap="RdYlBu_r", vmin=None, vmax=None, colorbar=True):
    """Shade zonal-mean temperature (K) and overlay labeled circulation contours.

    contours chooses 'zonal_wind' (m/s) or 'streamfunction' (10⁹ kg/s).
    levels sets contour values in those units. Negative contours are dashed.
    The default streamfunction levels omit zero where weak upper-level flow
    can change sign repeatedly. Include 0 in levels to show that boundary.
    Pressure increases downward. Returns the axes, e.g. ax.set_ylim(1000, 100).
    """
    from matplotlib import patheffects, pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    if contours == "zonal_wind":
        title, unit = "Zonal wind", "m/s"
    elif contours == "streamfunction":
        title, unit = "Mass streamfunction", "10⁹ kg/s"
    else:
        raise ValueError("contours must be 'zonal_wind' or 'streamfunction'")
    if pressure_scale not in ("log", "linear"):
        raise ValueError("pressure_scale must be 'log' or 'linear'")
    if ax is None:
        _, ax = plt.subplots(figsize=(7.6, 5), layout="constrained")

    lat, pressure = summary.lat.values, summary.pressure_hpa.values
    temp = summary.temperature.transpose("pressure_hpa", "lat")
    values = summary[contours].transpose("pressure_hpa", "lat").values
    shading = ax.pcolormesh(lat, pressure, temp, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    finite = values[np.isfinite(values)]
    if finite.size and finite.min() < finite.max():
        if levels is None:
            bound = np.max(np.abs(finite))
            positive = MaxNLocator(nbins=9, steps=[1, 2, 2.5, 5, 10]).tick_values(0, bound)[1:]
            levels = np.r_[-positive[::-1], positive]
            if contours == "zonal_wind":
                levels = np.sort(np.append(levels, 0.))
        levels = np.asarray(levels, dtype=float)
        if levels.ndim != 1 or not np.isfinite(levels).all() or np.any(np.diff(levels) <= 0):
            raise ValueError("levels must be a strictly increasing sequence of finite contour values")
        levels = levels[(levels > finite.min()) & (levels < finite.max())]
        if levels.size:
            lines = ax.contour(lat, pressure, values, levels=levels, colors="#202329",
                               linestyles=["dashed" if v < 0 else "solid" for v in levels],
                               linewidths=np.where(levels == 0, 1.4, .9))
            labels = ax.clabel(lines, fmt="%g", fontsize=8, inline_spacing=3)
            for label in labels:
                label.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground="white")])
    ax.set(yscale=pressure_scale, ylim=(pressure.max(), pressure.min()),
           xlabel="Latitude (°N)", ylabel="Pressure (hPa)", xticks=[-60, -30, 0, 30, 60],
           title=f"{title} contours ({unit})")
    if pressure_scale == "log":
        ticks = np.array([1, 3, 10, 30, 100, 300, 1000])
        ax.set_yticks(ticks[(ticks >= pressure.min()) & (ticks <= pressure.max())])
        ax.yaxis.set_major_formatter(ScalarFormatter())
    if colorbar:
        ax.figure.colorbar(shading, ax=ax, pad=.02, label="Temperature (K)")
    return ax


def plot_sections(summary, *, pressure_scale="log"):
    """Draw both circulation overlays using the same temperature scale. Return the figure."""
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), sharey=True, layout="constrained")
    for ax, field in zip(axes, ("zonal_wind", "streamfunction")):
        plot_section(summary, field, ax=ax, pressure_scale=pressure_scale, colorbar=False)
    axes[1].set_ylabel("")
    fig.colorbar(axes[0].collections[0], ax=axes, orientation="horizontal", shrink=.6,
                 label="Zonal-mean air temperature (K)", pad=.04)
    fig.suptitle("Zonal-mean atmosphere  ·  " + summary.attrs["period"])
    return fig


_SURFACE = {
    "surface_air": ("Lowest-interface air temperature", "°C", "coolwarm"),
    "sst": ("Sea surface temperature", "°C", "coolwarm"),
    "column_vapor": ("Column water vapor", "kg m⁻²", "YlGnBu"),
    "precipitation": ("Precipitation", "mm/day", "YlGnBu"),
}


def plot_zonal_mean(summary, field, *, ax=None):
    """Plot a surface field's area-weighted mean by latitude. Return the axes."""
    from matplotlib import pyplot as plt

    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    title, unit, _ = _SURFACE[field]
    ax.plot(summary.lat, summary[field + "_zonal"], label=title)
    ax.set(xlabel="Latitude (°N)", ylabel=unit, title=title, xlim=(-90, 90))
    return ax


def plot_surface(summary):
    """Pair each available surface map with its latitude profile. Return the figure."""
    import cartopy.crs as ccrs
    from . import plot_field
    from matplotlib import pyplot as plt

    fields = [name for name in _SURFACE if name in summary]
    fig = plt.figure(figsize=(9, 2.9 * len(fields)), layout="constrained")
    layout = fig.add_gridspec(len(fields), 2, width_ratios=(2.2, 1))
    for i, name in enumerate(fields):
        title, unit, cmap = _SURFACE[name]
        ax = fig.add_subplot(layout[i, 0], projection=ccrs.Robinson())
        plot_field(summary[name].assign_attrs(units=unit), summary, ax=ax, title=title, cmap=cmap)
        profile = plot_zonal_mean(summary, name, ax=fig.add_subplot(layout[i, 1]))
        profile.set_title("Zonal mean" if i == 0 else "")
        profile.grid(alpha=.2)
    fig.suptitle(summary.attrs["period"])
    return fig
