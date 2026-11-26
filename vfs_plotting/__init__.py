"""Plot native VFS fields from a live mesh or a campaign NetCDF dataset."""
import numpy as np
import xarray as xr

from ._geometry import Geometry, as_array, robinson_polygons
from .climate import open_state, summarize
from .climate_plots import plot_section, plot_sections, plot_surface, plot_zonal_mean

__all__ = ["open_state", "summarize", "plot_field", "plot_section", "plot_sections", "plot_surface", "plot_zonal_mean"]


def _samples(field, mesh, grid):
    sizes = {"cell": len(grid.cells)}
    if grid.edge_nodes is not None:
        sizes["edge"] = len(grid.edge_nodes)
    if grid.ordered_vertices:
        sizes["vertex"] = len(grid.vertices)

    if isinstance(field, xr.DataArray):
        dims = [d for d in field.dims if d in ("cell", "edge", "vertex", "land_cell")]
        if len(dims) != 1:
            raise ValueError("expected a field with one named horizontal dimension")
        dim = dims[0]
        extra = [d for d in field.dims if d != dim and field.sizes[d] != 1]
        if extra:
            raise ValueError(f"select one index for {', '.join(extra)} with .isel() or --isel")
        field = field.squeeze([d for d in field.dims if d != dim], drop=True)
        if dim == "land_cell":
            if "land_cell_index" in field.coords:
                subset = field.land_cell_index.values
            elif isinstance(mesh, xr.Dataset):
                subset = mesh.land_cell_index.values
            else:
                raise ValueError("compact land fields need the land_cell_index coordinate")
            if subset.shape != field.shape or np.any((subset < 0) | (subset >= sizes["cell"])):
                raise ValueError("invalid land_cell_index")
            values = np.ma.masked_all(sizes["cell"], dtype=field.dtype)
            values[subset] = field.values
            return "cell", values
        if dim not in sizes:
            raise ValueError(f"{dim} fields need native connectivity with that location")
        if dim in field.indexes:
            # Positionally matching a shuffled DataArray silently swaps values.
            if isinstance(mesh, xr.Dataset) and dim in mesh.indexes:
                index = mesh.indexes[dim]
            else:
                import pandas as pd
                index = pd.RangeIndex(sizes[dim])
            if not field.indexes[dim].is_unique or not field.indexes[dim].isin(index).all():
                raise ValueError(f"{dim} labels do not match the mesh")
            field = field.reindex({dim: index})
        location, values = dim, field.values
    elif hasattr(field, "location"):  # VFS Field
        location, values = field.location, as_array(field.values).squeeze()
    else:
        values = as_array(field).squeeze()
        matches = [name for name, count in sizes.items() if values.shape == (count,)]
        if len(matches) != 1:
            raise ValueError("supply one scalar per mesh location; slice levels, times and components first")
        location = matches[0]
    if location not in sizes:
        raise ValueError(f"{location} fields need native connectivity with that location")
    if values.shape != (sizes[location],):
        raise ValueError(f"expected one scalar per {location}; select a single level, time and component")
    return location, values


def plot_field(field, mesh, *, projection="robinson", central_longitude=0., elevation=20.,
               cmap="viridis", vmin=None, vmax=None, title=None, ax=None, colorbar=True):
    """Color native mesh polygons by scalar value. Return a Matplotlib collection.

    Pass a one-dimensional xarray DataArray, NumPy array, tensor or VFS Field.
    Supply the matching snapshot Dataset, live VoronoiMesh, mesh cache Dataset,
    or (for cell data) ordered Cartesian generators. Slice extra dimensions first.
    Masked and nonfinite samples appear gray. For an existing Robinson axis,
    its central longitude determines the view. A 3D axis can be rotated live.
    """
    from matplotlib import pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    grid = Geometry.read(mesh)
    location, values = _samples(field, mesh, grid)
    if values.dtype.kind not in "biuf":
        raise ValueError("supply a real-valued scalar field")
    values = np.ma.masked_invalid(values)
    rings, owners = grid.rings(location)
    colors = plt.get_cmap(cmap).with_extremes(bad="#d6d9dd")
    style = dict(cmap=colors, norm=Normalize(vmin, vmax), edgecolors="none", antialiaseds=False)
    if projection == "robinson":
        import cartopy.crs as ccrs
        if ax is None:
            fig = plt.figure(figsize=(10, 5.4), layout="constrained")
            ax = fig.add_subplot(projection=ccrs.Robinson(central_longitude=central_longitude))
        if not isinstance(getattr(ax, "projection", None), ccrs.Robinson):
            raise ValueError("Robinson mode requires a Cartopy Robinson axis")
        polygons, owners = robinson_polygons(rings, owners, ax.projection)
        artist = PolyCollection(polygons, **style)
        ax.add_collection(artist, autolim=False)
        ax.set_global()
    elif projection == "3d":
        if ax is None:
            fig = plt.figure(figsize=(6.5, 6), layout="constrained")
            ax = fig.add_subplot(projection="3d")
        if ax.name != "3d":
            raise ValueError("3d mode requires a Matplotlib 3D axis")
        artist = Poly3DCollection(rings, **style)
        ax.add_collection3d(artist)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        ax.set_box_aspect((1, 1, 1), zoom=1.24)
        ax.set_proj_type("ortho")
        ax.view_init(elev=elevation, azim=central_longitude)
        ax.set_axis_off()
    else:
        raise ValueError("projection must be 'robinson' or '3d'")
    artist.set_array(values[owners])
    artist.autoscale_None()
    attrs = getattr(field, "attrs", {})
    name = attrs.get("long_name") or getattr(field, "name", None) or ""
    if title is not None or name:
        ax.set_title(name if title is None else title)
    if colorbar:
        units = attrs.get("units") or getattr(field, "units", "") or ""
        ax.figure.colorbar(artist, ax=ax, orientation="horizontal", shrink=.74, pad=.04, label=units)
    return artist
