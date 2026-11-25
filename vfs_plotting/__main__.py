"""Plot a native field from a VFS NetCDF file."""
import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import xarray as xr

from . import plot_field


def _wet_mask(ds, mesh, variable, index):
    grids = [ds]
    if mesh is not ds and isinstance(mesh, xr.Dataset):
        grids.append(mesh)

    def find(*names):
        for grid in grids:
            for name in names:
                if name in grid:
                    var = grid[name]
                    return var.isel({d: pos for d, pos in index.items() if d in var.dims})
        return None

    land = find("land_mask", "land")
    if land is None:
        raise ValueError("--wet-only needs a land mask")
    wet = land == 0
    if "ocean_level" in variable.dims:
        count = find("ocean_active_layer_count")
        if count is not None:
            k = range(variable.sizes["ocean_level"])[index.get("ocean_level", 0)]
            return wet & (count > k)
        thickness = find("ocean_thickness", "ocean_h")
        if thickness is None:
            raise ValueError("--wet-only needs ocean layer counts or thickness")
        return wet & (thickness > 0)
    return wet


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="campaign snapshot, restart, or geography NetCDF")
    parser.add_argument("variable", help="field or coordinate name")
    parser.add_argument("--mesh", type=Path, help="separate VFS mesh NetCDF or ordered cell generators (.npy)")
    parser.add_argument("--isel", action="append", default=[], metavar="DIM=INDEX",
                        help="select a time, layer or component by zero-based index; repeat as needed")
    parser.add_argument("--projection", choices=("robinson", "3d"), default="robinson")
    parser.add_argument("--central-longitude", type=float, default=0.)
    parser.add_argument("--elevation", type=float, default=20., help="3D camera latitude in degrees")
    parser.add_argument("--wet-only", action="store_true", help="mask land and inactive layers for a cell field")
    parser.add_argument("--title", help="override the field name above the plot")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument("--output", type=Path, help="save PNG, PDF, SVG, etc.; omit to show interactively")
    args = parser.parse_args(argv)
    if args.output:
        import matplotlib
        matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    with ExitStack() as stack:
        ds = stack.enter_context(xr.open_dataset(args.input))
        mesh = ds
        if args.mesh and args.mesh.suffix == ".npy":
            mesh = np.load(args.mesh, allow_pickle=False)
        elif args.mesh:
            mesh = stack.enter_context(xr.open_dataset(args.mesh))
            identity = ds.attrs.get("mesh_sha256")
            if identity and mesh.attrs.get("mesh_sha256", identity) != identity:
                parser.error("field and mesh files have different mesh_sha256 values")
        try:
            field = ds[args.variable]
            index = {}
            for spec in args.isel:
                dim, pos = spec.split("=", 1)
                index[dim] = int(pos)
            field = field.isel(index)
            if args.wet_only:
                if "cell" not in field.dims:
                    raise ValueError("--wet-only requires a cell field")
                field = field.where(_wet_mask(ds, mesh, ds[args.variable], index))
            artist = plot_field(field, mesh, projection=args.projection, central_longitude=args.central_longitude,
                                elevation=args.elevation, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax, title=args.title)
        except (KeyError, ValueError, IndexError) as error:
            parser.error(str(error))
        if args.output:
            artist.figure.savefig(args.output, dpi=160, bbox_inches="tight")
            plt.close(artist.figure)
        else:
            plt.show()


if __name__ == "__main__":
    main()
