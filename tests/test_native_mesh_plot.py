"""Rendering must preserve the association between each sample and its spherical support."""
import numpy as np
import pytest
import xarray as xr

pytest.importorskip("matplotlib").use("Agg")
from matplotlib import pyplot as plt
import torch

from vfs.field import Field
from vfs.geometry.mesh.grid import VoronoiMesh
from vfs_plotting import plot_field
from vfs_plotting.__main__ import main


@pytest.fixture(scope="module")
def mesh():
    return VoronoiMesh(64, lloyd_iters=0, verbose=False)


@pytest.fixture
def saved(mesh):
    v = mesh.vertex_xyz.numpy()
    return xr.Dataset(
        {"value": (("time", "cell"), np.arange(mesh.nCells)[None, :], {"units": "K"})},
        coords={
            "cell": np.arange(mesh.nCells),
            "cell_xyz": (("cell", "xyz"), mesh.cell_xyz.numpy()),
            "vertices_on_cell": (("cell", "corner"), np.where(mesh.verticesOnCell >= 0, mesh.verticesOnCell, np.nan)),
            "vertices_on_edge": (("edge", "pair"), mesh.verticesOnEdge.numpy()),
            "cells_on_edge": (("edge", "pair"), mesh.cellsOnEdge.numpy()),
            "vertex_longitude": ("vertex", np.rad2deg(np.arctan2(v[:, 1], v[:, 0]))),
            "vertex_latitude": ("vertex", np.rad2deg(np.arcsin(v[:, 2]))),
        },
    )


@pytest.mark.parametrize("center", [0, 135, 180])
def test_robinson_seam_and_poles_retain_owner(mesh, saved, center):
    ccrs = pytest.importorskip("cartopy.crs")
    art = plot_field(saved.value, saved, central_longitude=center, colorbar=False)
    paths, owners = art.get_paths(), art.get_array()
    assert set(owners) == set(range(mesh.nCells))
    assert len(paths) > mesh.nCells
    lon, lat = np.meshgrid(center + np.array([-179.98, -176., -90., 0., 90., 176., 179.98]),
                           [-89.98, -68., -20., 0., 20., 68., 89.98])
    lon, lat = lon.ravel(), lat.ravel()
    xy = art.axes.projection.transform_points(ccrs.PlateCarree(), lon, lat)[:, :2]
    l, p = np.deg2rad(lon), np.deg2rad(lat)
    query = np.column_stack((np.cos(l)*np.cos(p), np.sin(l)*np.cos(p), np.sin(p)))
    dots = query @ mesh.cell_xyz.numpy().T
    for pos, ranks in zip(xy, dots):
        # Exclude samples effectively on a shared edge at the drawing resolution.
        best, second = np.argsort(ranks)[-2:][::-1]
        if ranks[best] - ranks[second] > .002:
            assert [owners[i] for i, path in enumerate(paths) if path.contains_point(pos)] == [best]
    plt.close(art.figure)


def test_globe_preserves_staggering_and_masks(mesh, saved):
    for location, size, pieces in [("cell", mesh.nCells, 1), ("edge", mesh.nEdges, 2),
                                   ("vertex", mesh.nVertices, 1)]:
        wrapped = Field(torch.arange(size, dtype=torch.float64), location=location, units="Pa")
        art = plot_field(wrapped, mesh, projection="3d")
        np.testing.assert_array_equal(art.get_array(), np.repeat(np.arange(size), pieces))
        assert art.colorbar.ax.get_xlabel() == "Pa"
        art.figure.canvas.draw()
        plt.close(art.figure)
    subset = saved.value.isel(cell=[10, 0, 20])
    art = plot_field(subset, saved, projection="3d", colorbar=False)
    np.testing.assert_array_equal(np.flatnonzero(~art.get_array().mask), [0, 10, 20])
    np.testing.assert_array_equal(art.get_array().compressed(), [0, 10, 20])
    plt.close(art.figure)
    land = xr.DataArray([6., 4., np.nan], dims="land_cell", coords={"land_cell_index": ("land_cell", [9, 2, 5])})
    art = plot_field(land, saved, projection="3d", colorbar=False)
    np.testing.assert_array_equal(np.flatnonzero(~art.get_array().mask), [2, 9])
    np.testing.assert_array_equal(art.get_array().compressed(), [4., 6.])
    plt.close(art.figure)


def test_legacy_edge_order_and_generator_only_cells(mesh, saved):
    ds = saved[["cell_xyz", "cells_on_edge"]].expand_dims(time=[0])
    field = xr.DataArray(np.arange(mesh.nEdges), dims="edge")
    art = plot_field(field, ds, projection="3d", colorbar=False)
    np.testing.assert_array_equal(art.get_array(), np.repeat(field, 2))
    plt.close(art.figure)
    pts = mesh.cell_xyz.numpy()
    art = plot_field(pts[:, 2], pts, projection="3d", colorbar=False)
    np.testing.assert_array_equal(art.get_array(), pts[:, 2])
    plt.close(art.figure)
    with pytest.raises(ValueError, match="connectivity"):
        plot_field(field, pts)
    with pytest.raises(ValueError, match="ocean_level"):
        plot_field(saved.value.expand_dims(ocean_level=[0, 1]), saved)


def test_netcdf_cli_uses_selected_wet_layer(saved, tmp_path):
    saved["temperature"] = saved.value.expand_dims(ocean_level=[0, 1])
    saved = saved.assign_coords(land_mask=("cell", np.zeros(saved.sizes["cell"], dtype="i1")),
                                ocean_active_layer_count=("cell", np.where(saved.cell < 4, 1, 2)))
    path = tmp_path / "input.nc"
    saved.to_netcdf(path)
    out = tmp_path / "layer.png"
    main([str(path), "temperature", "--isel", "ocean_level=-1", "--wet-only", "--projection", "3d", "--output", str(out)])
    assert out.read_bytes().startswith(b"\x89PNG")


def test_cli_separate_mesh_and_field_file(saved, tmp_path):
    base = saved.drop_vars("value")
    base.attrs["mesh_sha256"] = "test-grid"
    base.to_netcdf(tmp_path / "mesh.nc")
    ds = xr.Dataset({"field": ("cell", np.arange(saved.sizes["cell"]))}, attrs={"mesh_sha256": "test-grid"})
    ds.to_netcdf(tmp_path / "field.nc")
    args = [str(tmp_path / "field.nc"), "field", "--mesh", str(tmp_path / "mesh.nc"),
            "--projection", "3d", "--output", str(tmp_path / "field.png")]
    main(args)
    assert (tmp_path / "field.png").exists()
    ds.attrs["mesh_sha256"] = "other-grid"
    ds.to_netcdf(tmp_path / "field.nc")
    with pytest.raises(SystemExit):
        main(args)
