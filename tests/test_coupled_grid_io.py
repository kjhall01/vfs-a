"""An existing ordered mesh and its topography survive lossless conversion together."""
import hashlib
import json

import netCDF4
import numpy as np
import pytest
import xarray as xr

from workflows.coupled.grid_io import read_geography, read_mesh, write_geography, write_mesh


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_paired_earth_input_round_trip(tmp_path, dtype):
    xyz = np.random.default_rng(71).normal(size=(32, 3))
    xyz = np.asfortranarray((xyz / np.linalg.norm(xyz, axis=1, keepdims=True)).astype(dtype))
    np.save(tmp_path / "mesh.npy", xyz)
    original, identity = read_mesh(tmp_path / "mesh.npy")
    assert identity == hashlib.sha256((tmp_path / "mesh.npy").read_bytes()).hexdigest()
    write_mesh(tmp_path / "mesh.nc", original, mesh_sha256=identity)
    restored, digest = read_mesh(tmp_path / "mesh.nc")
    np.testing.assert_array_equal(restored, xyz)
    assert restored.dtype == xyz.dtype and identity == digest

    land = np.arange(len(xyz)) % 3 == 0
    raw = np.where(land, 0., np.arange(32) * 10.)
    attrs = {"min_wet_depth_m": 40., "mesh_sha256": identity, "source_urls": ["a", "b"]}
    fields = dict(land=land, land_fraction=land.astype(float), raw_depth_m=raw,
                  depth_m=np.where(land, 0., np.maximum(raw, 40.)), height_m=100. * land)
    np.savez_compressed(tmp_path / "geo.npz", **fields, metadata=json.dumps(attrs))
    old = read_geography(tmp_path / "geo.npz", mesh_sha256=identity)
    out = tmp_path / "geo.nc"
    write_geography(out, old, xyz, identity)
    result = read_geography(out, mesh_sha256=digest, xyz=restored)
    for key, value in fields.items():
        np.testing.assert_array_equal(result[key], value)
        assert result[key].dtype == value.dtype
    assert json.loads(result["metadata"]) == attrs
    with xr.open_dataset(out) as ds:
        assert ds.latitude.dims == ("cell",) and ds.depth_m.attrs["positive"] == "down"
    with pytest.raises(ValueError, match="mesh_sha256"):
        read_geography(out, mesh_sha256="different")
    with pytest.raises(ValueError, match="cell order"):
        read_geography(out, xyz=xyz[::-1])
    with netCDF4.Dataset(tmp_path / "mesh.nc", "a") as ds:
        assert ds.data_model == "NETCDF4"
        ds["cell_xyz"][:] = xyz[::-1]
    with pytest.raises(ValueError, match="generator_sha256"):
        read_mesh(tmp_path / "mesh.nc")


def test_operator_cache_can_supply_campaign_generators(tmp_path):
    from vfs.geometry.mesh.grid import VoronoiMesh

    mesh = VoronoiMesh(32, lloyd_iters=0, verbose=False)
    path = tmp_path / "cache.nc"
    mesh.save(path)
    xyz, identity = read_mesh(path)
    np.testing.assert_array_equal(xyz, mesh.cell_xyz.numpy())
    write_mesh(tmp_path / "generators.nc", xyz)
    assert read_mesh(tmp_path / "generators.nc")[1] == identity
