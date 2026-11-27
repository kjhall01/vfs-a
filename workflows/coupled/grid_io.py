"""File boundary for the coupled campaign's ordered generators and surface geography."""
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import xarray as xr


@contextmanager
def atomic_path(path):
    """Publish only a closed, complete file, on the same filesystem."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    tmp.touch(exist_ok=False)
    try:
        yield tmp
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def write_netcdf(path, ds):
    encoding = {k: {"zlib": True, "complevel": 1, "_FillValue": v.encoding.get("_FillValue")}
                for k, v in ds.variables.items() if v.ndim and v.dtype.kind in "fiu"}
    with atomic_path(path) as tmp:
        ds.to_netcdf(tmp, engine="netcdf4", format="NETCDF4", encoding=encoding)


def horizontal_coords(xyz):
    # Keep the original generator values and ordering, including floating-point precision.
    return dict(cell=np.arange(len(xyz)), xyz=["x", "y", "z"],
                cell_xyz=(("cell", "xyz"), xyz, {"long_name": "unit sphere Cartesian cell centers", "units": "1"}),
                latitude=("cell", np.rad2deg(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))),
                          {"standard_name": "latitude", "units": "degrees_north"}),
                longitude=("cell", np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0])),
                           {"standard_name": "longitude", "units": "degrees_east"}))


def _check_xyz(xyz):
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 16 or xyz.dtype.kind != "f"
            or not np.isfinite(xyz).all() or not np.allclose(np.linalg.norm(xyz, axis=1), 1., rtol=0, atol=1e-6)
            or len(np.unique(xyz, axis=0)) != len(xyz)):
        raise ValueError("mesh requires N>=16 distinct finite floating-point unit 3-vectors")


def _generator_hash(xyz):
    encoded = io.BytesIO()
    np.save(encoded, np.ascontiguousarray(xyz), allow_pickle=False)
    return hashlib.sha256(encoded.getbuffer()).hexdigest()


def read_mesh(path):
    """Return ordered generators and the mesh identity used by paired inputs/restarts."""
    path = Path(path)
    if path.suffix == ".npy":
        raw = path.read_bytes()
        xyz = np.load(io.BytesIO(raw), allow_pickle=False)
        _check_xyz(xyz)
        return xyz, hashlib.sha256(raw).hexdigest()
    if path.suffix != ".nc":
        raise ValueError("expected a generator mesh ending in .nc or .npy")
    with xr.open_dataset(path, engine="netcdf4", decode_cf=False) as ds:
        if ds.attrs.get("vfs_format") == "coupled_restart_v1":
            xyz = ds.cell_xyz.values
            _check_xyz(xyz)
            return xyz, ds.attrs.get("mesh_sha256") or _generator_hash(xyz)
        # A VFS operator cache also contains the ordered generator positions.
        # Campaign initialization always rebuilds its configured operators.
        if ds.attrs.get("mesh_schema_version") == 2 and ds.cell_xyz.dims == ("cell", "three"):
            xyz = ds.cell_xyz.values
            _check_xyz(xyz)
            return xyz, _generator_hash(xyz)
        if ds.attrs.get("vfs_format") != "generator_mesh_v1" or ds.cell_xyz.dims != ("cell", "xyz"):
            raise ValueError("expected a VFS generator mesh with cell_xyz(cell, xyz)")
        xyz = ds.cell_xyz.values
        _check_xyz(xyz)
        if _generator_hash(xyz) != ds.attrs["generator_sha256"]:
            raise ValueError("generator coordinates disagree with stored generator_sha256")
        return xyz, ds.attrs["mesh_sha256"]


def write_mesh(path, xyz, *, mesh_sha256=None):
    _check_xyz(xyz)
    if xyz.dtype not in (np.float32, np.float64):
        raise ValueError("NetCDF generator files require float32 or float64 coordinates")
    digest = _generator_hash(xyz)
    ds = xr.Dataset(coords=horizontal_coords(xyz), attrs={
        "vfs_format": "generator_mesh_v1", "generator_sha256": digest,
        "mesh_sha256": mesh_sha256 or digest, "grid": "spherical Voronoi generators",
    })
    if Path(path).suffix != ".nc":
        raise ValueError("write_mesh expects a .nc destination")
    write_netcdf(path, ds)


GEO_ATTRS = {
    "land": {"long_name": "land mask", "flag_values": np.array([0, 1], dtype="i1"), "flag_meanings": "ocean land"},
    "height_m": {"long_name": "land surface height above sea level", "units": "m", "positive": "up"},
    "depth_m": {"long_name": "resting wet ocean depth", "units": "m", "positive": "down"},
    "raw_depth_m": {"long_name": "wet ocean depth before applying the floor", "units": "m", "positive": "down"},
    "land_fraction": {"long_name": "fractional land area", "units": "1"},
}


def read_geography(path, *, mesh_sha256=None, xyz=None):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            result = {k: data[k] for k in data.files}
    elif path.suffix == ".nc":
        with xr.open_dataset(path, engine="netcdf4", decode_cf=False) as ds:
            if ds.attrs.get("vfs_format") == "coupled_restart_v1":
                if xyz is not None and (xyz.shape != ds.cell_xyz.shape or
                        not np.allclose(xyz, ds.cell_xyz.values, rtol=0, atol=16 * np.finfo(xyz.dtype).eps)):
                    raise ValueError("restart geography and supplied mesh coordinates differ")
                return dict(land=ds.land_mask.values.astype(bool), height_m=ds.surface_height_m.values,
                            depth_m=ds.ocean_depth_m.values)
            if ds.attrs.get("vfs_format") != "surface_geography_v1":
                raise ValueError("expected a VFS surface geography file")
            if ds.cell_xyz.dims != ("cell", "xyz") or _generator_hash(ds.cell_xyz.values) != ds.attrs["generator_sha256"]:
                raise ValueError("geography generator coordinates disagree with their stored hash")
            if xyz is not None and not np.array_equal(ds.cell_xyz.values, xyz):
                raise ValueError("geography cell order or coordinates differ from the supplied mesh")
            result = {}
            for k in GEO_ATTRS:
                if k in ds:
                    if ds[k].dims != ("cell",):
                        raise ValueError(f"expected geography {k}(cell)")
                    result[k] = ds[k].values
            if not np.isin(result["land"], [0, 1]).all():
                raise ValueError("invalid land mask flags")
            result["land"] = result["land"].astype(bool)
            meta = {k: v.tolist() if isinstance(v, (np.ndarray, np.generic)) else v
                    for k, v in ds.attrs.items() if k not in ("vfs_format", "generator_sha256", "coordinates")}
            result["metadata"] = json.dumps(meta)
    else:
        raise ValueError("expected geography ending in .nc or .npz")
    meta = json.loads(str(result.get("metadata", "{}")))
    if mesh_sha256 is not None and meta.get("mesh_sha256", mesh_sha256) != mesh_sha256:
        raise ValueError("geography mesh_sha256 differs from the supplied mesh")
    return result


def write_geography(path, result, xyz, mesh_sha256):
    """Retain the source fields, depth floor and provenance from the geography builder."""
    path = Path(path)
    metadata = json.loads(str(result.get("metadata", "{}")))
    if metadata.get("mesh_sha256", mesh_sha256) != mesh_sha256:
        raise ValueError("geography mesh_sha256 differs from the supplied mesh")
    metadata["mesh_sha256"] = mesh_sha256
    land = np.asarray(result["land"])
    if land.dtype != np.bool_ or any(np.shape(result[k]) != (len(xyz),) for k in ("land", "height_m", "depth_m")):
        raise ValueError("expected a boolean land mask and mesh-ordered height and depth")
    if path.suffix == ".npz":
        with atomic_path(path) as tmp, tmp.open("wb") as stream:
            np.savez_compressed(stream, **{**result, "metadata": json.dumps(metadata)})
        return
    if path.suffix != ".nc":
        raise ValueError("expected geography ending in .nc or .npz")
    extra = result.keys() - GEO_ATTRS.keys() - {"metadata"}
    if extra:
        raise ValueError(f"unrecognized geography fields: {sorted(extra)}")
    fields = {k: ("cell", result[k] if k != "land" else land.astype("i1"), attrs)
              for k, attrs in GEO_ATTRS.items() if k in result}
    ds = xr.Dataset(fields, coords=horizontal_coords(xyz), attrs={**metadata, "vfs_format": "surface_geography_v1", "generator_sha256": _generator_hash(xyz)})
    write_netcdf(path, ds)
