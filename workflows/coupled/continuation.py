"""In-place continuation of a coupled run and validation of its native checkpoint grid."""
import csv
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import xarray as xr

from .grid_io import atomic_path


def lock_run(directory, stack):
    handle = stack.enter_context((directory / ".campaign.lock").open("a"))
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, PermissionError) as error:
        raise ValueError(f"a campaign is already writing to {directory}") from error


def latest_checkpoint(directory):
    candidates = []
    for path in (directory / "restarts").glob("restart_*.nc"):
        stamp = path.stem.removeprefix("restart_")
        if stamp.isdigit():
            candidates.append((int(stamp), path))
    last = directory / "last_good_restart.nc"
    if last.exists():
        with xr.open_dataset(last, decode_cf=False) as ds:
            candidates.append((int(ds.time.item()), last))
    if not candidates:
        raise ValueError(f"no coupled checkpoint in {directory}")
    seconds, path = max(candidates, key=lambda entry: entry[0])
    with xr.open_dataset(path, decode_cf=False) as ds:
        if ds.attrs.get("vfs_format") != "coupled_restart_v1" or ds.time.item() != seconds:
            raise ValueError(f"invalid coupled checkpoint or inconsistent clock in {path}")
    return path


def recover_inputs(args, manifest):
    """A command-line argument overrides the corresponding stored input independently."""
    paths = manifest.get("inputs", {})
    for name in ("source", "mesh", "geography", "radiation_data"):
        if name == "geography" and args.island:
            continue
        if getattr(args, name) is None and paths.get(name):
            setattr(args, name, Path(paths[name]))
    if not paths and args.mesh is None and manifest.get("mesh", {}).get("path"):
        args.mesh = Path(manifest["mesh"]["path"])
    if args.geography is None and paths.get("island"):
        args.island = True
    if not manifest.get("radiation_enabled", True) and args.radiation_data is None:
        args.smoke_no_radiation = True


def check_grid(path, current, clock):
    """Compare the discretization carried by the state, independently of source or input path."""
    with xr.open_dataset(path, decode_times=False, mask_and_scale=False) as saved:
        if saved.attrs.get("vfs_format") != "coupled_restart_v1":
            raise ValueError("expected a coupled NetCDF restart")
        for key in ("units", "calendar"):
            if saved.time.attrs.get(key) != clock[key]:
                raise ValueError(f"restart clock mismatch: {key}")
        if saved.attrs.get("sphere_radius_m") != current.attrs["sphere_radius_m"]:
            raise ValueError("restart mesh mismatch: sphere_radius_m")
        names = ("cell_xyz", "edge_xyz", "cell_area", "cells_on_edge", "vertices_on_edge",
                 "air_hybrid_a", "air_hybrid_b", "ocean_reference_thickness",
                 "ocean_nominal_thickness", "ocean_depth_m", "land_mask", "surface_height_m",
                 "land_cell_index", "soil_layer_thickness", "category")
        for name in names:
            if name not in saved and name not in current:
                continue
            if name not in saved or name not in current or saved[name].dims != current[name].dims:
                raise ValueError(f"restart grid mismatch: {name}")
            left, right = saved[name].values, current[name].values
            if left.shape != right.shape:
                raise ValueError(f"restart grid shape mismatch: {name}")
            if left.dtype.kind == right.dtype.kind == "f":
                tol = 16 * max(np.finfo(left.dtype).eps, np.finfo(right.dtype).eps)
                equal = np.allclose(left, right, atol=tol, rtol=tol)
            else:
                equal = np.array_equal(left, right)
            if not equal:
                raise ValueError(f"restart grid values differ: {name}")


def archive_previous_output(directory, seconds, fields):
    """Keep valid history through the checkpoint. Retain any superseded output for inspection."""
    root = directory / "history"
    root.mkdir(exist_ok=True)
    archive = Path(tempfile.mkdtemp(prefix=f"restart_{seconds:08d}_", dir=root))
    for name in ("configuration.json", "manifest.json", "status.json", "last_good_restart.nc"):
        path = directory / name
        if path.exists():
            # Campaign metadata and checkpoints publish by atomic replacement,
            # so a hard link keeps the old version without copying large restarts.
            try:
                os.link(path, archive / name)
            except OSError:
                shutil.copy2(path, archive / name)
    trace = directory / "energy_stages.jsonl"
    if trace.exists():
        trace.rename(archive / trace.name)
    for folder, prefix in (("snapshots", "state_"), ("averages", "surface_")):
        for path in (directory / folder).glob(f"{prefix}*.nc"):
            stamp = path.stem.removeprefix(prefix)
            if stamp.isdigit() and int(stamp) > seconds:
                dest = archive / folder
                dest.mkdir(exist_ok=True)
                path.rename(dest / path.name)

    csv_path = directory / "diagnostics.csv"
    csv_last = None
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            header = reader.fieldnames or []
        retained = [row for row in rows if row.get("seconds") and int(row["seconds"]) <= seconds]
        fields = list(dict.fromkeys([*header, *fields]))
        if retained != rows or fields != header:
            shutil.copy2(csv_path, archive / csv_path.name)
            with atomic_path(csv_path) as tmp, tmp.open("w", newline="") as f:
                writer = csv.DictWriter(f, fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(retained)
        if retained:
            csv_last = int(retained[-1]["seconds"])
    nc_path = directory / "diagnostics.nc"
    nc_last = None
    if nc_path.exists():
        with xr.open_dataset(nc_path, decode_cf=False) as ds:
            times = ds.time.values
            mask = (times >= 0) & (times <= seconds)
            # An interrupted append can leave the time coordinate at its fill value.
            if not mask.all():
                trimmed = ds.isel(time=np.flatnonzero(mask)).load()
            else:
                trimmed = None
            if mask.any():
                nc_last = int(times[mask][-1])
        if trimmed is not None:
            shutil.copy2(nc_path, archive / nc_path.name)
            with atomic_path(nc_path) as tmp:
                trimmed.to_netcdf(tmp, engine="netcdf4", unlimited_dims=["time"])
    return fields, csv_last, nc_last


def check_state_layout(saved, fresh):
    """Require the prognostic and persistent arrays that the current components consume."""
    import torch

    def compare(old, new, path):
        if isinstance(new, dict):
            if not isinstance(old, dict):
                raise ValueError(f"incompatible restart state layout: {path}")
            for key, value in new.items():
                if key not in old:
                    raise ValueError(f"restart state is missing {path}.{key}")
                compare(old[key], value, f"{path}.{key}")
        elif isinstance(new, torch.Tensor):
            if not isinstance(old, torch.Tensor) or old.shape != new.shape or old.dtype != new.dtype:
                raise ValueError(f"incompatible restart state shape or dtype: {path}")
    for key in ("ocean", "ice", "snow_mass", "cold_content", "tke_h", "eddy_h", "land"):
        if key in fresh:
            compare(saved.get(key), fresh[key], key)
        elif key in saved:
            raise ValueError(f"unexpected restart state: {key}")
    for key in ("atmosphere", "cloud_fraction", "surface", "memory"):
        compare(saved["air"].get(key), fresh["air"][key], f"air.{key}")
