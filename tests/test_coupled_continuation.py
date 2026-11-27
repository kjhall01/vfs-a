"""File-level restart and surface-output comparisons through the campaign CLI."""
import csv
import json
from dataclasses import asdict, replace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import xarray as xr
import yaml

pytest.importorskip("pyam21")
from test_coupled_campaign import c, config, source
from test_pyam21_workflow import same_tree
from workflows.coupled.surface_output import SurfaceAverages


def test_rates_and_linear_sst_across_unaligned_windows():
    averages = SurfaceAverages(6, 0)
    records = []
    def emit(start, end, sums):
        records.append((start, end, {k: float(v) / (end - start) for k, v in sums.items()}))
    for start, end, precip, evap in [(0, 4, 8., -4.), (4, 12, 48., 16.)]:
        averages.accumulate(start, end, {"precipitation": torch.tensor(precip), "evaporation": torch.tensor(evap)},
                            torch.tensor(float(start)), torch.tensor(float(end)), emit)
    assert [(a, b) for a, b, _ in records] == [(0, 6), (6, 12)]
    assert [r["precipitation"] for _, _, r in records] == pytest.approx([20/6, 6])
    assert [r["evaporation"] for _, _, r in records] == pytest.approx([0, 2])
    assert [r["sst"] for _, _, r in records] == pytest.approx([3, 9])


def test_restart_in_place_replays_partial_average_without_duplicate_outputs(config, source, tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(asdict(replace(config, average_hours=0.05))))
    flags = ["--source", str(source), "--config", str(path), "--smoke-no-radiation", "--island"]
    full, run = tmp_path / "full", tmp_path / "split"
    def until(seconds):
        return ["--stop-after-days", str(seconds / c.DAY)]
    monkeypatch.setattr(c, "snapshot_times", lambda days: [0, 120, 360])
    assert c.main(flags + ["--output", str(full)] + until(360)) == 0
    assert c.main(flags + ["--output", str(run)] + until(120)) == 0
    start = run / "restarts/restart_00000120.nc"
    assert c.read_checkpoint(start)["surface_output"]["start"] == 0
    with xr.open_dataset(run / "snapshots/state_00000000.nc", decode_times=False) as initial:
        assert initial.precipitation.isnull().all() and initial.evaporation.isnull().all()
        assert initial.sst.where(initial.land_mask == 1).isnull().all()
        assert np.isfinite(initial.sst.values[0, initial.land_mask.values == 0]).all()
    # Simulate files written beyond the most recent durable checkpoint.
    (run / "snapshots/state_00000300.nc").write_bytes(b"superseded")
    with (run / "diagnostics.csv").open("a") as f:
        fieldnames = (run / "diagnostics.csv").read_text().splitlines()[0].split(",")
        csv.DictWriter(f, fieldnames).writerow({"seconds": 300})
    from netCDF4 import Dataset
    with Dataset(run / "diagnostics.nc", "a") as ds:
        ds["time"][len(ds.dimensions["time"])] = 300
    monkeypatch.chdir(run)
    with patch.object(c, "source_hashes", return_value={"changed.py": "development revision"}):
        assert c.main(["--restart"] + until(360)) == 0
    for seconds in (180, 360):
        file = f"averages/surface_{seconds:08d}.nc"
        with xr.open_dataset(full/file, decode_times=False) as a, xr.open_dataset(run/file, decode_times=False) as b:
            xr.testing.assert_identical(a, b)
            assert b.time_bounds.values.tolist() == [[seconds - 180, seconds]]
            assert b.precipitation.units == b.evaporation.units == "kg m-2 s-1"
    first = c.read_checkpoint(full / "restarts/restart_00000360.nc")
    with xr.open_dataset(run/"snapshots/state_00000360.nc", decode_times=False) as ds:
        assert ds.surface_flux_time_bounds.values.tolist() == [[300, 360]]
        last = first["surface_output"]["last"]
        for field in ("precipitation", "evaporation"):
            np.testing.assert_array_equal(ds[field].values[0], last[field].numpy()/last["dt"])
        np.testing.assert_equal(ds.sst.values[:, ds.land_mask.values == 0],
                                ds.ocean_temperature.isel(ocean_level=0).values[:, ds.land_mask.values == 0])
    continued = c.read_checkpoint(run / "restarts/restart_00000360.nc")
    assert first.pop("fingerprint") != continued.pop("fingerprint")
    same_tree(continued, first)
    with (run / "diagnostics.csv").open() as f:
        assert [int(row["seconds"]) for row in csv.DictReader(f)] == [0, 120, 360]
    with xr.open_dataset(run / "diagnostics.nc", decode_times=False) as ds:
        assert ds.time.values.tolist() == [0, 120, 360]
    assert list((run / "history").glob("*/snapshots/state_00000300.nc"))
    assert c.main(["--restart"] + until(360)) == 0


def test_bad_grid_preserves_run_and_last_good_is_selected(source, config, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(asdict(config)))
    run = tmp_path / "run"
    flags = ["--config", str(path), "--source", str(source), "--output", str(run),
             "--smoke-no-radiation", "--stop-after-days", str(180 / c.DAY)]
    step = c.advance
    def fail_second(*args, **kwargs):
        if args[1].time_seconds == 60:
            raise RuntimeError("test interruption")
        return step(*args, **kwargs)
    with patch.object(c, "advance", side_effect=fail_second):
        assert c.main(flags) == 1
    saved = {name: (run / name).read_bytes() for name in
             ("last_good_restart.nc", "manifest.json", "configuration.json", "status.json")}
    path.write_text(yaml.safe_dump(asdict(replace(config, atmosphere_top_pa=700.))))
    assert c.main(flags + ["--restart"]) == 1
    assert all((run/name).read_bytes() == contents for name, contents in saved.items())
    assert "air_hybrid" in (run/"run.log").read_text()
    assert c.main(["--restart", "--output", str(run), "--stop-after-days", str(120/c.DAY)]) == 0
    prior = json.loads((run / "manifest.json").read_text())["continued_from"]
    assert prior["seconds"] == 60 and prior["path"].endswith("last_good_restart.nc")


def test_fork_accepts_separate_grid_files_and_rejects_same_size_reordering(source, config, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(asdict(config)))
    flags = ["--config", str(path), "--source", str(source), "--smoke-no-radiation"]
    original = tmp_path / "original"
    assert c.main(flags + ["--island", "--output", str(original), "--stop-after-days", str(60/c.DAY)]) == 0
    checkpoint = original / "restarts/restart_00000060.nc"
    mesh, geography = tmp_path / "moved.npy", tmp_path / "moved.npz"
    with xr.open_dataset(checkpoint, decode_times=False) as ds:
        xyz = ds.cell_xyz.values
        np.save(mesh, xyz)
        np.savez(geography, land=ds.land_mask.values.astype(bool), height_m=ds.surface_height_m.values,
                 depth_m=ds.ocean_depth_m.values)
    args = flags + ["--mesh", str(mesh), "--geography", str(geography), "--resume", str(checkpoint),
                    "--stop-after-days", str(120/c.DAY)]
    assert c.main(args + ["--output", str(tmp_path / "fork")]) == 0
    np.save(mesh, np.roll(xyz, 1, axis=0))
    rejected = tmp_path / "rejected"
    assert c.main(args + ["--output", str(rejected)]) == 1
    assert "cell_xyz" in json.loads((rejected / "status.json").read_text())["error"]
