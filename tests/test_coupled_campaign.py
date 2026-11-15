"""Campaign calendar, exact sampling, radiation plumbing, restart and failure I/O."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
import yaml

pytest.importorskip("pyam21")
from test_pyam21_workflow import same_tree, PrescribedRadiationPackage
from workflows.coupled import campaign as c
from workflows.pyam21.radiation import RadiationSetup, build_radiation
from pyam21.radiation.driver import RadiationDriver
from pyam21.time_manager import set_date, set_time, get_date


@pytest.fixture
def source(tmp_path):
    p = tmp_path / "source/reference/input/gcss_bomex/input.nml"
    p.parent.mkdir(parents=True)
    p.write_text(
        "&moist_processes_nml\n do_lsc=.false., do_mca=.false., do_ras=.true., do_strat=.true., do_cmt=.true.\n/\n&vert_turb_driver_nml\n do_mellor_yamada=.false., do_entrain=.true., do_stable_bl=.true., use_tau=.false., gust_scheme='beljaars'\n/\n&damping_driver_nml\n do_conserve_energy=.true.\n/\n"
    )
    return tmp_path / "source"


@pytest.fixture
def config():
    return c.CampaignConfig(
        cells=16,
        atmosphere_levels=12,
        ocean_levels=3,
        ocean_depth_m=200.0,
        lloyd_iterations=0,
        radiation_dt=120,
    )


def synthetic(nml, model, data_dir, **options):
    package = PrescribedRadiationPackage()
    return RadiationSetup(
        RadiationDriver.from_nml(nml, package),
        "synthetic-campaign-test",
        {"synthetic": True, "namelist": nml},
    )


def test_output_schedule_and_nonleap_year():
    times = c.snapshot_times()
    assert len(times) == 408 and len(set(times)) == 408
    assert times[:57] == list(range(0, 14 * c.DAY + 1, 21600))
    assert times[57:] == list(range(15 * c.DAY, 366 * c.DAY, c.DAY))
    assert get_date(set_date(2001, 1, 1) + set_time(365 * c.DAY)) == (
        2002,
        1,
        1,
        0,
        0,
        0,
    )
    with pytest.raises(ValueError):
        c.CampaignConfig(start_year=2000).validate()
    with pytest.raises(ValueError):
        c.CampaignConfig(physics_dt=59).validate()


def test_actual_astronomy_season_and_fixed_solar_configuration(source, config):
    with patch("workflows.pyam21.radiation.build_radiation", side_effect=synthetic):
        adapter, air, receiver, sea = c.build(config, source, Path("synthetic"))
    astro = adapter.radiation.astro
    june = astro.declination(astro.angle(astro.orbital_time(set_date(2001, 6, 21))))
    december = astro.declination(
        astro.angle(astro.orbital_time(set_date(2001, 12, 21)))
    )
    assert june > 0.4 and december < -0.4
    assert astro.period == 365 * c.DAY
    assert (
        adapter.radiation_manifest["namelist"]["radiation_driver_nml"]["solar_constant"]
        == 1361.0
    )
    with pytest.raises(ValueError, match="radiation-data"):
        c.build(config, source, None)


def test_builder_applies_solar_constant_to_upstream_package(source, config, tmp_path):
    adapter, *_ = c.build(config, source, None, smoke_no_radiation=True)
    for f in ("stdlvls", "h2ocoeff_ckd_speccombwidebds_hi00", "h2ockd2.1_data"):
        (tmp_path / f).write_text("synthetic constructor fixture")
    package = SimpleNamespace(sw=SimpleNamespace(solar_constant=1365.0))
    nml = {
        "radiation_driver_nml": {
            "solar_constant": 1361.0,
            "solar_dataset_entry": [1, 1, 1, 0, 0, 0],
        }
    }
    with patch("workflows.pyam21.radiation.SeaEsfRad", return_value=package):
        result = build_radiation(nml, adapter.model, tmp_path)
    assert package.sw.solar_constant == 1361.0
    assert (
        result.manifest["namelist"]["radiation_driver_nml"]["solar_constant"] == 1361.0
    )


def test_coupled_restart_including_cached_radiation(source, config, tmp_path):
    with patch("workflows.pyam21.radiation.build_radiation", side_effect=synthetic):
        adapter, air, receiver, sea = c.build(config, source, Path("synthetic"))
    with torch.no_grad():
        air, sea, _ = c.advance(adapter, air, receiver, sea, 60, config)
        p = c.pack(adapter, air, receiver, sea, "id", {"upstream": 1.0})
        c.checkpoint(tmp_path / "state.pt", p)
        ra, rs, ledger = c.unpack(
            torch.load(tmp_path / "state.pt", weights_only=True),
            adapter,
            receiver,
            sea,
            "id",
        )
        left, water, r = c.advance(adapter, air, receiver, sea, 60, config)
        right, replay, t = c.advance(adapter, ra, receiver, rs, 60, config)
    same_tree(
        c.pack(adapter, left, receiver, water, "id", ledger),
        c.pack(adapter, right, receiver, replay, "id", ledger),
    )
    assert r == t
    with pytest.raises(ValueError, match="does not match"):
        c.unpack(p, adapter, receiver, sea, "different")
    c.snapshot(tmp_path / "snapshot.nc", adapter, left, receiver, water)
    import xarray as xr

    with xr.open_dataset(tmp_path / "snapshot.nc", decode_times=False) as ds:
        assert ds.time.item() == 120 and ds.time.attrs["calendar"] == "julian"
        assert ds.ice_cold_content.shape == (1, config.cells, 2)
        torch.testing.assert_close(
            torch.tensor(ds.ice_temperature.values[0]), receiver.snow.temperature(water)
        )
        assert ds.air_temperature.sizes["air_interface"] == config.atmosphere_levels + 1


def test_cli_resume_and_last_good_on_failure(source, config, tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(c.asdict(config)))
    common = ["--source", str(source), "--config", str(path), "--smoke-no-radiation"]
    first = tmp_path / "first"
    assert (
        c.main(common + ["--output", str(first), "--stop-after-days", str(120 / c.DAY)])
        == 0
    )
    assert "Initializing" not in capsys.readouterr().out
    assert "Initializing" in (first / "run.log").read_text()
    assert "Run finished" in (first / "run.log").read_text()
    restart = first / "restarts/restart_00000120.pt"
    assert restart.exists()
    second = tmp_path / "second"
    assert (
        c.main(
            common
            + [
                "--output",
                str(second),
                "--resume",
                str(restart),
                "--stop-after-days",
                str(240 / c.DAY),
            ]
        )
        == 0
    )
    assert json.loads((second / "status.json").read_text())["latest"]["seconds"] == 240
    assert len(list((first / "snapshots").glob("*.nc"))) == 1
    real = c.advance
    count = 0

    def fails(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError("injected failure")
        return real(*args, **kwargs)

    failed = tmp_path / "failed"
    with patch.object(c, "advance", side_effect=fails):
        assert (
            c.main(
                common
                + ["--output", str(failed), "--stop-after-days", str(120 / c.DAY)]
            )
            == 1
        )
    p = torch.load(failed / "last_good_restart.pt", weights_only=True)
    assert p["air"]["time_seconds"] == 60 and p["ledger"]["steps"] == 1
    log = (failed / "run.log").read_text()
    assert "Traceback" in log and "injected failure" in log
    assert "Saved last_good_restart.pt at second 60" in log


def test_saved_mesh_loading_and_fingerprint(source, config, tmp_path):
    import numpy as np
    from vfs.geometry.mesh.grid import _fibonacci_xyz

    path = tmp_path / "mesh.npy"
    xyz = _fibonacci_xyz(32)
    np.save(path, xyz)
    effective, loaded, info = c.load_mesh(replace(config, lloyd_iterations=10), path)
    assert effective.cells == 32 and effective.lloyd_iterations == 0
    assert np.array_equal(loaded, xyz)
    adapter, air, receiver, sea = c.build(
        effective, source, None, smoke_no_radiation=True, generator_xyz=loaded
    )
    expected = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    np.testing.assert_array_equal(adapter.model.mesh.cell_xyz.numpy(), expected)
    assert adapter.model.mesh.lloyd_diagnostics is None
    assert sea.coupled.ocean.ocean.mesh is adapter.model.mesh
    renamed = tmp_path / "renamed.npy"
    renamed.write_bytes(path.read_bytes())
    again, _, moved = c.load_mesh(config, renamed)
    identity = c.fingerprint(effective, adapter, {}, True, info["sha256"])
    assert identity == c.fingerprint(again, adapter, {}, True, moved["sha256"])
    np.save(renamed, xyz[::-1].copy())
    _, _, changed = c.load_mesh(config, renamed)
    assert identity != c.fingerprint(effective, adapter, {}, True, changed["sha256"])


@pytest.mark.parametrize(
    "kind", ["shape", "nan", "zero", "duplicate", "nonunit", "object"]
)
def test_invalid_saved_mesh_rejects(config, tmp_path, kind):
    import numpy as np
    from vfs.geometry.mesh.grid import _fibonacci_xyz

    xyz = _fibonacci_xyz(16)
    if kind == "shape":
        xyz = xyz[:, :2]
    if kind == "nan":
        xyz[0, 0] = np.nan
    if kind == "zero":
        xyz[0] = 0
    if kind == "duplicate":
        xyz[0] = xyz[1]
    if kind == "nonunit":
        xyz *= 2
    if kind == "object":
        xyz = xyz.astype(object)
    path = tmp_path / "bad.npy"
    np.save(path, xyz)
    with pytest.raises(ValueError):
        c.load_mesh(config, path)


def test_mesh_cli_preflight_logs_and_resolved_configuration(
    source, config, tmp_path, capsys
):
    import numpy as np
    from vfs.geometry.mesh.grid import _fibonacci_xyz

    mesh = tmp_path / "custom.npy"
    np.save(mesh, _fibonacci_xyz(32))
    cfg = tmp_path / "campaign.yaml"
    cfg.write_text(yaml.safe_dump(c.asdict(replace(config, lloyd_iterations=10))))
    out = tmp_path / "preflight"
    assert (
        c.main(
            [
                "--source",
                str(source),
                "--config",
                str(cfg),
                "--mesh",
                str(mesh),
                "--output",
                str(out),
                "--smoke-no-radiation",
                "--preflight",
                "--verbose",
            ]
        )
        == 0
    )
    assert "Initializing 32 cells" in capsys.readouterr().out
    saved = json.loads((out / "configuration.json").read_text())
    assert saved["cells"] == 32 and saved["lloyd_iterations"] == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["mesh"]["kind"] == "saved_generators"
    log = (out / "run.log").read_text()
    assert "Initializing 32 cells" in log and "Preflight passed" in log
    assert "INFO" in log and "SHA256" in log
