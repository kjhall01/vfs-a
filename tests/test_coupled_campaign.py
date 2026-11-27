"""Campaign calendar, exact sampling, radiation plumbing, restart and failure I/O."""

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
import yaml

pytest.importorskip("pyam21")
from test_pyam21_workflow import same_tree, PrescribedRadiationPackage
from test_coupled_restart_compat import vapor_sources
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
    aerosol = p.parents[1] / "gcss_astex/input.nml"
    aerosol.parent.mkdir()
    shutil.copyfile(
        Path(__file__).parent / "fixtures/pyam21/astex-aerosol.nml", aerosol
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
        physics_dt=60,
        atmosphere_dt=15,
        ocean_dt=60,
        mixing_substeps=2,
        eddy_substeps=2,
    )


def synthetic(nml, model, data_dir, **options):
    package = PrescribedRadiationPackage()
    return RadiationSetup(
        RadiationDriver.from_nml(nml, package),
        "synthetic-campaign-test",
        {"synthetic": True, "namelist": nml},
    )


def test_campaign_script_prefers_checkout_over_older_install(tmp_path):
    import vfs

    # Build a checkout layout even when pytest itself runs against a staged wheel.
    checkout = tmp_path / "checkout"
    shutil.copytree(c.ROOT / "workflows", checkout / "workflows")
    (checkout / "vfs").symlink_to(Path(vfs.__file__).resolve().parent, target_is_directory=True)
    stale = tmp_path / "old-install"
    (stale / "vfs").mkdir(parents=True)
    for path in (stale / "vfs/__init__.py", stale / "roquet.py"):
        path.write_text("raise RuntimeError('imported a stale campaign dependency')")

    # The staged checkout contains only the packaged physics and workflows.
    paths = [stale, checkout]
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, paths)))
    result = subprocess.run([sys.executable, str(checkout / "workflows/coupled/campaign.py"), "--help"],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "--preflight" in result.stdout


def test_radiation_off_needs_only_bomex_namelist(source, config):
    (source / "reference/input/gcss_astex/input.nml").unlink()
    adapter, *_ = c.build(config, source, None, smoke_no_radiation=True)
    assert adapter.radiation is None


@pytest.mark.parametrize("air_method", ["ssprk3", "ssprk43"])
def test_opt_in_ssprk43_survives_thermal_ice_coupled_interval(source, config, air_method):
    # Full campaign path: ThermalSnowOceanIce must forward the selected
    # integrator to the moving snow/ocean/ice dynamics, without rejecting it
    # as an independent surface forcing option.
    candidate = replace(config, ocean_dynamics_integrator="ssprk43", atmosphere_dynamics_integrator=air_method)
    assert candidate.validate() == candidate
    with patch("workflows.pyam21.radiation.build_radiation", side_effect=synthetic):
        adapter, air, receiver, sea = c.build(candidate, source, Path("synthetic"))
    with torch.no_grad():
        final_air, final_sea, report = c.advance(
            adapter, air, receiver, sea, candidate.physics_dt, candidate
        )
    assert final_air.time_seconds == candidate.physics_dt
    receiver.snow.check(final_sea)
    assert report["ale_steps"] == 1
    assert torch.isfinite(final_sea.coupled.ocean.ocean.h).all()


def test_explicit_legacy_cadences_reproduce_default_coupled_step(source, config):
    # The scheduler MUST not change physics ordering or numerical trajectory
    # when every closure is scheduled at its original frequency.
    explicit = replace(
        config, mixing_dt=config.ocean_dt, mesoscale_dt=config.ocean_dt,
        ale_dt=config.physics_dt,
    ).validate()
    with patch("workflows.pyam21.radiation.build_radiation", side_effect=synthetic):
        adapter, air, receiver, sea = c.build(config, source, Path("synthetic"))
    with torch.no_grad():
        a, sa, ra = c.advance(adapter, air, receiver, sea, config.physics_dt, config)
        b, sb, rb = c.advance(adapter, air, receiver, sea, config.physics_dt, explicit)
    same_tree(
        c.pack(adapter, a, receiver, sa, "equivalence", {}),
        c.pack(adapter, b, receiver, sb, "equivalence", {}),
    )
    assert ra == rb


def test_40_minute_physics_multirate_smoke(source, config):
    schedule = replace(
        config,
        physics_dt=2400, radiation_dt=7200,
        atmosphere_dt=600, ocean_dt=600,
        mixing_dt=1200, mesoscale_dt=1200, ale_dt=7200,
    ).validate()
    with patch("workflows.pyam21.radiation.build_radiation", side_effect=synthetic):
        adapter, air, receiver, sea = c.build(schedule, source, Path("synthetic"))
    with torch.no_grad():
        air, sea, report = c.advance(
            adapter, air, receiver, sea, schedule.physics_dt, schedule
        )
    assert air.time_seconds == 2400
    assert report["gmredi_steps"] == schedule.eddy_substeps * 2
    assert report["ale_steps"] == 0
    receiver.snow.check(sea)
    adapter.check(air)
    assert torch.isfinite(torch.tensor(report["surface_residual"]))


def test_multirate_throughput_paired_small_coupled_case(source, config, capsys):
    """Exploratory full-physics CPU timing; no fragile speed threshold.

    Each method advances 2400 seconds from the SAME initial state. AM2 column
    physics, radiative cache, surface/ice exchange, vertical mixing, GM–Redi,
    phase and ALE are included. No optical SeaEsf tables are available in CI:
    the PyAM21 radiation interface uses its tested analytic fixture.
    """
    import statistics
    import time

    reference = replace(
        config, cells=64, ocean_levels=8, physics_dt=1200,
        atmosphere_dt=600, ocean_dt=600, radiation_dt=7200,
        ale_dt=None, mixing_dt=None, mesoscale_dt=None,
    ).validate()
    faster = replace(
        reference, physics_dt=2400, mixing_dt=1200, mesoscale_dt=1200,
        ale_dt=7200,
    ).validate()
    with patch("workflows.pyam21.radiation.build_radiation", side_effect=synthetic):
        adapter, air, receiver, sea = c.build(reference, source, Path("synthetic"))

    def measure(cfg):
        with torch.no_grad():
            result, water = air, sea
            totals = {}
            for _ in range(2400 // cfg.physics_dt):
                result, water, report = c.advance(
                    adapter, result, receiver, water, cfg.physics_dt, cfg
                )
                totals = report
        assert result.time_seconds == 2400
        receiver.snow.check(water)
        return totals

    # Warm up kernels, avoiding first-time initialization effects.
    measure(reference)
    measure(faster)
    timings = {}
    for name, cfg in (("reference", reference), ("multirate", faster)):
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            measure(cfg)
            samples.append(time.perf_counter() - start)
        timings[name] = dict(samples_s=samples, median_s=statistics.median(samples))
    ratio = timings["reference"]["median_s"] / timings["multirate"]["median_s"]
    print("COUPLED_MULTIRATE_TIMING " + json.dumps({
        "cells": 64, "atmosphere_levels": reference.atmosphere_levels,
        "ocean_levels": reference.ocean_levels,
        "equal_model_seconds": 2400, "threads": reference.threads,
        **timings, "observed_speedup": ratio,
        "radiation": "synthetic PyAM21 driver, no SeaEsf tables",
    }), flush=True)


def test_4096_40_multirate_pilot_yaml_is_complete_and_valid():
    path = Path(__file__).resolve().parents[1] / (
        "workflows/coupled/aquaplanet_multirate_4096_40.yaml"
    )
    cfg = c.CampaignConfig(**yaml.safe_load(path.read_text())).validate()
    assert (cfg.cells, cfg.atmosphere_levels, cfg.ocean_levels) == (4096, 32, 40)
    assert (cfg.physics_dt, cfg.atmosphere_dt, cfg.ocean_dt) == (2400, 600, 600)
    assert (cfg.mixing_dt, cfg.mesoscale_dt, cfg.ale_dt, cfg.radiation_dt) == (
        1200, 1200, 7200, 7200
    )
    assert cfg.gm_redi and cfg.backscatter and cfg.ale_remap
    assert cfg.dtype == "float64"


def test_noncommensurate_process_cadences_fail_closed(config):
    for changes in (
        dict(mixing_dt=59),
        dict(mixing_dt=180),
        dict(mesoscale_dt=61),
        dict(mesoscale_dt=180),
        dict(ale_dt=61),
        dict(ale_dt=90),
    ):
        with pytest.raises(ValueError):
            replace(config, **changes).validate()


@pytest.mark.parametrize("scheme", ["midpoint", "quadratic_funded_donor_parcel"])
def test_multirate_closures_and_ale_through_actual_coupled_campaign(source, config, scheme):
    # Both closures run half as often while retaining their full elapsed dt.
    # Radiation and atmosphere--ocean transfer remain one atomic physics call.
    scheduled = replace(
        config,
        gm_energy_scheme=scheme,
        physics_dt=120,
        radiation_dt=120,
        ocean_dt=60,
        atmosphere_dt=15,
        mixing_dt=120,
        mesoscale_dt=120,
        ale_dt=240,
    ).validate()
    with patch("workflows.pyam21.radiation.build_radiation", side_effect=synthetic):
        adapter, air, receiver, sea = c.build(scheduled, source, Path("synthetic"))
    with torch.no_grad():
        air1, sea1, r1 = c.advance(
            adapter, air, receiver, sea, scheduled.physics_dt, scheduled
        )
        air2, sea2, r2 = c.advance(
            adapter, air1, receiver, sea1, scheduled.physics_dt, scheduled
        )
    receiver.snow.check(sea1)
    receiver.snow.check(sea2)
    assert air1.time_seconds == 120 and air2.time_seconds == 240
    assert r1["gmredi_steps"] == r2["gmredi_steps"] == scheduled.eddy_substeps
    assert r1["eddy_decay_steps"] == r2["eddy_decay_steps"] == 1
    assert r1["ale_steps"] == 0
    assert r2["ale_steps"] == 1
    for report in (r1, r2):
        assert all(
            torch.isfinite(torch.tensor(report[key]))
            for key in ("external", "upstream", "surface_residual", "air_residual")
        )
    assert (sea2.coupled.ocean.ocean.salt_h >= 0).all()


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
    long_run = c.CampaignConfig(days=395).validate()
    assert c.snapshot_times(long_run.days)[-31:] == list(range(365 * c.DAY, 396 * c.DAY, c.DAY))
    assert get_date(set_date(2001, 1, 1) + set_time(395 * c.DAY)) == (2002, 1, 31, 0, 0, 0)
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
    epoch = set_date(config.start_year, 1, 1)
    assert astro.orbital_time(epoch + set_time(395*c.DAY)) == pytest.approx(
        astro.orbital_time(epoch + set_time(30*c.DAY))
    )
    assert (
        adapter.radiation_manifest["namelist"]["radiation_driver_nml"]["solar_constant"]
        == 1361.0
    )
    with pytest.raises(ValueError, match="radiation-data"):
        c.build(config, source, None)


def test_coupled_clock_crosses_new_year(source, config):
    config = replace(config, days=395)
    with patch("workflows.pyam21.radiation.build_radiation", side_effect=synthetic):
        adapter, air, receiver, sea = c.build(config, source, Path("synthetic"))
    boundary = 365 * c.DAY
    air = replace(air, time_seconds=boundary - config.physics_dt)
    with torch.no_grad():
        for end in (boundary, boundary + config.physics_dt):
            air, sea, _ = c.advance(adapter, air, receiver, sea, config.physics_dt, config)
            assert air.time_seconds == end
            assert get_date(adapter.start + set_time(end))[:3] == (2002, 1, 1)


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


@pytest.mark.parametrize("native_grid", [False, True])
def test_coupled_restart_including_cached_radiation(source, config, tmp_path, native_grid):
    def builder(nml, model, data_dir, **options):
        if not native_grid:
            return synthetic(nml, model, data_dir, **options)
        from test_radiation_mapping import NativeGridRadiationPackage
        from workflows.pyam21.radiation import HostRadiationDriver

        return RadiationSetup(
            HostRadiationDriver.from_nml(nml, NativeGridRadiationPackage()),
            "analytic-native-grid-campaign-test", {"synthetic": True},
        )

    with patch("workflows.pyam21.radiation.build_radiation", side_effect=builder):
        adapter, air, receiver, sea = c.build(config, source, Path("synthetic"))
    with torch.no_grad():
        air, sea, _ = c.advance(adapter, air, receiver, sea, 60, config)
        p = c.pack(adapter, air, receiver, sea, "id", {"upstream": 1.0})
        c.save_restart(tmp_path / "state.nc", adapter, air, receiver, sea, "id", {"upstream": 1.0})
        same_tree(c.read_checkpoint(tmp_path / "state.nc"), p)
        ra, rs, ledger = c.unpack(
            c.read_checkpoint(tmp_path / "state.nc"),
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
    if native_grid:
        assert r["radiation_residual"] < 2e-10
        assert abs(r["radiation_energy_residual"]) < 2e-10 * 60
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
        torch.testing.assert_close(
            torch.tensor(ds.ocean_eddy_energy.values[0]),
            water.coupled.ocean.specific_eddy_energy(),
        )


def test_extend_campaign_horizon_with_original_manifest(source, config, tmp_path):
    from workflows.coupled.restart_compat import NATIVE_VAPOR_VERIFIER, PRE_CLOUD_ADAPTER_SHA, VERIFIER, PRE_DRAG_HASHES, pre_shelf_sources

    path = tmp_path / "config.yaml"
    args = ["--source", str(source), "--config", str(path), "--smoke-no-radiation"]
    old = tmp_path / "pilot"
    new = tmp_path / "continued"
    control = tmp_path / "control"
    path.write_text(yaml.safe_dump(c.asdict(replace(config, days=30))))
    previous_sources = pre_shelf_sources({**c.source_hashes(), **PRE_DRAG_HASHES})
    previous_sources["workflows/pyam21/adapter.py"] = PRE_CLOUD_ADAPTER_SHA
    previous_sources[VERIFIER] = NATIVE_VAPOR_VERIFIER
    previous_sources["workflows/coupled/campaign.py"] = "200c9497dab13b00f3e7f33e5faf19fcbd30fa611a3912d0630d7c0c303dab9d"
    previous_sources["workflows/pyam21/dynamics.py"] = "3db19be53cafe3f86798392c5efb8d8ad46ded1710f0cca2184208aab46459cd"
    asdict = c.asdict

    def pilot_asdict(obj):
        fields = asdict(obj)
        if isinstance(obj, c.CampaignConfig):
            del fields["atmosphere_dynamics_integrator"]
            del fields["matched_shelf_momentum"]
            del fields["ice_ocean_drag"]
        return fields

    with patch.object(c, "source_hashes", return_value=previous_sources), patch.object(c, "asdict", pilot_asdict):
        assert c.main(args + ["--output", str(old), "--stop-after-days", str(60/c.DAY)]) == 0
    resume = ["--resume", str(old / "restarts/restart_00000060.nc"),
              "--resume-compatible-manifest", str(old / "manifest.json")]
    end = ["--stop-after-days", str(120/c.DAY)]
    path.write_text(yaml.safe_dump(c.asdict(replace(config, days=395))))
    assert c.main(args + ["--output", str(new)] + end + resume) == 0
    assert c.main(args + ["--output", str(control)] + end) == 0
    restart = Path("restarts/restart_00000120.nc")
    same_tree(c.read_checkpoint(new/restart), c.read_checkpoint(control/restart))
    audit = json.loads((new/"compatible_resume_audit.json").read_text())
    assert audit["horizon_days"] == {"prior": 30, "current": 395}
    assert audit["numerical_change"] == "pyam21_cloud_area_boundary_roundoff"

    four_stage = tmp_path / "four_stage"
    path.write_text(yaml.safe_dump(c.asdict(replace(config, days=395, atmosphere_dynamics_integrator="ssprk43"))))
    with patch.object(c, "advance_dynamics", wraps=c.advance_dynamics) as advance_air:
        assert c.main(args + ["--output", str(four_stage)] + end + resume) == 0
    assert [call.kwargs["integrator"] for call in advance_air.call_args_list] == ["ssprk43"] * 4
    audit = json.loads((four_stage/"compatible_resume_audit.json").read_text())
    assert audit["atmosphere_dynamics_integrator"] == {"prior": "ssprk3", "current": "ssprk43", "from_seconds": 60}
    assert audit["cloud_area_boundary_roundoff"] is True
    assert c.main(args + ["--output", str(tmp_path/"strict"), "--resume", str(four_stage/restart),
                          "--stop-after-days", str(180/c.DAY)]) == 0

    changes = [
        {"days": 29}, {"days": 395, "atmosphere_dt": 12},
        {"days": 395, "ice_ocean_drag": "native_edge"},
    ]
    for index, change in enumerate(changes):
        path.write_text(yaml.safe_dump(c.asdict(replace(config, **change))))
        bad = tmp_path / f"rejected-{index}"
        assert c.main(args + ["--output", str(bad)] + end + resume) != 0
        assert "configuration differs" in json.loads((bad/"status.json").read_text())["error"]


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
    restart = first / "restarts/restart_00000120.nc"
    assert restart.exists()
    import csv
    import xarray as xr
    with (first / "diagnostics.csv").open() as csv_file, xr.open_dataset(first / "diagnostics.nc", decode_times=False) as ds:
        rows = list(csv.DictReader(csv_file))
        assert ds.time.values.tolist() == [int(row["seconds"]) for row in rows]
        assert ds.numerical_energy_residual_J_m2.values.tolist() == [float(row["numerical_energy_residual_J_m2"]) for row in rows]
        assert ds.numerical_energy_residual_J_m2.units == "J m-2"
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
    p = c.read_checkpoint(failed / "last_good_restart.nc")
    assert p["air"]["time_seconds"] == 60 and p["ledger"]["steps"] == 1
    log = (failed / "run.log").read_text()
    assert "Traceback" in log and "injected failure" in log
    assert "Saved last_good_restart.nc at second 60" in log


@pytest.mark.parametrize("pre_trace", [False, True])
def test_cli_resumes_audited_pre_endpoint_checkpoint(source, config, tmp_path, pre_trace):
    from workflows.coupled.restart_compat import APPROVED_PRE_TRACE_HASHES, PRE_ENDPOINT_HASHES

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(c.asdict(config)))
    args = ["--source", str(source), "--config", str(path), "--smoke-no-radiation"]
    initial = tmp_path / "initial"
    # Exercise the historical transition, even when current development has
    # advanced beyond the exact target allowed by its verifier.
    archive = Path(__file__).parent / "fixtures"
    prior_hashes = json.loads((archive / "coupled_restart_sources-6e68cb3.json").read_text())
    endpoint_hashes = {**prior_hashes, **json.loads((archive / "coupled_endpoint_sources-ed46ac5.json").read_text())}
    if pre_trace:
        prior_hashes.update(APPROVED_PRE_TRACE_HASHES)
        del prior_hashes["workflows/coupled/restart_compat.py"]
    with patch.object(c, "source_hashes", return_value=prior_hashes):
        assert c.main(args + ["--output", str(initial), "--stop-after-days", str(60 / c.DAY)]) == 0
    resumed = tmp_path / "resumed"
    manifest = initial / "manifest.json"
    with patch.object(c, "source_hashes", return_value=endpoint_hashes):
        assert c.main(args + ["--output", str(resumed), "--stop-after-days", str(120 / c.DAY),
                              "--resume", str(initial / "restarts/restart_00000060.nc"),
                              "--resume-compatible-manifest", str(manifest)]) == 0
    status = json.loads((resumed / "status.json").read_text())
    audit = json.loads((resumed / "compatible_resume_audit.json").read_text())
    assert status["latest"]["seconds"] == 120
    assert audit["prior_fingerprint"] == json.loads(manifest.read_text())["fingerprint"]
    assert audit["new_fingerprint"] != audit["prior_fingerprint"]
    assert audit["endpoint_fix_files"] == sorted(PRE_ENDPOINT_HASHES)
    legacy_audit = resumed / "instrumentation_resume_audit.json"
    if pre_trace:
        assert json.loads(legacy_audit.read_text()) == audit
    else:
        assert not legacy_audit.exists()


@pytest.mark.parametrize("revision", ["dcda5cd", "5329122"])
def test_cli_resumes_audited_zero_vapor_transition(revision, source, config, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(c.asdict(config)))
    args = ["--source", str(source), "--config", str(path), "--smoke-no-radiation"]
    initial, resumed = tmp_path / "initial", tmp_path / "resumed"
    prior_hashes, fixed_hashes = vapor_sources(revision)
    with patch.object(c, "source_hashes", return_value=prior_hashes):
        assert c.main(args + ["--output", str(initial), "--stop-after-days", str(60 / c.DAY)]) == 0
    restart = initial / "restarts/restart_00000060.nc"
    old = c.read_checkpoint(restart)
    for key in c.VAPOR_SUMS + c.VAPOR_PEAKS:
        old["ledger"].pop(key)  # historical checkpoints lack these counters
    restart = restart.with_suffix(".pt")  # Historical payload and container.
    torch.save(old, restart)
    manifest = initial / "manifest.json"
    with patch.object(c, "source_hashes", return_value=fixed_hashes):
        assert c.main(args + ["--output", str(resumed), "--stop-after-days", str(120 / c.DAY),
                             "--resume", str(restart), "--resume-compatible-manifest", str(manifest)]) == 0
    audit = json.loads((resumed / "compatible_resume_audit.json").read_text())
    status = json.loads((resumed / "status.json").read_text())
    assert audit["numerical_change"] == "funded_zero_vapor_v1"
    assert audit["prior_fingerprint"] == old["fingerprint"]
    assert audit["new_fingerprint"] != audit["prior_fingerprint"]
    assert status["latest"]["seconds"] == 120
    for key in c.VAPOR_SUMS + c.VAPOR_PEAKS:
        assert status["latest"][key] == 0


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
    assert "GM-Redi=True" in log and "backscatter=True" in log and "ALE=True" in log


def test_full_ocean_schedule_calls_backscatter_and_changes_state(source, config):
    config = replace(config, ocean_dt=30)
    adapter, air, receiver, sea = c.build(config, source, None, smoke_no_radiation=True)
    initial = c.pack(adapter, air, receiver, sea, "id", {})
    actual = c.EddyBackscatter.tendency
    powers = []

    def tracked(op, model, state, *, transported_eddy_h, dt):
        assert transported_eddy_h.shape == state.eddy_h.shape and dt == config.ocean_dt
        du, de, report = actual(op, model, state, transported_eddy_h=transported_eddy_h, dt=dt)
        powers.append(float(report.kinetic_power))
        return du, de, report

    with torch.no_grad(), patch.object(c.EddyBackscatter, "tendency", tracked):
        end_air, end, report = c.advance(adapter, air, receiver, sea, 60, config)
    assert len(powers) == 6  # three actual RK stages per ocean step
    assert any(p != 0 for p in powers)
    assert report["gmredi_steps"] == 2 * config.eddy_substeps
    assert report["eddy_decay_steps"] == 2
    assert report["ale_steps"] == 1
    assert report["ale_changed_columns"] > 0
    assert report["gm_outflow_fraction"] > 0 and report["redi_step_number"] > 0
    assert report["gm_exchange_J_m2"] != 0 and report["redi_exchange_J_m2"] != 0
    assert report["eddy_dissipation_J_m2"] > 0
    assert not torch.equal(end.coupled.ocean.eddy_h, sea.coupled.ocean.eddy_h)
    same_tree(initial, c.pack(adapter, air, receiver, sea, "id", {}))


def test_mesoscale_and_nontrivial_ale_conserve_combined_ice_energy(source, config):
    _, _, receiver, sea = c.build(config, source, None, smoke_no_radiation=True)
    original = sea.coupled.ocean
    s = original.ocean
    # A deliberately displaced layer interface tests actual ALE, not identity.
    h = s.h.clone()
    h[:, 0] += 1
    h[:, 1] -= 1
    physical = s.with_tensors(
        u_e=s.u_e,
        h=h,
        temp_h=h * s.temperature(),
        salt_h=h * s.salinity(),
        tracer_h=h[..., None] * s.tracer_h / s.h[..., None],
    )
    ocean = replace(
        original,
        ocean=physical,
        tke_h=h * original.tke_h / s.h,
        eddy_h=h * original.eddy_h / s.h,
    )
    sea = replace(sea, coupled=replace(sea.coupled, ocean=ocean))
    before_energy = receiver.snow.total_energy(sea)
    area = s.mesh.cell_area[:, None]
    with torch.no_grad():
        mixed, gm = c.mesoscale_step(receiver, sea, 60, config)
        end, ale = c.ale_step(receiver, mixed, config)
    final = end.coupled.ocean.ocean
    torch.testing.assert_close(
        receiver.snow.total_energy(end), before_energy, rtol=5e-13, atol=0
    )
    torch.testing.assert_close(final.h.sum(-1), physical.h.sum(-1), rtol=2e-15, atol=0)
    torch.testing.assert_close(
        (area * final.salt_h).sum(), (area * physical.salt_h).sum(), rtol=2e-15, atol=0
    )
    expected = c.rest_thickness(config) / c.rest_thickness(config).sum()
    torch.testing.assert_close(
        final.h / final.h.sum(-1, keepdim=True),
        expected[None].expand_as(final.h),
        rtol=2e-15,
        atol=0,
    )
    assert ale["ale_changed_columns"] == config.cells
    assert gm["eddy_dissipation_J_m2"] > 0
    assert end.coupled.ice is sea.coupled.ice
    assert end.snow_mass is sea.snow_mass and end.cold_content is sea.cold_content
    # ALE carries the reservoirs conservatively, independently of its heat solve.
    for name in ("eddy_h", "tke_h"):
        torch.testing.assert_close(
            getattr(end.coupled.ocean, name).sum(-1),
            getattr(mixed.coupled.ocean, name).sum(-1),
            rtol=2e-15,
            atol=0,
        )


def test_ocean_switches_and_failure_keep_inputs_intact(source, config):
    disabled = replace(
        config,
        gm_redi=False,
        eddy_dissipation=False,
        backscatter=False,
        ale_remap=False,
        eddy_initial_energy_J_kg=0.0,
    )
    adapter, air, receiver, sea = c.build(
        disabled, source, None, smoke_no_radiation=True
    )
    assert receiver.H.model.backscatter is None
    with (
        patch.object(c.GMRedi, "apply", side_effect=AssertionError("disabled GM")),
        patch.object(
            c.EddyDissipation, "apply", side_effect=AssertionError("disabled decay")
        ),
        patch.object(
            receiver.H.model,
            "remap_with_report",
            side_effect=AssertionError("disabled ALE"),
        ),
        torch.no_grad(),
    ):
        _, _, report = c.advance(adapter, air, receiver, sea, 60, disabled)
    assert all(report[k] == 0 for k in c.OCEAN_SUMS + c.OCEAN_PEAKS)
    assert c.fingerprint(config, adapter, {}, True) != c.fingerprint(
        disabled, adapter, {}, True
    )
    adapter, air, receiver, sea = c.build(config, source, None, smoke_no_radiation=True)
    initial = c.pack(adapter, air, receiver, sea, "id", {})
    with (
        patch.object(
            receiver.H.model, "remap_with_report", side_effect=ValueError("ALE failure")
        ),
        torch.no_grad(),
        pytest.raises(ValueError, match="ALE failure"),
    ):
        c.advance(adapter, air, receiver, sea, 60, config)
    same_tree(initial, c.pack(adapter, air, receiver, sea, "id", {}))


@pytest.mark.parametrize(
    "changes",
    [
        {"gm_redi": "false"},
        {"ale_remap": 1},
        {"eddy_substeps": 0},
        {"gm_length_m": 0},
        {"gm_efficiency": -1},
        {"gm_kinetic_fraction": 1.1},
        {"redi_diffusivity_m2_s": -1},
        {"eddy_initial_energy_J_kg": -1},
        {"eddy_decay_days": 0},
        {"backscatter_fraction": 1.1},
        {"backscatter_days": 1e-6},
        {"biharmonic_viscosity_m4_s": float("nan")},
    ],
)
def test_invalid_ocean_process_settings_reject(config, changes):
    with pytest.raises(ValueError):
        replace(config, **changes).validate()


def test_published_config_enables_full_ocean_and_requested_timesteps():
    path = Path(c.__file__).with_name("coupled_aquaplanet_1yr.yaml")
    configured = c.CampaignConfig(**yaml.safe_load(path.read_text())).validate()
    assert configured == c.CampaignConfig().validate()
    assert configured.gm_redi and configured.eddy_dissipation
    assert configured.backscatter and configured.ale_remap
    assert configured.biharmonic_viscosity_m4_s > 0
    assert configured.physics_dt == configured.ocean_dt == 1200
    assert configured.atmosphere_dt == 240 and configured.radiation_dt == 3600
    assert configured.mixing_substeps == 10 and configured.eddy_substeps == 10


def test_full_depth_aquaplanet_preset():
    path = Path(c.__file__).with_name("aquaplanet_full_depth_4096.yaml")
    cfg = c.CampaignConfig(**yaml.safe_load(path.read_text())).validate()
    assert (cfg.cells, cfg.atmosphere_levels, cfg.atmosphere_top_pa) == (4096, 32, 100.0)
    assert (cfg.ocean_levels, cfg.ocean_depth_m, cfg.top_layer_m) == (60, 5000.0, 10.0)
    assert cfg.gm_redi and cfg.eddy_dissipation and cfg.backscatter and cfg.ale_remap
    assert cfg.restart_days == 1
    assert cfg.physics_dt == 1200 and cfg.ocean_dt == 300 and cfg.atmosphere_dt == 240
    assert torch.isclose(c.rest_thickness(cfg).sum(), torch.tensor(5000.0, dtype=torch.float64))


def test_atmospheric_top_is_configurable(source, config):
    cfg = replace(config, atmosphere_top_pa=100.0)
    adapter, *_ = c.build(cfg, source, None, smoke_no_radiation=True)
    p = adapter.model.interface_pressure(100000.0)
    assert float(p.min()) == pytest.approx(100.0, abs=1.e-8)
    for bad in (-1.0, 0.0, 100000.0):
        with pytest.raises(ValueError):
            replace(config, atmosphere_top_pa=bad).validate()


def test_fresh_full_depth_two_substep_preset():
    path = Path(c.__file__).with_name("aquaplanet_full_depth_4096_gm2.yaml")
    cfg = c.CampaignConfig(**yaml.safe_load(path.read_text())).validate()
    assert (cfg.days, cfg.cells, cfg.atmosphere_levels, cfg.ocean_levels) == (30, 4096, 32, 60)
    assert (cfg.atmosphere_top_pa, cfg.ocean_depth_m, cfg.top_layer_m) == (100.0, 5000.0, 10.0)
    assert (cfg.physics_dt, cfg.atmosphere_dt, cfg.ocean_dt, cfg.radiation_dt) == (1200, 600, 600, 1200)
    assert cfg.gm_energy_scheme == "funded_donor_parcel" and cfg.eddy_substeps == 2
    assert cfg.mixing_substeps == 10 and cfg.gm_redi and cfg.redi_positive_transport
    assert cfg.eddy_dissipation and cfg.backscatter and cfg.ale_remap
    with pytest.raises(ValueError, match="gm_energy_scheme"):
        replace(cfg, gm_energy_scheme="unknown").validate()


def test_campaign_warns_and_commits_energy_limited_redi(source, config, tmp_path):
    # Isolate Redi work with a small stored reservoir and negligible GM flux.
    cfg = replace(config, eddy_initial_energy_J_kg=1e-11, eddy_substeps=1,
                  gm_efficiency=1e-20, gm_energy_scheme="funded_donor_parcel", backscatter=False)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(c.asdict(cfg)))
    out = tmp_path / "run"
    assert c.main(["--config", str(path), "--source", str(source), "--smoke-no-radiation",
                   "--output", str(out), "--stop-after-days", str(120 / c.DAY)]) == 0
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "segment_completed"
    assert (out / "run.log").read_text().count("local eddy energy limited Redi") == 2
    ledger = c.read_checkpoint(out / "restarts/restart_00000120.nc")["ledger"]
    for key in c.REDI_ENERGY_SUMS + c.REDI_ENERGY_PEAKS:
        assert ledger[key] == status["latest"][key] > 0
    assert ledger["redi_energy_blocked_rows"] <= ledger["redi_energy_limited_rows"]


@pytest.mark.parametrize("mix_dt", [60, 120])
def test_campaign_exposes_accepted_tke_funding(source, config, mix_dt):
    cfg = replace(config, physics_dt=120, mixing_dt=mix_dt)
    adapter, air, receiver, sea = c.build(cfg, source, None, smoke_no_radiation=True)
    model = receiver.H.model
    model.closure = replace(model.closure, max_reductions=0, predictor_steps=0)
    ocean = sea.coupled.ocean
    ocean = replace(ocean, tke_h=1e-12 * ocean.ocean.h)
    sea = replace(sea, coupled=replace(sea.coupled, ocean=ocean))
    collected = []
    original = type(model.closure).apply
    def record(self, *args):
        end, k, report = original(self, *args)
        collected.append(report.mixing_limiter)
        return end, k, report
    with torch.no_grad(), patch.object(type(model.closure), "apply", record):
        _, _, report = c._advance_water(adapter, air, receiver, sea, 120, cfg, None)
    assert len(collected) == 120 // mix_dt
    for key in ("limited_interfaces", "deferred_columns"):
        assert report["tke_" + key] == sum(float(r[key]) for r in collected) > 0


def test_campaign_records_native_vapor_recovery_when_resuming(source, config, tmp_path):
    from netCDF4 import Dataset

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(c.asdict(config)))
    args = ["--config", str(path), "--source", str(source), "--smoke-no-radiation"]
    first, second = tmp_path / "first", tmp_path / "second"
    assert c.main(args + ["--output", str(first), "--stop-after-days", str(60/c.DAY)]) == 0
    restart = first / "restarts/restart_00000060.nc"
    # Emulate an old accepted state with positive vapor below the dynamics floor.
    with Dataset(restart, "r+") as ds:
        water = ds["air_atmosphere_water_content"]
        water[0, 0, 2] += water[0, 0, 0] - 1e-20
        water[0, 0, 0] = 1e-20
    assert c.main(args + ["--resume", str(restart), "--output", str(second),
                         "--stop-after-days", str(120/c.DAY)]) == 0
    assert "WARNING Day" in (second / "run.log").read_text()
    assert "native vapor floor funded during restart; interfaces=1" in (second / "run.log").read_text()
    ledger = c.read_checkpoint(second / "restarts/restart_00000120.nc")["ledger"]
    assert ledger["interface_vapor_correction_events"] >= 1 and ledger["interface_vapor_correction_kg_m2"] > 0
    with Dataset(second / "diagnostics.nc") as ds:
        for key in ("interface_vapor_correction_events", "interface_vapor_correction_kg_m2"):
            assert ds[key][0] > 0 and ds[key][-1] == ledger[key]
