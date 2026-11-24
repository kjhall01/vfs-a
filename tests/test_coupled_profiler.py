"""Measurement fidelity and failure semantics of the standalone coupled profiler.

Reuses the real PyAM21 synthetic-radiation fixture already exercised by the
coupled campaign suite. No physical components or installed VFS files altered.
"""
from dataclasses import replace
from pathlib import Path
import json
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
pytest.importorskip("pyam21")
from tools.profile_coupled import SectionClock, Instrument, run_profile, _config
from workflows.coupled import campaign as c
from test_coupled_campaign import source, config, synthetic


def test_nested_exclusive_timing_and_inclusive_accounting():
    timer = SectionClock("cpu")
    with timer.section("root"):
        with timer.section("child"):
            sum(range(10))
    result = timer.results()
    assert result["root"]["inclusive"]["calls"] == 1
    assert result["child"]["inclusive"]["calls"] == 1
    assert result["root"]["inclusive"]["total_s"] >= result["child"]["inclusive"]["total_s"]
    assert (
        abs(result["root"]["exclusive"]["total_s"]
            + result["child"]["inclusive"]["total_s"]
            - result["root"]["inclusive"]["total_s"]) < 1e-6
    )


def test_config_override_validation(tmp_path):
    import argparse
    import yaml

    config_path = tmp_path / "pilot.yaml"
    config_path.write_text(yaml.safe_dump(dict(
        physics_dt=2400, atmosphere_dt=600, ocean_dt=600,
        radiation_dt=4800, mixing_dt=1200, mesoscale_dt=1200,
        ale_dt=4800, ocean_dynamics_integrator="ssprk43",
    )))
    args = argparse.Namespace(
        config=config_path, set=["atmosphere_dt=800", "ocean_dt=1200"],
        device=None, dtype=None, threads=None, compile=None,
    )
    value = _config(args)
    assert value.atmosphere_dt == 800
    assert value.ocean_dt == 1200
    assert value.ocean_dynamics_integrator == "ssprk43"
    args.set = ["some_fake_control=1"]
    with pytest.raises(ValueError, match="unknown campaign field"):
        _config(args)


def test_real_coupled_profile_has_nested_operator_counters(
    source, config, tmp_path
):
    short = replace(
        config, physics_dt=120, ocean_dt=60, atmosphere_dt=15,
        radiation_dt=120, mixing_dt=120, mesoscale_dt=120,
        ale_dt=240, gm_redi=True, eddy_dissipation=True,
    ).validate()
    with patch("workflows.pyam21.radiation.build_radiation", side_effect=synthetic):
        adapter, air, receiver, sea = c.build(short, source, Path("synthetic"))

    original = c.advance
    with torch.no_grad():
        # Check returning/committing the same physical evolution, within
        # deterministic PyTorch exactness, with and without observational wrappers.
        expected_air, expected_sea, expected_report = original(
            adapter, air, receiver, sea, short.physics_dt, short
        )
        result, measured_air, measured_sea = run_profile(
            adapter, air, receiver, sea, short,
            output=tmp_path, identity="test-profile-only",
            warmup_intervals=0, measure_intervals=1
        )
    assert measured_air.time_seconds == expected_air.time_seconds
    assert torch.equal(
        measured_sea.coupled.ocean.ocean.h, expected_sea.coupled.ocean.ocean.h
    )
    assert torch.equal(
        measured_air.atmosphere.ps, expected_air.atmosphere.ps
    )
    assert result["stages"]["coupled_interval"]["inclusive"]["calls"] == 1
    assert result["stages"]["atmosphere_dynamics"]["inclusive"]["calls"] == 8
    assert result["stages"]["ocean_ice_dynamics"]["inclusive"]["calls"] == 2
    assert result["stages"]["ocean_vertical_mixing"]["inclusive"]["calls"] == 1
    assert result["stages"]["mesoscale"]["inclusive"]["calls"] == 1
    assert result["stages"]["shared_surface_exchange"]["inclusive"]["calls"] == 1
    assert result["stages"]["PyAM21_total"]["inclusive"]["calls"] == 1
    assert result["stages"]["GM_Redi_internal_substep"]["inclusive"]["calls"] == short.eddy_substeps
    assert result["throughput"]["SYPD"] > 0
    assert (tmp_path / "profile.json").exists()
    assert (tmp_path / "summary.txt").exists()
    assert len((tmp_path / "intervals.jsonl").read_text().splitlines()) == 1

    # Every observational patch must be removed when profiling exits.
    assert c.advance is original
    before = getattr(type(adapter.physics), "down")
    with Instrument(SectionClock("cpu"), adapter, receiver):
        assert c.advance is original  # advance is not wrapped itself
    assert getattr(type(adapter.physics), "down") is before


def test_profile_does_not_touch_numerical_source_hashes():
    # The profiler must not appear in a source fingerprint and cannot change
    # an existing numeric checkpoint's identity.
    hashes = c.source_hashes()
    assert not any(key.endswith("profile_coupled.py") for key in hashes)
