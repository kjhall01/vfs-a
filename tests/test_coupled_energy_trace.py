"""Isolated trace-ledger tests; do not require PyAM21 or radiation data."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "workflows" / "coupled" / "energy_trace.py"
spec = importlib.util.spec_from_file_location("vfs_energy_trace_standalone", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
EnergyTrace = module.EnergyTrace


def state(energy=100.0):
    return SimpleNamespace(energy=energy, water=8.0, salt=3.0,
                           speed=0.1, time_seconds=0)


def measure(_adapter, air, _receiver, _sea):
    return dict(energy=air.energy, water=air.water, salt=air.salt,
                atmosphere_energy=air.energy / 2,
                ocean_energy=air.energy / 4,
                ice_snow_energy=air.energy / 4)


def summarize(_adapter, air, _receiver, _sea):
    return dict(max_ocean_speed_m_s=air.speed)


def test_trace_matches_external_upstream_and_component_changes(tmp_path):
    path = tmp_path / "trace.jsonl"
    air = state()
    trace = EnergyTrace(path, measure, summarize)
    trace.begin(None, air, None, None)
    air.energy = 104.0
    air.time_seconds = 600
    assert trace.observe("dynamics", None, air, None, None) == 4
    air.energy = 110.0
    air.time_seconds = 1200
    assert trace.observe("exchange", None, air, None, None,
                         external=5.0, upstream=0.5) == 0.5
    assert trace.commit(dict(initial=dict(energy=100.0), external=5.0, upstream=0.5),
                        seconds=air.time_seconds) == 4.5
    trace.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["begin", "stage", "stage", "commit"]
    assert rows[1]["component_delta_J_m2"]["ocean_energy"] == 1.0
    assert rows[2]["stage_numerical_residual_J_m2"] == 0.5
    assert rows[2]["interval_residual_J_m2"] == 4.5
    assert rows[-1]["campaign_cumulative_numerical_residual_J_m2"] == 4.5


def test_partial_failed_interval_is_not_marked_committed(tmp_path):
    path = tmp_path / "trace.jsonl"
    air = state()
    trace = EnergyTrace(path, measure, summarize, substeps=True)
    trace.begin(None, air, None, None)
    air.energy = 102.0
    trace.observe("gmredi", None, air, None, None)
    trace.close()
    assert [json.loads(line)["event"] for line in path.read_text().splitlines()] == [
        "begin", "stage"
    ]


def test_nonfinite_energy_rejected(tmp_path):
    air = state()
    trace = EnergyTrace(tmp_path / "trace.jsonl", measure, summarize)
    trace.begin(None, air, None, None)
    air.energy = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite energy ledger"):
        trace.observe("bad", None, air, None, None)
    trace.close()
