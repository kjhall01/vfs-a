"""Non-invasive nested RHS instrumentation through the real coupled step."""
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
import torch

pytest.importorskip("pyam21")
sys.path.append(str(Path(__file__).resolve().parents[1]))
from workflows.coupled import campaign as c
from tools.profile_coupled import SectionClock
from tools.profile_ocean_rhs import RHSInstrument
from test_coupled_campaign import source, config, synthetic


def test_ocean_rhs_instrument_preserves_coupled_state_and_covers_real_rhs(
    source, config, tmp_path
):
    with patch("workflows.pyam21.radiation.build_radiation", side_effect=synthetic):
        adapter, air, receiver, sea = c.build(config, source, Path("synthetic"))
    device = adapter.model.mesh.cell_area.device
    with torch.no_grad():
        a1, s1, r1 = c.advance(adapter, air, receiver, sea,
                               config.physics_dt, config)
        clock = SectionClock(device)
        with RHSInstrument(clock, adapter, receiver):
            a2, s2, r2 = c.advance(adapter, air, receiver, sea,
                                   config.physics_dt, config)
    for field in ("u_e", "ps", "entropy_content", "water_content"):
        assert torch.equal(
            getattr(a1.atmosphere, field), getattr(a2.atmosphere, field)
        ), field
    for field in ("h", "u_e", "temp_h", "salt_h", "tracer_h"):
        assert torch.equal(
            getattr(s1.coupled.ocean.ocean, field),
            getattr(s2.coupled.ocean.ocean, field)
        ), field
    assert torch.equal(s1.coupled.ice.area, s2.coupled.ice.area)
    assert r1 == r2
    counts = {k: v["inclusive"]["calls"] for k, v in clock.results().items()}
    for region in (
        "ocean_ice_rhs", "rhs_coupled_ocean_ice",
        "rhs_ocean_temperature_tendency", "rhs_temperature_entropy_fields",
        "rhs_conservative_transport_total", "rhs_parent_gibbs_transport",
        "rhs_ice_transport", "rhs_ice_ridging",
    ):
        assert counts.get(region, 0) > 0, (region, counts)
    assert counts["rhs_coupled_ocean_ice"] == counts["ocean_ice_rhs"]
    assert counts["rhs_ocean_temperature_tendency"] == counts["ocean_ice_rhs"]


def test_optional_transport_source_line_trace_observes_real_coupled_update(
    source, config, tmp_path
):
    """Measure line-level transport work in a separate, uncommitted interval."""
    from tools.profile_ocean_rhs import _profile_line_pass

    with patch("workflows.pyam21.radiation.build_radiation", side_effect=synthetic):
        adapter, air, receiver, sea = c.build(config, source, Path("synthetic"))
    with torch.no_grad():
        expected_air, expected_sea, _ = c.advance(
            adapter, air, receiver, sea, config.physics_dt, config
        )
    trace = _profile_line_pass(
        dict(adapter=adapter, air=air, receiver=receiver, sea=sea,
             config=config, output=tmp_path), 1
    )
    assert trace["measured_intervals"] == 1
    assert trace["model_start_seconds"] == air.time_seconds
    assert trace["model_end_seconds"] == expected_air.time_seconds
    assert (tmp_path / "ocean_transport_lines.json").exists()
    assert (tmp_path / "ocean_transport_lines.txt").exists()
    regions = {x["region"] for x in trace["source_lines"]}
    assert {"local.evaluate", "parent.evaluate", "parent.bounded_transport",
            "local._local_target_fields"} <= regions
    assert all(row["elapsed_s"] >= 0 and row["calls"] > 0
               for row in trace["source_lines"])
    assert all(row["source"] for row in trace["source_lines"])


def test_deeper_profiler_observes_donor_and_parent_limiting(
    source, config
):
    with patch("workflows.pyam21.radiation.build_radiation", side_effect=synthetic):
        adapter, air, receiver, sea = c.build(config, source, Path("synthetic"))
    with torch.no_grad():
        timer = SectionClock(adapter.model.mesh.cell_area.device)
        with RHSInstrument(timer, adapter, receiver):
            c.advance(adapter, air, receiver, sea,
                      config.physics_dt, config)
    calls = {key: d["inclusive"]["calls"]
             for key, d in timer.results().items()}
    assert calls["rhs_parent_admissibility_limiter"] > 0
    assert calls["rhs_parent_flux_divergence"] > 0
    assert calls["rhs_local_flux_divergence"] > 0
    assert calls["rhs_Gibbs_face_traces"] > 0
    assert calls["rhs_entropy_domain_validation"] > 0
    assert calls["rhs_temperature_ocean_check"] > 0
    assert calls["rhs_ridge_validation"] > 0
    assert calls["rhs_caloric_reference_evaluate"] > 0


def test_nested_temperature_validation_reuse_is_exact(source, config, monkeypatch):
    """The same coupled SSP stage has one owner of full ocean validation.

    Reference path reinstates both redundant checks, including every caloric
    reference evaluation. Compare all prognostic tensors, physical source
    reports and the number of temperature/entropy field evaluations.
    """
    from dataclasses import fields, is_dataclass
    from vfs.ocean.state import OceanState
    with patch("workflows.pyam21.radiation.build_radiation", side_effect=synthetic):
        adapter, air, receiver, sea = c.build(config, source, Path("synthetic"))
    snow = receiver.snow
    coupled = snow.coupled
    ridge = coupled.ridge
    core = coupled.ocean.dynamics
    original_fields = core.fields
    original_snow_tendency = snow.tendency
    original_ridge = ridge.tendency
    calls = []

    def counted(state):
        calls.append(1)
        return original_fields(state)

    def exact(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, OceanState):
            assert a.mesh is b.mesh and a.column is b.column
            for key in ("u_e", "h", "temp_h", "salt_h", "tracer_h"):
                exact(getattr(a, key), getattr(b, key))
        elif is_dataclass(a):
            for field in fields(a):
                if field.name not in ("mesh", "column"):
                    exact(getattr(a, field.name), getattr(b, field.name))
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                exact(x, y)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                exact(a[key], b[key])
        else:
            assert a == b

    def legacy_snow_tendency(state, dt, **options):
        snow.check(state)
        return original_snow_tendency(state, dt, **options)

    def legacy_ridge(state, *, _validated=False):
        return original_ridge(state, _validated=False)

    with torch.no_grad():
        with monkeypatch.context() as p:
            p.setattr(core, "fields", counted)
            p.setattr(snow, "tendency", legacy_snow_tendency)
            p.setattr(ridge, "tendency", legacy_ridge)
            old_air, old_sea, old_report = c.advance(
                adapter, air, receiver, sea, config.physics_dt, config
            )
            old_count = len(calls)

        calls.clear()
        with monkeypatch.context() as p:
            p.setattr(core, "fields", counted)
            new_air, new_sea, new_report = c.advance(
                adapter, air, receiver, sea, config.physics_dt, config
            )
            new_count = len(calls)

    exact(old_air.atmosphere, new_air.atmosphere)
    exact(old_sea, new_sea)
    exact(old_report, new_report)
    assert new_count < old_count, (old_count, new_count)
    # Each ocean–ice RHS stage removes exactly one full check from snow and
    # one from ridging, without changing any RK stage or endpoint validation.
    assert old_count - new_count == 2 * (config.physics_dt // config.ocean_dt) * (
        4 if config.ocean_dynamics_integrator == "ssprk43" else 3
    )


def test_public_ridging_still_validates_state(source, config, monkeypatch):
    with patch("workflows.pyam21.radiation.build_radiation", side_effect=synthetic):
        adapter, air, receiver, sea = c.build(config, source, Path("synthetic"))
    ridge = receiver.snow.coupled.ridge
    called = []
    original = ridge.check

    def checked(state):
        called.append(True)
        return original(state)

    with monkeypatch.context() as p:
        p.setattr(ridge, "check", checked)
        with torch.no_grad():
            ridge.tendency(sea.coupled)
    assert called == [True]
