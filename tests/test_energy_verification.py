"""Test measurement independence and the observer, not scientific thresholds."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("energy_verification", Path(__file__).resolve().parents[1] / "tools/run_energy_verification.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


@pytest.fixture
def case():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    return driver.make_case("sharp", 16, 3, 20261030)


def test_observer_preserves_step_and_records_actual_active_ledgers(case):
    model, state = case
    original = driver.archive(driver.payload(state))
    expected = model(state, 1.)
    observed, rows, raw = driver.record_stages(model, state, 1.)
    assert len(rows) == 3
    assert all(row["dt_s"] == 1. and row["exact_water_ledger"] for row in rows)
    assert all(row["limited_faces"] > 0 for row in rows)
    for key in driver.fields(state):
        assert torch.equal(getattr(expected, key), getattr(observed, key))
        assert torch.equal(getattr(state, key), original[key])
        assert torch.equal(raw[1]["initial"][key], raw[0]["euler_output"][key])
    assert "euler_with_info" not in model.stepper.transport.__dict__


@pytest.mark.parametrize("bad", ["water", "count", "beta", "nan"])
def test_stage_audit_rejects_corrupt_observation(case, bad):
    model, state = case
    transport = model.stepper.transport
    updated, info = transport.euler_with_info(state, 1.)
    if bad == "water":
        updated = driver.state_with(updated, water_content=updated.water_content+1.)
    elif bad == "count":
        info = dict(info, limited_faces=-1)
    else:
        beta = info["beta"].clone()
        beta[0] = -1 if bad == "beta" else float("nan")
        info = dict(info, beta=beta)
    with pytest.raises(ValueError):
        driver.stage_budget(model.dynamics, state, updated, info, 1., transport.dry_floor, transport.vapor_floor)


def test_incidence_audit_is_sensitive_to_corrupt_flux(case):
    model, state = case
    transport = model.stepper.transport
    updated, info = transport.euler_with_info(state, 1.)
    original, _ = driver.stage_budget(model.dynamics, state, updated, info, 1., transport.dry_floor, transport.vapor_floor)
    flux = info["integrated_flux"].clone()
    flux[0, 1] += 1e12
    changed, _ = driver.stage_budget(model.dynamics, state, updated, dict(info, integrated_flux=flux), 1., transport.dry_floor, transport.vapor_floor)
    assert changed["species_incidence_max_error_kg"][1] > original["species_incidence_max_error_kg"][1] + 1e10
    assert changed["species_flux_sum_minus_carrier_max_kg"] > original["species_flux_sum_minus_carrier_max_kg"] + 1e10


def test_derivative_reference_does_not_reuse_analytic_gradient(monkeypatch):
    torch.set_default_dtype(torch.float64)
    model, state = driver.make_case("smooth", 16, 2, 71)
    reference, directions = driver.derivative_sweeps(model, state, [1e-3], 1, 17)
    original = driver.analytic_work
    monkeypatch.setattr(driver, "analytic_work", lambda *args: original(*args)+1e15)
    altered, again = driver.derivative_sweeps(model, state, [1e-3], 1, 17)
    assert directions.keys() == again.keys()
    for before, after in zip(reference, altered):
        assert before["ad_J"] == after["ad_J"]
        assert before["fd_J"] == after["fd_J"]
        if before["energy"] == "total":
            assert after["analytic_minus_ad_J"] > before["analytic_minus_ad_J"] + 9e14


def test_component_audit_detects_tendency_decomposition_error(case, monkeypatch):
    model, state = case
    reference, _ = driver.work_closure(model, state, 1.)
    original = model.dynamics.tendency_terms
    def wrong(s):
        terms = original(s)
        terms["u_e"]["pgf"] = terms["u_e"]["pgf"]+1.
        return terms
    monkeypatch.setattr(model.dynamics, "tendency_terms", wrong)
    altered, _ = driver.work_closure(model, state, 1.)
    assert reference["ad_total_W"] == altered["ad_total_W"]
    assert altered["assembly_max_error_native_per_s"]["u_e"] > .9


def test_failed_run_retains_manifest_and_refuses_overwrite(tmp_path, monkeypatch):
    common = driver.common
    monkeypatch.setattr(common, "verify_package", lambda *args: {})
    monkeypatch.setattr(common, "git", lambda *args: b"test-only")
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="deliberate"):
        with common.recorded_run(tmp_path, output, "test", {}, driver.__file__):
            common.write_json(output / "partial.json", {"partial": True})
            raise RuntimeError("deliberate")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "partial.json" in manifest["artifacts_sha256"]
    assert manifest["baseline_commit"] == common.BASELINE
    with pytest.raises(FileExistsError):
        with common.recorded_run(tmp_path, output, "test", {}, driver.__file__):
            pass
