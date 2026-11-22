"""Local funding of shared GM circulations on a finite-amplitude ocean front."""
from dataclasses import fields, replace
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "validation/temperature_coordinate")]
from eddy_campaign import make
from eddy import GMRedi
from eddy_energy import EddyMixingLength
import gm_direct_funding
from gm_parcel_limiter import make_experimental_gm_class
from vfs.ocean.transport import divergence


@pytest.fixture
def front():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        yield make(front="cold", eddy_energy=1e-4)
    finally:
        torch.set_default_dtype(previous)


def operators(*, prescribed=False, redi=100.0):
    params = dict(kappa_redi=redi, positive_redi=True)
    if prescribed:
        params["kappa_gm"] = 100.0
    else:
        params["energy_closure"] = EddyMixingLength(10000.0, 0.1, 0.5)
    events = []
    direct = gm_direct_funding.make_direct_gm_class(GMRedi, divergence, events=events)
    return GMRedi(**params), direct(**params), events, params


def deplete(state, sites):
    local = state.eddy_h.clone()
    for cell, layer in sites:
        local[cell, layer] = 1e-8 * state.ocean.h[cell, layer]
    return replace(state, eddy_h=local)


def check_identical(left, right):
    a, report_a = left
    b, report_b = right
    for name in ("u_e", "h", "temp_h", "salt_h", "tracer_h"):
        assert torch.equal(getattr(a.ocean, name), getattr(b.ocean, name))
    assert torch.equal(a.eddy_h, b.eddy_h)
    assert torch.equal(a.tke_h, b.tke_h)
    for field in fields(report_a):
        x, y = getattr(report_a, field.name), getattr(report_b, field.name)
        if torch.is_tensor(x):
            assert torch.equal(x, y), field.name
        else:
            assert x == y, field.name


def check_budget(model, before, after, report):
    # Independent finite-work ledger across the two accepted GM and Redi updates.
    inflow = report.gm_energy_flux_divergence + report.redi_energy_flux_divergence
    mass = model.hamiltonian.mass_area(before.ocean)
    specific_residual = (model.hamiltonian.finite_change(before, after) - inflow) / (mass * before.ocean.h)
    assert float(specific_residual.abs().max()) < 1e-12
    assert torch.equal(before.ocean.h, after.ocean.h)
    assert (after.eddy_h >= 0).all()
    assert (after.tke_h >= 0).all()
    assert (after.ocean.salt_h >= 0).all()
    assert (after.ocean.tracer_h >= 0).all()
    eps = torch.finfo(before.ocean.h.dtype).eps
    transport_scale = report.horizontal_transport.abs().max() + report.vertical_transport.abs().max()
    assert report.volume_residual.abs().max() <= 32 * eps * transport_scale
    for name in ("temp_h", "salt_h", "tracer_h"):
        a, b = getattr(before.ocean, name), getattr(after.ocean, name)
        weight = mass.unsqueeze(-1) if a.ndim == 3 else mass
        assert (weight * (b - a)).sum().abs() <= 64 * eps * (weight * a.abs()).sum()


def test_funded_front_is_native_bit_for_bit(front):
    model, state = front
    native, direct, events, _ = operators()
    check_identical(native.apply(model, state, 60.0), direct.apply(model, state, 60.0))
    assert not events


@pytest.mark.parametrize("sites,patch_count", [
    ([(13, 5), (13, 6)], 1),    # Two depleted layers share a streamfunction interface.
    ([(13, 1), (13, 6)], 2),    # Well-separated in the same water column.
])
def test_overlapping_and_disjoint_funding(front, sites, patch_count):
    model, state = front
    state = deplete(state, sites)
    native, direct, events, _ = operators(redi=0.0)
    with pytest.raises(FloatingPointError, match="GM requires unavailable"):
        native.apply(model, state, 60.0)
    after, report = direct.apply(model, state, 60.0)
    assert len(events) == 1 and events[0]["patch_count"] == patch_count
    assert not events[0]["fallback"]
    check_budget(model, state, after, report)


def test_donor_reversal_is_checked_on_the_new_circulation(front):
    model, state = front
    state = deplete(state, [(0, 0)])
    native, direct, events, _ = operators(prescribed=True)
    original_vertical = native._geometry(model, state, part="gm")[3]
    after, report = direct.apply(model, state, 60.0)
    assert events
    assert ((original_vertical * report.vertical_transport < 0)
            & (original_vertical.abs() > 1e-9) & (report.vertical_transport.abs() > 1e-9)).any()
    check_budget(model, state, after, report)


def test_unresolved_patch_preserves_checked_halving(front, monkeypatch, tmp_path):
    model, state = front
    state = deplete(state, [(13, 6)])
    _, direct, events, params = operators()
    monkeypatch.setattr(gm_direct_funding, "solve_patch_scale", lambda *args, **kwargs: None)
    halving = make_experimental_gm_class(GMRedi, divergence, tmp_path / "events.jsonl")(**params)
    check_identical(direct.apply(model, state, 60.0), halving.apply(model, state, 60.0))
    assert len(events) == 1 and events[0]["fallback"]

def test_direct_proposal_must_also_fund_redi(front, tmp_path):
    model, state = front
    state = deplete(state, [(13, 5), (13, 6)])
    _, direct, events, params = operators(redi=100.0)
    halving = make_experimental_gm_class(GMRedi, divergence, tmp_path / "events.jsonl")(**params)
    accepted = direct.apply(model, state, 60.0)
    check_identical(accepted, halving.apply(model, state, 60.0))
    assert events[0]["fallback"] and "Redi" in events[0]["fallback_reason"]
    check_budget(model, state, *accepted)
