"""Independent Hamiltonian, local first-law and whole-model reservoir checks."""

import sys
from pathlib import Path
import pytest
import torch
from dataclasses import replace

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "validation/temperature_coordinate"),
    str(Path(__file__).resolve().parents[2] / "validation/ocean_column"),
]
from eddy_energy import EddyState
from eddy import GMRedi
from eddy_campaign import make
from roquet import (
    RoquetOcean,
    RoquetMixing,
    RoquetHeating,
    RoquetExchange,
    TemperatureBoundary,
)
from vfs.ocean.stratified import TKEState
from test_jmd95_hamiltonian import mesh, case


@pytest.mark.parametrize("coast", [False, True])
def test_single_hamiltonian_all_state_derivatives(mesh, coast):
    old, s = case(mesh, coast)
    model = RoquetOcean(mesh, old.column, domain=old.domain)
    fields = ("u_e", "h", "temp_h", "salt_h")
    leaves = [getattr(s, n).clone().requires_grad_() for n in fields]
    s = s.with_tensors(*leaves, s.tracer_h)
    wet = (model.energy.mass_area(s) > 0).expand_as(s.h)
    kh = torch.where(wet, s.h.detach() * 1e-4, 0.0).requires_grad_()
    eh = torch.where(wet, s.h.detach() * 0.1, 0.0).requires_grad_()
    state = EddyState(s, kh, eh)
    model.check(state)
    actual = torch.autograd.grad(model.total_energy(state), (*leaves, kh, eh))
    resolved = model.dynamics.gradients(s)
    for name, g in zip(fields, actual[:4]):
        torch.testing.assert_close(g, resolved[name], rtol=3e-12, atol=0.1)
    for g in actual[-2:]:
        torch.testing.assert_close(
            g, model.energy.mass_area(s).expand_as(s.h), rtol=0, atol=0
        )
    terms = model.hamiltonian.terms(state)
    torch.testing.assert_close(terms["total"], model.total_energy(state))
    assert terms["mesoscale_eddy"] > terms["turbulent_kinetic"]


@pytest.mark.parametrize("front", ["cold", "warm"])
def test_redi_pays_actual_hamiltonian_work_in_both_directions(front):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    model, initial = make(front=front)
    end, r = GMRedi(0.0, 1000.0).apply(model, initial, 60.0)
    mass = model.energy.mass_area(initial.ocean) * initial.ocean.h
    # These are the published default EOS coefficients, not a modified test EOS.
    if front == "cold":
        assert r.redi_energy_change > 0 and r.reservoir_change < 0
    else:
        assert r.redi_energy_change < 0 and r.reservoir_change > 0
    eps = torch.finfo(mass.dtype).eps
    assert (
        abs(r.energy_change) / mass.sum()
        < 64 * eps * initial.specific_eddy_energy().max()
    )
    assert (
        r.redi_local_energy_residual.abs() / mass
    ).max() < 64 * eps * initial.specific_eddy_energy().max()
    # Removing the reservoir term must expose a nonzero energy change.
    assert (
        model.hamiltonian.finite_change(initial, end, include_eddy=False).sum().abs()
        > 1e6 * r.energy_change.abs()
    )
    assert (end.eddy_h >= 0).all()


def test_insufficient_local_reservoir_rejects_without_mutation():
    model, initial = make(front="cold", eddy_energy=0.0)
    original = [
        x.clone() for x in (initial.ocean.temp_h, initial.ocean.salt_h, initial.eddy_h)
    ]
    with pytest.raises(FloatingPointError, match="unavailable local eddy energy"):
        GMRedi(0.0, 1000.0).apply(model, initial, 60.0)
    for x, y in zip(
        original, (initial.ocean.temp_h, initial.ocean.salt_h, initial.eddy_h)
    ):
        assert torch.equal(x, y)
    # Energy elsewhere in the domain cannot silently pay a local deficit.
    funded = initial.eddy_h.clone()
    funded[:1] = 1e12
    with pytest.raises(FloatingPointError, match="unavailable local eddy energy"):
        GMRedi(1000.0, 0.0).apply(model, replace(initial, eddy_h=funded), 1.0)


def test_redi_refreshes_slopes_after_gm_and_preserves_density_nullspace():
    model, initial = make()
    gm, _ = GMRedi(1000.0, 0.0).apply(model, initial, 3600.0)
    both, r = GMRedi(1000.0, 1000.0).apply(model, initial, 3600.0)
    expected = GMRedi(0.0, 1000.0)._geometry(model, gm)[0]
    torch.testing.assert_close(r.redi_slope, expected, rtol=0, atol=0)
    assert (r.redi_slope - r.slope).abs().max() > 1e-9
    torch.testing.assert_close(both.ocean.salt_h, gm.ocean.salt_h, rtol=0, atol=0)
    assert (both.ocean.tracer_h - gm.ocean.tracer_h).abs().max() > 0


def test_reservoir_survives_forcing_mixing_dynamics_and_ale():
    from complete_campaign import make as forced_case

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    old, _, s, k, forcing = forced_case(16, 4, False)
    model = RoquetOcean(
        old.mesh,
        old.column,
        closure=RoquetMixing(2.0, substeps=2),
        boundary=TemperatureBoundary(0.0025),
        heating=RoquetHeating(0.6, (0.5, 20.0)),
        exchange=RoquetExchange(),
    )
    initial = model.with_eddy_reservoir(
        TKEState(s, s.h * k), 0.1 + 0.01 * old.mesh.cell_xyz[:, :1]
    )
    state, report = model.step_with_report(initial, 10.0, **forcing)
    area_mass = model.energy.mass_area(s)
    torch.testing.assert_close(
        (area_mass * state.eddy_h).sum(),
        (area_mass * initial.eddy_h).sum(),
        rtol=3e-15,
        atol=0,
    )
    state, eddy_report = GMRedi(1000.0, 1000.0).apply(model, state, 10.0)
    target = state.ocean.h.sum(-1, keepdim=True).expand_as(state.ocean.h) / 4
    final, _ = model.remap_with_report(state, target)
    torch.testing.assert_close(
        (area_mass * final.eddy_h).sum(),
        (area_mass * state.eddy_h).sum(),
        rtol=3e-15,
        atol=0,
    )
    external = (
        report.boundary.surface_work
        + report.heating.external_heat
        + report.exchange.external_energy
    )
    mass = (area_mass * s.h).sum()
    # Existing RK/physics/ALE tolerances, not an assertion of exact time integration.
    assert (
        abs(model.total_energy(final) - model.total_energy(initial) - external) / mass
        < 1e-7
    )
    assert isinstance(final, EddyState) and (final.eddy_h >= 0).all()
    assert final.ocean.tracer_h.shape == initial.ocean.tracer_h.shape


def test_optional_state_no_passives_identity_and_invalid_reservoir(mesh):
    old, s = case(mesh)
    m = RoquetOcean(mesh, old.column)
    state = m.state(
        u_e=s.u_e,
        h=s.h,
        conservative_temperature=s.temperature(),
        absolute_salinity=s.salinity(),
        eddy_energy=0.1,
    )
    assert state.ocean.tracer_h is None
    end, _ = m.step_with_report(state, 1.0)
    end, _ = m.remap_with_report(end, end.ocean.h)
    assert end.ocean.tracer_h is None and isinstance(end, EddyState)
    end, r = GMRedi().apply(m, state, 1.0)
    assert torch.equal(end.eddy_h, state.eddy_h) and r.energy_change == 0
    with pytest.raises(TypeError, match="explicit EddyState"):
        GMRedi().apply(m, TKEState(state.ocean, state.tke_h), 1.0)
    with pytest.raises(ValueError, match="finite/nonnegative"):
        m.check(replace(state, eddy_h=-state.eddy_h))
    with pytest.raises(ValueError, match="unchanged geometry"):
        m.hamiltonian.finite_change(
            state,
            replace(
                state, ocean=s.with_tensors(s.u_e, s.h * 1.01, s.temp_h, s.salt_h, None)
            ),
        )


def test_one_step_reservoir_gradient_against_finite_difference():
    model, initial = make()
    eddy = GMRedi(1000.0, 1000.0)
    x = initial.ocean.mesh.cell_xyz[:, :1]

    def loss(alpha):
        s = initial.ocean
        changed = replace(
            initial,
            ocean=s.with_tensors(
                s.u_e, s.h, s.temp_h, s.salt_h + alpha * s.h * x, s.tracer_h
            ),
        )
        end, _ = eddy.apply(model, changed, 60.0)
        return (x * end.specific_eddy_energy()).mean()

    alpha = torch.tensor(0.0, requires_grad=True)
    (analytic,) = torch.autograd.grad(loss(alpha), alpha)
    step = 1e-4
    fd = (loss(torch.tensor(step)) - loss(torch.tensor(-step))) / (2 * step)
    torch.testing.assert_close(analytic, fd, rtol=1e-5, atol=1e-10)
