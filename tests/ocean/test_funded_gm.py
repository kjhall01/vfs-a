"""Donor energy transport and local funding at a finite-amplitude ocean front."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "validation/temperature_coordinate"))
from vfs.ocean.temperature.eddy import GMRedi, LocalEddyEnergyError
from eddy_campaign import make
from vfs.ocean.temperature.eddy_energy import EddyMixingLength
from vfs.ocean.transport import divergence


@pytest.fixture
def front():
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        yield make(front="cold", eddy_energy=1e-4)
    finally:
        torch.set_default_dtype(dtype)


def operator(scheme):
    return GMRedi(
        gm_energy_scheme=scheme, energy_closure=EddyMixingLength(10000, 0.1, 0.5),
        positive_redi=True, kappa_redi=100.0,
    )


@pytest.mark.parametrize("scheme", ["funded_donor_parcel", "quadratic_funded_donor_parcel"])
def test_unlimited_donor_energy_from_independent_parcel_quadrature(front, scheme):
    model, state = front
    strict = operator("donor_parcel")
    dt = 60.0
    end, report = strict.apply(model, state, dt)
    funded, accepted = operator(scheme).apply(model, state, dt)
    for name in ("temp_h", "salt_h", "tracer_h"):
        assert torch.equal(getattr(end.ocean, name), getattr(funded.ocean, name))
    assert torch.equal(end.eddy_h, funded.eddy_h)
    assert torch.equal(report.gm_energy_flux_divergence, accepted.gm_energy_flux_divergence)
    assert accepted.gm_limiting_passes == accepted.gm_limited_interfaces == 0

    # Independent two-node quadrature at both face-adjacent layer geometries.
    ocean = state.ocean
    geom = strict._geometry(model, state, part="gm")
    i, j, carrier = geom[4:7]
    donor = torch.where(carrier >= 0, i, j)
    ct = ocean.temperature().flatten()[donor]
    sa = ocean.salinity().flatten()[donor]
    base = model.ocean.energy.base
    z, h = base.height(ocean).flatten(), ocean.h.flatten()
    nodes, weights = base._quadrature(ct)
    potential = torch.zeros_like(ct)
    for adjacent in (i, j):
        points = z[adjacent, None] + h[adjacent, None] * (nodes - 0.5)
        psi, *_ = base.parcel_potential(ct[:, None], sa[:, None], points)
        potential = potential + 0.5 * (psi * weights).sum(-1)
    energy = model.config.cp * (ct - model.config.CT0) + potential
    energy = energy + ((state.tke_h + state.eddy_h) / ocean.h).flatten()[donor]
    influx = divergence(i, j, ocean.column.rho0 * dt * carrier * energy, h.numel())
    torch.testing.assert_close(report.gm_energy_flux_divergence.flatten(), influx, rtol=2e-12, atol=2e-6)


def test_undersupplied_patch_limits_only_incident_streamfunctions(front):
    model, state = front
    energy = state.eddy_h.clone()
    energy[13, 6] = 1e-8 * state.ocean.h[13, 6]
    state = replace(state, eddy_h=energy)
    strict = operator("donor_parcel")
    with pytest.raises(LocalEddyEnergyError, match="GM requires unavailable") as failure:
        strict.apply(model, state, 60.0)
    assert failure.value.bad.nonzero().tolist() == [[13, 6]]

    end, report = operator("funded_donor_parcel").apply(model, state, 60.0)
    geom = strict._geometry(model, state, part="gm")
    i, j = strict._metrics(model, state)[3:5]
    adjacent = failure.value.bad[i] | failure.value.bad[j]
    incident = adjacent[:, :-1] | adjacent[:, 1:]
    initial_psi, accepted = geom[1][:, 1:-1], report.streamfunction[:, 1:-1]
    assert report.gm_limiting_passes > 0 and report.gm_limited_interfaces == int(incident.sum())
    assert torch.equal(initial_psi[~incident], accepted[~incident])
    assert (accepted[incident].abs() < initial_psi[incident].abs()).any()
    assert torch.equal(end.ocean.h, state.ocean.h)
    assert (end.specific_eddy_energy() >= 0).all()
    assert (end.ocean.tracer_h >= 0).all()
    scale = report.horizontal_transport.abs().max() + report.vertical_transport.abs().max()
    assert report.volume_residual.abs().max() <= 32 * torch.finfo(scale.dtype).eps * scale
    inflow = report.gm_energy_flux_divergence + report.redi_energy_flux_divergence
    mass = model.hamiltonian.mass_area(state.ocean) * state.ocean.h
    specific_error = (model.hamiltonian.finite_change(state, end) - inflow) / mass
    assert specific_error.abs().max() < 1e-12


@pytest.mark.parametrize("gm,redi,message", [(1e20, 0, "GM outflow"), (0, 1e20, "Redi explicit")])
@pytest.mark.parametrize("scheme", ["funded_donor_parcel", "quadratic_funded_donor_parcel"])
def test_funding_retains_explicit_stability_checks(front, gm, redi, message, scheme):
    model, state = front
    with pytest.raises(FloatingPointError, match=message):
        GMRedi(gm, redi, gm_energy_scheme=scheme).apply(model, state, 60)
    with pytest.raises(ValueError, match="Unknown GM energy scheme"):
        GMRedi(gm_energy_scheme="unknown")


@pytest.mark.parametrize("layers", [[6], [5, 6]])
def test_quadratic_funding_preserves_single_and_adjacent_layer_updates(front, layers):
    model, state = front
    energy = state.eddy_h.clone()
    energy[13, layers] = 1e-8 * state.ocean.h[13, layers]
    state = replace(state, eddy_h=energy)
    reference, halved = operator("funded_donor_parcel").apply(model, state, 60)
    accepted, report = operator("quadratic_funded_donor_parcel").apply(model, state, 60)
    assert 1 == report.gm_limiting_passes < halved.gm_limiting_passes
    for key in ("streamfunction", "gm_energy_flux_divergence", "redi_energy_flux_divergence",
                "gm_local_energy_residual", "redi_local_energy_residual"):
        assert torch.equal(getattr(report, key), getattr(halved, key))
    for key in ("temp_h", "salt_h", "tracer_h"):
        assert torch.equal(getattr(accepted.ocean, key), getattr(reference.ocean, key))
    assert torch.equal(accepted.tke_h, reference.tke_h)
    assert torch.equal(accepted.eddy_h, reference.eddy_h)


def test_rejected_quadratic_proposal_replays_the_original_halvings(front, monkeypatch):
    from vfs.ocean.temperature import eddy

    model, state = front
    energy = state.eddy_h.clone()
    energy[13, 6] = 1e-8 * state.ocean.h[13, 6]
    state = replace(state, eddy_h=energy)
    # Deliberately propose full strength again. The ordinary settlement must
    # reject it and resume at the first half with all twenty halvings available.
    def unreduced(model, before, after, failure, metrics):
        return before.eddy_h.new_ones(len(metrics[3]), before.ocean.h.shape[1] - 1)

    monkeypatch.setattr(eddy, "_funded_factors", unreduced)
    reference, halved = operator("funded_donor_parcel").apply(model, state, 60)
    actual, report = operator("quadratic_funded_donor_parcel").apply(model, state, 60)
    assert report.gm_limiting_passes == halved.gm_limiting_passes + 2
    assert torch.equal(report.streamfunction, halved.streamfunction)
    assert torch.equal(actual.eddy_h, reference.eddy_h)


def test_quadratic_budget_matches_full_donor_flux_and_finite_work(front):
    model, state = front
    step = operator("donor_parcel")
    h = state.ocean.h
    metrics = step._metrics(model, state)
    psi = step._geometry(model, state, part="gm", metrics=metrics)[1]
    volume = (state.ocean.mesh.cell_area[:, None] * h).flatten()
    initial = (step._contents(state) / h[..., None]).reshape(h.numel(), -1)
    mass = model.hamiltonian.mass_area(state.ocean)
    a = model.hamiltonian.base.coefficients(state.ocean)[0]

    # Make finite-amplitude transport so the nonlinear term is resolved.
    i, j, carriers = step._carriers(psi, metrics, h)[2:]
    outflow = (torch.zeros_like(volume).index_add(0, i, carriers.clamp_min(0))
               .index_add(0, j, (-carriers).clamp_min(0)))
    psi = psi * (0.25 / (60 * outflow / volume).max())

    def trial(scale):
        i, j, carriers = step._carriers(psi * scale, metrics, h)[2:]
        transported, inflow, _ = step._gm_transport(model, state, 60, initial, volume, i, j, carriers)
        work = model.hamiltonian.finite_change(state, transported)
        return transported, transported.eddy_h - (work - inflow) / mass

    full, remaining = trial(1.)
    quadratic = -a * (full.ocean.temp_h - state.ocean.temp_h).square() / h
    assert (quadratic / h).abs().max() > 1e-9
    # Forming the donor caloric flux and storing CT round at the cp*CT scale.
    ct_scale = (state.ocean.temperature() - model.config.CT0).abs().max().item()
    atol = 4 * torch.finfo(h.dtype).eps * model.config.cp * ct_scale
    for scale in (0.125, 0.5, 0.875):
        _, actual = trial(scale)
        predicted = (1 - scale) * state.eddy_h + scale * remaining + scale * (scale - 1) * quadratic
        torch.testing.assert_close(predicted / h, actual / h, atol=atol, rtol=0)


@pytest.mark.parametrize("scheme", ["funded_donor_parcel", "quadratic_funded_donor_parcel"])
def test_gm_funding_does_not_mask_an_unfunded_redi_step(front, scheme):
    model, state = front
    energy = state.eddy_h.clone()
    energy[13, 6] = 1e-9 * state.ocean.h[13, 6]
    with pytest.raises(LocalEddyEnergyError, match="Redi requires unavailable"):
        operator(scheme).apply(model, replace(state, eddy_h=energy), 60.0)


@pytest.mark.parametrize("specific_energy", [0., 1e-9, 1e-4])
def test_redi_can_reduce_only_rows_touching_a_depleted_layer(front, specific_energy):
    model, state = front
    energy = state.eddy_h.clone()
    h = state.ocean.h
    energy[13, 6] = specific_energy * h[13, 6]
    state = replace(state, eddy_h=energy)
    strict = GMRedi(kappa_redi=100., positive_redi=True)
    funded = GMRedi(kappa_redi=100., positive_redi=True, fund_redi=True)
    end, report = funded.apply(model, state, 60.)

    if specific_energy == 1e-4:
        reference, old = strict.apply(model, state, 60.)
        assert torch.equal(funded._contents(end), strict._contents(reference))
        for key in ("redi_energy_flux_divergence", "redi_local_energy_residual", "redi_variance_change"):
            assert torch.equal(getattr(report, key), getattr(old, key))
        assert report.redi_energy_limiting_passes == report.redi_energy_limited_rows == 0
        return

    with pytest.raises(LocalEddyEnergyError) as failure:
        strict.apply(model, state, 60.)
    bad = failure.value.bad
    assert bad.nonzero().tolist() == [[13, 6]]
    geometry = strict._geometry(model, state, part="redi")
    stencil = geometry[-2]
    incident = stencil.row_min((~bad).to(h)) == 0
    assert report.redi_energy_limited_rows == incident.sum()
    assert report.redi_energy_blocked_rows == (int(incident.sum()) if specific_energy == 0 else 0)
    assert 0 < report.redi_energy_deficit_max_J_kg
    assert (end.eddy_h >= 0).all()
    # Away from the limited rows, temperature still gets its original increment.
    far = stencil.transpose(incident.to(h)[..., None], absolute=True)[..., 0] == 0
    full, _ = strict.apply(model, replace(state, eddy_h=1e-4*h), 60.)
    assert torch.equal(end.ocean.temp_h[far], full.ocean.temp_h[far])
    assert (end.ocean.temp_h[far] != state.ocean.temp_h[far]).any()
    mass = model.hamiltonian.mass_area(state.ocean) * h
    assert ((model.hamiltonian.finite_change(state, end) - report.redi_energy_flux_divergence)
            / mass).abs().max() < 1e-12
    assert report.energy_change.abs() / mass.sum() < 1e-12
    assert (report.redi_variance_change[:2] <= 0).all()
    with pytest.raises(FloatingPointError, match="Redi explicit step"):
        GMRedi(kappa_redi=1e20, fund_redi=True).apply(model, state, 60.)
