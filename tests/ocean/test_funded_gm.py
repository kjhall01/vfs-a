"""Donor energy transport and local funding at a finite-amplitude ocean front."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "validation/temperature_coordinate"))
from eddy import GMRedi, LocalEddyEnergyError
from eddy_campaign import make
from eddy_energy import EddyMixingLength
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


def test_unlimited_donor_energy_from_independent_parcel_quadrature(front):
    model, state = front
    strict = operator("donor_parcel")
    dt = 60.0
    end, report = strict.apply(model, state, dt)
    funded, accepted = operator("funded_donor_parcel").apply(model, state, dt)
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
def test_funding_retains_explicit_stability_checks(front, gm, redi, message):
    model, state = front
    with pytest.raises(FloatingPointError, match=message):
        GMRedi(gm, redi, gm_energy_scheme="funded_donor_parcel").apply(model, state, 60)
    with pytest.raises(ValueError, match="Unknown GM energy scheme"):
        GMRedi(gm_energy_scheme="unknown")
