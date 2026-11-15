"""Independent full-functional checks for stationary ocean/ice columns."""

from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
import torch

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "validation" / p)
    for p in ("sea_ice", "temperature_coordinate", "ocean_column")
]
from ice import SeaIceState, IceOceanState, SeaIceThermodynamics, ConductiveIceBoundary
from test_gmredi import setup, mesh


def cold_case(mesh, coast=False, mass=0.0, temperature=None):
    model, old = setup(mesh, coast)
    s = old.ocean
    h = torch.cat(
        (
            torch.full_like(s.h[:, :1], 20.0),
            s.h[:, 1:2],
            s.h[:, 2:] + s.h[:, :1] - 20.0,
        ),
        -1,
    )
    t = (
        -0.054 * s.salinity()
        if temperature is None
        else torch.full_like(h, temperature)
    )
    physical = s.with_tensors(
        torch.zeros_like(s.u_e),
        h,
        t * h,
        s.salinity() * h,
        s.tracer_h / s.h[..., None] * h[..., None],
    )
    wet = model.energy.mass_area(s) > 0
    ice = SeaIceState(torch.where(wet, torch.full_like(s.h[:, :1], mass), 0.0))
    return model, IceOceanState(replace(old, ocean=physical), ice)


@pytest.mark.parametrize("coast", [False, True])
@pytest.mark.parametrize(
    "mass,flux", [(0.0, -200.0), (917.0, -100.0), (917.0, 100.0), (10.0, 1000.0)]
)
def test_actual_energy_and_material_budgets(mesh, coast, mass, flux):
    model, initial = cold_case(mesh, coast, mass)
    thermo = SeaIceThermodynamics(model)
    H = thermo.hamiltonian
    terms = H.column_terms(initial)
    ocean = sum(v.sum() for k, v in terms.items() if not k.startswith("ice_"))
    torch.testing.assert_close(
        ocean, model.total_energy(initial.ocean), rtol=3e-15, atol=0
    )
    wet = model.energy.mass_area(initial.ocean.ocean) > 0
    q = torch.where(wet, torch.full_like(initial.ice.mass, flux), 0.0)
    with patch("torch.autograd.grad", side_effect=AssertionError("forward AD")):
        end, report = thermo.apply(initial, 86400.0, heat_flux=q)
    eps = torch.finfo(q.dtype).eps
    scale = sum(v.abs().sum() for v in terms.values())
    actual = H.total_energy(end) - H.total_energy(initial) - report.external_heat.sum()
    assert actual.abs() < 64 * eps * scale
    local_scale = sum(v.abs() for v in terms.values())
    assert (report.energy_residual.abs() <= 64 * eps * local_scale).all()
    s, t = initial.ocean.ocean, end.ocean.ocean
    water_scale = mesh.cell_area * (
        s.column.rho0 * s.h.sum(-1) + initial.ice.mass[:, 0]
    )
    assert (report.water_residual.abs() <= 8 * eps * water_scale).all()
    assert torch.count_nonzero(report.salt_residual) == 0
    for name in ("salt_h", "tracer_h"):
        assert torch.equal(getattr(s, name), getattr(t, name))
    assert torch.equal(initial.ocean.tke_h, end.ocean.tke_h)
    assert torch.equal(initial.ocean.eddy_h, end.ocean.eddy_h)
    covered = end.ice.mass > 0
    torch.testing.assert_close(
        t.temperature()[:, :1][covered],
        -0.054 * t.salinity()[:, :1][covered],
        rtol=2e-15,
        atol=1e-15,
    )
    if mass == 10.0:
        assert torch.count_nonzero(end.ice.mass) == 0
    if flux < 0:
        assert (end.ice.mass[wet] > initial.ice.mass[wet]).all()
    if coast:
        assert torch.equal(s.h[~wet[:, 0]], t.h[~wet[:, 0]])
        assert torch.equal(s.temp_h[~wet[:, 0]], t.temp_h[~wet[:, 0]])


def test_phase_derivative_independent_autograd(mesh):
    model, initial = cold_case(mesh, mass=917.0)
    H = SeaIceThermodynamics(model).hamiltonian
    m = initial.ice.mass.clone().requires_grad_()
    s = initial.ocean.ocean
    h = torch.cat((s.h[:, :1] + (initial.ice.mass - m) / s.column.rho0, s.h[:, 1:]), -1)
    th = torch.cat((-0.054 * s.salt_h[:, :1], s.temp_h[:, 1:]), -1)
    physical = s.with_tensors(s.u_e, h, th, s.salt_h, s.tracer_h)
    state = IceOceanState(replace(initial.ocean, ocean=physical), SeaIceState(m))
    (expected,) = torch.autograd.grad(H.energy_per_area(state).sum(), m)
    torch.testing.assert_close(
        H.phase_mass_derivative(state), expected, rtol=2e-14, atol=1e-8
    )


def test_phase_sensitivity_and_reversible_heat_cycle(mesh):
    model, initial = cold_case(mesh, temperature=-1.8)
    thermo = SeaIceThermodynamics(model)
    q = torch.tensor(-200.0, requires_grad=True)

    def evolve(flux):
        return thermo.apply(
            initial, 86400.0, heat_flux=flux * torch.ones_like(initial.ice.mass)
        )[0]

    end = evolve(q)
    (derivative,) = torch.autograd.grad(end.ice.mass.sum(), q)
    difference = (
        evolve(q.detach() + 0.01).ice.mass.sum()
        - evolve(q.detach() - 0.01).ice.mass.sum()
    ) / 0.02
    torch.testing.assert_close(derivative, difference, rtol=2e-7, atol=1e-7)
    back, _ = thermo.apply(
        end, 86400.0, heat_flux=torch.full_like(initial.ice.mass, 200.0)
    )
    assert torch.count_nonzero(back.ice.mass) == 0
    torch.testing.assert_close(
        back.ocean.ocean.h, initial.ocean.ocean.h, rtol=0, atol=2e-12
    )
    torch.testing.assert_close(
        back.ocean.ocean.temp_h, initial.ocean.ocean.temp_h, rtol=0, atol=2e-10
    )


def test_conductive_resistance_and_open_water(mesh):
    _, thin = cold_case(mesh, mass=917.0)
    thick = replace(thin, ice=SeaIceState(2 * thin.ice.mass))
    boundary = ConductiveIceBoundary()
    air = torch.full_like(thin.ice.mass, -20.0)
    assert (boundary.flux(thin, air) < boundary.flux(thick, air)).all()
    warm = boundary.flux(thin, torch.full_like(air, 5.0))
    torch.testing.assert_close(warm, torch.full_like(air, 50.0))
    water = replace(thin, ice=SeaIceState(torch.zeros_like(thin.ice.mass)))
    torch.testing.assert_close(
        boundary.flux(water, air), 10 * (air - water.ocean.ocean.temperature()[:, :1])
    )


def test_rejects_unsupported_motion_and_exhaustion(mesh):
    model, initial = cold_case(mesh)
    thermo = SeaIceThermodynamics(model)
    s = initial.ocean.ocean
    moving = s.with_tensors(torch.ones_like(s.u_e), s.h, s.temp_h, s.salt_h, s.tracer_h)
    with pytest.raises(ValueError, match="stationary"):
        thermo.check(replace(initial, ocean=replace(initial.ocean, ocean=moving)))
    with pytest.raises(ValueError, match="exhausts"):
        thermo.apply(
            initial, 86400.0, heat_flux=torch.full_like(initial.ice.mass, -1e8)
        )
    with pytest.raises(ValueError, match="nonnegative"):
        thermo.check(
            replace(initial, ice=SeaIceState(-torch.ones_like(initial.ice.mass)))
        )


def test_unforced_supercooling_releases_latent_heat(mesh):
    model, initial = cold_case(mesh, temperature=-2.0)
    thermo = SeaIceThermodynamics(model)
    end, report = thermo.apply(
        initial, 1.0, heat_flux=torch.zeros_like(initial.ice.mass)
    )
    assert (end.ice.mass > 0).all()
    assert (end.ocean.ocean.temperature()[:, :1] > -2.0).all()
    # Unforced phase equilibration exchanges energy internally, including PE.
    assert torch.count_nonzero(report.external_heat) == 0
    scale = sum(
        t.abs().sum() for t in thermo.hamiltonian.column_terms(initial).values()
    )
    assert (
        abs(
            thermo.hamiltonian.total_energy(end)
            - thermo.hamiltonian.total_energy(initial)
        )
        < 32 * torch.finfo(torch.float64).eps * scale
    )


def test_no_flux_phase_equilibrium_is_rest(mesh):
    model, initial = cold_case(mesh, mass=917.0)
    thermo = SeaIceThermodynamics(model)
    end, _ = thermo.apply(
        initial, 86400.0, heat_flux=torch.zeros_like(initial.ice.mass)
    )
    torch.testing.assert_close(end.ice.mass, initial.ice.mass, rtol=0, atol=2e-10)
    torch.testing.assert_close(
        end.ocean.ocean.temp_h, initial.ocean.ocean.temp_h, rtol=0, atol=2e-12
    )


def test_salinity_ceiling_allows_warming_but_not_more_freezing(mesh):
    model, initial = cold_case(mesh)
    s = initial.ocean.ocean
    salinity = 42.0 - 1e-13
    physical = s.with_tensors(
        s.u_e, s.h, -0.054 * salinity * s.h, salinity * s.h, s.tracer_h
    )
    initial = replace(initial, ocean=replace(initial.ocean, ocean=physical))
    thermo = SeaIceThermodynamics(model)
    end, _ = thermo.apply(initial, 60.0, heat_flux=torch.ones_like(initial.ice.mass))
    assert torch.count_nonzero(end.ice.mass) == 0
    assert (end.ocean.ocean.temperature() > physical.temperature()).any()
    with pytest.raises(ValueError, match="exhausts"):
        thermo.apply(initial, 60.0, heat_flux=-torch.ones_like(initial.ice.mass))
