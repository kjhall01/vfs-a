"""Independent mass, momentum and full-H checks of the freshwater boundary."""

from dataclasses import replace
from unittest.mock import patch
import pytest
import torch
from test_snow_ice import snowy, ridge_mesh
from vfs.ice.freshwater import FreshwaterBoundary, FreshwaterFluxes, LiquidWater


def forcing(boundary, state, kind="mixed"):
    s = state.coupled.ocean.ocean
    z = torch.zeros_like(s.h[:, :1])
    rate = torch.full_like(z, 1e-4)
    wet = boundary.H.model.energy.mass_area(s) > 0
    velocity = torch.full_like(state.coupled.ice.velocity, 0.3)
    data = dict(
        rain=boundary.liquid(rate, z + 5, velocity),
        runoff=boundary.liquid(torch.where(wet, 2 * rate, 0), z + 10, -velocity),
        snowfall=rate,
        evaporation=0.4 * rate,
        sublimation=0.2 * rate,
    )
    return FreshwaterFluxes(**(data if kind == "mixed" else {kind: data[kind]}))


@pytest.mark.parametrize("coast", [False, True])
@pytest.mark.parametrize(
    "kind", ["rain", "runoff", "snowfall", "evaporation", "sublimation", "mixed"]
)
def test_independent_boundary_budgets(ridge_mesh, coast, kind):
    snow, state = snowy(ridge_mesh, coast)
    boundary = FreshwaterBoundary(snow)
    H = boundary.H
    s = state.coupled.ocean.ocean
    with patch("torch.autograd.grad", side_effect=AssertionError("forward AD")):
        end, report = boundary.apply(state, 100.0, forcing(boundary, state, kind))
    t = end.coupled.ocean.ocean
    A = ridge_mesh.cell_area[:, None]
    mass = A * (
        s.column.rho0 * (t.h - s.h).sum(-1, keepdim=True)
        + H.mass(end.coupled)
        - H.mass(state.coupled)
    )
    expected = sum(report.inputs.values()) - sum(report.outputs.values())
    torch.testing.assert_close(mass, expected, rtol=2e-8, atol=1.0)
    mechanical = H.model.energy.base.mechanical
    momentum = (
        mechanical.metric(t)[:, :1] * t.u_e[:, :1]
        + H.metric(end.coupled) * end.coupled.ice.velocity
        - mechanical.metric(s)[:, :1] * s.u_e[:, :1]
        - H.metric(state.coupled) * state.coupled.ice.velocity
    )
    torch.testing.assert_close(
        momentum, report.ocean_impulse + report.ice_impulse, rtol=2e-8, atol=1e-3
    )
    error = (
        H.total_energy(end.coupled)
        - H.total_energy(state.coupled)
        - report.external_energy
    )
    scale = sum(v.abs().sum() for v in H.terms(state.coupled).values())
    assert error.abs() < 128 * torch.finfo(s.h.dtype).eps * scale
    for a, b in [
        (t.salt_h, s.salt_h),
        (t.tracer_h, s.tracer_h),
        (end.coupled.ocean.tke_h, state.coupled.ocean.tke_h),
        (end.coupled.ocean.eddy_h, state.coupled.ocean.eddy_h),
        (t.temp_h[:, 1:], s.temp_h[:, 1:]),
    ]:
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    if coast:
        dry = ~H.model.ocean.domain.wet_cell
        for a, b in [
            (t.h, s.h),
            (t.temp_h, s.temp_h),
            (end.snow_mass, state.snow_mass),
        ]:
            torch.testing.assert_close(a[dry], b[dry], rtol=0, atol=0)


@pytest.mark.parametrize("coverage", [0.0, 0.4, 1.0])
def test_precipitation_routing_and_evaporating_area(ridge_mesh, coverage):
    snow, state = snowy(ridge_mesh)
    ice = state.coupled.ice
    a = torch.zeros_like(ice.area)
    a[:, 0] = coverage
    ice = replace(ice, area=a, volume=0.5 * a, velocity=torch.zeros_like(ice.velocity))
    state = snow.bare(replace(state.coupled, ice=ice))
    boundary = FreshwaterBoundary(snow)
    rate = torch.full_like(a[:, :1], 1e-4)
    dt = 100.0
    end, r = boundary.apply(
        state, dt, FreshwaterFluxes(snowfall=rate, evaporation=0.2 * rate)
    )
    A = ridge_mesh.cell_area[:, None]
    torch.testing.assert_close(sum(r.inputs.values()), A * dt * rate)
    torch.testing.assert_close(r.inputs["snowfall_on_ice"], coverage * A * dt * rate)
    torch.testing.assert_close(
        r.outputs["evaporation"], (1 - coverage) * A * dt * 0.2 * rate
    )
    torch.testing.assert_close(
        r.energy_inputs["snowfall_open_water"],
        -snow.hamiltonian.slab.parameters.latent_heat * (1 - coverage) * A * dt * rate,
    )
    torch.testing.assert_close(end.snow_mass[:, :1], dt * coverage * rate)


def test_under_ice_loading_work_is_explicit(ridge_mesh):
    snow, state = snowy(ridge_mesh)
    boundary = FreshwaterBoundary(snow)
    flux = forcing(boundary, state, "rain")
    s = state.coupled.ocean.ocean
    dt = 100.0
    end, r = boundary.apply(state, dt, flux)
    dm = dt * flux.rain.rate
    dh = dm / s.column.rho0
    A = ridge_mesh.cell_area[:, None]
    surface = A * s.column.gravity * dm * (s.column.eta(s.h) + 0.5 * dh)
    loading = A * s.column.gravity * boundary.H.mass(state.coupled) * dh
    torch.testing.assert_close(
        r.energy_inputs["liquid_incorporation"] - surface,
        loading,
        rtol=2e-13,
        atol=1e-5,
    )
    assert loading.min() > 0
    # Salt is retained, while freshwater actually raises sea level.
    torch.testing.assert_close(end.coupled.ocean.ocean.h[:, :1] - s.h[:, :1], dh)
    assert (end.coupled.ocean.ocean.salinity()[:, :1] < s.salinity()[:, :1]).all()


def test_sublimation_uses_snow_before_ice_and_can_exhaust_cover(ridge_mesh):
    snow, state = snowy(ridge_mesh)
    boundary = FreshwaterBoundary(snow)
    rho = boundary.H.slab.parameters.density
    rate = torch.full_like(state.snow_mass[:, :1], 1e-4)
    end, r = boundary.apply(state, 10.0, FreshwaterFluxes(sublimation=rate))
    torch.testing.assert_close(
        end.coupled.ice.volume - end.snow_mass / rho,
        state.coupled.ice.volume - state.snow_mass / rho,
        rtol=1e-14,
        atol=1e-15,
    )
    torch.testing.assert_close(
        end.coupled.ice.area, state.coupled.ice.area, rtol=0, atol=0
    )
    # Equal thickness makes a single per-covered-area sublimation rate exhaust both categories.
    area = torch.tensor([0.5, 0.25]).expand_as(state.coupled.ice.area).clone()
    ice = replace(state.coupled.ice, area=area, volume=area / 256)
    state = snow.bare(replace(state.coupled, ice=ice))
    end, r = boundary.apply(
        state, 1.0, FreshwaterFluxes(sublimation=rate * 0 + rho / 256)
    )
    assert (end.coupled.ice.volume == 0).all()
    assert (end.coupled.ice.area == 0).all()
    assert (end.coupled.ice.velocity == 0).all()
    torch.testing.assert_close(
        r.energy_inputs["sublimation_latent"],
        -(boundary.vaporization_heat + boundary.H.slab.parameters.latent_heat)
        * r.outputs["sublimation"],
    )


def test_runoff_mass_and_enthalpy_mapping(ridge_mesh):
    snow, state = snowy(ridge_mesh, True)
    boundary = FreshwaterBoundary(snow)
    wet = torch.where(boundary.H.model.ocean.domain.wet_cell)[0]
    receivers = torch.stack((wet[:2], wet[1:3]))
    Q = torch.tensor([5.0, 11.0])
    T = torch.tensor([3.0, 15.0])
    weights = torch.tensor([[0.3, 0.7], [0.4, 0.6]])
    v = torch.zeros_like(state.coupled.ice.velocity)
    runoff = boundary.route_runoff(state, Q, T, receivers, weights, v)
    A = ridge_mesh.cell_area[:, None]
    rho = state.coupled.ocean.ocean.column.rho0
    torch.testing.assert_close((A * runoff.rate).sum(), rho * Q.sum())
    torch.testing.assert_close(
        (A * runoff.rate * runoff.enthalpy).sum(),
        rho * boundary.H.model.config.cp * (Q * T).sum(),
    )
    end, r = boundary.apply(state, 100.0, FreshwaterFluxes(runoff=runoff))
    torch.testing.assert_close(r.inputs["runoff"].sum(), 100 * rho * Q.sum())
    bad = receivers.clone()
    bad[0, 0] = torch.where(~boundary.H.model.ocean.domain.wet_cell)[0][0]
    with pytest.raises(ValueError, match="wet"):
        boundary.route_runoff(state, Q, T, bad, weights, v)
    with pytest.raises(ValueError, match="normalized"):
        boundary.route_runoff(state, Q, T, receivers, 0.5 * weights, v)


def test_invalid_forcing_and_no_input_mutation(ridge_mesh):
    snow, state = snowy(ridge_mesh, True)
    boundary = FreshwaterBoundary(snow)
    copy = state.coupled.ocean.ocean.h.clone()
    rate = torch.ones_like(state.snow_mass[:, :1])
    with pytest.raises(ValueError, match="nonnegative"):
        boundary.apply(state, 1.0, FreshwaterFluxes(snowfall=-rate))
    with pytest.raises(ValueError, match="exhausts"):
        boundary.apply(state, 1e8, FreshwaterFluxes(evaporation=rate))
    with pytest.raises(ValueError, match="category"):
        boundary.apply(state, 1e8, FreshwaterFluxes(sublimation=rate))
    with pytest.raises(ValueError, match="wet"):
        boundary.apply(
            state,
            1.0,
            FreshwaterFluxes(
                runoff=boundary.liquid(
                    rate, rate * 0, torch.zeros_like(state.coupled.ice.velocity)
                )
            ),
        )
    with pytest.raises(ValueError, match="not both"):
        snow.step(state, 1.0, snowfall=rate, freshwater=FreshwaterFluxes())
    torch.testing.assert_close(copy, state.coupled.ocean.ocean.h, rtol=0, atol=0)


def test_zero_flux_step_is_bitwise_unchanged(ridge_mesh):
    snow, state = snowy(ridge_mesh)
    boundary = FreshwaterBoundary(snow)
    end, r = boundary.apply(state, 1.0, FreshwaterFluxes())
    for a, b in [
        (end.coupled.ocean.ocean.temp_h, state.coupled.ocean.ocean.temp_h),
        (end.coupled.ice.volume, state.coupled.ice.volume),
        (end.coupled.ice.area, state.coupled.ice.area),
    ]:
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert r.external_energy == 0
    plain, _ = snow.step(state, 0.1)
    coupled, _ = snow.step(state, 0.1, freshwater=FreshwaterFluxes())
    torch.testing.assert_close(
        plain.coupled.ocean.ocean.temp_h,
        coupled.coupled.ocean.ocean.temp_h,
        rtol=0,
        atol=0,
    )


def test_boundary_backward_against_finite_difference(ridge_mesh):
    snow, state = snowy(ridge_mesh)
    boundary = FreshwaterBoundary(snow)
    sample = torch.ones_like(state.snow_mass[:, :1])
    u = torch.zeros_like(state.coupled.ice.velocity)

    def result(value):
        flux = FreshwaterFluxes(rain=boundary.liquid(sample * 1e-4, sample * value, u))
        return (
            boundary.apply(state, 100.0, flux)[0]
            .coupled.ocean.ocean.temperature()
            .sum()
        )

    temp = torch.tensor(5.0, requires_grad=True)
    g = torch.autograd.grad(result(temp), temp)[0]
    finite = (result(5.01) - result(4.99)) / 0.02
    torch.testing.assert_close(g, finite, rtol=2e-5, atol=1e-9)


def test_full_step_boundary_ledger(ridge_mesh):
    snow, state = snowy(ridge_mesh)
    boundary = FreshwaterBoundary(snow)
    end, report = snow.step(state, 0.1, freshwater=forcing(boundary, state))
    assert len(report.freshwater) == 2
    A = ridge_mesh.cell_area[:, None]
    s, t = state.coupled.ocean.ocean, end.coupled.ocean.ocean
    mass = (
        A
        * (
            s.column.rho0 * (t.h - s.h).sum(-1, keepdim=True)
            + boundary.H.mass(end.coupled)
            - boundary.H.mass(state.coupled)
        )
    ).sum()
    expected = sum(
        (sum(r.inputs.values()) - sum(r.outputs.values())).sum()
        for r in report.freshwater
    )
    torch.testing.assert_close(mass, expected, rtol=2e-6, atol=1.0)
    delta = boundary.H.total_energy(end.coupled) - boundary.H.total_energy(
        state.coupled
    )
    torch.testing.assert_close(delta, report.external_energy, rtol=2e-6, atol=1e3)
