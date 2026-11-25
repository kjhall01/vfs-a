"""Snow is a subset of the frozen carrier: independent material/work checks."""

from dataclasses import replace
import torch
import pytest
from test_ocean_ice_coupling import setup, ridge_mesh
from vfs.ice.snow import SnowOceanIce, SnowState, SnowParameters


def snowy(mesh, coast=False):
    coupled, state = setup(mesh, coast)
    model = SnowOceanIce(coupled)
    bare = model.bare(state)
    return (
        model,
        model.accumulate(bare, 86400.0, torch.full_like(state.ice.area[:, :1], 2e-4))[
            0
        ],
    )


@pytest.mark.parametrize("coast", [False, True])
def test_snowfall_water_energy_and_momentum(ridge_mesh, coast):
    coupled, initial = setup(ridge_mesh, coast)
    model = SnowOceanIce(coupled)
    state = model.bare(initial)
    rate = torch.full_like(initial.ice.area[:, :1], 1e-4)
    end, added, energy = model.accumulate(state, 3600.0, rate)
    H = coupled.hamiltonian
    A = ridge_mesh.cell_area[:, None]
    expected = (A * initial.ice.area * rate * 3600.0).sum()
    torch.testing.assert_close(added, expected)
    torch.testing.assert_close(
        (A * (H.mass(end.coupled) - H.mass(initial))).sum(),
        added,
        rtol=1e-12,
        atol=1e-3,
    )
    torch.testing.assert_close(
        H.total_energy(end.coupled) - H.total_energy(initial),
        energy,
        rtol=2e-8,
        atol=1.0,
    )
    torch.testing.assert_close(end.snow_mass, 3600 * rate * initial.ice.area)
    torch.testing.assert_close(
        end.coupled.ice.velocity, initial.ice.velocity, rtol=0, atol=0
    )
    torch.testing.assert_close(
        end.coupled.ocean.ocean.salt_h, initial.ocean.ocean.salt_h, rtol=0, atol=0
    )
    # Co-moving deposition brings its own horizontal generalized momentum.
    impulse = (H.metric(end.coupled) - H.metric(initial)) * initial.ice.velocity
    dm = H.mass(end.coupled) - H.mass(initial)
    torch.testing.assert_close(
        impulse,
        (ridge_mesh.dc_edge * ridge_mesh.dv_edge)[:, None]
        * ridge_mesh.cell_to_edge(dm)
        * initial.ice.velocity,
    )


def test_snow_insulates_and_reflects(ridge_mesh):
    model, state = snowy(ridge_mesh)
    bare = model.bare(state.coupled)
    air = torch.full_like(state.snow_mass[:, :1], -20.0)
    flux = model.surface_flux(state, air)
    exposed = model.surface_flux(bare, air)
    assert (flux.heat > exposed.heat).all()  # less negative winter loss
    sw = torch.full_like(air, 300.0)
    snow_flux = model.surface_flux(state, air, sw)
    bare_flux = model.surface_flux(bare, air, sw)
    assert (snow_flux.albedo > bare_flux.albedo).all()
    assert (snow_flux.absorbed_shortwave < bare_flux.absorbed_shortwave).all()
    torch.testing.assert_close(
        snow_flux.absorbed_shortwave + snow_flux.reflected_shortwave, sw
    )
    hi, hs = model.depths(state)
    rho = model.hamiltonian.slab.parameters.density
    torch.testing.assert_close(
        state.coupled.ice.area * (rho * hi + model.parameters.density * hs),
        rho * state.coupled.ice.volume,
    )


@pytest.mark.parametrize("coast", [False, True])
def test_zero_snow_zero_solar_matches_existing_surface(ridge_mesh, coast):
    coupled, state = setup(ridge_mesh, coast)
    model = SnowOceanIce(coupled)
    for temperature in [-20.0, 5.0]:
        air = torch.full_like(state.ice.area[:, :1], temperature)
        torch.testing.assert_close(
            model.surface_flux(model.bare(state), air).heat,
            model.phase.air_flux(state, air),
            rtol=1e-14,
            atol=1e-12,
        )


@pytest.mark.parametrize("heat", [100.0, 5000.0])
def test_melt_snow_first_and_close_phase_budget(ridge_mesh, heat):
    model, state = snowy(ridge_mesh)
    initial = state.coupled
    end, energy, melted = model.phase_step(
        state, 86400.0, torch.full_like(state.snow_mass[:, :1], heat)
    )
    H = model.hamiltonian
    A = ridge_mesh.cell_area[:, None]
    torch.testing.assert_close(
        H.total_energy(end.coupled) - H.total_energy(initial),
        energy,
        rtol=2e-8,
        atol=1.0,
    )
    rho = H.slab.parameters.density
    loss = rho * (initial.ice.volume - end.coupled.ice.volume)
    expected = torch.minimum(loss.clamp_min(0), state.snow_mass)
    torch.testing.assert_close(state.snow_mass - end.snow_mass, expected)
    torch.testing.assert_close(melted, (A * expected).sum())
    water = (
        initial.ocean.ocean.column.rho0
        * (end.coupled.ocean.ocean.h - initial.ocean.ocean.h).sum(-1, keepdim=True)
        + H.mass(end.coupled)
        - H.mass(initial)
    )
    assert water.abs().max() < 1e-10
    if heat == 5000:
        assert (end.snow_mass == 0).all()
        assert (end.coupled.ice.volume == 0).all()


def test_freezing_does_not_create_snow(ridge_mesh):
    model, state = snowy(ridge_mesh)
    end, _, melt = model.phase_step(
        state, 3600.0, torch.full_like(state.snow_mass[:, :1], -100.0)
    )
    torch.testing.assert_close(end.snow_mass, state.snow_mass, rtol=0, atol=0)
    assert melt == 0
    assert (end.coupled.ice.volume >= state.coupled.ice.volume).all()


@pytest.mark.parametrize("coast", [False, True])
def test_snow_carrier_transport_and_ridging(ridge_mesh, coast):
    model, state = snowy(ridge_mesh, coast)
    rho = model.hamiltonian.slab.parameters.density
    # Uniform constituent fraction must remain uniform through redistribution.
    state = replace(state, snow_mass=0.2 * rho * state.coupled.ice.volume)
    rhs, _ = model.tendency(state, 1.0, ocean_dynamics=False, drag=False)
    torch.testing.assert_close(
        rhs.snow_mass, 0.2 * rho * rhs.coupled.ice.volume, rtol=1e-12, atol=1e-15
    )
    A = ridge_mesh.cell_area[:, None]
    assert (A * rhs.snow_mass).sum().abs() < 1e-12 * (A * rhs.snow_mass).abs().sum()
    end, _ = model.dynamics_step(state, 1.0, ocean_dynamics=False, drag=False)
    model.check(end)
    torch.testing.assert_close(
        end.snow_mass, 0.2 * rho * end.coupled.ice.volume, rtol=1e-13, atol=1e-13
    )


def test_bare_step_reproduces_existing_coupling(ridge_mesh):
    coupled, state = setup(ridge_mesh)
    snow = SnowOceanIce(coupled)
    expected, work = coupled.step(state, 1.0)
    actual, report = snow.step(snow.bare(state), 1.0)
    for x, y in [
        (actual.coupled.ice.volume, expected.ice.volume),
        (actual.coupled.ice.area, expected.ice.area),
        (actual.coupled.ice.velocity, expected.ice.velocity),
        (actual.coupled.ocean.ocean.u_e, expected.ocean.ocean.u_e),
        (actual.coupled.ocean.ocean.temp_h, expected.ocean.ocean.temp_h),
    ]:
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    torch.testing.assert_close(report.external_energy, work, rtol=0, atol=0)


def test_snow_surface_backward(ridge_mesh):
    model, state = snowy(ridge_mesh)
    mass = state.snow_mass.clone().requires_grad_()
    state = replace(state, snow_mass=mass)
    air = torch.full_like(mass[:, :1], -10.0)
    sw = torch.full_like(air, 200.0)
    value = model.surface_flux(state, air, sw).heat.sum()
    grad = torch.autograd.grad(value, mass)[0].sum()

    def shifted(eps):
        return model.surface_flux(
            replace(state, snow_mass=mass.detach() + eps), air, sw
        ).heat.sum()

    finite = (shifted(1e-5) - shifted(-1e-5)) / 2e-5
    torch.testing.assert_close(grad, finite, rtol=1e-7, atol=1e-7)


def test_bad_inputs_and_open_water(ridge_mesh):
    model, state = snowy(ridge_mesh)
    with pytest.raises(ValueError):
        model.check(replace(state, snow_mass=-state.snow_mass))
    with pytest.raises(ValueError):
        SnowParameters(conductivity=0)
    with pytest.raises(ValueError):
        SnowParameters(albedo=2)
    with pytest.raises(ValueError):
        model.step(state, 1.0, shortwave=torch.ones_like(state.snow_mass[:, :1]))
    zero = replace(
        state.coupled.ice,
        area=torch.zeros_like(state.snow_mass),
        volume=torch.zeros_like(state.snow_mass),
        velocity=torch.zeros_like(state.coupled.ice.velocity),
    )
    bare = model.bare(replace(state.coupled, ice=zero))
    end, added, energy = model.accumulate(
        bare, 1.0, torch.ones_like(state.snow_mass[:, :1])
    )
    assert added == 0 and energy == 0
    assert (end.snow_mass == 0).all()
    flux = model.surface_flux(
        end,
        torch.ones_like(state.snow_mass[:, :1]),
        torch.full_like(state.snow_mass[:, :1], 100.0),
    )
    torch.testing.assert_close(
        flux.albedo, torch.full_like(flux.albedo, model.parameters.water_albedo)
    )


def test_snow_only_melt_preserves_ice_volume_and_footprint(ridge_mesh):
    model, state = snowy(ridge_mesh)
    end, _, melt = model.phase_step(
        state, 100.0, torch.full_like(state.snow_mass[:, :1], 100.0)
    )
    assert melt > 0
    assert (end.snow_mass > 0).all()
    torch.testing.assert_close(
        end.coupled.ice.area, state.coupled.ice.area, rtol=0, atol=0
    )
    rho = model.hamiltonian.slab.parameters.density
    torch.testing.assert_close(
        end.coupled.ice.volume - end.snow_mass / rho,
        state.coupled.ice.volume - state.snow_mass / rho,
        rtol=1e-14,
        atol=1e-15,
    )


def test_snow_phase_energy_derivative(ridge_mesh):
    from vfs.ice.snow import SnowPhase

    model, state = snowy(ridge_mesh)
    phase = SnowPhase(model.coupled, state.snow_mass)
    mass = (model.hamiltonian.mass(state.coupled) * 0.999).requires_grad_()
    energy = phase.root_energy(state.coupled, mass)
    independent = torch.autograd.grad(energy.sum(), mass)[0]
    candidate = phase.phase_state(state.coupled, mass)
    analytic = phase.derivative(state.coupled, candidate, mass)
    torch.testing.assert_close(analytic, independent, rtol=2e-13, atol=1e-7)
