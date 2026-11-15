"""Independent full-H coupling, reciprocal work and phase-momentum checks."""

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
from ocean_ice import OceanIce, vector_transpose
from category_phase import CategoryThermodynamics
from ridging import RidgingState, CategoryIce
from test_ice_ridging import case, ridge_mesh


def setup(mesh, coast=False):
    r, state = case(mesh, coast)
    s = state.ocean.ocean
    h = s.h * (1 + 0.001 * mesh.cell_xyz[:, :1])
    velocity = 0.02 * torch.sin(
        torch.arange(mesh.nEdges)[:, None] + torch.arange(3)[None, :]
    )
    if coast:
        velocity = r.hamiltonian.model.ocean.domain.apply_edge(velocity)
    physical = s.with_tensors(
        velocity, h, -0.054 * s.salinity() * h, s.salinity() * h, s.tracer_h
    )
    factor = 1 + 0.05 * mesh.cell_xyz[:, :1]
    state = replace(
        state,
        ocean=replace(state.ocean, ocean=physical),
        ice=replace(
            state.ice, area=state.ice.area * factor, volume=state.ice.volume * factor
        ),
    )
    return OceanIce(r.hamiltonian.model), state


def leaf_state(state):
    s = state.ocean.ocean
    leaves = [
        x.clone().requires_grad_()
        for x in (
            s.u_e,
            s.h,
            s.temp_h,
            s.salt_h,
            state.ocean.tke_h,
            state.ocean.eddy_h,
            state.ice.area,
            state.ice.volume,
            state.ice.velocity,
        )
    ]
    u, h, t, salt, k, e, a, v, ui = leaves
    physical = s.with_tensors(u, h, t, salt, s.tracer_h)
    return (
        RidgingState(
            replace(state.ocean, ocean=physical, tke_h=k, eddy_h=e),
            CategoryIce(a, v, ui),
        ),
        leaves,
    )


def rates(rhs):
    s = rhs.ocean.ocean
    return (
        s.u_e,
        s.h,
        s.temp_h,
        s.salt_h,
        rhs.ocean.tke_h,
        rhs.ocean.eddy_h,
        rhs.ice.area,
        rhs.ice.volume,
        rhs.ice.velocity,
    )


@pytest.mark.parametrize("coast", [False, True])
@pytest.mark.parametrize("parts", ["transport", "drag", "ocean", "all", "wind"])
def test_full_H_tendency_work(ridge_mesh, coast, parts):
    coupled, original = setup(ridge_mesh, coast)
    state, leaves = leaf_state(original)
    g = torch.autograd.grad(coupled.hamiltonian.total_energy(state), leaves)
    options = dict(
        ocean_dynamics=parts in ("ocean", "all", "wind"),
        transport=parts in ("transport", "all", "wind"),
        drag=parts in ("drag", "all", "wind"),
        ridging=parts in ("all", "wind"),
    )
    wind = 5 * ridge_mesh.cell_rec_basis1 if parts == "wind" else None
    with patch("torch.autograd.grad", side_effect=AssertionError("forward AD")):
        rhs, report = coupled.tendency(state, 1.0, wind_velocity=wind, **options)
    products = [a * b for a, b in zip(g, rates(rhs))]
    residual = sum(x.sum() for x in products) - report.wind_power
    scale = sum(x.abs().sum() for x in products) + report.wind_power.abs()
    assert residual.abs() < 512 * torch.finfo(torch.float64).eps * scale
    if parts == "transport":
        for q in (rhs.ice.volume, rhs.ice.area):
            assert (ridge_mesh.cell_area[:, None] * q).sum().abs() < 128 * torch.finfo(
                torch.float64
            ).eps * (ridge_mesh.cell_area[:, None] * q).abs().sum()
    if parts == "drag":
        H = coupled.hamiltonian
        ice = H.metric(state) * rhs.ice.velocity
        ocean = (
            H.model.energy.base.mechanical.metric(state.ocean.ocean)[:, :1]
            * rhs.ocean.ocean.u_e[:, :1]
        )
        torch.testing.assert_close(ice, -ocean, rtol=2e-15, atol=1e-8)
        assert (rhs.ocean.tke_h >= 0).all()
        assert report.drag_tke_power > 0
    if parts == "ocean":
        assert report.loading_power.abs() > 0


def test_vector_reconstruction_transpose(ridge_mesh):
    torch.manual_seed(31)
    u = torch.randn(ridge_mesh.nEdges, 1)
    force = torch.randn(ridge_mesh.nCells, 3)
    left = (ridge_mesh.cell_wind_vector(u)[:, 0, :] * force).sum()
    right = (u * vector_transpose(ridge_mesh, force)).sum()
    torch.testing.assert_close(left, right, rtol=2e-14, atol=1e-12)


@pytest.mark.parametrize("coast", [False, True])
@pytest.mark.parametrize("flux", [-100.0, 100.0, 5000.0])
def test_category_phase_actual_budget_and_momentum(ridge_mesh, coast, flux):
    coupled, initial = setup(ridge_mesh, coast)
    phase = CategoryThermodynamics(coupled)
    s = initial.ocean.ocean
    wet = coupled.ocean.energy.mass_area(s) > 0
    q = torch.where(wet, torch.full_like(s.h[:, :1], flux), 0.0)
    with patch("torch.autograd.grad", side_effect=AssertionError("forward AD")):
        end, report = phase.apply(initial, 86400.0, heat_flux=q)
    H = coupled.hamiltonian
    residual = (
        H.total_energy(end) - H.total_energy(initial) - report.external_heat.sum()
    )
    scale = sum(x.abs().sum() for x in H.terms(initial).values())
    assert residual.abs() < 128 * torch.finfo(torch.float64).eps * scale
    assert report.kinetic_to_tke.min() >= 0
    water_scale = ridge_mesh.cell_area * s.column.rho0 * s.h.sum(-1)
    assert (
        report.water_residual.abs() < 16 * torch.finfo(torch.float64).eps * water_scale
    ).all()
    momentum_scale = (H.metric(initial) * initial.ice.velocity).abs() + (
        H.model.energy.base.mechanical.metric(s)[:, :1] * s.u_e[:, :1]
    ).abs()
    assert (
        report.momentum_residual.abs()
        <= 16 * torch.finfo(torch.float64).eps * momentum_scale
    ).all()
    assert torch.equal(end.ocean.ocean.salt_h, s.salt_h)
    assert torch.equal(end.ocean.ocean.tracer_h, s.tracer_h)
    assert torch.equal(end.ocean.eddy_h, initial.ocean.eddy_h)
    if flux == 5000.0:
        assert torch.count_nonzero(end.ice.volume) == 0
        assert torch.count_nonzero(end.ice.velocity) == 0


def test_phase_derivative_and_backward(ridge_mesh):
    coupled, initial = setup(ridge_mesh)
    phase = CategoryThermodynamics(coupled)
    m = (0.9 * coupled.hamiltonian.mass(initial)).requires_grad_()
    state = phase.phase_state(initial, m)
    (expected,) = torch.autograd.grad(phase.thermal_energy(state).sum(), m)
    torch.testing.assert_close(
        phase.derivative(initial, state, m), expected, rtol=2e-14, atol=1e-8
    )
    q = torch.tensor(-100.0, requires_grad=True)

    def evolve(value):
        return phase.apply(initial, 3600.0, heat_flux=torch.ones_like(m) * value)[0]

    end = evolve(q)
    (derivative,) = torch.autograd.grad(end.ice.volume.sum(), q)
    fd = (
        evolve(q.detach() + 0.01).ice.volume.sum()
        - evolve(q.detach() - 0.01).ice.volume.sum()
    ) / 0.02
    torch.testing.assert_close(derivative, fd, rtol=1e-6, atol=1e-8)


def test_freeze_from_open_water_and_ice_free_dynamics(ridge_mesh):
    coupled, state = setup(ridge_mesh)
    state = replace(
        state,
        ice=CategoryIce(
            torch.zeros_like(state.ice.area),
            torch.zeros_like(state.ice.volume),
            torch.zeros_like(state.ice.velocity),
        ),
    )
    phase = CategoryThermodynamics(coupled)
    end, _ = phase.apply(
        state, 86400.0, heat_flux=torch.full_like(state.ice.area[:, :1], -200.0)
    )
    assert (end.ice.volume[:, 0] > 0).all()
    torch.testing.assert_close(
        end.ice.velocity, end.ocean.ocean.u_e[:, :1], rtol=1e-12, atol=1e-12
    )
    empty, _ = coupled.dynamics_step(state, 1.0)
    assert torch.count_nonzero(empty.ice.volume) == 0


def test_packing_still_resists_when_thin_category_empty(ridge_mesh):
    coupled, state = setup(ridge_mesh)
    a = state.ice.area.clone()
    v = state.ice.volume.clone()
    a[:, 0] = 0
    v[:, 0] = 0
    state = replace(state, ice=replace(state.ice, area=a, volume=v))
    rhs, report = coupled.tendency(
        state, 1.0, ocean_dynamics=False, transport=False, drag=False
    )
    assert (rhs.ice.area[:, 1] < 0).any()
    assert report.ridge_heat_power > 0
    assert torch.count_nonzero(rhs.ice.volume) == 0


def test_phase_root_uses_exact_full_H_difference(ridge_mesh):
    coupled, initial = setup(ridge_mesh, True)
    phase = CategoryThermodynamics(coupled)
    m = 1.3 * coupled.hamiltonian.mass(initial)
    candidate = phase.phase_state(initial, m)
    excluded = ("kinetic", "ice_kinetic", "turbulent_kinetic", "mesoscale_eddy")

    def full(state):
        return sum(
            v for k, v in coupled.hamiltonian.terms(state).items() if k not in excluded
        )

    actual = full(candidate) - full(initial)
    fast = (phase.thermal_energy(candidate) - phase.thermal_energy(initial))[
        :, 0
    ] * ridge_mesh.cell_area
    torch.testing.assert_close(fast, actual, rtol=3e-13, atol=1.0)
    torch.testing.assert_close(
        phase.root_energy(initial, m),
        phase.thermal_energy(candidate),
        rtol=3e-15,
        atol=1e-6,
    )


def test_fully_packed_growth_and_step(ridge_mesh):
    coupled, state = setup(ridge_mesh)
    phase = CategoryThermodynamics(coupled)
    end, _ = phase.apply(
        state, 86400.0, heat_flux=torch.full_like(state.ice.area[:, :1], -3000.0)
    )
    assert (end.ice.area.sum(-1) <= 1).all()
    assert (end.ice.area.sum(-1) > 0.999999).all()
    final, work = coupled.step(
        end, 0.1, heat_flux=torch.zeros_like(state.ice.area[:, :1])
    )
    assert torch.isfinite(coupled.hamiltonian.total_energy(final)) and work == 0


def test_conductive_boundary_and_wind_input(ridge_mesh):
    coupled, state = setup(ridge_mesh, True)
    phase = CategoryThermodynamics(coupled)
    q = phase.air_flux(state, torch.full_like(state.ice.area[:, :1], -20.0))
    wet = coupled.ocean.energy.mass_area(state.ocean.ocean) > 0
    assert (q[wet] < 0).all() and torch.count_nonzero(q[~wet]) == 0
    with pytest.raises(ValueError, match="wind_velocity"):
        coupled.tendency(state, 1.0, wind_velocity=torch.ones(ridge_mesh.nCells, 2))


def test_coupled_step_backward_smooth_branch(ridge_mesh):
    coupled, state = setup(ridge_mesh)
    phase = CategoryThermodynamics(coupled)
    # Phase-equilibrated starting point; no category births in this check.
    state, _ = phase.apply(
        state, 1.0, heat_flux=torch.zeros_like(state.ice.area[:, :1])
    )
    amplitude = torch.tensor(2.0, requires_grad=True)

    def run(x):
        return (
            coupled.step(state, 0.1, wind_velocity=x * ridge_mesh.cell_rec_basis1)[0]
            .ice.velocity.square()
            .sum()
        )

    value = run(amplitude)
    (gradient,) = torch.autograd.grad(value, amplitude)
    fd = (run(amplitude.detach() + 0.001) - run(amplitude.detach() - 0.001)) / 0.002
    torch.testing.assert_close(gradient, fd, rtol=2e-5, atol=1e-9)


def test_configured_backscatter_work(ridge_mesh):
    from backscatter import EddyBackscatter

    coupled, state = setup(ridge_mesh)
    coupled.ocean.backscatter = EddyBackscatter(
        viscosity=100.0, timescale=86400.0, fraction=0.1, biharmonic_viscosity=1e10
    )
    state, leaves = leaf_state(state)
    g = torch.autograd.grad(coupled.hamiltonian.total_energy(state), leaves)
    rhs, report = coupled.tendency(state, 0.1)
    products = [a * b for a, b in zip(g, rates(rhs))]
    assert abs(sum(p.sum() for p in products)) < 512 * torch.finfo(
        torch.float64
    ).eps * sum(p.abs().sum() for p in products)


def test_existing_mixing_can_follow_coupled_dynamics(ridge_mesh):
    from roquet import RoquetMixing

    coupled, state = setup(ridge_mesh)
    coupled.ocean.closure = RoquetMixing(2.0, substeps=2)
    H = coupled.hamiltonian
    end, external = coupled.step(
        state, 0.1, heat_flux=torch.zeros_like(state.ice.area[:, :1])
    )
    scale = sum(x.abs().sum() for x in H.terms(state).values())
    assert (
        abs(H.total_energy(end) - H.total_energy(state) - external)
        < 256 * torch.finfo(torch.float64).eps * scale
    )
    assert (end.ocean.tke_h >= 0).all()


def test_transport_and_ridging_category_birth_directional_work(ridge_mesh):
    coupled, state = setup(ridge_mesh)
    a, v = state.ice.area.clone(), state.ice.volume.clone()
    a[0, 1] = 0
    v[0, 1] = 0
    potential = torch.zeros(ridge_mesh.nCells, 1)
    potential[0] = 1
    state = replace(state, ice=CategoryIce(a, v, 1e4 * ridge_mesh.grad(potential)))
    state, leaves = leaf_state(state)
    g = torch.autograd.grad(coupled.hamiltonian.total_energy(state), leaves)
    rhs, _ = coupled.tendency(state, 0.1, ocean_dynamics=False, drag=False)
    products = [x * y for x, y in zip(g, rates(rhs))]
    empty = state.ice.area == 0
    # Independent directional derivative of V^2/a at (0,0).
    da = rhs.ice.area
    dv = rhs.ice.volume
    safe_da = torch.where(da > 0, da, torch.ones_like(da))
    born = (
        ridge_mesh.cell_area[:, None]
        * coupled.hamiltonian.coefficient(state)
        * torch.where(empty, dv.square() / safe_da, 0.0)
    ).sum()
    residual = sum(x.sum() for x in products) + born
    scale = sum(x.abs().sum() for x in products) + born.abs()
    assert residual.abs() < 512 * torch.finfo(torch.float64).eps * scale


def test_almost_complete_melt_does_not_divide_cancelling_momenta(ridge_mesh):
    coupled, initial = setup(ridge_mesh)
    phase = CategoryThermodynamics(coupled)
    m = coupled.hamiltonian.mass(initial) * 1e-20
    end = phase.phase_state(initial, m)
    transferred, _, _ = phase.exchange_momentum(initial, end)
    # Ice is the donor during melt, so its velocity must remain EXACTLY itself.
    assert torch.equal(transferred.ice.velocity, initial.ice.velocity)
    assert torch.isfinite(transferred.ocean.ocean.u_e).all()


@pytest.mark.parametrize("coast", [False, True])
@pytest.mark.parametrize("dt", [1.0, 3600.0])
def test_implicit_drag_budget(ridge_mesh, coast, dt):
    coupled, state = setup(ridge_mesh, coast)
    end, report = coupled.drag_step(state, dt)
    scale = report.tke_gain.sum().abs().clamp_min(1.0)
    assert report.energy_residual.abs() < 2e-10 * scale
    assert report.relative_solver_residual < 1e-12
    assert (report.tke_gain >= 0).all()
    Mi = coupled.hamiltonian.metric(state)
    Mo = coupled.ocean.energy.base.mechanical.metric(state.ocean.ocean)[:, :1]
    pscale = (Mi * state.ice.velocity).abs().max() + (
        Mo * state.ocean.ocean.u_e[:, :1]
    ).abs().max()
    assert report.momentum_residual.abs().max() < 1e-14 * pscale
    assert torch.isfinite(end.ice.velocity).all()


def test_implicit_drag_zero_ice(ridge_mesh):
    coupled, state = setup(ridge_mesh)
    state = replace(
        state,
        ice=CategoryIce(
            torch.zeros_like(state.ice.area),
            torch.zeros_like(state.ice.volume),
            torch.zeros_like(state.ice.velocity),
        ),
    )
    end, report = coupled.drag_step(state, 3600.0)
    torch.testing.assert_close(
        end.ocean.ocean.u_e, state.ocean.ocean.u_e, rtol=0, atol=0
    )
    assert report.tke_gain.sum() == 0


def test_implicit_drag_backward(ridge_mesh):
    coupled, state = setup(ridge_mesh)
    velocity = state.ice.velocity.clone().requires_grad_()
    state = replace(state, ice=replace(state.ice, velocity=velocity))
    end, _ = coupled.drag_step(state, 100.0)
    objective = end.ice.velocity.square().sum()
    derivative = torch.autograd.grad(objective, velocity)[0]
    direction = torch.cos(
        torch.arange(velocity.numel(), dtype=velocity.dtype)
    ).reshape_as(velocity)

    def value(offset):
        shifted = replace(
            state,
            ice=replace(state.ice, velocity=velocity.detach() + offset * direction),
        )
        result, _ = coupled.drag_step(shifted, 100.0)
        return result.ice.velocity.square().sum()

    finite = (value(1e-6) - value(-1e-6)) / 2e-6
    torch.testing.assert_close(
        (derivative * direction).sum(), finite, rtol=1e-6, atol=1e-9
    )


@pytest.mark.parametrize("fraction", [1e-6, 1e-12, 0.0])
def test_rotation_stays_bounded_at_vanishing_ice_margin(ridge_mesh, fraction):
    coupled, state = setup(ridge_mesh)
    factor = torch.where(ridge_mesh.cell_xyz[:, :1] > 0, fraction, 1.0)
    state = replace(
        state,
        ice=replace(
            state.ice, area=state.ice.area * factor, volume=state.ice.volume * factor
        ),
    )
    active = coupled.ridge.active_edges(state)
    state = replace(
        state,
        ice=replace(state.ice, velocity=torch.where(active, state.ice.velocity, 0.0)),
    )
    zero = torch.zeros_like(state.ice.area)
    with patch.object(coupled, "conjugates", return_value=(zero, zero)):
        _, _, acceleration, _ = coupled.ice_transport(state)
    # Rotation must stay finite on a physical Coriolis/advective timescale,
    # independently of how little ice remains. No inverse-mass blowup.
    mesh = ridge_mesh
    frequency = mesh.curl(state.ice.velocity).abs().max() + 2 * abs(
        coupled.ocean.ocean.base.dynamics.mechanical.rotation_rate
    )
    assert acceleration.abs().max() <= 100 * frequency * state.ice.velocity.abs().max()
    work = (coupled.hamiltonian.metric(state) * state.ice.velocity * acceleration).sum()
    scale = (
        (coupled.hamiltonian.metric(state) * state.ice.velocity * acceleration)
        .abs()
        .sum()
    )
    assert work.abs() <= 1e-13 * scale


def test_drag_at_partial_ice_margin(ridge_mesh):
    coupled, state = setup(ridge_mesh)
    present = ridge_mesh.cell_xyz[:, :1] > 0
    state = replace(
        state,
        ice=replace(
            state.ice,
            area=torch.where(present, state.ice.area, 0.0),
            volume=torch.where(present, state.ice.volume, 0.0),
        ),
    )
    active = coupled.ridge.active_edges(state)
    state = replace(
        state,
        ice=replace(state.ice, velocity=torch.where(active, state.ice.velocity, 0.0)),
    )
    state, leaves = leaf_state(state)
    gradients = torch.autograd.grad(coupled.hamiltonian.total_energy(state), leaves)
    rhs, _ = coupled.tendency(
        state, 1.0, ocean_dynamics=False, transport=False, ridging=False
    )
    products = [a * b for a, b in zip(gradients, rates(rhs))]
    assert sum(x.sum() for x in products).abs() < 1e-13 * sum(
        x.abs().sum() for x in products
    )
    end, report = coupled.drag_step(state, 100.0)
    assert report.energy_residual.abs() < 2e-10 * report.tke_gain.sum()
    assert torch.isfinite(end.ice.velocity).all()
