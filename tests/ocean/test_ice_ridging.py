"""Independent work, material, stage and derivative checks of compression ridging."""

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
from ridging import CategoryIce, RidgingState, CompressionRidging
from ice import IceOceanState, SeaIceState
from roquet import RoquetOcean
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn, OceanDomain


@pytest.fixture(scope="module")
def ridge_mesh():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    return VoronoiMesh(n_cells=32, radius=1e5, lloyd_iters=2, verbose=False)


def case(mesh, coast=False, empty=False):
    domain = OceanDomain(mesh, mesh.cell_xyz[:, 2] > -0.4) if coast else None
    model = RoquetOcean(mesh, OceanColumn(3, total_depth=200.0), domain=domain)
    h = torch.tensor([20.0, 60.0, 120.0]).expand(mesh.nCells, -1).clone()
    ocean = model.state(
        u_e=torch.zeros(mesh.nEdges, 3),
        h=h,
        conservative_temperature=torch.full_like(h, -1.8),
        absolute_salinity=torch.full_like(h, 34.0),
        tracer=torch.ones_like(h)[..., None],
        tke=1e-4 if domain is None else 1e-4 * domain.wet_cell[:, None],
        eddy_energy=0.01 if domain is None else 0.01 * domain.wet_cell[:, None],
    )
    area = torch.tensor([0.7, 0.0 if empty else 0.15]).expand(mesh.nCells, -1).clone()
    volume = area * torch.tensor([0.8, 4.0])
    velocity = mesh.grad(mesh.cell_xyz[:, :1]) * 1e4
    if domain:
        area = domain.apply_cell(area)
        volume = domain.apply_cell(volume)
        velocity = domain.apply_edge(velocity)
    return CompressionRidging(model), RidgingState(
        ocean, CategoryIce(area, volume, velocity)
    )


@pytest.mark.parametrize("coast", [False, True])
@pytest.mark.parametrize("friction,viscosity", [(0.0, 0.0), (1.0, 0.0), (1.0, 1e7)])
def test_independent_full_hamiltonian_contraction(
    ridge_mesh, coast, friction, viscosity
):
    original, state = case(ridge_mesh, coast)
    op = CompressionRidging(
        original.hamiltonian.model, friction=friction, bulk_viscosity=viscosity
    )
    s = state.ocean.ocean
    a, v, u, t = [
        x.clone().requires_grad_()
        for x in (state.ice.area, state.ice.volume, state.ice.velocity, s.temp_h)
    ]
    physical = s.with_tensors(s.u_e, s.h, t, s.salt_h, s.tracer_h)
    state = RidgingState(replace(state.ocean, ocean=physical), CategoryIce(a, v, u))
    gradients = torch.autograd.grad(op.hamiltonian.total_energy(state), (a, v, u, t))
    with patch("torch.autograd.grad", side_effect=AssertionError("forward AD")):
        rhs, report = op.tendency(state)
    works = [
        (g * r).sum()
        for g, r in zip(gradients, (rhs.area, rhs.volume, rhs.velocity, rhs.temp_h))
    ]
    scale = sum(x.abs() for x in works)
    # The full-H volume conjugates contain a large common latent datum.
    # Roundoff bound uses products BEFORE cancellation of category transfers.
    uncancelled = sum(
        (g * r).abs().sum()
        for g, r in zip(gradients, (rhs.area, rhs.volume, rhs.velocity, rhs.temp_h))
    )
    assert abs(sum(works)) < 32 * torch.finfo(t.dtype).eps * uncancelled
    assert (
        abs(report.energy_redistribution.sum()) < 256 * torch.finfo(t.dtype).eps * scale
    )
    assert (report.potential_power >= 0).all() and (report.heating_power >= 0).all()
    assert torch.count_nonzero(rhs.volume.sum(-1)) == 0
    assert (rhs.area.sum(-1) <= 0).all()
    # A missing ridge PE term is detectable; the thermal store is not a residual patch.
    assert report.potential_power.sum() > 1e8 * abs(sum(works))
    if coast:
        dry = ~op.hamiltonian.model.ocean.domain.wet_cell
        assert torch.count_nonzero(rhs.area[dry]) == 0
        assert torch.count_nonzero(rhs.temp_h[dry]) == 0


def test_slab_limit_and_variance_energy(ridge_mesh):
    op, state = case(ridge_mesh)
    a = torch.zeros_like(state.ice.area)
    a[:, 0] = 1.0
    v = a * 2.0
    state = replace(state, ice=CategoryIce(a, v, torch.zeros_like(state.ice.velocity)))
    H = op.hamiltonian
    expected = H.slab.total_energy(
        IceOceanState(state.ocean, SeaIceState(H.mass(state)))
    )
    torch.testing.assert_close(H.total_energy(state), expected, rtol=0, atol=0)
    # Same cell mass, thinner footprint -> greater subcell gravitational PE.
    reduced = replace(state, ice=replace(state.ice, area=0.5 * a))
    assert (
        H.terms(reduced)["ridge_potential"] > H.terms(state)["ridge_potential"]
    ).all()


def test_empty_receiver_birth_and_donor_empty(ridge_mesh):
    op, state = case(ridge_mesh, empty=True)
    rhs, report = op.tendency(state)
    end = op.increment(state, rhs, 1e-3)
    op.check(end)
    created = rhs.volume[:, 1] > 0
    torch.testing.assert_close(
        end.ice.thickness()[created, 1],
        op.ridge_ratio * state.ice.thickness()[created, 0],
    )
    # Independent one-sided potential-energy derivative at category birth.
    change = (
        op.hamiltonian.terms(end)["ridge_potential"]
        - op.hamiltonian.terms(state)["ridge_potential"]
    ) / 1e-3
    torch.testing.assert_close(change, report.potential_power, rtol=2e-5, atol=0.1)
    swapped = replace(
        state,
        ice=replace(
            state.ice, area=state.ice.area.flip(-1), volume=state.ice.volume.flip(-1)
        ),
    )
    rhs, _ = op.tendency(swapped)
    assert torch.count_nonzero(rhs.volume) == 0


def test_rest_and_diverging_cells_do_not_ridge(ridge_mesh):
    op, state = case(ridge_mesh)
    resting = replace(
        state, ice=replace(state.ice, velocity=torch.zeros_like(state.ice.velocity))
    )
    rhs, _ = op.tendency(resting)
    for name in ("area", "volume", "velocity", "temp_h"):
        assert torch.count_nonzero(getattr(rhs, name)) == 0
    rhs, report = op.tendency(state)
    separating = ridge_mesh.div(state.ice.velocity)[:, 0] >= 0
    assert torch.count_nonzero(report.pressure[separating]) == 0
    assert torch.count_nonzero(rhs.volume[separating]) == 0


def test_step_material_budget_and_backward(ridge_mesh):
    op, state = case(ridge_mesh)
    amplitude = torch.tensor(1.0, requires_grad=True)

    def advance(x):
        return op.step(
            replace(state, ice=replace(state.ice, velocity=x * state.ice.velocity)),
            10.0,
        )

    end = advance(amplitude)
    (grad,) = torch.autograd.grad(end.ice.volume[:, 1].sum(), amplitude)
    fd = (
        advance(amplitude.detach() + 1e-4).ice.volume[:, 1].sum()
        - advance(amplitude.detach() - 1e-4).ice.volume[:, 1].sum()
    ) / 2e-4
    torch.testing.assert_close(grad, fd, rtol=2e-7, atol=1e-9)
    torch.testing.assert_close(
        end.ice.volume.sum(-1), state.ice.volume.sum(-1), rtol=3e-16, atol=0
    )
    for name in ("h", "salt_h", "tracer_h", "u_e"):
        assert torch.equal(
            getattr(end.ocean.ocean, name), getattr(state.ocean.ocean, name)
        )
    assert torch.equal(end.ocean.tke_h, state.ocean.tke_h)
    assert torch.equal(end.ocean.eddy_h, state.ocean.eddy_h)


def test_invalid_stores_reject_without_clipping(ridge_mesh):
    op, state = case(ridge_mesh)
    for bad in [
        replace(state.ice, area=2 * state.ice.area),
        replace(state.ice, area=-state.ice.area),
        replace(state.ice, volume=state.ice.area * torch.tensor([4.0, 0.8])),
        replace(state.ice, area=torch.zeros_like(state.ice.area)),
    ]:
        with pytest.raises(ValueError):
            op.check(replace(state, ice=bad))
    with pytest.raises(ValueError):
        op.step(state, 1e8)


def test_ice_free_limit(ridge_mesh):
    op, state = case(ridge_mesh)
    ice = CategoryIce(
        torch.zeros_like(state.ice.area),
        torch.zeros_like(state.ice.volume),
        torch.zeros_like(state.ice.velocity),
    )
    state = replace(state, ice=ice)
    end = op.step(state, 1000.0)
    assert torch.equal(end.ocean.ocean.temp_h, state.ocean.ocean.temp_h)
    assert torch.equal(end.ice.area, state.ice.area)
    assert torch.equal(end.ice.volume, state.ice.volume)
    assert torch.equal(end.ice.velocity, state.ice.velocity)


def test_finite_step_error_converges_without_repair():
    from run_ridging import initial_case, stable_change

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    op, initial = initial_case(32, False)
    errors = []
    for dt in (500.0, 250.0, 125.0):
        state = initial
        for _ in range(round(1000.0 / dt)):
            state = op.step(state, dt)
        stable = sum(stable_change(op.hamiltonian, initial, state))
        direct = op.hamiltonian.total_energy(state) - op.hamiltonian.total_energy(
            initial
        )
        datum = op.hamiltonian.total_energy(initial).abs()
        assert (stable - direct).abs() < 16 * torch.finfo(torch.float64).eps * datum
        errors.append(float(stable.abs()))
    # Smooth-branch RK4 refinement, not an assertion of physical calibration.
    assert errors[0] / errors[1] > 10 and errors[1] / errors[2] > 10
