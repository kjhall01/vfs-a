"""Zero-opening shelf slots carry no momentum, KE, or circulation information."""

from dataclasses import replace

import pytest
import torch

from vfs import VoronoiMesh
from vfs.ocean.column import OceanColumn
from vfs.ocean.domain import OceanDomain
from vfs.ocean.temperature.backscatter import EddyBackscatter
from vfs.ocean.temperature.roquet import RoquetMixing, RoquetOcean
from vfs.ocean.temperature.shelf_momentum import (
    install_opening_momentum,
    project_changed_faces,
    wet_ocean_peak,
)
from vfs.ocean.temperature.shelf_transport import ShelfTemperatureOcean


@pytest.fixture
def shelf():
    mesh = VoronoiMesh(
        n_cells=48, lloyd_iters=0, rotation_scheme="compatible", verbose=False
    )
    depth = mesh.cell_area.new_full((mesh.nCells,), 100.0)
    edge = 15
    neighbors = mesh.cellsOnEdge[edge]
    depth[neighbors[0]] = 50.0
    col = OceanColumn(
        2,
        total_depth=depth,
        rest_thickness=depth.new_tensor([30.0, 70.0]),
        partial_cells=True,
    )
    domain = OceanDomain.global_ocean(mesh, depth=depth)
    model = RoquetOcean(
        mesh,
        col,
        domain=domain,
        closure=RoquetMixing(10.0, substeps=1),
        backscatter=EddyBackscatter(1e4, 30 * 86400.0, 0.1, 1e13),
    )
    model.dynamics = ShelfTemperatureOcean(model.ocean)
    en = install_opening_momentum(model)
    h = col.layer_reference
    h2 = h.clone()
    change = 0.85 * h[neighbors, 1]
    h2[neighbors, 0] += change
    h2[neighbors, 1] -= change
    u = (
        torch.sin(torch.arange(mesh.nEdges, device=h.device, dtype=h.dtype))[:, None]
        .expand(-1, 2)
        .clone()
    )
    u[edge] = h.new_tensor([1.0, 3.0])
    state = model.state(
        u_e=u,
        h=h,
        conservative_temperature=torch.full_like(h, 2.0),
        absolute_salinity=torch.full_like(h, 35.0),
        tke=0.1,
        eddy_energy=0.1,
    )
    s = state.ocean
    closed = s.with_tensors(u, h2, 2 * h2, 35 * h2, s.tracer_h)
    opening = en.edge_thickness(closed, h2)
    assert domain.active_edge_layers[edge, 1] and opening[edge, 1] == 0
    assert en.edge_thickness(s, h)[edge, 1] > 0
    return model, state, closed, edge


def velocity(s, u):
    return s.with_tensors(u, s.h, s.temp_h, s.salt_h, s.tracer_h)


def test_rotation_and_backscatter_ignore_stale_normal(shelf):
    model, state, closed, edge = shelf
    energy = model.energy.base.mechanical
    wet = energy.metric(closed) > 0
    a = velocity(closed, torch.where(wet, closed.u_e, 0.0))
    b = velocity(a, torch.where(wet, a.u_e, 123.0))
    rot = model.ocean.base.dynamics.mechanical.rotation_acceleration
    rhs0, rep = model.dynamics.tendency_with_report(a, 1)
    rhs1, _ = model.dynamics.tendency_with_report(b, 1)
    assert torch.equal(rhs0.u_e, rhs1.u_e) and torch.equal(rhs0.h, rhs1.h)
    assert torch.equal(rhs0.temp_h, rhs1.temp_h) and torch.equal(
        rhs0.salt_h, rhs1.salt_h
    )
    peak, peak_edge, level = wet_ocean_peak(model, b)
    assert wet[peak_edge, level] and peak == float(a.u_e.abs().max())
    peak, peak_edge, level = wet_ocean_peak(
        model, velocity(b, torch.where(wet, 0.0, b.u_e))
    )
    assert peak == 0.0 and wet[peak_edge, level]
    assert torch.count_nonzero(rot(b)[~wet]) == 0
    work = rep.metric * a.u_e * rot(a)
    assert work.sum().abs() < 5e-14 * work.abs().sum()
    f0, e0, _ = model.backscatter.tendency(model, replace(state, ocean=a))
    f1, e1, r = model.backscatter.tendency(model, replace(state, ocean=b))
    assert torch.equal(f0, f1) and torch.equal(e0, e1)
    assert f1[edge, 1] == 0
    torch.testing.assert_close(
        r.biharmonic_kinetic_power + r.biharmonic_reservoir_power,
        r.biharmonic_reservoir_power * 0,
        rtol=0,
        atol=float(r.biharmonic_reservoir_power) * 1e-12,
    )
    # Closed mass is decoupled from the tridiagonal mixing solve.
    mixa, ka, _ = model.closure.apply(model.ocean, a, state.tke_h, 1.0)
    mixb, kb, _ = model.closure.apply(model.ocean, b, state.tke_h, 1.0)
    torch.testing.assert_close(mixa.u_e, mixb.u_e, rtol=0, atol=0)
    torch.testing.assert_close(ka, kb, rtol=0, atol=0)


def test_close_and_reopen_momentum_and_thermal_budget(shelf):
    model, state, closed, edge = shelf
    s = state.ocean
    en = model.energy.base.mechanical
    before = en.metric(s)
    after = en.metric(closed)
    # Same sea level and bottom; only interfaces moved. All edges except the
    # target have vertically uniform velocity, so rezoning cannot change their KE.
    merged = model.dynamics.finish_euler(s, closed)
    torch.testing.assert_close(
        (before * s.u_e).sum(1), (after * merged.u_e).sum(1), rtol=8e-15, atol=64.0
    )
    loss = 0.5 * (before * s.u_e.square() - after * merged.u_e.square()).sum()
    base = model.ocean.energy.base
    a, b, _, _ = base.coefficients(closed)
    t = closed.temperature()
    delta = merged.temperature() - t
    heat = (
        model.ocean.energy.mass_area(closed)
        * closed.h
        * delta
        * (model.ocean.reference.cp_theta + 2 * a * t + b + a * delta)
    ).sum()
    assert loss > 0
    torch.testing.assert_close(heat, loss, rtol=3e-10, atol=0.0)
    dry = after == 0
    stored = torch.where(dry, 123.0, merged.u_e)
    ghost = velocity(merged, stored)
    target = s.with_tensors(stored, s.h, s.temp_h, s.salt_h, s.tracer_h)
    reopened = model.dynamics.finish_euler(ghost, target)
    assert reopened.u_e[edge, 1] == pytest.approx(float(merged.u_e[edge, 0]), rel=1e-14)
    torch.testing.assert_close(
        (after * merged.u_e).sum(1),
        (before * reopened.u_e).sum(1),
        rtol=8e-15,
        atol=64.0,
    )
    torch.testing.assert_close(
        (after * merged.u_e.square()).sum(),
        (before * reopened.u_e.square()).sum(),
        rtol=8e-15,
        atol=0.0,
    )
    for other in (s, merged):
        clean = model.dynamics.finish_blend(
            merged, other, 0.25 * merged + 0.75 * other, (0.25, 0.75)
        )
        dirty = model.dynamics.finish_blend(
            ghost, other, 0.25 * ghost + 0.75 * other, (0.25, 0.75)
        )
        assert torch.equal(clean.u_e, dirty.u_e)
        assert torch.equal(clean.temp_h, dirty.temp_h)


def test_exposed_surface_and_disappearing_water():
    old = torch.tensor([[0.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    new = torch.tensor([[1.0, 3.0, 2.0], [0.0, 1.0, 2.0]], dtype=torch.float64)
    u = torch.tensor([[999.0, 2.0, -1.0], [999.0, -999.0, 999.0]], dtype=old.dtype)
    mapped, heat = project_changed_faces(old, new, u)
    # One metre entering at the old surface inherits its 2 m/s speed.
    torch.testing.assert_close(
        (new * mapped).sum(1), (old * u).sum(1) + old.new_tensor([2.0, 0.0])
    )
    torch.testing.assert_close(
        (0.5 * new * mapped.square() + heat).sum(1),
        (0.5 * old * u.square()).sum(1) + old.new_tensor([2.0, 0.0]),
    )
    assert torch.all(heat >= 0) and torch.count_nonzero(mapped[1]) == 0
    mapped, heat = project_changed_faces(new, new * 0, u)
    assert torch.count_nonzero(mapped) == torch.count_nonzero(heat) == 0


def test_ale_reopens_from_water_with_identical_clean_and_dirty_results(shelf):
    model, state, closed, edge = shelf
    wet = model.energy.base.mechanical.metric(closed) > 0
    clean = replace(
        state,
        ocean=velocity(closed, torch.where(wet, closed.u_e, 0.0)),
        tke_h=closed.h * 0.1,
        eddy_h=closed.h * 0.1,
    )
    dirty = replace(clean, ocean=velocity(closed, torch.where(wet, closed.u_e, 123.0)))
    reopened, report = model.remap_with_report(clean, state.ocean.h)
    comparison, _ = model.remap_with_report(dirty, state.ocean.h)
    assert torch.equal(reopened.ocean.u_e, comparison.ocean.u_e)
    assert torch.equal(reopened.ocean.temp_h, comparison.ocean.temp_h)
    assert reopened.ocean.u_e[edge, 1] == pytest.approx(float(clean.ocean.u_e[edge, 0]))
    momentum = model.energy.base.mechanical.metric(closed) * clean.ocean.u_e
    assert report.momentum_residual.abs().max() < 1e-14 * momentum.abs().max()
    end, _ = model.remap_with_report(dirty, dirty.ocean.h)
    assert torch.equal(end.ocean.u_e, clean.ocean.u_e)
