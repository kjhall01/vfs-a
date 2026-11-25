from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("pyam21")
from test_coupled_campaign import config, source, synthetic
from test_pyam21_workflow import same_tree
from workflows.coupled import campaign as c
from vfs.ocean.temperature.shelf_redi import CoupledShelfRedi


@torch.no_grad()
def test_projected_shelf_redi_coupled_and_restart(config, source):
    cfg = replace(config, shelf_redi=True, ocean_levels=6)
    land = np.arange(cfg.cells) == 0
    depths = np.linspace(8.0, cfg.ocean_depth_m, cfg.cells)
    geography = dict(land=land, depth_m=depths, height_m=1200.0 * land)
    with patch("workflows.pyam21.radiation.build_radiation", side_effect=synthetic):
        adapter, air, receiver, sea = c.build(
            cfg, source, Path("synthetic"), geography=geography
        )
    model = receiver.H.model
    state = sea.coupled.ocean
    op = CoupledShelfRedi(model, state, c.rest_thickness(cfg), c.gm_redi_operator(cfg))
    torch.manual_seed(212)
    x = torch.randn_like(state.ocean.h)[..., None]
    g = op.apply(x)
    y = torch.randn_like(g)
    torch.testing.assert_close(
        (g * y).sum(), (x * op.transpose(y)).sum(), atol=1e-14, rtol=1e-12
    )
    assert (g.square() * op.rows.measure[:, None]).sum() > 0
    assert op.transpose(y)[land].count_nonzero() == 0
    torch.testing.assert_close(
        op.apply(torch.ones_like(x)), torch.zeros_like(g), rtol=0, atol=1e-13
    )
    # Force a positive-field withdrawal that exceeds the native inventory.
    transfer = 1.0e10 * y
    amounts = torch.ones_like(x)
    scale = op.positive_flux_scale(transfer, amounts)
    assert (scale < 1.0).any()
    updated = amounts + op.transpose(scale * transfer)
    assert updated.min() >= 0.0
    torch.testing.assert_close(updated.sum(), amounts.sum(), rtol=2.0e-15, atol=1.0e-13)
    saved = c.pack(adapter, air, receiver, sea, "shelf", {})
    with patch.object(
        CoupledShelfRedi, "apply", autospec=True, side_effect=CoupledShelfRedi.apply
    ) as ran:
        a, s, report = c.advance(adapter, air, receiver, sea, cfg.physics_dt, cfg)
        assert ran.call_count == report["gmredi_steps"] > 0
    ra, rs, _ = c.unpack(saved, adapter, receiver, sea, "shelf")
    b, t, replay = c.advance(adapter, ra, receiver, rs, cfg.physics_dt, cfg)
    same_tree(
        c.pack(adapter, a, receiver, s, "shelf", {}),
        c.pack(adapter, b, receiver, t, "shelf", {}),
    )
    assert report == replay and s.land is not None


def test_mesh_order_check(tmp_path):
    path = tmp_path / "geography.npz"
    np.savez(
        path,
        land=[True, False],
        height_m=[100.0, 0.0],
        depth_m=[0.0, 80.0],
        metadata='{"mesh_sha256":"expected"}',
    )
    with pytest.raises(ValueError, match="mesh_sha256"):
        c.load_geography(path, mesh_sha256="different")
    assert c.load_geography(path, mesh_sha256="expected")["height_m"][0] == 100.0


@torch.no_grad()
def test_shelf_resolved_and_gm_transport_share_physical_openings(config, source):
    from vfs.ocean.transport import divergence
    from vfs.ocean.temperature.shelf_transport import ShelfTemperatureOcean

    cfg = replace(config, shelf_redi=True, ocean_levels=6)
    land = np.arange(cfg.cells) == 0
    depths = np.linspace(8.0, cfg.ocean_depth_m, cfg.cells)
    geography = dict(land=land, depth_m=depths, height_m=1200.0 * land)
    with patch("workflows.pyam21.radiation.build_radiation", side_effect=synthetic):
        _, _, receiver, sea = c.build(
            cfg, source, Path("synthetic"), geography=geography
        )
    model = receiver.H.model
    state = sea.coupled.ocean
    s = state.ocean
    assert isinstance(model.dynamics, ShelfTemperatureOcean)
    m = s.mesh
    wet = model.ocean.domain.active_edge
    generator = torch.Generator().manual_seed(255)

    def rand(x):
        return torch.randn(x.shape, generator=generator, dtype=x.dtype)

    u = 0.03 * rand(s.u_e) * wet[:, None]
    q = torch.stack(((rand(s.h) > 0).to(s.h), torch.full_like(s.h, 0.25)), -1)
    s = s.with_tensors(u, s.h, s.temp_h, s.salt_h, s.h[..., None] * q)
    rhs, report = model.dynamics.tendency_with_report(s, 60.0)
    faces = report.faces
    k = s.h.shape[1]
    edge = faces.velocity_slot // k
    assert (faces.left % k != faces.right % k).any()
    torch.testing.assert_close(m.cellsOnEdge[edge, 0], faces.left // k)
    torch.testing.assert_close(m.cellsOnEdge[edge, 1], faces.right // k)
    assert wet[edge].all()
    area = s.h.new_zeros(m.nEdges).index_add(0, edge, faces.area)
    depth = model.ocean.domain.depth[:, 0]
    expected = torch.minimum(depth[m.cellsOnEdge[:, 0]], depth[m.cellsOnEdge[:, 1]])
    torch.testing.assert_close(area, wet * m.dv_edge * expected)
    torch.testing.assert_close(rhs.tracer_h[:, :, 1], 0.25 * rhs.h)
    end = report.transport.install(s)
    # Compare finite entropy with the actual donor-water inflow in each cell.
    entropy = torch.log1p(s.temperature() / 273.15)
    flux = 60.0 * report.transport.carrier
    donor = torch.where(flux >= 0, faces.left, faces.right)
    advected = (
        s.h * entropy
        + divergence(
            faces.left, faces.right, flux * entropy.flatten()[donor], s.h.numel()
        ).reshape_as(s.h)
        / m.cell_area[:, None]
    )
    assert (end.h * torch.log1p(end.temperature() / 273.15) >= advected - 1e-12).all()
    assert end.tracer_h.min() >= 0
    assert (end.tracer_h[..., 0] <= end.h + 1e-12).all()
    for field in (rhs.h, rhs.temp_h, rhs.salt_h, rhs.tracer_h[..., 0]):
        amount = m.cell_area[:, None] * field
        assert amount.sum().abs() < 1e-13 * amount.abs().sum()
    load = 100 * rand(s.h[:, :1])
    load_force = model.dynamics.loading_acceleration(s, load, report)
    grad = model.energy.base.gradients(s)
    work = (
        (grad["h"] + load * m.cell_area[:, None]) * rhs.h
        + grad["temp_h"] * rhs.temp_h
        + grad["salt_h"] * rhs.salt_h
    ).sum()
    kinetic = grad["u_e"] * (rhs.u_e + load_force)
    assert (work + kinetic.sum()).abs() < 2e-12 * kinetic.abs().sum()

    eddy = c.gm_redi_operator(cfg)
    metrics = eddy._metrics(model, state)
    _, psi, horizontal, vertical, i, j, carrier, _, _ = eddy._geometry(
        model, state, metrics=metrics
    )
    assert horizontal.abs().max() > 0
    assert horizontal[~wet].count_nonzero() == 0
    assert vertical[:, [0, -1]].count_nonzero() == 0
    assert psi[:, [0, -1]].count_nonzero() == 0
    residual = divergence(i, j, carrier, s.h.numel())
    assert residual.abs().max() < 5e-14 * carrier.abs().max()


@torch.no_grad()
def test_prognostic_partial_columns_through_coupling_ale_and_disk_restart(config, source, tmp_path):
    """One-layer, merged-bottom, 40 m and abyssal water share a physical shelf."""
    import xarray as xr

    cfg = replace(config, ocean_levels=40, ocean_depth_m=5000., top_layer_m=10.,
                  shelf_redi=True, partial_cell_layers=True, ale_dt=config.physics_dt)
    depths = np.resize([8., 40., 11., 5000., 5600., 300., 600., 2000.], cfg.cells)
    land = np.arange(cfg.cells) == 3
    geo = dict(land=land, depth_m=depths, height_m=300. * land)
    with patch("workflows.pyam21.radiation.build_radiation", side_effect=synthetic):
        adapter, air, receiver, sea = c.build(cfg, source, Path("synthetic"), geography=geo)
    model = receiver.H.model
    domain = model.ocean.domain
    initial = sea.coupled.ocean.ocean
    ref = initial.column.layer_reference
    active = initial.column.active_layers
    torch.testing.assert_close(ref[:3, :2], ref.new_tensor([[8., 0.], [10., 30.], [11., 0.]]))
    torch.testing.assert_close(ref[~land].sum(-1), ref.new_tensor(depths[~land]))
    assert active[4].all() and active[2].sum() == 1
    assert sea.coupled.ice.volume[~land].sum() > 0

    def check_state(water):
        st = water.coupled.ocean
        for x in (st.ocean.h, st.ocean.temp_h, st.ocean.salt_h, st.ocean.tracer_h, st.tke_h, st.eddy_h):
            assert torch.isfinite(x).all()
            assert torch.count_nonzero(x[~active]) == 0
        assert torch.all(st.ocean.h[active] > 0)
        assert torch.count_nonzero(st.ocean.u_e[~domain.active_edge_layers]) == 0
        model.check(st)

    check_state(sea)
    # Exercise the pressure/transport work identity with moving water at the shelf wall.
    generator = torch.Generator().manual_seed(92)
    u = .02 * torch.randn(initial.u_e.shape, dtype=initial.u_e.dtype, generator=generator)
    water = initial.with_tensors(domain.apply_layer_edge(u), initial.h, initial.temp_h,
                                 initial.salt_h, initial.tracer_h)
    # Vary only the land dummy column. It has no physical volume and must
    # contribute neither vertex stress nor eddy energy at adjacent wet faces.
    moving = replace(sea.coupled.ocean, ocean=water)
    land_factor = torch.where(domain.wet_cell[:, None], 1., 2.).to(water.h)
    dummy = water.with_tensors(water.u_e, water.h*land_factor, water.temp_h*land_factor,
                               water.salt_h*land_factor, water.tracer_h*land_factor[..., None])
    du, de, _ = model.backscatter.tendency(model, moving)
    other_du, other_de, _ = model.backscatter.tendency(model, replace(moving, ocean=dummy))
    assert torch.equal(du, other_du) and torch.equal(de, other_de)
    rhs, rhs_report = model.dynamics.tendency_with_report(water, cfg.ocean_dt)
    faces = rhs_report.faces
    assert domain.active_edge_layers.flatten()[faces.velocity_slot].all()
    assert active.flatten()[faces.left].all() and active.flatten()[faces.right].all()
    grad = model.energy.base.gradients(water)
    work = sum((grad[k] * getattr(rhs, k)).sum() for k in ('h', 'temp_h', 'salt_h', 'u_e'))
    scale = (grad['u_e'] * rhs.u_e).abs().sum()
    assert work.abs() <= 2e-12 * scale
    sea = replace(sea, coupled=replace(sea.coupled, ocean=replace(sea.coupled.ocean, ocean=water)))
    air, sea, report = c.advance(adapter, air, receiver, sea, cfg.physics_dt, cfg)
    check_state(sea)
    assert report['ale_steps'] == 1 and report['gmredi_steps'] == cfg.eddy_substeps
    assert report['ale_column_residual_J_m2'] < 1e-3
    after = sea.coupled.ocean.ocean.h
    torch.testing.assert_close(after / after.sum(-1, keepdim=True), ref / ref.sum(-1, keepdim=True))
    c.snapshot(tmp_path / "active.nc", adapter, air, receiver, sea)
    with xr.open_dataset(tmp_path / "active.nc", decode_times=False) as ds:
        np.testing.assert_array_equal(ds.ocean_active_layer_count, (active & domain.wet_cell[:, None]).sum(-1))
        wet = (active & domain.wet_cell[:, None]).cpu().numpy()
        np.testing.assert_array_equal(ds.ocean_layer_depth.notnull()[0], wet)
        np.testing.assert_array_equal(ds.ocean_reference_thickness > 0, wet)
    identity = c.fingerprint(cfg, adapter, {}, False)
    path = tmp_path / "restart.nc"
    c.save_restart(path, adapter, air, receiver, sea, identity, {})
    restored_air, restored_sea, _ = c.unpack(c.read_checkpoint(path), adapter, receiver, sea, identity)
    a, s, r = c.advance(adapter, air, receiver, sea, cfg.physics_dt, cfg)
    b, t, replay = c.advance(adapter, restored_air, receiver, restored_sea, cfg.physics_dt, cfg)
    check_state(s)
    assert r == replay
    same_tree(c.pack(adapter, a, receiver, s, identity, {}), c.pack(adapter, b, receiver, t, identity, {}))


@torch.no_grad()
def test_single_layer_defers_infeasible_ale_at_shared_moving_face(config, source):
    """A faster shelf current cannot be funded by cooling its only layer."""
    from vfs.ocean.stratified.remap import ThermalProjectionInfeasible

    cfg = replace(config, ocean_levels=5, shelf_redi=True, partial_cell_layers=True)
    depth = np.full(cfg.cells, cfg.ocean_depth_m)
    depth[0] = 8.
    land = np.arange(cfg.cells) == cfg.cells - 1
    geo = dict(land=land, depth_m=depth, height_m=100. * land)
    _, _, receiver, sea = c.build(cfg, source, None, geography=geo, smoke_no_radiation=True)
    model = receiver.H.model
    s = sea.coupled.ocean.ocean
    domain = model.ocean.domain
    edge = torch.where((s.mesh.cellsOnEdge == 0).any(-1) & domain.active_edge)[0][0]
    h = s.h.clone()
    deep = domain.wet_cell & (s.column.active_layers.sum(-1) > 1)
    # Returning all deep upper layers to reference makes the moving face
    # lighter. Conserving its momentum then increases the shelf column's KE.
    h[deep, 0] += 1.
    h[deep, 1] -= 1.
    u = torch.zeros_like(s.u_e)
    u[edge, 0] = 1.
    cells = s.mesh.cellsOnEdge
    shelf_edges = (cells == 0).any(-1)
    neighbors = torch.zeros_like(deep)
    neighbors[cells[shelf_edges].flatten()] = True
    remote = torch.where(domain.active_edge & ~neighbors[cells].any(-1))[0][0]
    profile = torch.linspace(.1, .3, cfg.ocean_levels, dtype=h.dtype)
    u[remote] = profile
    tracer = profile.expand_as(h)[..., None]
    state = model.state(u_e=u, h=h, conservative_temperature=torch.full_like(h, 10.),
                        absolute_salinity=torch.full_like(h, 35.), tracer=tracer,
                        tke=1e-4 * domain.wet_cell[:, None], eddy_energy=.1)
    ref = s.column.layer_reference
    target = h.sum(-1, keepdim=True) * (ref / ref.sum(-1, keepdim=True))
    with pytest.raises(ThermalProjectionInfeasible) as failure:
        model.remap_with_report(state, target)
    assert failure.value.columns[0]
    sea = replace(sea, coupled=replace(sea.coupled, ocean=state))
    remapped, report = c.ale_step(receiver, sea, cfg)
    end = remapped.coupled.ocean.ocean
    assert report["ale_changed_columns"] > 0 and report["ale_deferred_columns"] > 0
    assert report["ale_changed_columns"] + report["ale_deferred_columns"] == int(deep.sum())
    assert torch.equal(end.u_e[edge], u[edge])
    assert torch.equal(end.temp_h[0], state.ocean.temp_h[0])
    assert torch.equal(end.h[:, 0] != h[:, 0], deep & (end.h[:, 0] == target[:, 0]))
    # Compare the model's actual generalized momentum before and after ALE.
    metric = model.energy.base.mechanical.metric
    momentum0 = metric(state.ocean) * u
    momentum1 = metric(end) * end.u_e
    assert not torch.equal(momentum0[remote], momentum1[remote])
    torch.testing.assert_close(momentum1.sum(-1), momentum0.sum(-1), rtol=2e-14, atol=0.)
    for old, new in ((state.ocean.salt_h, end.salt_h),
                     (state.ocean.tracer_h[..., 0], end.tracer_h[..., 0]),
                     (state.tke_h, remapped.coupled.ocean.tke_h),
                     (state.eddy_h, remapped.coupled.ocean.eddy_h)):
        torch.testing.assert_close(new.sum(-1), old.sum(-1), rtol=2e-14, atol=0.)
