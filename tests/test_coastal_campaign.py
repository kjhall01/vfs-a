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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_earth_pressure_matches_moist_profile(source, config, dtype):
    from pyam21.sat_vapor_pres import get_sat_vapor_pres

    # Equal-height mountains span different latitudes, with sea-level controls.
    heights = np.resize([0., 1500., 4500.], config.cells)
    geography = dict(land=heights > 0, height_m=heights, depth_m=np.full(config.cells, 200.))
    config = replace(config, dtype=str(dtype).removeprefix("torch."))
    adapter, air, *_ = c.build(config, source, None, geography=geography, smoke_no_radiation=True)
    model = adapter.model
    ps = air.atmosphere.ps.reshape(-1, 1)
    assert ps.dtype == dtype
    assert (ps[heights == 0] == 100000.).all()

    # Independently recover the geopotential by summing specific volume * dp
    # between the initialized surface pressure and the sea-level reference pressure.
    ps = ps.double()
    p = ps + (100000. - ps) * torch.linspace(0., 1., 4097, dtype=ps.dtype)
    sea_level_temperature = 300. - 25. * model.mesh.cell_xyz[:, 2:3].double().square()
    temperature = 205. + (sea_level_temperature - 205.) * (p / 100000.) ** .7
    vapor = .7 * get_sat_vapor_pres().compute_qs(temperature, p)
    zero = torch.zeros_like(vapor)
    fractions = torch.stack((1 - vapor, vapor, zero, zero), -1)
    alpha = model.dynamics.thermodynamics.specific_volume(temperature, p, fractions)
    reconstructed_phi = torch.trapezoid(alpha, p, dim=-1)
    torch.testing.assert_close(reconstructed_phi, model.column.phi_s.double(), rtol=0, atol=.1)


def test_terrain_reaching_lid_is_rejected(source):
    from workflows.pyam21.setup import example

    with pytest.raises(ValueError, match="terrain reaches the atmospheric pressure lid"):
        example(source, cells=16, levels=12, top_pressure_pa=95000., surface_height_m=2000.)


@torch.no_grad()
def test_coastal_runoff_ice_and_restart(config, source, tmp_path):
    # One elevated island cell adjacent to a partly ice-covered, sloping shelf.
    n = config.cells
    mask = np.arange(n) == 2
    geography = dict(land=mask, height_m=1000. * mask, depth_m=np.linspace(180., 220., n))
    with patch("workflows.pyam21.radiation.build_radiation", side_effect=synthetic):
        adapter, air, receiver, sea = c.build(config, source, Path("synthetic"), geography=geography)
    l = receiver.land_surface
    domain = receiver.H.model.ocean.domain
    mesh = adapter.model.mesh
    A = mesh.cell_area
    s = sea.coupled.ocean.ocean
    assert air.atmosphere.ps[l.land].max() < air.atmosphere.ps[l.wet].min()
    torch.testing.assert_close(s.h.sum(-1)[l.wet], domain.depth[l.wet, 0], atol=.1, rtol=0)
    cover = .3 * l.wet[:, None] * s.h.new_tensor([.8, .2])
    # Convergent flow on the wet approaches to the closed coastline.
    near = torch.zeros_like(A)
    near[mesh.cellsOnEdge[domain.coast_edge].reshape(-1)] = 1.
    velocity = .01 * domain.apply_edge(mesh.grad(near[:, None]).sign())
    assert velocity.abs().max() > 0
    ice = c.CategoryIce(cover, cover * s.h.new_tensor([.5, 1.]), velocity)
    s = s.with_tensors(s.u_e, s.h, s.h * (-.054 * 34.), s.salt_h, s.tracer_h)
    sea = replace(sea, coupled=replace(sea.coupled, ocean=replace(sea.coupled.ocean, ocean=s), ice=ice),
                  cold_content=-2. * receiver.snow.heat_capacity * receiver.H.slab.parameters.density * ice.volume,
                  land=l.column.initialize(sea.land.skin, soil_water=150.))
    initial = c.inventories(adapter, air, receiver, sea)
    for _ in range(2):
        air, sea, report = c.advance(adapter, air, receiver, sea, 60, config)
        assert report["runoff_kg_m2"] > 0
        assert report["runoff_gravity_heat_J_m2"] == pytest.approx(9800. * report["runoff_kg_m2"])
        assert report["surface_residual"] < 1e-4
        assert sea.coupled.ice.velocity[domain.coast_edge].count_nonzero() == 0
        assert sea.coupled.ocean.ocean.u_e[domain.coast_edge].count_nonzero() == 0
        assert sea.coupled.ice.volume[l.land].count_nonzero() == 0
        assert sea.coupled.ice.volume[l.wet].sum() > 0
    current = c.inventories(adapter, air, receiver, sea)
    assert abs(current["water"] - initial["water"]) < 1e-8
    assert abs(current["salt"] - initial["salt"]) < 1e-9
    identity = c.fingerprint(config, adapter, {}, False)
    payload = c.pack(adapter, air, receiver, sea, identity, {})
    path = tmp_path / "restart.pt"
    c.checkpoint(path, payload)
    ra, rs, _ = c.unpack(torch.load(path, weights_only=True), adapter, receiver, sea, identity)
    a, s, r = c.advance(adapter, air, receiver, sea, 60, config)
    b, t, u = c.advance(adapter, ra, receiver, rs, 60, config)
    same_tree(c.pack(adapter, a, receiver, s, identity, {}), c.pack(adapter, b, receiver, t, identity, {}))
    assert r == u
    c.snapshot(tmp_path / "coast.nc", adapter, a, receiver, s)
    import xarray as xr
    with xr.open_dataset(tmp_path / "coast.nc", decode_times=False) as ds:
        assert ds.land_mask.dims == ("cell",) and "land_mask" in ds.coords
        assert ds.land_cell_index.dims == ("land_cell",)
        assert ds.soil_layer.attrs["positive"] == "down"
        assert ds.air_pressure.dims == ("time", "cell", "air_interface")
        assert ds.mesh.attrs["cf_role"] == "mesh_topology"
        assert ds.ocean_normal_wind.attrs["location"] == "edge"
        assert ds.ocean_temperature.attrs["location"] == "face"
        np.testing.assert_array_equal(ds.vertices_on_cell.count("max_vertices"), mesh.nEdgesOnCell)
        np.testing.assert_array_equal(ds.cells_on_edge, mesh.cellsOnEdge)
        np.testing.assert_array_equal(ds.land_mask, mask)
        np.testing.assert_array_equal(ds.land_soil_water[0], s.land.soil_water)


@torch.no_grad()
def test_runoff_enters_only_wet_outlet_and_its_hamiltonian(source, config):
    from workflows.exchange import Exchange

    adapter, air, receiver, sea = c.build(config, source, None, island=True, smoke_no_radiation=True)
    land = receiver.land_surface
    A = land.mesh.cell_area
    soil = land.column.initialize(sea.land.skin, soil_water=150.)
    sea = replace(sea, land=soil)
    tiles = receiver.tiles(sea)
    zeros = torch.zeros_like(A)
    tiled = torch.zeros_like(tiles.fraction)
    vectors = torch.zeros_like(tiles.velocity)
    terms = {k: zeros for k in ("sensible", "vapor_enthalpy", "rain_enthalpy", "snow_enthalpy",
                                "water_potential", "water_kinetic", "radiation", "stress_work")}
    flux = Exchange(60, 0, tiled, zeros, zeros, tiled, vectors, vectors, vectors[:, 0],
                    terms, zeros, zeros, zeros, tiled)
    candidate, flux, ctx = receiver.prepare(sea, flux, tiles)
    end, report = receiver.finish(sea, candidate, flux, ctx)
    assert report.runoff_received.max() > 0
    loss = soil.water() - end.land.water()
    expected = land.cells(0., land.route(loss))
    torch.testing.assert_close(report.runoff_received, expected, rtol=0, atol=1e-12)
    s0, s1 = sea.coupled.ocean.ocean, end.coupled.ocean.ocean
    torch.testing.assert_close((s1.h[:, 0] - s0.h[:, 0])*s0.column.rho0, expected, rtol=0, atol=1e-10)
    torch.testing.assert_close(s0.salt_h, s1.salt_h, rtol=0, atol=0)
    old, new = receiver.H.terms(sea.coupled), receiver.H.terms(end.coupled)
    change = sum(new[k] - old[k] for k in old) / A
    change = change + (end.cold_content - sea.cold_content).sum(-1)
    export = loss * (land.column.surface_geopotential + land.column.thermo.cp_l*(soil.skin-land.column.thermo.T_ref))
    torch.testing.assert_close(change, land.cells(0., land.route(export)), rtol=1e-9, atol=1e-5)


def test_restart_identity_includes_bathymetry(source, config):
    n = config.cells
    mask = np.arange(n) == 2
    geometry = dict(land=mask, height_m=mask * 500., depth_m=np.full(n, 200.))
    def ident():
        adapter, *_ = c.build(config, source, None, geography=geometry, smoke_no_radiation=True)
        return c.fingerprint(config, adapter, {}, True)
    first = ident()
    geometry["depth_m"][0] -= 1.
    assert ident() != first
