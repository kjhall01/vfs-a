from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("pyam21")
from test_pyam21_workflow import PrescribedRadiationPackage, initial, model
from pyam21.radiation.driver import RadiationDriver, RadiationDriverConfig
from workflows.coupled.driver import exchange_step
from workflows.coupled.land import LandSlabReceiver


@pytest.mark.parametrize("radiation", [False, True])
def test_mixed_column_exchange(model, radiation):
    driver = RadiationDriver(RadiationDriverConfig(rad_time_step=120), PrescribedRadiationPackage()) if radiation else None
    adapter, air = initial(model, radiation=driver, cloudy=True)
    wind = torch.full_like(air.atmosphere.u_e, 2.)
    air = replace(air, atmosphere=replace(air.atmosphere, u_e=wind))
    mesh = model.mesh
    area = mesh.cell_area
    mask = mesh.cell_xyz[:, 0] > 0.1
    receiver = LandSlabReceiver(adapter, mask)
    surface = receiver.initialize(air.surface["t_surf"], soil_water=150.)
    w0, e0 = receiver.inventories(surface)
    d0 = model.diagnostics(air.atmosphere)
    external = torch.zeros_like(area)
    delivered = torch.zeros_like(area)
    for _ in range(3):
        air, surface, x, ar, sr = exchange_step(adapter, air, receiver, surface, 60)
        assert torch.equal(air.surface["land"], mask)
        assert sr.water_residual.abs().max() < 1e-10
        assert sr.energy_residual.abs().max() < 2e-6
        assert (sr.runoff[~mask] == 0).all() and (sr.runoff_received[mask] == 0).all()
        torch.testing.assert_close((sr.runoff * area).sum(), (sr.runoff_received * area).sum())
        assert x.stress[mask].abs().max() > 0
        torch.testing.assert_close(sr.support_impulse, -x.water_impulse - x.stress.sum(1))
        external += x.toa + ar.energy_changes["am2_energy_residual"]
        delivered += sr.runoff_received
    assert delivered.sum() > 0
    w1, e1 = receiver.inventories(surface)
    d1 = model.diagnostics(air.atmosphere)
    assert abs(((d1["species"].sum() - d0["species"].sum()) + (area * (w1-w0)).sum()) / area.sum()) < 1e-9
    assert abs(((d1["energy"] - d0["energy"]) + (area * (e1-e0-external)).sum()) / area.sum()) < 2e-6


def test_supply_bound_is_used_by_implicit_air_solve(model):
    adapter, air = initial(model)
    mask = model.mesh.cell_xyz[:, 0] > 0.1
    receiver = LandSlabReceiver(adapter, mask)
    surface = receiver.initialize(air.surface["t_surf"], soil_water=0.)
    # Tiny wet canopy on otherwise dry land. Saturated skin with tiny
    # available water stresses the supply branch of the implicit solve.
    s = replace(surface.land, canopy=torch.full_like(surface.land.canopy, 1e-7))
    surface = replace(surface, land=s)
    tiles = receiver.tiles(surface)
    tiles = replace(tiles, availability=torch.ones_like(tiles.availability))
    with patch.object(receiver, "tiles", return_value=tiles):
        _, end, x, ar, sr = exchange_step(adapter, air, receiver, surface, 600)
    torch.testing.assert_close(x.vapor[mask, 0], s.canopy, atol=1e-16, rtol=0)
    torch.testing.assert_close(600 * ar.fluxes["flux_q"][mask], s.canopy, atol=1e-16, rtol=0)
    assert (x.vapor[~mask, 0] > s.canopy.max()).all()
    assert sr.water_residual.abs().max() < 1e-10
    assert (surface.land.canopy == 1e-7).all()
    assert end.land.canopy.min() >= 0
