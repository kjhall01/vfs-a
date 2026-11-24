from dataclasses import replace

import pytest
import torch

from vfs.land import LandColumn


def test_land_column_balances_dew_frost_runoff_and_melt():
    column = LandColumn()
    s = column.initialize(torch.tensor([270., 280., 290.], dtype=torch.float64), soil_water=145.)
    s = replace(s, snow=s.skin.new_tensor([0., 8., 0.]),
                soil_temperature=s.soil_temperature - 2)
    zero = torch.zeros_like(s.skin)
    rain, snow = s.skin.new_tensor([0., 0., 200.]), s.skin.new_tensor([2., 0., 0.])
    vapor, energy = s.skin.new_tensor([-0.3, 0.1, -0.3]), s.skin.new_tensor([50., 1.e6, -2.e4])
    end, runoff, heat = column.advance(s, 7200, rain=rain, snow=snow, vapor=vapor, energy=energy)
    assert end.snow[0] == pytest.approx(2.3)
    assert end.snow[1] < s.snow[1] - vapor[1]
    assert runoff[2] > 100 and end.canopy[2] > 0
    torch.testing.assert_close(end.water() - s.water() + runoff, rain + snow - vapor, atol=2e-13, rtol=0)
    torch.testing.assert_close(column.energy(end) - column.energy(s) + heat, energy, atol=2e-7, rtol=0)
    with pytest.raises(ValueError, match="evaporation exceeds"):
        column.advance(s, 600, rain=zero, snow=zero, vapor=column.boundary(s)[3] + 1., energy=zero)


def test_all_evaporatable_water_can_be_withdrawn_once():
    column = LandColumn()
    s = column.initialize(torch.tensor([282.], dtype=torch.float64))
    s = replace(s, canopy=s.skin.new_tensor([0.2]), snow=s.skin.new_tensor([1.]))
    available = column.boundary(s)[3]
    zero = torch.zeros_like(s.skin)
    end, runoff, heat = column.advance(s, 60, rain=zero, snow=zero, vapor=available, energy=zero)
    torch.testing.assert_close(s.water() - end.water(), available, atol=1e-13, rtol=0)
    torch.testing.assert_close(column.boundary(end)[3], zero, atol=1e-13, rtol=0)
    assert runoff.item() == heat.item() == end.canopy.item() == end.snow.item() == 0.


def test_archived_grassland_column_at_2400_seconds():
    # Values evaluated from hydrology and thermal diffusion at VFS 62095de.
    column = LandColumn()
    s = column.initialize(torch.tensor([282.], dtype=torch.float64), soil_water=135.)
    s = replace(s, soil_temperature=s.soil_temperature - 2.)
    zero = torch.zeros_like(s.skin)
    end, runoff, _ = column.advance(s, 2400, rain=zero, snow=zero, vapor=zero, energy=zero)
    torch.testing.assert_close(end.soil_water, zero.new_tensor([[5.25, 15.75, 31.5, 58.5]]))
    torch.testing.assert_close(runoff, zero.new_tensor([24.]))
    torch.testing.assert_close(end.soil_temperature, zero.new_tensor([[280.72, 280., 280., 280.]]))
