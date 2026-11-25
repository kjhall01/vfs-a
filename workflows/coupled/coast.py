"""One atmospheric column per mesh cell over land, open water or category ice."""
from dataclasses import replace

import torch

from workflows.coupled.ice_temperature import ThermalIceReceiver
from workflows.coupled.land import LandCatchment
from workflows.pyam21.surface import CoastalTiles


class CoastalReceiver(ThermalIceReceiver):
    def __init__(self, owner, adapter, land_mask):
        super().__init__(owner)
        if not torch.equal(self.H.model.ocean.domain.wet_cell, ~land_mask):
            raise ValueError("land and ocean masks must be complementary")
        self.land_surface = LandCatchment(adapter, land_mask)

    def check_geometry(self, adapter, atmosphere, state):
        super().check_geometry(adapter, atmosphere, state)
        self.land_surface.column.check(state.land)

    def tiles(self, state, ice_temperature=None):
        tiles = super().tiles(state, ice_temperature)
        catchment = self.land_surface
        albedo, rough, availability, supply = catchment.column.boundary(state.land)
        temp = tiles.temperature.clone()
        alpha = tiles.albedo.clone()
        vel = tiles.velocity.clone()
        temp[catchment.land, 0] = state.land.skin
        alpha[catchment.land, 0] = albedo
        vel[catchment.land] = 0
        return CoastalTiles(tiles.fraction, temp, vel, alpha, catchment.land,
                           catchment.cells(rough, 1.), catchment.cells(availability, 1.), catchment.cells(supply, 0.))

    def prepare(self, initial, exchange, tiles):
        catchment = self.land_surface
        mask = catchment.land
        land, runoff, energy = catchment.column.advance(
            initial.land, exchange.dt, rain=exchange.rain[mask], snow=exchange.snow[mask],
            vapor=exchange.vapor[mask, 0], energy=exchange.surface_energy[mask])
        received = catchment.cells(0., catchment.route(runoff))
        delivered = catchment.cells(0., catchment.route(energy))
        wet = catchment.wet
        ocean_flux = replace(exchange, rain=exchange.rain * wet, snow=exchange.snow * wet,
                             vapor=exchange.vapor * wet[:, None], sensible=exchange.sensible * wet[:, None],
                             stress=exchange.stress * wet[:, None, None], toa=exchange.toa * wet,
                             surface_thermal=exchange.surface_thermal * wet[:, None],
                             terms={k: v * wet for k, v in exchange.terms.items()})
        candidate, ocean_flux, impulses = super().prepare(initial, ocean_flux, tiles, runoff=received)
        # Runoff arrives without horizontal momentum. Its exported energy
        # enters the top-layer Hamiltonian solve after the freshwater dilution.
        gravity_heat = catchment.cells(0., catchment.route(runoff * catchment.runoff_drop))
        candidate = replace(candidate, land=land)
        land_error = catchment.column.energy(land) - catchment.column.energy(initial.land) + energy - exchange.surface_energy[mask]
        water_error = land.water() - initial.land.water() + runoff + exchange.water[mask]
        full = exchange.with_stress_work(ocean_flux.terms["stress_work"])
        context = (ocean_flux, impulses, received, delivered, gravity_heat, land_error, water_error)
        return candidate, full, context

    def finish(self, initial, candidate, exchange, context):
        ocean_flux, impulses, received, delivered, gravity_heat, land_error, water_error = context
        end, report = super().finish(initial, candidate, ocean_flux, impulses,
                                     runoff=received, runoff_energy=delivered)
        catchment = self.land_surface
        # Stationary land and closed coastal edges absorb the reaction impulse
        # that the ocean and ice cannot take. Their boundary work is zero.
        force = -(exchange.water_impulse + exchange.stress.sum(1))
        support = self.wind.transpose((force * catchment.mesh.cell_area[:, None])[:, None]) - sum(impulses[:2])
        return end, replace(report, runoff_received=received, runoff_gravity_heat=gravity_heat,
                            support_impulse=support, energy_residual=catchment.cells(land_error, report.energy_residual[catchment.wet]),
                            water_residual=catchment.cells(water_error, report.water_residual[catchment.wet]))
