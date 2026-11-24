"""Shared PyAM21 surface exchange over land and stationary ocean slabs."""
from dataclasses import dataclass
import torch
from scipy.spatial import cKDTree

from vfs.land import LandColumn, LandState
from workflows.pyam21.surface import ColumnTiles


@dataclass(frozen=True)
class ColumnSurfaceState:
    land: LandState
    slab_mass: torch.Tensor  # kg/m² of wet area
    slab_heat: torch.Tensor  # J/m² relative to liquid at T_ref


@dataclass(frozen=True)
class ColumnSurfaceReport:
    energy_residual: torch.Tensor   # J/m², after accounting for routed runoff
    water_residual: torch.Tensor    # kg/m²
    runoff: torch.Tensor            # kg/m², positive at donating land cells
    runoff_received: torch.Tensor   # kg/m², positive at receiving wet cells
    runoff_gravity_heat: torch.Tensor  # J/m² dissipated at receiving wet cells
    support_impulse: torch.Tensor   # N s/m² taken by the fixed surface


class LandSlabReceiver:
    """Binary land cells alongside a finite, horizontally stationary mixed layer."""
    def __init__(self, adapter, land_mask):
        self.mesh = adapter.model.mesh
        ref = self.mesh.cell_area
        if (land_mask.shape != ref.shape or land_mask.dtype != torch.bool
                or land_mask.device != ref.device or land_mask.all() or not land_mask.any()):
            raise ValueError("a boolean land mask with both land and wet cells is required")
        self.land = land_mask.clone()
        self.wet = ~self.land
        # Nearest wet center by great-circle distance (same ordering as chord).
        # The tree is built once. All interval updates use native tensors.
        xyz = self.mesh.cell_xyz.detach().cpu().numpy()
        mask = self.land.cpu().numpy()
        _, outlet = cKDTree(xyz[~mask]).query(xyz[mask])
        self.outlet = torch.as_tensor(outlet, device=ref.device, dtype=torch.long)

        phi = adapter.model.column.phi_s.expand_as(ref)
        self.wet_phi = phi[self.wet]
        self.column = LandColumn(adapter.model.dynamics.thermodynamics, surface_geopotential=phi[self.land])
        self.runoff_drop = phi[self.land] - self.wet_phi[self.outlet]
        if (self.runoff_drop < 0).any():
            raise ValueError("nearest wet outlet lies uphill from a land cell")

    def check_geometry(self, adapter, atmosphere, state):
        if adapter.model.mesh is not self.mesh:
            raise ValueError("land and atmosphere must share the same mesh")
        if (state.slab_mass.shape != self.mesh.cell_area[self.wet].shape
                or state.land.skin.shape != self.mesh.cell_area[self.land].shape):
            raise ValueError("land or slab state differs from its mask")
        if (atmosphere.surface["land"].any()
                and not torch.equal(atmosphere.surface["land"], self.land)):
            raise ValueError("atmospheric land mask differs from receiver")

    def initialize(self, temperature, *, soil_water=50., slab_depth=50.):
        ref = self.mesh.cell_area
        if (temperature.shape != ref.shape or temperature.dtype != ref.dtype
                or temperature.device != ref.device or not torch.isfinite(temperature).all()
                or (temperature <= 0).any() or not 0 < slab_depth < float("inf")):
            raise ValueError("positive finite initial temperatures and slab depth required")
        th = self.column.thermo
        mass = torch.full_like(temperature[self.wet], slab_depth * 1000.)
        heat = mass * th.cp_l * (temperature[self.wet] - th.T_ref)
        return ColumnSurfaceState(self.column.initialize(temperature[self.land], soil_water=soil_water),
                                  mass, heat)

    def _cells(self, land_values, wet_values):
        cells = torch.zeros_like(self.mesh.cell_area)
        cells[self.land] = land_values
        cells[self.wet] = wet_values
        return cells

    def tiles(self, state, ice_temperature=None):
        if ice_temperature is not None:
            raise ValueError("the stationary slab receiver has no sea ice")
        albedo, rough, beta, supply = self.column.boundary(state.land)
        th = self.column.thermo
        slab_t = th.T_ref + state.slab_heat / (th.cp_l * state.slab_mass)
        skin = self._cells(state.land.skin, slab_t)
        return ColumnTiles(torch.ones_like(skin[:, None]), skin[:, None],
                           skin.new_zeros(len(skin), 1, 3), self._cells(albedo, 0.14)[:, None],
                           self.land, self._cells(rough, 1.), self._cells(beta, 1.),
                           self._cells(supply, 0.))

    def inventories(self, state):
        return (self._cells(state.land.water(), state.slab_mass),
                self._cells(self.column.energy(state.land), state.slab_heat + state.slab_mass * self.wet_phi))

    def prepare(self, initial, exchange, tiles):
        # Both surfaces are stationary. The support takes the reaction impulse
        # at zero velocity; the producer has already recorded zero stress work.
        support_impulse = -(exchange.water_impulse + exchange.stress.sum(1))
        return initial, exchange, support_impulse

    def _route(self, amount):
        area = self.mesh.cell_area
        total = torch.zeros_like(area[self.wet]).index_add(0, self.outlet, amount * area[self.land])
        return total / area[self.wet]

    def finish(self, initial, candidate, exchange, support_impulse):
        land, wet = self.land, self.wet
        new_land, runoff, runoff_energy = self.column.advance(
            initial.land, exchange.dt, rain=exchange.rain[land], snow=exchange.snow[land],
            vapor=exchange.vapor[land, 0], energy=exchange.surface_energy[land],
        )
        received = self._route(runoff)
        received_energy = self._route(runoff_energy)
        gravity_heat = self._route(runoff * self.runoff_drop)
        mass = initial.slab_mass - exchange.water[wet] + received
        net_water = received - exchange.water[wet]
        # The outlet stores water at its own geopotential. The remaining
        # delivered energy includes liquid enthalpy and the downhill loss.
        heat = initial.slab_heat + exchange.surface_energy[wet] + received_energy - net_water * self.wet_phi
        if (not torch.isfinite(mass).all() or (mass <= 0).any() or not torch.isfinite(heat).all()
                or (heat <= -mass * self.column.thermo.cp_l * self.column.thermo.T_ref).any()):
            raise ValueError("exchange exhausts the receiving slab mass or heat")
        end = ColumnSurfaceState(new_land, mass, heat)
        w0, e0 = self.inventories(initial)
        w1, e1 = self.inventories(end)
        water = w1 - w0 + exchange.water - self._cells(-runoff, received)
        energy = e1 - e0 - exchange.surface_energy - self._cells(-runoff_energy, received_energy)
        # Land heat is diagnosed from absolute temperatures, while the slab
        # stores heat relative to the liquid datum. Budget roundoff must use
        # the magnitude of the stored quantities used by each calculation.
        eps = torch.finfo(energy.dtype).eps
        capacity = self.column.skin_heat_capacity + self.column.soil_heat_capacity * sum(self.column.thickness)
        datum_scale = self._cells(capacity * self.column.thermo.T_ref, 0.)
        water_scale = w0 + w1 + exchange.water.abs() + self._cells(runoff, received)
        energy_scale = e0.abs() + e1.abs() + datum_scale + exchange.surface_energy.abs() + self._cells(runoff_energy.abs(), received_energy.abs())
        if (not torch.isfinite(energy).all() or not torch.isfinite(water).all()
                or (water.abs() > 64 * eps * water_scale.clamp_min(1.)).any()
                or (energy.abs() > 64 * eps * energy_scale.clamp_min(1.)).any()):
            raise FloatingPointError("land/slab surface exchange failed its local budget")
        return end, ColumnSurfaceReport(energy, water, self._cells(runoff, 0.),
                                        self._cells(0., received), self._cells(0., gravity_heat), support_impulse)
