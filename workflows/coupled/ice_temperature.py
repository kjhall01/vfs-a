"""Settle tiled atmospheric heat against the prognostic ice thermal store."""
from .ocean_ice import OceanIceReceiver


class ThermalIceReceiver(OceanIceReceiver):
    def tiles(self, state, ice_temperature=None):
        if ice_temperature is not None:
            raise ValueError("prognostic ice temperature cannot be overridden")
        return super().tiles(state, self.ocean_ice.temperature(state))

    def finish(self, initial, candidate, exchange, impulses, *, runoff=None, runoff_energy=None):
        if exchange.surface_thermal is None:
            raise ValueError("thermal ice requires tiled surface thermal energy")
        # H already includes -Lf*m. Remove that contribution from incoming heat
        # to obtain the sensible-storage forcing: sublimation pays Lv+Lf,
        # while intercepted snowfall supplies its sensible enthalpy.
        dm = exchange.snow[:, None] * initial.ice.area - exchange.vapor[:, 1:]
        q = exchange.surface_thermal[:, 1:] + self.H.slab.parameters.latent_heat * dm
        candidate, _, _ = self.ocean_ice.surface_heat(initial, candidate, q, exchange.dt)
        storage = (candidate.ice.cold_content - initial.ice.cold_content).sum(-1)
        return super().finish(
            initial, candidate, exchange, impulses, storage_change=storage, runoff=runoff, runoff_energy=runoff_energy
        )
