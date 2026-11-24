"""Grassland column adapted from VFS 62095de, vfs/land.

Fluxes are interval amounts per unit land area. Heat capacity includes soil
moisture implicitly (the archived fixed-capacity convention). Snow stores
negative fusion enthalpy relative to liquid at the atmospheric datum.
"""
from dataclasses import dataclass
import math
import torch

from vfs.atmosphere.mixture_thermodynamics import MixtureThermodynamics


@dataclass(frozen=True)
class LandState:
    soil_water: torch.Tensor       # [land cell, layer], kg/m2
    soil_temperature: torch.Tensor # K
    canopy: torch.Tensor           # liquid, kg/m2
    snow: torch.Tensor             # snow water equivalent, kg/m2
    skin: torch.Tensor             # K

    def water(self):
        return self.soil_water.sum(-1) + self.canopy + self.snow


class LandColumn:
    """Four archived soil layers, grass roots, canopy, snow and conductive skin."""
    thickness = (0.10, 0.30, 0.60, 1.00)
    soil_heat_capacity = 2.e6  # J/m3/K
    skin_heat_capacity = 2.e6  # J/m2/K
    conductivity = 1.5         # W/m/K
    canopy_capacity = 0.25     # kg/m2, archived grassland class

    def __init__(self, thermodynamics=None):
        self.thermo = thermodynamics or MixtureThermodynamics()

    def initialize(self, temperature, *, soil_water=50.):
        if temperature.ndim != 1 or not temperature.is_floating_point():
            raise ValueError("land temperature must be a floating cell vector")
        if not 0 <= soil_water <= 150:
            raise ValueError("initial column soil water must lie in [0, 150] kg/m2")
        dz = temperature.new_tensor(self.thickness)
        state = LandState(
            (soil_water * dz / dz.sum()).expand(len(temperature), -1).clone(),
            temperature[:, None].expand(-1, len(dz)).clone(),
            torch.zeros_like(temperature), torch.zeros_like(temperature), temperature.clone(),
        )
        self.check(state)
        return state

    def check(self, state):
        ref = state.skin
        for x, shape in ((ref, ref.shape), (state.canopy, ref.shape), (state.snow, ref.shape),
                         (state.soil_temperature, (ref.numel(), len(self.thickness))),
                         (state.soil_water, (ref.numel(), len(self.thickness)))):
            if (ref.ndim != 1 or not ref.is_floating_point() or x.shape != shape
                    or x.dtype != ref.dtype or x.device != ref.device or not torch.isfinite(x).all()):
                raise ValueError("invalid land state shape, precision or value")
        if ((ref <= 0).any() or (state.soil_temperature <= 0).any()
                or (state.soil_water < 0).any() or (state.canopy < 0).any()
                or (state.snow < 0).any()):
            raise ValueError("inadmissible land temperature or water storage")

    def _soil_properties(self, ref):
        dz = ref.new_tensor(self.thickness)
        capacity = 150. * dz / dz.sum()
        bottom = dz.cumsum(0)
        roots = torch.exp(-(bottom - dz) / 0.60) - torch.exp(-bottom / 0.60)
        return capacity, roots / roots.sum()

    def boundary(self, state):
        capacity, roots = self._soil_properties(state.skin)
        available = (state.soil_water - 0.15 * capacity).clamp_min(0.)
        beta = ((available / (0.55 * capacity)).clamp_max(1.) * roots).sum(-1)
        cover = (state.snow / 10.).clamp_max(1.)
        beta = torch.maximum(beta, torch.maximum(cover, (state.canopy / self.canopy_capacity).clamp_max(1.)))
        return (0.20 + (0.80 - 0.20) * cover, 0.10 + (0.001 - 0.10) * cover,
                beta.clamp_max(1.), state.canopy + state.snow + available.sum(-1))

    def energy(self, state):
        dz = state.skin.new_tensor(self.thickness)
        t0 = self.thermo.T_ref
        return (self.skin_heat_capacity * (state.skin - t0)
                + ((state.soil_temperature - t0) * dz * self.soil_heat_capacity).sum(-1)
                - self.thermo.L_f_ref * state.snow)

    def _soil_heat(self, state, dt):
        # Archive's explicit finite-volume diffusion. Accumulate the heat taken
        # from the skin across *all* substeps, with an insulated bottom.
        dz = state.skin.new_tensor(self.thickness)
        n = max(1, math.ceil(dt * self.conductivity /
                             (0.45 * self.soil_heat_capacity * min(self.thickness)**2)))
        temperature = state.soil_temperature
        insulation = 1. / (1. + state.snow / 20.)
        for _ in range(n):
            top = (state.skin - temperature[:, 0]) * insulation / (0.5 * dz[0])
            inner = (temperature[:, :-1] - temperature[:, 1:]) / (0.5 * (dz[:-1] + dz[1:]))
            flux = self.conductivity * torch.cat((top[:, None], inner, torch.zeros_like(top[:, None])), -1)
            temperature = temperature + (dt / n) * (flux[:, :-1] - flux[:, 1:]) / (dz * self.soil_heat_capacity)
        ground_heat = ((temperature - state.soil_temperature) * dz * self.soil_heat_capacity).sum(-1)
        return temperature, ground_heat

    @staticmethod
    def _saturation_cascade(layers, incoming, capacity):
        retained = []
        for k, layer in enumerate(layers):
            water = layer + incoming
            incoming = (water - capacity[k]).clamp_min(0.)
            retained.append(water - incoming)
        return retained, incoming

    def advance(self, state, dt, *, rain, snow, vapor, energy):
        """Receive PyAM21's signed vapor and net energy. Return state, runoff kg/m2 and J/m2."""
        self.check(state)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("positive finite land timestep required")
        for x in (rain, snow, vapor, energy):
            if (x.shape != state.skin.shape or x.dtype != state.skin.dtype
                    or x.device != state.skin.device or not torch.isfinite(x).all()):
                raise ValueError("land fluxes must be matching finite cell amounts")
        if (rain < 0).any() or (snow < 0).any():
            raise ValueError("negative precipitation")
        capacity, roots = self._soil_properties(state.skin)
        extractable = (state.soil_water - 0.15 * capacity).clamp_min(0.)
        supply = state.canopy + state.snow + extractable.sum(-1)
        tol = 16 * torch.finfo(vapor.dtype).eps * supply
        if (vapor > supply + tol).any():
            raise ValueError("atmospheric evaporation exceeds available land water")
        evaporation = vapor.clamp_min(0.)
        canopy_loss = torch.minimum(evaporation, state.canopy)
        snow_loss = torch.minimum(evaporation - canopy_loss, state.snow)
        demand = evaporation - canopy_loss - snow_loss
        uptake = torch.minimum(demand[:, None] * roots, extractable)
        remaining = extractable - uptake
        total = remaining.sum(-1)
        # Redistribute unmet root demand over layers that can still supply it.
        fraction = ((demand - uptake.sum(-1)).clamp_min(0.) /
                    torch.where(total > 0, total, 1.)).clamp_max(1.)
        soil = state.soil_water - (uptake + fraction[:, None] * remaining)
        dew = (-vapor).clamp_min(0.)
        cold = state.skin < self.thermo.T_ref
        frozen = state.snow - snow_loss + snow + torch.where(cold, dew, 0.)
        liquid = rain + torch.where(cold, 0., dew)
        canopy = state.canopy - canopy_loss
        captured = torch.minimum((self.canopy_capacity - canopy).clamp_min(0.), liquid)
        canopy = canopy + captured
        temperature, ground = self._soil_heat(state, dt)
        lf = self.thermo.L_f_ref
        sensible = energy - ground + lf * (frozen - state.snow)
        warm = (state.skin - self.thermo.T_ref).clamp_min(0.)
        melt = torch.minimum(frozen, 2.e-5 * warm * dt)
        melt = torch.minimum(melt, (self.skin_heat_capacity * warm + sensible).clamp_min(0.) / lf)
        frozen = frozen - melt
        rain_to_soil = liquid - captured + melt
        # Archive's saturation cascade and gravity drainage. Inter-layer water
        # is internal. Infiltration excess and bottom outflow leave as runoff.
        incoming = torch.minimum(rain_to_soil, torch.full_like(liquid, 0.01 * dt))
        runoff = rain_to_soil - incoming
        layers, excess = self._saturation_cascade(soil.unbind(-1), incoming, capacity)
        runoff = runoff + excess
        # Drainage can temporarily push a receiving layer above saturation.
        # Preserve the original ordering: first cascade, drain, then cascade.
        incoming = torch.zeros_like(runoff)
        drained = []
        for k, layer in enumerate(layers):
            water = layer + incoming
            saturation = (water / capacity[k]).clamp_max(1.)
            incoming = torch.minimum((water - 0.70 * capacity[k]).clamp_min(0.),
                                     0.01 * dt * saturation.square())
            drained.append(water - incoming)
        layers, excess = self._saturation_cascade(drained, torch.zeros_like(runoff), capacity)
        runoff = runoff + incoming + excess
        runoff_heat = runoff * self.thermo.cp_l * (state.skin - self.thermo.T_ref)
        skin = state.skin + (sensible - lf * melt - runoff_heat) / self.skin_heat_capacity
        result = LandState(torch.stack(layers, -1), temperature, canopy, frozen, skin)
        self.check(result)
        return result, runoff, runoff_heat
