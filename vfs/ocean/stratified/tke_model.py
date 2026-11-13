"""Composition of existing active ocean transport/ALE with optional TKE physics."""
from dataclasses import dataclass, replace
import math
import torch
from .model import StratifiedOcean
from .equations import EntropyState
from .tke import TKEClosure
from .boundary import OceanBoundaryStress
from .surface_heating import OceanSurfaceHeating
from .surface_exchange import OceanSurfaceExchange


@dataclass(frozen=True)
class TKEState:
    """Explicit resolved state and cell-layer TKE content h*k [m³/s²]."""
    ocean: EntropyState
    tke_h: torch.Tensor

    def specific_tke(self):
        return self.tke_h/self.ocean.h


@dataclass(frozen=True)
class TKEStepReport:
    ocean: object
    turbulence: object
    tke_transport_residual: torch.Tensor
    boundary: object = None
    heating: object = None
    exchange: object = None


@dataclass(frozen=True)
class TKERemapReport:
    ocean: object
    tke_change: torch.Tensor
    total_energy_change: torch.Tensor


class TKEOcean:
    """Optional composition; the original StratifiedOcean and its states stay intact.

    First-order splitting: dynamics, requested ALE, water exchange, boundary stresses, surface heating, vertical TKE
    physics. closure=None disables vertical mixing/decay; boundary drag may still
    produce TKE when explicitly configured.
    Existing ocean SSPRK3 energy error is not corrected by the closure.
    """
    def __init__(self, ocean, *, closure=None, boundary=None, heating=None, exchange=None):
        if not isinstance(ocean, StratifiedOcean):
            raise TypeError('TKEOcean requires a StratifiedOcean')
        if closure is not None and not isinstance(closure, TKEClosure):
            raise TypeError('closure must be TKEClosure or None')
        if boundary is not None and not isinstance(boundary, OceanBoundaryStress):
            raise TypeError("boundary must be OceanBoundaryStress or None")
        if heating is not None and not isinstance(heating, OceanSurfaceHeating):
            raise TypeError("heating must be OceanSurfaceHeating or None")
        self.ocean, self.closure, self.boundary = ocean, closure, boundary
        self.heating = heating
        if exchange is not None and not isinstance(exchange, OceanSurfaceExchange):
            raise TypeError("exchange must be OceanSurfaceExchange or None")
        self.exchange = exchange

    def check_state(self, state, *, _physical=None):
        if not isinstance(state, TKEState):
            raise TypeError('expected TKEState')
        physical = (self.ocean.check_state(state.ocean) if _physical is None else
                    self.ocean.check_state(state.ocean, _physical=_physical))
        h, k = state.ocean.h, state.tke_h
        if k.shape != h.shape or k.dtype != h.dtype or k.device != h.device:
            raise ValueError('tke_h must match thickness shape, dtype and device')
        if not torch.isfinite(k).all() or (k < 0).any():
            raise ValueError('TKE must be finite and nonnegative')
        if self.ocean.domain is not None and (k[~self.ocean.domain.wet_cell] != 0).any():
            raise ValueError('dry-cell TKE must be zero')
        return physical

    def state(self, resolved, *, tke):
        """Explicit initialization from specific k [m²/s²], never an implicit floor."""
        if not isinstance(tke, torch.Tensor):
            raise TypeError('tke must be a tensor with the same layout as h')
        if tke.shape != resolved.h.shape or tke.dtype != resolved.h.dtype or tke.device != resolved.h.device:
            raise ValueError("specific tke must match thickness shape, dtype and device")
        state = TKEState(resolved, resolved.h*tke)
        self.check_state(state)
        return state

    @staticmethod
    def _pack(state):
        # Reuse authoritative conservative passive budgets, including ALE.
        q = state.ocean.tracer_h
        contents = state.tke_h[..., None] if q is None else torch.cat((q, state.tke_h[..., None]), -1)
        return replace(state.ocean, tracer_h=contents)

    @staticmethod
    def _unpack(packed, template):
        q = None if template.ocean.tracer_h is None else packed.tracer_h[..., :-1]
        return TKEState(replace(packed, tracer_h=q), packed.tracer_h[..., -1])

    def budgets(self, state):
        self.check_state(state)
        result = self.ocean.budgets(state.ocean)
        result['resolved_total'] = result['total']
        result['tke'] = (self.ocean.energy.mass_area(state.ocean)*state.tke_h).sum()
        result['total'] = result['resolved_total']+result['tke']
        return result

    def mix_with_report(self, state, dt):
        """Vertical physics alone, useful for column qualification."""
        self.check_state(state)
        if not math.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError("dt must be finite and positive")
        if self.closure is None:
            return state, None
        resolved, tke_h, report = self.closure.apply(self.ocean, state.ocean, state.tke_h, dt)
        result = TKEState(resolved, tke_h)
        self.check_state(result)
        return result, report

    def boundary_with_report(self, state, dt, *, surface_stress=None):
        """Boundary momentum/work alone; no hidden call to vertical mixing."""
        self.check_state(state)
        if not math.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError('dt must be finite and positive')
        if self.boundary is None:
            if surface_stress is not None:
                raise ValueError('surface_stress requires an explicit OceanBoundaryStress')
            return state, None
        resolved, tke_h, report = self.boundary.apply(self.ocean, state.ocean, state.tke_h,
            dt, surface_stress=surface_stress)
        result = TKEState(resolved, tke_h)
        self.check_state(result)
        return result, report

    def heat_with_report(self, state, dt, *, shortwave=None, surface_heat=None):
        """Prescribed external heating alone; TKE is unchanged."""
        self.check_state(state)
        if not math.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError('dt must be finite and positive')
        if self.heating is None:
            if shortwave is not None or surface_heat is not None:
                raise ValueError('heat fluxes require an explicit OceanSurfaceHeating')
            return state, None
        resolved, report = self.heating.apply(self.ocean, state.ocean, dt,
            shortwave=shortwave, surface_heat=surface_heat)
        return TKEState(resolved, state.tke_h), report

    def exchange_with_report(self, state, dt, *, surface_fluxes=None):
        """Prescribed water/heat exchange; extensive TKE content is retained.

        No unresolved energy is carried through the boundary in this version.
        Water energy flux therefore refers to the resolved Hamiltonian only.
        """
        self.check_state(state)
        if not math.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError('dt must be finite and positive')
        if surface_fluxes is None:
            return state, None
        if self.exchange is None:
            raise ValueError('surface_fluxes requires an explicit OceanSurfaceExchange')
        resolved, report = self.exchange.apply(self.ocean, state.ocean, dt, surface_fluxes)
        result = TKEState(resolved, state.tke_h)
        self.check_state(result)
        return result, report

    def step_with_report(self, state, dt, *, target_h=None, surface_stress=None, shortwave=None, surface_heat=None, surface_fluxes=None):
        if type(self) is not TKEOcean or type(self.ocean) is not StratifiedOcean:
            return self._legacy_step_with_report(state,dt,target_h=target_h,
                surface_stress=surface_stress,shortwave=shortwave,
                surface_heat=surface_heat,surface_fluxes=surface_fluxes)

        physical = self.check_state(state)
        packed_input = self._pack(state)
        physical = physical.with_tensors(physical.u_e,physical.h,physical.temp_h,
                                        physical.salt_h,packed_input.tracer_h)
        packed, report, physical = self.ocean._step_with_physical(
            packed_input, dt, target_h=target_h, physical=physical)
        transported = self._unpack(packed, state)
        physical = physical.with_tensors(physical.u_e,physical.h,physical.temp_h,
                                        physical.salt_h,transported.ocean.tracer_h)
        self.check_state(transported, _physical=physical)
        end = transported
        exchange = boundary = heating = mixing = None
        if surface_fluxes is not None:
            if type(self.exchange) is OceanSurfaceExchange:
                resolved, exchange, physical = self.exchange.apply(self.ocean,end.ocean,dt,
                    surface_fluxes,_physical=physical,_return_physical=True)
                end = TKEState(resolved,end.tke_h)
            else:
                end, exchange = self.exchange_with_report(end,dt,surface_fluxes=surface_fluxes)
                physical = self.ocean.check_state(end.ocean)
            self.check_state(end,_physical=physical)
        if self.boundary is not None or surface_stress is not None:
            if type(self.boundary) is OceanBoundaryStress:
                resolved, k, boundary, physical = self.boundary.apply(self.ocean,end.ocean,end.tke_h,dt,
                    surface_stress=surface_stress,_physical=physical,_return_physical=True)
                end = TKEState(resolved,k)
            else:
                end, boundary = self.boundary_with_report(end,dt,surface_stress=surface_stress)
                physical = self.ocean.check_state(end.ocean)
            self.check_state(end,_physical=physical)
        if self.heating is not None or shortwave is not None or surface_heat is not None:
            if type(self.heating) is OceanSurfaceHeating:
                resolved, heating, physical = self.heating.apply(self.ocean,end.ocean,dt,
                    shortwave=shortwave,surface_heat=surface_heat,_physical=physical,_return_physical=True)
                end = TKEState(resolved,end.tke_h)
            else:
                end, heating = self.heat_with_report(end,dt,shortwave=shortwave,surface_heat=surface_heat)
                physical = self.ocean.check_state(end.ocean)
            self.check_state(end,_physical=physical)
        # Closures retain their own validation protocol (including research subclasses).
        if self.closure is not None:
            end, mixing = self.mix_with_report(end,dt)
        transport_residual = (self.ocean.energy.mass_area(state.ocean)*(transported.tke_h-state.tke_h)).sum()
        return end, TKEStepReport(report, mixing, transport_residual, boundary, heating, exchange)

    def _legacy_step_with_report(self, state, dt, *, target_h=None, surface_stress=None, shortwave=None, surface_heat=None, surface_fluxes=None):
        self.check_state(state)
        packed, report = self.ocean.step_with_report(self._pack(state), dt, target_h=target_h)
        transported = self._unpack(packed, state)
        exchanged, exchange = self.exchange_with_report(transported, dt, surface_fluxes=surface_fluxes)
        forced, boundary = self.boundary_with_report(exchanged, dt, surface_stress=surface_stress)
        heated, heating = self.heat_with_report(forced, dt, shortwave=shortwave, surface_heat=surface_heat)
        end, mixing = self.mix_with_report(heated, dt)
        transport_residual = (self.ocean.energy.mass_area(state.ocean)*(transported.tke_h-state.tke_h)).sum()
        return end, TKEStepReport(report, mixing, transport_residual, boundary, heating, exchange)


    def step(self, state, dt, *, target_h=None, surface_stress=None, shortwave=None, surface_heat=None, surface_fluxes=None):
        return self.step_with_report(state, dt, target_h=target_h, surface_stress=surface_stress, shortwave=shortwave, surface_heat=surface_heat, surface_fluxes=surface_fluxes)[0]

    def remap_with_report(self, state, target_h):
        self.check_state(state)
        packed, report = self.ocean.remap_with_report(self._pack(state), target_h)
        end = self._unpack(packed, state)
        self.check_state(end)
        change = (self.ocean.energy.mass_area(state.ocean)*(end.tke_h-state.tke_h)).sum()
        return end, TKERemapReport(report, change, report.energy_change["total"]+change)
