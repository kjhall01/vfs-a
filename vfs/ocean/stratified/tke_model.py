"""Composition of existing active ocean transport/ALE with optional TKE physics."""
from dataclasses import dataclass, replace
import math
import torch
from .model import StratifiedOcean
from .equations import EntropyState
from .tke import TKEClosure


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


@dataclass(frozen=True)
class TKERemapReport:
    ocean: object
    tke_change: torch.Tensor
    total_energy_change: torch.Tensor


class TKEOcean:
    """Optional composition; the original StratifiedOcean and its states stay intact.

    First-order splitting: checked dynamics, requested ALE, then vertical TKE
    physics. closure=None transports/remaps TKE without production or decay.
    Existing ocean SSPRK3 energy error is not corrected by the closure.
    """
    def __init__(self, ocean, *, closure=None):
        if not isinstance(ocean, StratifiedOcean):
            raise TypeError('TKEOcean requires a StratifiedOcean')
        if closure is not None and not isinstance(closure, TKEClosure):
            raise TypeError('closure must be TKEClosure or None')
        self.ocean, self.closure = ocean, closure

    def check_state(self, state):
        if not isinstance(state, TKEState):
            raise TypeError('expected TKEState')
        self.ocean.check_state(state.ocean)
        h, k = state.ocean.h, state.tke_h
        if k.shape != h.shape or k.dtype != h.dtype or k.device != h.device:
            raise ValueError('tke_h must match thickness shape, dtype and device')
        if not torch.isfinite(k).all() or (k < 0).any():
            raise ValueError('TKE must be finite and nonnegative')
        if self.ocean.domain is not None and (k[~self.ocean.domain.wet_cell] != 0).any():
            raise ValueError('dry-cell TKE must be zero')

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

    def step_with_report(self, state, dt, *, target_h=None):
        self.check_state(state)
        packed, report = self.ocean.step_with_report(self._pack(state), dt, target_h=target_h)
        transported = self._unpack(packed, state)
        end, mixing = self.mix_with_report(transported, dt)
        transport_residual = (self.ocean.energy.mass_area(state.ocean)*(transported.tke_h-state.tke_h)).sum()
        return end, TKEStepReport(report, mixing, transport_residual)

    def step(self, state, dt, *, target_h=None):
        return self.step_with_report(state, dt, target_h=target_h)[0]

    def remap_with_report(self, state, target_h):
        self.check_state(state)
        packed, report = self.ocean.remap_with_report(self._pack(state), target_h)
        end = self._unpack(packed, state)
        self.check_state(end)
        change = (self.ocean.energy.mass_area(state.ocean)*(end.tke_h-state.tke_h)).sum()
        return end, TKERemapReport(report, change, report.energy_change["total"]+change)
