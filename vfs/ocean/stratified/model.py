"""Explicit facade for the integrated unforced entropy/JMD95 ocean.

Temperature arguments are potential temperature in degrees Celsius, IPTS-68.
The surface caloric fit is bundled; GSW is not a runtime dependency. A step is
checked SSPRK3 followed, only when requested, by the adaptive thermal ALE solve.
"""
from dataclasses import dataclass
from importlib.resources import files
import json
import math
import torch
from ..model import Ocean
from ..transport import HorizontalTransport
from ..jmd95 import JMD95Dynamics
from .thermodynamics import SurfaceGibbs
from .equations import EntropyState, EntropyEnergy, validate
from . import transport
from .remap import AdaptiveThermalRemapper


@dataclass(frozen=True)
class StepReport:
    """Three Euler-stage work ledgers and an optional post-step remap ledger."""
    stages: tuple
    remap: object = None


class StratifiedOcean:
    """Opt-in supported integration; scientific qualification remains limited.

    No forcing, mixing, wetting/drying or implicit timestep. Salt must satisfy
    0 < SP < 42; temperature must lie strictly inside the fitted reference domain.
    Geometry is borrowed, states own tensors, and conversion is explicit.
    """
    def __init__(self, mesh, column, *, domain=None, remapper=None, warm_start=False):
        dynamics = JMD95Dynamics(domain=domain, gravity=column.gravity, rho0=column.rho0)
        self.base = Ocean(mesh, column, domain=domain, dynamics=dynamics)
        self.mesh, self.column, self.domain = mesh, column, domain
        data = json.loads(files(__package__).joinpath('surface-gibbs-reference.json').read_text())
        self.reference = SurfaceGibbs(data, temperature_scale='IPTS-68')
        self.energy = EntropyEnergy(dynamics.energy, self.reference)
        if remapper is not None and not isinstance(remapper, AdaptiveThermalRemapper):
            raise TypeError('use AdaptiveThermalRemapper or None')
        self.remapper = remapper
        self.warm_start = bool(warm_start)
        self.base._stratified_transport = HorizontalTransport(mesh, domain=domain,
            scheme='donor', min_thickness=dynamics.min_thickness)
        self.base._stratified_transport.prepare(column.n_layers)

    def check_state(self, state, *, _physical=None):
        if not isinstance(state, EntropyState):
            raise TypeError('StratifiedOcean requires EntropyState; use from_temperature explicitly')
        if state.mesh is not self.mesh or state.column is not self.column:
            raise ValueError('state must borrow this model geometry')
        return validate(self.base, state, self.energy, physical=_physical)

    def from_temperature(self, state):
        """Convert an existing OceanState explicitly, without modifying it."""
        self.base.check_state(state)
        result = EntropyState.from_temperature(state, self.reference)
        self.check_state(result)
        return result

    def state(self, *, u_e, h, temperature, salinity, tracer=None):
        return self.from_temperature(self.base.state(u_e=u_e, h=h,
            temperature=temperature, salinity=salinity, tracer=tracer))

    def physical_fields(self, state):
        return self.base.physical_fields(self.check_state(state))

    def budgets(self, state):
        self.check_state(state)
        mass = self.energy.mass_area(state)
        result = dict(self.energy.terms(state))
        result.update(volume=(mass*state.h/self.column.rho0).sum(),
            salt=(mass*state.salt_h).sum(), entropy=(mass*state.entropy_h).sum(),
            tracers=None if state.tracer_h is None else
                (mass[..., None]*state.tracer_h).sum((0, 1)))
        return result

    def step_with_report(self, state, dt, *, target_h=None):
        end, report, _ = self._step_with_physical(state, dt, target_h=target_h)
        return end, report

    def _step_with_physical(self, state, dt, *, target_h=None, physical=None):
        """target_h is an explicit post-step target; no hidden remap schedule."""
        if not math.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError('dt must be finite and positive')
        physical = (self.check_state(state) if physical is None else
                    self.check_state(state, _physical=physical))
        if target_h is not None and self.remapper is None:
            raise ValueError('target_h requires an explicit remapper')
        end, stages, physical = transport.step_with_physical(self.base, state, dt, self.energy,
            physical=physical, warm_start=self.warm_start)
        report = None
        if target_h is not None:
            end, report = self.remap_with_report(end, target_h)
            physical = self.check_state(end)
        return end, StepReport(stages, report), physical

    def step(self, state, dt, *, target_h=None):
        return self.step_with_report(state, dt, target_h=target_h)[0]

    def remap_with_report(self, state, target_h):
        self.check_state(state)
        if self.remapper is None:
            raise ValueError('select an explicit AdaptiveThermalRemapper')
        if not isinstance(target_h, torch.Tensor) or target_h.dtype != state.h.dtype or target_h.device != state.h.device:
            raise ValueError('target_h must match state dtype and device')
        return self.remapper.remap_with_report(self.base, state, target_h, self.energy)
