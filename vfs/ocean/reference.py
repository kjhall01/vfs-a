"""Explicit ownership around the recovered ocean equations.

Ocean owns geometry, equation/source objects and a shared explicit RK function.
States borrow the same geometry and own their tensor references. There is no
registry, inference of the component, implicit device conversion, or projection.
"""
from __future__ import annotations

import math
import torch
from vfs.integrators.explicit import ssprk3
from .state import OceanState
from .dynamics import MultilayerOceanDynamics


from .layers import LayerGrid


class Ocean(LayerGrid):
    """Experimental layered ocean; default equations preserve the archive.

    T is in degrees Celsius. Freshwater forcing in the recovered sources is a
    virtual salinity flux, not added water volume. The reference mechanical
    energy excludes stratification and is not an invariant of these equations.
    Time-dependent forcing requires a later explicit stage-time interface.
    """

    def __init__(self, mesh, column, *, domain=None, dynamics=None,
                 sources=(), stepper=ssprk3):
        if dynamics is None:
            dynamics = MultilayerOceanDynamics(domain=domain, gravity=column.gravity,
                                               rho0=column.rho0)
        super().__init__(mesh, column, domain, dynamics)
        self.sources = tuple(sources)
        self.stepper = stepper
        if self.dynamics.domain is not domain:
            raise ValueError("dynamics and Ocean must share the same domain")
        if self.dynamics.gravity != column.gravity or self.dynamics.rho0 != column.rho0:
            raise ValueError("dynamics and column must agree on gravity and reference density")
        def check_source(source):
            if hasattr(source, 'domain') and source.domain is not domain:
                raise ValueError("sources and Ocean must share the same domain")
            for child in getattr(source, 'components', ()):
                check_source(child)
        for source in self.sources:
            check_source(source)

    def tendency(self, state):
        """Sum equations then sources, in the caller's declared order."""
        self.check_state(state)
        out = self.dynamics.tendency(state)
        for source in self.sources:
            out = out + source.tendency(state)
        return out

    def physical_fields(self, state):
        """Recover fields with the selected dynamics' density convention."""
        self.check_state(state)
        return {'u_e':state.u_e, 'h':state.h, 'temperature':state.temperature(),
                'salinity':state.salinity(), 'tracer':state.tracers(),
                'eta':state.eta(), 'density':self.dynamics.density(state)}

    def to_latlon(self, state, remapper, *, fields='default'):
        """Remap model output with the selected dynamics' density convention.

        Field remapping remains differentiable and nonconservative. No EOS is
        inferred from the state, and no density calculation enters time stepping.
        """
        from .regrid import state_to_latlon
        self.check_state(state)
        output = state_to_latlon(remapper, state, fields=fields,
                                 density_provider=self.dynamics.density)
        if 'density' in output:
            output['attrs']['density_convention'] = 'selected_dynamics'
            output['attrs']['density_provider'] = type(self.dynamics).__name__
            if hasattr(self.dynamics, 'eos'):
                output['attrs']['density_eos'] = self.dynamics.eos
        return output

    def tendency_terms(self, state):
        """Expose equation terms and each source without changing the fast RHS."""
        self.check_state(state)
        return {'dynamics': self.dynamics.tendency_terms(state),
                'sources': tuple(source.tendency(state) for source in self.sources)}

    def step(self, state, dt):
        """Advance with the selected shared integrator; reject invalid outputs.

        No clipping/repair or hidden coast projection is performed. A rejection
        does not establish a general positivity guarantee or stable timestep.
        """
        if not math.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError("dt must be finite and positive (seconds)")
        self.check_state(state)
        result = self.stepper(self, state, dt)
        self.check_state(result)
        return result

    def budgets(self, state):
        """Connected tensor inventories; temperature content is m3 degC, not J."""
        self.check_state(state)
        owner = state if self.domain is None else self.domain
        args = () if self.domain is None else (state,)
        return {'volume': owner.total_volume(*args),
                'temperature_content': owner.total_heat_content(*args),
                'salinity_content': owner.total_salt_content(*args),
                'tracer_content': owner.total_tracer_content(*args),
                getattr(self.dynamics, 'energy_label', 'reference_mechanical_energy'):
                    self.dynamics.total_energy(state)}
