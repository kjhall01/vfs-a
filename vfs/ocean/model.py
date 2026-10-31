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


class Ocean:
    """Experimental layered ocean; default equations preserve the archive.

    T is in degrees Celsius. Freshwater forcing in the recovered sources is a
    virtual salinity flux, not added water volume. The reference mechanical
    energy excludes stratification and is not an invariant of these equations.
    Time-dependent forcing requires a later explicit stage-time interface.
    """

    def __init__(self, mesh, column, *, domain=None, dynamics=None,
                 sources=(), stepper=ssprk3):
        if domain is not None and domain.mesh is not mesh:
            raise ValueError("domain must borrow this exact mesh")
        self.mesh = mesh
        self.column = column
        self.domain = domain
        self.dynamics = (MultilayerOceanDynamics(domain=domain,
            gravity=column.gravity, rho0=column.rho0) if dynamics is None else dynamics)
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

    def check_state(self, state):
        if not isinstance(state, OceanState):
            raise TypeError("Ocean requires OceanState")
        if state.mesh is not self.mesh or state.column is not self.column:
            raise ValueError("state must borrow this Ocean's exact mesh and column")
        state.validate()
        for name in ('u_e', 'h', 'temp_h', 'salt_h', 'tracer_h'):
            value = getattr(state, name)
            if value is None:
                continue
            expected_ndim = 3 if name == 'tracer_h' else 2
            if value.ndim != expected_ndim:
                raise ValueError("Ocean supports unbatched states with an explicit tracer axis")
            if value.dtype != self.mesh.cell_xyz.dtype or value.device != self.mesh.cell_xyz.device:
                raise ValueError(f"{name} must match mesh dtype/device; convert explicitly")
            if not value.is_floating_point() or not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must contain finite floating-point values")
        for value in (self.column.rest_thickness, self.column.total_depth):
            if value.dtype != state.h.dtype or value.device != state.h.device:
                raise ValueError("column must match state dtype/device; convert explicitly")
        self.column.validate_thickness(state.h, min_thickness=self.dynamics.min_thickness)

    def state(self, *, u_e, h, temperature, salinity, tracer=None):
        """Form conservative contents without casting, detaching or regridding."""
        if not isinstance(h, torch.Tensor):
            raise TypeError("h must be a torch.Tensor")
        for name,value in (('u_e',u_e),('h',h),('temperature',temperature),
                           ('salinity',salinity),('tracer',tracer)):
            if value is not None and not isinstance(value,torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value is not None and (value.dtype != h.dtype or value.device != h.device):
                raise ValueError(f"{name} must match h dtype/device")
        if temperature.shape != h.shape or salinity.shape != h.shape:
            raise ValueError("temperature and salinity must have exactly h's shape")
        if tracer is not None and (tracer.ndim != 3 or tracer.shape[:2] != h.shape):
            raise ValueError("tracer must have shape [cell, layer, tracer]")
        state = OceanState.from_temperature(u_e,h,temperature,salinity,tracer,
                                            mesh=self.mesh,column=self.column)
        self.check_state(state)
        return state

    def tendency(self, state):
        """Sum equations then sources, in the caller's declared order."""
        self.check_state(state)
        out = self.dynamics.tendency(state)
        for source in self.sources:
            out = out + source.tendency(state)
        return out

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
                'reference_mechanical_energy': self.dynamics.total_energy(state)}
