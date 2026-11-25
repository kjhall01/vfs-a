"""Geometry and conservative tensor construction shared by layered dynamics."""
import torch
from .state import OceanState


class LayerGrid:
    def __init__(self, mesh, column, domain, dynamics):
        if domain is not None and domain.mesh is not mesh:
            raise ValueError("domain must borrow this exact mesh")
        if column.active_layers is not None:
            if domain is None:
                raise ValueError("partial ocean layers require a wet ocean domain")
            domain.set_active_layers(column.active_layers)
        elif domain is not None and domain.active_edge_layers is not None:
            raise ValueError("a layered domain must use the matching partial ocean column")
        self.mesh = mesh
        self.column = column
        self.domain = domain
        self.dynamics = dynamics

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
        if self.column.active_layers is not None:
            padding = ~self.column.active_layers
            occupied = (state.temp_h != 0) | (state.salt_h != 0)
            if state.tracer_h is not None:
                occupied = occupied | (state.tracer_h != 0).any(-1)
            if (occupied & padding).any():
                raise ValueError("nonzero inventory in inactive ocean layer")
            if self.domain is not None and (state.u_e.masked_select(~self.domain.active_edge_layers) != 0).any():
                raise ValueError("nonzero velocity on a closed layer face")

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
