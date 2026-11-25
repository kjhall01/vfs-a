"""Public ocean state. Contents are per horizontal area on the layer grid."""
from dataclasses import dataclass, field, replace
from functools import cached_property

import torch

from .helpers import nonzero_divisor
from .state import OceanState as LayerState


@dataclass(frozen=True, eq=False)
class OceanState:
    mesh: object
    column: object
    u_e: torch.Tensor
    h: torch.Tensor
    temp_h: torch.Tensor
    salt_h: torch.Tensor
    tracer_h: torch.Tensor | None
    tke_h: torch.Tensor
    attrs: dict = field(default_factory=dict, kw_only=True)

    @classmethod
    def from_layers(cls, layers, tke_h, **reservoir):
        return cls(layers.mesh, layers.column, layers.u_e, layers.h, layers.temp_h,
                   layers.salt_h, layers.tracer_h, tke_h, attrs=layers.attrs, **reservoir)

    @cached_property
    def resolved(self):
        """Tensor-sharing view for operators that evolve only the resolved fluid."""
        return LayerState(self.u_e, self.h, self.temp_h, self.salt_h, self.tracer_h,
                          mesh=self.mesh, column=self.column, attrs=self.attrs)

    @property
    def ocean(self):
        """Resolved layer view for existing component-level operators."""
        return self.resolved

    def with_layers(self, layers, **reservoirs):
        return replace(self, mesh=layers.mesh, column=layers.column, attrs=layers.attrs,
                       u_e=layers.u_e, h=layers.h, temp_h=layers.temp_h,
                       salt_h=layers.salt_h, tracer_h=layers.tracer_h, **reservoirs)

    def temperature(self):
        return self.resolved.temperature()

    def salinity(self):
        return self.resolved.salinity()

    def specific_tke(self):
        return self.tke_h / nonzero_divisor(self.h)


@dataclass(frozen=True, eq=False)
class EddyState(OceanState):
    eddy_h: torch.Tensor

    @classmethod
    def from_layers(cls, layers, tke_h, eddy_h):
        return super().from_layers(layers, tke_h, eddy_h=eddy_h)

    def specific_eddy_energy(self):
        return self.eddy_h / nonzero_divisor(self.h)
