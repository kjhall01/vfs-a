"""Shared pressure-coordinate transport input, with no thermodynamic disguise.

Dry potential-temperature content is Pa K; moist entropy content is
Pa J/(kg K). Both occupy cell interfaces and divide by the same CP pressure
mass. This borrowed tensor view owns no geometry, caches or conversions.
"""
from dataclasses import dataclass
from typing import Any
import torch
from vfs.atmosphere.state import interface_mass


@dataclass(frozen=True)
class ScalarTransportState:
    """u_e: edge/full m/s; ps: cell/1 Pa; scalar_content: cell/interface.

    The caller specifies the scalar's physical interpretation; geometry and
    tensors are shared by identity. This is transport input, not a dry State,
    and deliberately has no theta/mtheta_i or dry temperature interface.
    """
    u_e: torch.Tensor
    ps: torch.Tensor
    scalar_content: torch.Tensor
    mesh: Any
    column: Any

    def layer_dp(self):
        return self.column.layer_thickness(self.ps)

    def specific_scalar(self):
        """K for dry theta; J/(kg K) for moist entropy, at cell interfaces."""
        return self.scalar_content / interface_mass(self.layer_dp())
