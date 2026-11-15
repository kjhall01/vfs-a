"""Explicit workflow state; neither VFS nor PyAM21 owns the simulation clock."""

from dataclasses import dataclass
import torch
from vfs.atmosphere.moist_conservative import MoistState


def copy_tree(value):
    """Clone tensors without detaching gradients; copy plain restart containers."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {k: copy_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(copy_tree(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported workflow state type: {type(value)}")


@dataclass(frozen=True)
class PhysicsState:
    atmosphere: MoistState
    cloud_fraction: torch.Tensor  # [cell, CP interface]; passive during dynamics
    memory: dict  # PyAM21 persistent diffusivity/convection fields
    surface: dict  # roughness, albedos, prescribed SST
    radiation: dict  # cached radiation fields and integer alarms
    time_seconds: int  # elapsed since the adapter's explicit start date


@dataclass(frozen=True)
class PhysicsReport:
    fluxes: dict  # surface rates, positive into atmosphere unless named otherwise
    water_increment: torch.Tensor  # [cell,3] kg/m2, vapor/liquid/ice
    dry_air_residual: torch.Tensor  # [cell] kg/m2
    water_mapping_residual: torch.Tensor  # [cell,3] kg/m2
    water_flux_residual: torch.Tensor  # [cell] kg/m2, actual increment - dt*(E-P)
    energy_changes: dict  # [cell] J/m2; independent boundary/upstream/transfer ledgers
    edge_impulse: torch.Tensor  # [edge,level], kg m/s, work-adjoint physics impulse
    carrier_impulse: torch.Tensor  # explicit surface-water impulse; legacy: co-moving
    momentum_residual: torch.Tensor
    column_result: dict  # exact outputs of the staged PyAM21 call
    energy_transfer: dict | None = None  # layer constraints and nonlinear solve report
