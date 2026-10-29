"""Global mass fixers for split coupled integrations."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from ..core.registry import register_component
from ..core.state import State, interface_mass
from ..process import PhysicsStepResult
from .state import CoupledState


def _as_component_state(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


class DryAirMassCorrectionProcess(nn.Module):
    """Remove global dry-air mass drift with a uniform surface-pressure offset."""

    recipe_name = "dry_air_mass_correction_process"

    def __init__(self, *, min_surface_pressure: float = 1.0) -> None:
        super().__init__()
        if float(min_surface_pressure) <= 0.0:
            raise ValueError("min_surface_pressure must be positive")
        self.min_surface_pressure = float(min_surface_pressure)

    def initial_physics_state(self, state: CoupledState) -> dict[str, torch.Tensor]:
        atmosphere = state.atmosphere
        return {
            "target_dry_air_mass": atmosphere.total_mass().detach().clone(),
            "cumulative_surface_pressure_correction": atmosphere.ps.new_zeros(()),
        }

    def step(
        self,
        state: CoupledState,
        dt: float,
        *,
        physics_state: Any | None = None,
    ) -> PhysicsStepResult:
        atmosphere = state.atmosphere
        component_state = _as_component_state(physics_state)
        target_mass = component_state.get("target_dry_air_mass")
        if not torch.is_tensor(target_mass):
            target_mass = atmosphere.total_mass().detach().clone()

        mass_before = atmosphere.total_mass()
        area = atmosphere.mesh.cell_area.to(dtype=atmosphere.ps.dtype, device=atmosphere.ps.device).sum()
        gravity = float((atmosphere.attrs or {}).get("gravity", 9.80665))
        dps = (target_mass.to(dtype=atmosphere.ps.dtype, device=atmosphere.ps.device) - mass_before) * gravity
        dps = dps / area.clamp_min(torch.finfo(atmosphere.ps.dtype).tiny)
        ps = (atmosphere.ps + dps).clamp_min(self.min_surface_pressure)

        next_atmosphere = _with_preserved_theta(atmosphere, ps)
        mass_after = next_atmosphere.total_mass()
        cumulative = component_state.get("cumulative_surface_pressure_correction")
        if not torch.is_tensor(cumulative):
            cumulative = atmosphere.ps.new_zeros(())
        component_state["target_dry_air_mass"] = target_mass
        component_state["cumulative_surface_pressure_correction"] = cumulative.to(
            dtype=atmosphere.ps.dtype,
            device=atmosphere.ps.device,
        ) + dps

        residual_before = (mass_before - target_mass) / target_mass.abs().clamp_min(1.0)
        residual_after = (mass_after - target_mass) / target_mass.abs().clamp_min(1.0)
        diagnostics = {
            "target_dry_air_mass": target_mass,
            "mass_before": mass_before,
            "mass_after": mass_after,
            "relative_residual_before": residual_before,
            "relative_residual_after": residual_after,
            "surface_pressure_correction": dps,
            "cumulative_surface_pressure_correction": component_state["cumulative_surface_pressure_correction"],
        }
        return PhysicsStepResult(
            CoupledState(next_atmosphere, state.ocean, attrs=state._attrs_copy()),
            component_state,
            diagnostics,
        )


def _with_preserved_theta(atmosphere: State, ps: torch.Tensor) -> State:
    theta = atmosphere.theta()
    layer_dp = atmosphere.column.layer_thickness(ps)
    dm = interface_mass(layer_dp)
    return atmosphere._with_tensors(
        atmosphere.u_e,
        dm * theta,
        ps,
        atmosphere.tracer_m,
    )


register_component("dry_air_mass_correction_process", DryAirMassCorrectionProcess)


__all__ = ["DryAirMassCorrectionProcess"]
