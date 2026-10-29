"""Post-dynamics saturation adjustment for coupled moist integrations."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..core.registry import register_component
from ..moisture import cloud_liquid_tracer_index, saturation_mixing_ratio, vapor_tracer_index
from ..physics import _theta_full_and_exner_factor
from ..process import PhysicsStepResult
from .state import CoupledState


class SaturationAdjustmentProcess(nn.Module):
    """Condense grid-cell supersaturation into cloud liquid after dynamics."""

    recipe_name = "saturation_adjustment_process"

    def __init__(
        self,
        *,
        relative_humidity_limit: float = 1.0,
        latent_heat_vaporization: float = 2.5e6,
        require_cloud_liquid: bool = True,
    ) -> None:
        super().__init__()
        if float(relative_humidity_limit) <= 0.0:
            raise ValueError("relative_humidity_limit must be positive")
        self.relative_humidity_limit = float(relative_humidity_limit)
        self.latent_heat_vaporization = float(latent_heat_vaporization)
        self.require_cloud_liquid = bool(require_cloud_liquid)

    def initial_physics_state(self, state: CoupledState) -> None:
        return None

    def step(
        self,
        state: CoupledState,
        dt: float,
        *,
        physics_state: Any | None = None,
    ) -> PhysicsStepResult:
        atmosphere = state.atmosphere
        q = atmosphere.tracers()
        vapor_index = vapor_tracer_index(atmosphere)
        cloud_index = cloud_liquid_tracer_index(atmosphere)
        diagnostics: dict[str, Any] = {
            "relative_humidity_limit": self.relative_humidity_limit,
            "has_vapor": q is not None and vapor_index is not None,
            "has_cloud_liquid": cloud_index is not None,
        }
        if q is None or vapor_index is None or atmosphere.tracer_m is None:
            return PhysicsStepResult(state, physics_state, diagnostics)
        if cloud_index is None:
            if self.require_cloud_liquid:
                raise ValueError("SaturationAdjustmentProcess requires a cloud-liquid tracer")
            return PhysicsStepResult(state, physics_state, diagnostics)

        temperature = atmosphere.temperature()
        pressure = atmosphere.full_pressure()
        r_v = q[..., int(vapor_index)]
        r_sat = saturation_mixing_ratio(temperature, pressure)
        rh_before = r_v / r_sat.clamp_min(torch.finfo(r_v.dtype).tiny)
        excess = (r_v - self.relative_humidity_limit * r_sat).clamp_min(0.0)
        adjusted = excess > 0.0
        condensed_m = atmosphere.layer_dp() * excess
        condensed_mass = atmosphere._require_mesh().integrate_cell(condensed_m.sum(dim=-1, keepdim=True)) / float(
            (atmosphere.attrs or {}).get("gravity", 9.80665)
        )
        diagnostics.update(
            {
                "rh_before_max": rh_before.max(),
                "adjusted_cell_level_fraction": adjusted.to(dtype=temperature.dtype).mean(),
                "excess_mixing_ratio_max": excess.max(),
                "condensed_water_mass": condensed_mass.sum(),
            }
        )

        tracer_m = atmosphere.tracer_m.clone()
        tracer_m[..., int(vapor_index)] = (tracer_m[..., int(vapor_index)] - condensed_m).clamp_min(0.0)
        tracer_m[..., int(cloud_index)] = tracer_m[..., int(cloud_index)] + condensed_m

        mtheta_i = atmosphere.mtheta_i + self._latent_dmtheta(atmosphere, excess)
        next_atmosphere = atmosphere._with_tensors(atmosphere.u_e, mtheta_i, atmosphere.ps, tracer_m)
        next_state = CoupledState(next_atmosphere, state.ocean, attrs=state._attrs_copy())

        q_after = next_atmosphere.tracers()
        assert q_after is not None
        r_v_after = q_after[..., int(vapor_index)]
        r_sat_after = saturation_mixing_ratio(next_atmosphere.temperature(), next_atmosphere.full_pressure())
        diagnostics["rh_after_max"] = (r_v_after / r_sat_after.clamp_min(torch.finfo(r_v_after.dtype).tiny)).max()
        diagnostics["temperature_increment_max"] = (
            self.latent_heat_vaporization / next_atmosphere.column.c_p
        ) * excess.max()
        return PhysicsStepResult(next_state, physics_state, diagnostics)

    def _latent_dmtheta(self, atmosphere, condensed_mixing_ratio: torch.Tensor) -> torch.Tensor:
        _, exner_factor = _theta_full_and_exner_factor(atmosphere)
        d_temperature = (self.latent_heat_vaporization / atmosphere.column.c_p) * condensed_mixing_ratio
        dtheta_full = d_temperature / exner_factor.clamp_min(torch.finfo(d_temperature.dtype).tiny)
        layer_increment = atmosphere.layer_dp() * dtheta_full
        dmtheta_i = torch.zeros_like(atmosphere.mtheta_i)
        dmtheta_i[..., :-1] += 0.5 * layer_increment
        dmtheta_i[..., 1:] += 0.5 * layer_increment
        return dmtheta_i


register_component("saturation_adjustment_process", SaturationAdjustmentProcess)


__all__ = ["SaturationAdjustmentProcess"]
