"""Column-physics tendency providers for VFS."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from vfs.geometry.column import CharneyPhillipsColumn
from vfs.core.integrators import ssprk3
from vfs.moisture import (
    CLOUD_LIQUID_QUANTITY_DRY_MIXING_RATIO,
    VAPOR_QUANTITY_DRY_MIXING_RATIO,
    cloud_liquid_mixing_ratio,
    cloud_liquid_tracer_index,
    saturation_mixing_ratio,
    specific_humidity_from_mixing_ratio,
    vapor_mixing_ratio,
    vapor_tracer_index,
)
from vfs.process import PhysicsStepResult
from vfs.core.state import State, StateRequirement, TracerRequirement, interface_mass


def _vapor_state_requirement() -> StateRequirement:
    return StateRequirement(
        tracers=(
            TracerRequirement(
                name="vapor",
                quantity=VAPOR_QUANTITY_DRY_MIXING_RATIO,
            ),
        )
    )


def _vapor_cloud_state_requirement() -> StateRequirement:
    return StateRequirement(
        tracers=(
            TracerRequirement(
                name="vapor",
                quantity=VAPOR_QUANTITY_DRY_MIXING_RATIO,
            ),
            TracerRequirement(
                name="cloud_liquid",
                quantity=CLOUD_LIQUID_QUANTITY_DRY_MIXING_RATIO,
            ),
        )
    )


def _component_state_requirement(component: Any) -> StateRequirement | None:
    method = getattr(component, "required_state_requirement", None)
    if callable(method):
        return method()
    requirement = getattr(component, "state_requirement", None)
    if requirement is None:
        return None
    return StateRequirement.merge(requirement)


@dataclass(frozen=True)
class HeldSuarezConstants:
    """Standard dry Held-Suarez forcing constants."""

    p0: float = 1.0e5
    T0: float = 315.0
    T_min: float = 200.0
    delta_T_y: float = 60.0
    delta_theta_z: float = 10.0
    sigma_b: float = 0.7
    k_a: float = 1.0 / (40.0 * 86400.0)
    k_s: float = 1.0 / (4.0 * 86400.0)
    k_f: float = 1.0 / 86400.0

    @property
    def T_surface_equator(self) -> float:
        return self.T0

    @property
    def k_a_per_day(self) -> float:
        return self.k_a * 86400.0

    @property
    def k_s_per_day(self) -> float:
        return self.k_s * 86400.0

    @property
    def k_f_per_day(self) -> float:
        return self.k_f * 86400.0


def _lat_for_pressure(lat: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
    lat_aligned = lat
    while lat_aligned.ndim < pressure.ndim:
        lat_aligned = lat_aligned[..., None]
    return lat_aligned


def held_suarez_equilibrium_temperature(
    column: CharneyPhillipsColumn,
    constants: HeldSuarezConstants,
    lat: torch.Tensor,
    pressure: torch.Tensor,
) -> torch.Tensor:
    """Held-Suarez radiative-equilibrium temperature at pressure points."""

    lat = _lat_for_pressure(lat, pressure)
    sigma = (pressure / float(constants.p0)).clamp_min(1.0e-12)
    raw = (
        float(constants.T0)
        - float(constants.delta_T_y) * torch.sin(lat).square()
        - float(constants.delta_theta_z) * torch.log(sigma) * torch.cos(lat).square()
    ) * sigma.pow(column.kappa)
    floor = torch.as_tensor(constants.T_min, dtype=raw.dtype, device=raw.device)
    return torch.maximum(raw, floor)


class HeldSuarezPhysics(nn.Module):
    """Held-Suarez thermal relaxation and Rayleigh drag as a source tendency."""

    def __init__(
        self,
        constants: HeldSuarezConstants | None = None,
        *,
        Teq: torch.Tensor | None = None,
        kT: torch.Tensor | None = None,
        kv: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.constants = constants or HeldSuarezConstants()
        self.register_buffer("Teq", None if Teq is None else torch.as_tensor(Teq))
        self.register_buffer("kT", None if kT is None else torch.as_tensor(kT))
        self.register_buffer("kv", None if kv is None else torch.as_tensor(kv))

    def _cell_lat(self, state: State) -> torch.Tensor:
        mesh = state._require_mesh()
        return torch.asin(torch.clamp(mesh.cell_xyz[:, 2:3], -1.0, 1.0))

    def _optional_tensor(self, value: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor | None:
        if value is None:
            return None
        return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)

    def equilibrium_temperature(self, state: State, pressure: torch.Tensor) -> torch.Tensor:
        Teq = self._optional_tensor(self.Teq, pressure)
        if Teq is not None:
            return Teq
        return held_suarez_equilibrium_temperature(state._require_column(), self.constants, self._cell_lat(state), pressure)

    def thermal_relaxation_rate(self, state: State, sigma: torch.Tensor) -> torch.Tensor:
        kT = self._optional_tensor(self.kT, sigma)
        if kT is not None:
            return kT
        c = self.constants
        ramp = ((sigma - float(c.sigma_b)) / (1.0 - float(c.sigma_b))).clamp(0.0, 1.0)
        return float(c.k_a) + (float(c.k_s) - float(c.k_a)) * ramp * torch.cos(self._cell_lat(state)).pow(4)

    def drag_rate(self, state: State, sigma: torch.Tensor) -> torch.Tensor:
        kv = self._optional_tensor(self.kv, sigma)
        if kv is not None:
            return kv
        c = self.constants
        ramp = ((sigma - float(c.sigma_b)) / (1.0 - float(c.sigma_b))).clamp(0.0, 1.0)
        return float(c.k_f) * ramp

    def tendency(self, state: State) -> State:
        theta = state.theta()
        dp = state.layer_dp()
        dm = interface_mass(dp)

        p_i = state.interface_pressure()
        sigma_i = (p_i / state.ps.clamp_min(1.0)).clamp_min(1.0e-12)
        theta_eq = self.equilibrium_temperature(state, p_i) / (
            state.exner_interface() / state._require_column().c_p
        ).clamp_min(1.0e-12)
        dmtheta = dm * self.thermal_relaxation_rate(state, sigma_i) * (theta_eq - theta)

        p_full = state.full_pressure()
        sigma_full = (p_full / state.ps.clamp_min(1.0)).clamp_min(1.0e-12)
        du = -state.cell_to_edge(self.drag_rate(state, sigma_full)) * state.u_e
        tracer = None if state.tracer_m is None else torch.zeros_like(state.tracer_m)
        return State(
            du,
            dmtheta,
            torch.zeros_like(state.ps),
            tracer,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )

    def forward(self, state: State) -> State:
        return self.tendency(state)


def _theta_full_and_exner_factor(state: State) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full-level theta and Exner/cp using the active column placement."""

    column = state._require_column()
    theta_i = state.theta()
    temperature = column.temperature_full(theta_i, state.ps)
    if column.phi_full_mode in ("linear_pfull_adjoint", "direct_u_linear_adjoint", "direct_u_gl2_adjoint"):
        pi_full = column.exner_full_pmid(state.ps)
    else:
        pi_full = column.exner_full(state.ps)
    exner_factor = (pi_full / column.c_p).clamp_min(torch.finfo(temperature.dtype).tiny)
    return temperature / exner_factor, exner_factor


def _broadcast_to_ref(name: str, value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
    try:
        return torch.broadcast_to(tensor, ref.shape)
    except RuntimeError as exc:
        raise ValueError(f"{name} must broadcast to {tuple(ref.shape)}, got {tuple(tensor.shape)}") from exc


def _prepare_param(name: str, value: torch.Tensor | float | int, ref: torch.Tensor, n_cells: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
    if tensor.ndim == 0:
        return tensor
    if tensor.ndim == 1 and int(tensor.shape[0]) == int(n_cells):
        return tensor[:, None]
    if tensor.ndim >= 2 and int(tensor.shape[-2]) == int(n_cells):
        return tensor
    raise ValueError(
        f"SCM parameter {name!r} must be scalar, [nCells], or [..., nCells, ...]; "
        f"got {tuple(tensor.shape)} for nCells={int(n_cells)}"
    )


class ToyCondensationSCM(nn.Module):
    """Simple vectorized saturation-relaxation column model.

    This is a test/shim column model, not a validated cloud parameterization.
    It removes supersaturated vapor on a configurable timescale and returns the
    corresponding dry-theta heating using a constant latent heat.
    """

    def __init__(
        self,
        *,
        condensation_rate: float = 1.0 / 3600.0,
        latent_heat_vaporization: float = 2.5e6,
        c_p: float = 1004.0,
    ) -> None:
        super().__init__()
        self.register_buffer("condensation_rate", torch.as_tensor(float(condensation_rate)))
        self.latent_heat_vaporization = float(latent_heat_vaporization)
        self.c_p = float(c_p)
        self.state_requirement = _vapor_state_requirement()

    def required_state_requirement(self) -> StateRequirement:
        return self.state_requirement

    def _param(self, columns: Mapping[str, Any], name: str, default: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        params = columns.get("params", {})
        if isinstance(params, Mapping) and name in params:
            return _broadcast_to_ref(name, params[name], ref)
        return torch.as_tensor(default, dtype=ref.dtype, device=ref.device)

    def forward(self, columns: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        temperature = columns["temperature_full"]
        pressure = columns["p_full"]
        r_v = columns["r_v"]
        exner_factor = columns["exner_full_over_cp"]

        rate = self._param(columns, "condensation_rate", self.condensation_rate, r_v)
        r_sat = saturation_mixing_ratio(temperature, pressure)
        condensation = rate * (r_v - r_sat).clamp_min(0.0)
        dT_dt = (float(self.latent_heat_vaporization) / float(self.c_p)) * condensation
        return {
            "dtheta_full_dt": dT_dt / exner_factor.clamp_min(torch.finfo(dT_dt.dtype).tiny),
            "dr_v_dt": -condensation,
            "condensation_rate": condensation,
            "saturation_mixing_ratio": r_sat,
        }


class KesslerWarmRainSCM(nn.Module):
    """Single-moment warm-rain column physics with vapor and cloud water.

    The prognostic water variables are dry-air mixing ratios: water vapor
    ``r_v`` and cloud liquid ``r_l``.  Condensation and evaporation exchange
    mass between those two tracers and apply matched latent heating/cooling.
    Cloud water above a threshold autoconverts to immediate surface
    precipitation; no rain-water tracer or rain sedimentation is represented.
    """

    def __init__(
        self,
        *,
        condensation_rate: float = 1.0 / 600.0,
        evaporation_rate: float = 1.0 / 600.0,
        autoconversion_rate: float = 1.0 / 1000.0,
        cloud_water_threshold: float = 1.0e-3,
        latent_heat_vaporization: float = 2.5e6,
        c_p: float = 1004.0,
        gravity: float = 9.80665,
    ) -> None:
        super().__init__()
        self.register_buffer("condensation_rate", torch.as_tensor(float(condensation_rate)))
        self.register_buffer("evaporation_rate", torch.as_tensor(float(evaporation_rate)))
        self.register_buffer("autoconversion_rate", torch.as_tensor(float(autoconversion_rate)))
        self.register_buffer("cloud_water_threshold", torch.as_tensor(float(cloud_water_threshold)))
        self.latent_heat_vaporization = float(latent_heat_vaporization)
        self.c_p = float(c_p)
        self.gravity = float(gravity)
        self.state_requirement = _vapor_cloud_state_requirement()

    def required_state_requirement(self) -> StateRequirement:
        return self.state_requirement

    def _param(self, columns: Mapping[str, Any], name: str, default: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        params = columns.get("params", {})
        if isinstance(params, Mapping) and name in params:
            return _broadcast_to_ref(name, params[name], ref)
        return _broadcast_to_ref(name, default, ref)

    def forward(self, columns: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        temperature = columns["temperature_full"]
        pressure = columns["p_full"]
        r_v = columns["r_v"]
        r_l = columns.get("r_l")
        if r_l is None:
            raise ValueError("KesslerWarmRainSCM requires columns['r_l'] from a named cloud-liquid tracer")
        r_l = _broadcast_to_ref("r_l", r_l, r_v)
        exner_factor = columns["exner_full_over_cp"]

        condensation_rate = self._param(columns, "condensation_rate", self.condensation_rate, r_v).clamp_min(0.0)
        evaporation_rate = self._param(columns, "evaporation_rate", self.evaporation_rate, r_v).clamp_min(0.0)
        autoconversion_rate = self._param(columns, "autoconversion_rate", self.autoconversion_rate, r_v).clamp_min(0.0)
        cloud_water_threshold = self._param(columns, "cloud_water_threshold", self.cloud_water_threshold, r_v).clamp_min(0.0)

        r_sat = saturation_mixing_ratio(temperature, pressure)
        r_v_nonnegative = r_v.clamp_min(0.0)
        r_l_nonnegative = r_l.clamp_min(0.0)

        condensation = condensation_rate * (r_v_nonnegative - r_sat).clamp_min(0.0)
        evaporation_demand = evaporation_rate * (r_sat - r_v_nonnegative).clamp_min(0.0)
        evaporation = torch.minimum(evaporation_demand, evaporation_rate * r_l_nonnegative)
        autoconversion = autoconversion_rate * (r_l_nonnegative - cloud_water_threshold).clamp_min(0.0)

        net_condensation = condensation - evaporation
        dT_dt = (float(self.latent_heat_vaporization) / float(self.c_p)) * net_condensation
        g = float(columns.get("gravity", self.gravity))
        surface_precipitation_rate = (columns["dp"] * autoconversion / g).sum(dim=-1, keepdim=True)
        return {
            "dtheta_full_dt": dT_dt / exner_factor.clamp_min(torch.finfo(dT_dt.dtype).tiny),
            "dr_v_dt": -net_condensation,
            "dr_l_dt": net_condensation - autoconversion,
            "condensation_mixing_ratio_tendency": condensation,
            "evaporation_mixing_ratio_tendency": evaporation,
            "autoconversion_mixing_ratio_tendency": autoconversion,
            "precipitation_mixing_ratio_tendency": autoconversion,
            "surface_precipitation_rate": surface_precipitation_rate,
            "saturation_mixing_ratio": r_sat,
            "relative_humidity": r_v_nonnegative / r_sat.clamp_min(torch.finfo(r_v.dtype).tiny),
        }


class VFSNativeSCM(nn.Module):
    """Compact VFS-native climate column physics.

    This is a differentiable tensor-native bridge component, not a mature
    parameterization package. It combines four minimal processes that are useful
    for coupled atmosphere/ocean experiments:

    - gray/Newtonian atmospheric radiation;
    - lower-tropospheric PBL scalar mixing and optional near-surface wind drag;
    - bulk turbulent plus simple radiative ocean surface fluxes;
    - immediate saturation-adjustment latent heating with precipitation sink.

    Water vapor is a dry-air mixing ratio carried in the named VFS vapor tracer.
    Condensed excess vapor precipitates immediately; no prognostic cloud/rain
    store is required.
    """

    # ``step_columns`` is provided for the conventional split-process API, but
    # this model has no private restart update: its ARK source evaluation is
    # exactly ``forward(columns)``.  CoupledSCMProcess requires this explicit
    # declaration before accepting any column model with ``step_columns``.
    ark_stage_contract = "stateless_tendency"

    def __init__(
        self,
        *,
        radiative_timescale: float = 20.0 * 86400.0,
        radiative_lapse_rate: float = 6.5e-3,
        radiative_scale_height: float = 7000.0,
        radiative_surface_temperature: float = 288.0,
        radiative_min_temperature: float = 200.0,
        pbl_mixing_timescale: float = 6.0 * 3600.0,
        pbl_sigma_top: float = 0.85,
        pbl_transition: float = 0.03,
        pbl_drag_timescale: float = 0.0,
        surface_exchange_coefficient_heat: float = 1.1e-3,
        surface_exchange_coefficient_moisture: float = 1.1e-3,
        surface_wind_min: float = 1.0,
        air_density: float = 1.225,
        c_p: float = 1004.0,
        latent_heat_vaporization: float = 2.5e6,
        water_density: float = 1000.0,
        reference_salinity: float = 35.0,
        absorbed_shortwave_surface_down: float = 160.0,
        shortwave_latitude_gradient: float = 0.35,
        surface_emissivity: float = 0.98,
        atmosphere_emissivity: float = 0.78,
        stefan_boltzmann: float = 5.670374419e-8,
        condensation_timescale: float = 600.0,
        relative_humidity_condensation_threshold: float = 1.0,
        gravity: float = 9.80665,
    ) -> None:
        super().__init__()
        for name, value in {
            "radiative_timescale": radiative_timescale,
            "radiative_scale_height": radiative_scale_height,
            "pbl_mixing_timescale": pbl_mixing_timescale,
            "surface_wind_min": surface_wind_min,
            "air_density": air_density,
            "c_p": c_p,
            "latent_heat_vaporization": latent_heat_vaporization,
            "water_density": water_density,
            "condensation_timescale": condensation_timescale,
            "gravity": gravity,
        }.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if float(pbl_transition) <= 0.0:
            raise ValueError("pbl_transition must be positive")
        if not 0.0 <= float(pbl_sigma_top) <= 1.0:
            raise ValueError("pbl_sigma_top must be in [0, 1]")
        if not 0.0 <= float(shortwave_latitude_gradient) <= 1.0:
            raise ValueError("shortwave_latitude_gradient must be in [0, 1]")
        self.register_buffer("radiative_timescale", torch.as_tensor(float(radiative_timescale)))
        self.register_buffer("radiative_lapse_rate", torch.as_tensor(float(radiative_lapse_rate)))
        self.register_buffer("radiative_scale_height", torch.as_tensor(float(radiative_scale_height)))
        self.register_buffer("radiative_surface_temperature", torch.as_tensor(float(radiative_surface_temperature)))
        self.register_buffer("radiative_min_temperature", torch.as_tensor(float(radiative_min_temperature)))
        self.register_buffer("pbl_mixing_rate", torch.as_tensor(1.0 / float(pbl_mixing_timescale)))
        self.register_buffer("pbl_sigma_top", torch.as_tensor(float(pbl_sigma_top)))
        self.register_buffer("pbl_transition", torch.as_tensor(float(pbl_transition)))
        drag_rate = 0.0 if float(pbl_drag_timescale) <= 0.0 else 1.0 / float(pbl_drag_timescale)
        self.register_buffer("pbl_drag_rate", torch.as_tensor(drag_rate))
        self.register_buffer("surface_exchange_coefficient_heat", torch.as_tensor(float(surface_exchange_coefficient_heat)))
        self.register_buffer(
            "surface_exchange_coefficient_moisture",
            torch.as_tensor(float(surface_exchange_coefficient_moisture)),
        )
        self.register_buffer("surface_wind_min", torch.as_tensor(float(surface_wind_min)))
        self.air_density = float(air_density)
        self.c_p = float(c_p)
        self.latent_heat_vaporization = float(latent_heat_vaporization)
        self.water_density = float(water_density)
        self.reference_salinity = float(reference_salinity)
        self.absorbed_shortwave_surface_down = float(absorbed_shortwave_surface_down)
        self.shortwave_latitude_gradient = float(shortwave_latitude_gradient)
        self.surface_emissivity = float(surface_emissivity)
        self.atmosphere_emissivity = float(atmosphere_emissivity)
        self.stefan_boltzmann = float(stefan_boltzmann)
        self.register_buffer("condensation_rate", torch.as_tensor(1.0 / float(condensation_timescale)))
        self.relative_humidity_condensation_threshold = float(relative_humidity_condensation_threshold)
        self.gravity = float(gravity)
        self.state_requirement = _vapor_state_requirement()

    def required_state_requirement(self) -> StateRequirement:
        return self.state_requirement

    def _param(self, columns: Mapping[str, Any], name: str, default: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        params = columns.get("params", {})
        if isinstance(params, Mapping) and name in params:
            return _broadcast_to_ref(name, params[name], ref)
        return _broadcast_to_ref(name, default, ref)

    def _surface_param(
        self,
        columns: Mapping[str, Any],
        name: str,
        default: torch.Tensor,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        params = columns.get("params", {})
        if isinstance(params, Mapping) and name in params:
            return _broadcast_to_ref(name, params[name], ref)
        return _broadcast_to_ref(name, default, ref)

    def _pbl_mask(self, columns: Mapping[str, Any], ref: torch.Tensor) -> torch.Tensor:
        sigma = (columns["p_full"] / columns["surface_pressure"].clamp_min(1.0)).clamp(0.0, 1.5)
        top = self._param(columns, "pbl_sigma_top", self.pbl_sigma_top, ref)
        transition = self._param(columns, "pbl_transition", self.pbl_transition, ref).clamp_min(
            torch.finfo(ref.dtype).eps
        )
        return torch.sigmoid((sigma - top) / transition)

    def _mix_to_pbl_mean(
        self,
        field: torch.Tensor,
        dp: torch.Tensor,
        mask: torch.Tensor,
        rate: torch.Tensor,
    ) -> torch.Tensor:
        weights = dp * mask
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(field.dtype).tiny)
        mean = (weights * field).sum(dim=-1, keepdim=True) / denom
        return rate * mask * (mean - field)

    def _surface_fluxes(
        self,
        columns: Mapping[str, Any],
        temperature: torch.Tensor,
        r_v: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        surface_ref = temperature[..., -1:]
        p_surface_level = columns["p_full"][..., -1:]
        surface_temperature = self._surface_param(
            columns,
            "surface_temperature",
            self.radiative_surface_temperature,
            surface_ref,
        )
        relative_wind = self._surface_param(
            columns,
            "relative_wind_speed_cell",
            torch.zeros_like(surface_ref),
            surface_ref,
        ).clamp_min(0.0)
        wind_min = self._surface_param(
            columns,
            "surface_wind_min",
            self.surface_wind_min,
            surface_ref,
        ).clamp_min(0.0)
        wind_speed = (relative_wind.square() + wind_min.square()).sqrt()

        heat_coeff = self._surface_param(
            columns,
            "surface_exchange_coefficient_heat",
            self.surface_exchange_coefficient_heat,
            surface_ref,
        ).clamp_min(0.0)
        moisture_coeff = self._surface_param(
            columns,
            "surface_exchange_coefficient_moisture",
            self.surface_exchange_coefficient_moisture,
            surface_ref,
        ).clamp_min(0.0)
        air_temperature = temperature[..., -1:]
        r_air = r_v[..., -1:].clamp_min(0.0)
        r_sat_surface = saturation_mixing_ratio(surface_temperature, p_surface_level)
        sensible_up = (
            float(self.air_density)
            * float(self.c_p)
            * heat_coeff
            * wind_speed
            * (surface_temperature - air_temperature)
        )
        evaporation_up = float(self.air_density) * moisture_coeff * wind_speed * (r_sat_surface - r_air)
        latent_up = float(self.latent_heat_vaporization) * evaporation_up

        lat = columns["lat"]
        shortwave_default = float(self.absorbed_shortwave_surface_down) * (
            1.0 - float(self.shortwave_latitude_gradient) * torch.sin(lat).square()
        )
        shortwave_down = self._surface_param(
            columns,
            "absorbed_shortwave_surface_down",
            shortwave_default,
            surface_ref,
        ).clamp_min(0.0)
        longwave_up = (
            float(self.surface_emissivity)
            * float(self.stefan_boltzmann)
            * surface_temperature.clamp_min(0.0).pow(4)
        )
        longwave_down = (
            float(self.atmosphere_emissivity)
            * float(self.stefan_boltzmann)
            * air_temperature.clamp_min(0.0).pow(4)
        )
        ocean_heat_down = shortwave_down + longwave_down - longwave_up - sensible_up - latent_up
        freshwater_down = -evaporation_up / float(self.water_density)
        return {
            "surface_temperature": surface_temperature,
            "surface_wind_speed": wind_speed,
            "surface_saturation_mixing_ratio": r_sat_surface,
            "sensible_heat_flux_up": sensible_up,
            "evaporation_mass_flux_up": evaporation_up,
            "latent_heat_flux_up": latent_up,
            "surface_shortwave_down": shortwave_down,
            "surface_longwave_down": longwave_down,
            "surface_longwave_up": longwave_up,
            "ocean_surface_heat_flux_down": ocean_heat_down,
            "freshwater_flux_down": freshwater_down,
        }

    def forward(self, columns: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        temperature = columns["temperature_full"]
        pressure = columns["p_full"]
        dp = columns["dp"]
        r_v = columns["r_v"].clamp_min(0.0)
        ref = temperature

        surface_temperature = self._surface_param(
            columns,
            "surface_temperature",
            self.radiative_surface_temperature,
            temperature[..., -1:],
        )
        radiation_surface_temperature = self._surface_param(
            columns,
            "radiative_surface_temperature",
            surface_temperature,
            temperature[..., -1:],
        )
        scale_height = self._param(columns, "radiative_scale_height", self.radiative_scale_height, ref).clamp_min(
            torch.finfo(ref.dtype).eps
        )
        height = scale_height * torch.log((columns["surface_pressure"] / pressure.clamp_min(1.0)).clamp_min(1.0))
        lapse = self._param(columns, "radiative_lapse_rate", self.radiative_lapse_rate, ref)
        t_min = self._param(columns, "radiative_min_temperature", self.radiative_min_temperature, ref)
        t_eq = torch.maximum(radiation_surface_temperature - lapse * height, t_min)
        tau_rad = self._param(columns, "radiative_timescale", self.radiative_timescale, ref).clamp_min(
            torch.finfo(ref.dtype).eps
        )
        dT_radiation = (t_eq - temperature) / tau_rad

        mask = self._pbl_mask(columns, ref)
        pbl_rate = self._param(columns, "pbl_mixing_rate", self.pbl_mixing_rate, ref).clamp_min(0.0)
        dT_pbl = self._mix_to_pbl_mean(temperature, dp, mask, pbl_rate)
        dr_v_pbl = self._mix_to_pbl_mean(r_v, dp, mask, pbl_rate)

        fluxes = self._surface_fluxes(columns, temperature, r_v)
        dT_surface = torch.zeros_like(temperature)
        dr_v_surface = torch.zeros_like(r_v)
        gravity = float(columns.get("gravity", self.gravity))
        bottom_dp = dp[..., -1:].clamp_min(torch.finfo(dp.dtype).tiny)
        dT_surface[..., -1:] = gravity * fluxes["sensible_heat_flux_up"] / (float(self.c_p) * bottom_dp)
        dr_v_surface[..., -1:] = gravity * fluxes["evaporation_mass_flux_up"] / bottom_dp

        r_sat = saturation_mixing_ratio(temperature, pressure)
        threshold = float(self.relative_humidity_condensation_threshold) * r_sat
        condensation_rate = self._param(columns, "condensation_rate", self.condensation_rate, ref).clamp_min(0.0)
        condensation = condensation_rate * (r_v - threshold).clamp_min(0.0)
        dT_latent = (float(self.latent_heat_vaporization) / float(self.c_p)) * condensation
        dr_v_latent = -condensation
        precipitation_rate = (dp * condensation / gravity).sum(dim=-1, keepdim=True)
        # ``_surface_fluxes`` can only see evaporation. Incorporate the
        # diagnosed column precipitation here so the reported freshwater flux
        # is the complete positive-downward ocean budget, not an evaporation
        # diagnostic mislabeled as a coupled transfer.
        fluxes["freshwater_flux_down"] = (
            precipitation_rate - fluxes["evaporation_mass_flux_up"]
        ) / float(self.water_density)
        fluxes["freshwater_flux_includes_precipitation"] = torch.ones_like(
            fluxes["freshwater_flux_down"],
            dtype=torch.bool,
        )

        outputs: dict[str, torch.Tensor] = {
            "dT_dt": dT_radiation + dT_pbl + dT_surface + dT_latent,
            "dr_v_dt": dr_v_pbl + dr_v_surface + dr_v_latent,
            "radiative_temperature_tendency": dT_radiation,
            "pbl_temperature_tendency": dT_pbl,
            "pbl_vapor_tendency": dr_v_pbl,
            "surface_temperature_tendency": dT_surface,
            "surface_vapor_tendency": dr_v_surface,
            "latent_temperature_tendency": dT_latent,
            "condensation_mixing_ratio_tendency": condensation,
            "surface_precipitation_rate": precipitation_rate,
            "saturation_mixing_ratio": r_sat,
            "relative_humidity": r_v / r_sat.clamp_min(torch.finfo(r_v.dtype).tiny),
            **fluxes,
        }
        drag_rate = self._param(columns, "pbl_drag_rate", self.pbl_drag_rate, ref).clamp_min(0.0)
        state = columns.get("state")
        if state is not None and torch.is_tensor(getattr(state, "u_e", None)):
            edge_mask = state.mesh.cell_to_edge(mask)
            edge_drag = state.mesh.cell_to_edge(drag_rate)
            outputs["u_e_dt"] = -edge_drag * edge_mask * state.u_e
        return outputs

    def step_columns(
        self,
        columns: Mapping[str, Any],
        *,
        dt: float,
        physics_state=None,
    ) -> dict[str, Any]:
        outputs = self.forward(columns)
        diagnostics = {
            "surface_sensible_heat_flux_up_mean": outputs["sensible_heat_flux_up"].mean(),
            "surface_latent_heat_flux_up_mean": outputs["latent_heat_flux_up"].mean(),
            "ocean_surface_heat_flux_down_mean": outputs["ocean_surface_heat_flux_down"].mean(),
            "surface_precipitation_rate_mean": outputs["surface_precipitation_rate"].mean(),
            "relative_humidity_mean": outputs["relative_humidity"].mean(),
        }
        return {"outputs": outputs, "physics_state": {} if physics_state is None else physics_state, "diagnostics": diagnostics}


class SCMPhysics(nn.Module):
    """Vectorized adapter from PyTorch column models to VFS State tendencies.

    The wrapped column model receives a dictionary of tensors with horizontal
    cells as the batch dimension.  It should return a mapping containing one or
    more known tendency fields, such as ``dtheta_i_dt``, ``dtheta_full_dt``,
    ``dT_dt``, ``dr_v_dt``, or ``dr_l_dt``.
    """

    def __init__(
        self,
        column_model: nn.Module | None = None,
        *,
        params: Mapping[str, torch.Tensor | float | int] | None = None,
        state_requirement: StateRequirement | Mapping[str, Any] | None = None,
        validate_outputs: bool = True,
        process_integrator=ssprk3,
    ) -> None:
        super().__init__()
        self.column_model = column_model or ToyCondensationSCM()
        self.validate_outputs = bool(validate_outputs)
        self.process_integrator = process_integrator
        self.state_requirement = StateRequirement.merge(
            _vapor_state_requirement(),
            _component_state_requirement(self.column_model),
            None if state_requirement is None else StateRequirement.merge(state_requirement),
        )
        self._param_names: list[tuple[str, str]] = []
        for raw_name, value in dict(params or {}).items():
            name = str(raw_name)
            safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
            if not safe or safe[0].isdigit():
                safe = f"p_{safe}"
            buffer_name = f"scm_param_{safe}"
            if hasattr(self, buffer_name):
                raise ValueError(f"duplicate or conflicting SCM parameter name {name!r}")
            self.register_buffer(buffer_name, torch.as_tensor(value))
            self._param_names.append((name, buffer_name))

    def required_state_requirement(self) -> StateRequirement:
        return self.state_requirement

    def check_state_compatible(self, state: State) -> None:
        self.state_requirement.validate(state, owner="SCMPhysics")

    def _params_for_state(
        self,
        state: State,
        ref: torch.Tensor,
        extra_params: Mapping[str, torch.Tensor | float | int] | None = None,
    ) -> dict[str, torch.Tensor]:
        params = {
            name: _prepare_param(name, getattr(self, buffer_name), ref, int(state.mesh.nCells))
            for name, buffer_name in self._param_names
        }
        for raw_name, value in dict(extra_params or {}).items():
            name = str(raw_name)
            params[name] = _prepare_param(name, value, ref, int(state.mesh.nCells))
        return params

    def column_state(
        self,
        state: State,
        extra_params: Mapping[str, torch.Tensor | float | int] | None = None,
    ) -> dict[str, Any]:
        """Return the vectorized column-model input dictionary."""

        self.check_state_compatible(state)
        theta_i = state.theta()
        dp = state.layer_dp()
        dm = interface_mass(dp)
        theta_full, exner_factor = _theta_full_and_exner_factor(state)
        temperature = state.temperature()
        r_v = vapor_mixing_ratio(state)
        if r_v is None:
            raise ValueError("SCMPhysics could not recover named vapor mixing ratio")
        r_l = cloud_liquid_mixing_ratio(state)

        xyz = state.mesh.cell_xyz.to(dtype=state.ps.dtype, device=state.ps.device)
        lat = torch.asin(xyz[:, 2:3].clamp(-1.0, 1.0))
        lon = torch.atan2(xyz[:, 1:2], xyz[:, 0:1])
        area = state.mesh.cell_area.to(dtype=state.ps.dtype, device=state.ps.device)[:, None]
        return {
            "state": state,
            "params": self._params_for_state(state, state.ps, extra_params=extra_params),
            "p_interface": state.interface_pressure(),
            "p_full": state.full_pressure(),
            "surface_pressure": state.ps,
            "dp": dp,
            "dm": dm,
            "theta_i": theta_i,
            "theta_full": theta_full,
            "temperature_full": temperature,
            "exner_interface_over_cp": state.exner_interface() / state.column.c_p,
            "exner_full_over_cp": exner_factor,
            "r_v": r_v,
            "r_l": r_l,
            "cloud_liquid_mixing_ratio": r_l,
            "total_water_mixing_ratio": r_v if r_l is None else r_v + r_l,
            "specific_humidity": specific_humidity_from_mixing_ratio(r_v),
            "lat": lat,
            "lon": lon,
            "cell_area": area,
            "gravity": float((state.attrs or {}).get("gravity", 9.80665)),
        }

    def _dmtheta_from_outputs(
        self,
        state: State,
        columns: Mapping[str, Any],
        outputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        dm = columns["dm"]
        dp = columns["dp"]
        if "dmtheta_i_dt" in outputs:
            return _broadcast_to_ref("dmtheta_i_dt", outputs["dmtheta_i_dt"], state.mtheta_i)
        for name in ("dtheta_i_dt", "dtheta_interface_dt", "theta_i_tendency", "theta_interface_tendency"):
            if name in outputs:
                return dm * _broadcast_to_ref(name, outputs[name], state.mtheta_i)

        dtheta_full = None
        for name in ("dtheta_full_dt", "theta_full_tendency"):
            if name in outputs:
                dtheta_full = _broadcast_to_ref(name, outputs[name], dp)
                break
        if dtheta_full is None:
            for name in ("dT_dt", "temperature_tendency"):
                if name in outputs:
                    dT_dt = _broadcast_to_ref(name, outputs[name], dp)
                    dtheta_full = dT_dt / columns["exner_full_over_cp"].clamp_min(torch.finfo(dT_dt.dtype).tiny)
                    break
        if dtheta_full is None:
            return torch.zeros_like(state.mtheta_i)

        out = torch.zeros_like(state.mtheta_i)
        layer_tendency = dp * dtheta_full
        out[..., :-1] += 0.5 * layer_tendency
        out[..., 1:] += 0.5 * layer_tendency
        return out

    def _tracer_from_outputs(
        self,
        state: State,
        columns: Mapping[str, Any],
        outputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor | None:
        if state.tracer_m is None:
            return None
        if "tracer_m_dt" in outputs:
            return _broadcast_to_ref("tracer_m_dt", outputs["tracer_m_dt"], state.tracer_m)
        tracer = torch.zeros_like(state.tracer_m)
        vapor_dt = None
        for name in ("dr_v_dt", "vapor_mixing_ratio_tendency"):
            if name in outputs:
                vapor_dt = _broadcast_to_ref(name, outputs[name], columns["dp"])
                break
        if vapor_dt is not None:
            index = vapor_tracer_index(state)
            assert index is not None
            tracer[..., int(index)] = columns["dp"] * vapor_dt
        cloud_dt = None
        for name in ("dr_l_dt", "dcloud_liquid_dt", "cloud_liquid_mixing_ratio_tendency"):
            if name in outputs:
                cloud_dt = _broadcast_to_ref(name, outputs[name], columns["dp"])
                break
        if cloud_dt is not None:
            index = cloud_liquid_tracer_index(state)
            if index is None:
                raise ValueError(
                    "SCM output includes cloud-liquid tendency but State lacks "
                    "state.attrs['cloud_liquid_tracer_index']"
                )
            tracer[..., int(index)] += columns["dp"] * cloud_dt
        return tracer

    def tendency_from_outputs(
        self,
        state: State,
        columns: Mapping[str, Any],
        outputs: Mapping[str, torch.Tensor],
    ) -> State:
        """Map column-model outputs to a State-shaped source tendency."""

        dmtheta = self._dmtheta_from_outputs(state, columns, outputs)
        tracer = self._tracer_from_outputs(state, columns, outputs)
        du = torch.zeros_like(state.u_e)
        ps = torch.zeros_like(state.ps)
        if "u_e_dt" in outputs:
            du = _broadcast_to_ref("u_e_dt", outputs["u_e_dt"], state.u_e)
        if "ps_dt" in outputs:
            ps = _broadcast_to_ref("ps_dt", outputs["ps_dt"], state.ps)
        if self.validate_outputs:
            tensors = [du, dmtheta, ps]
            if tracer is not None:
                tensors.append(tracer)
            if not all(torch.isfinite(t).all().item() for t in tensors):
                raise FloatingPointError("SCMPhysics produced non-finite State tendency")
        return State(
            du,
            dmtheta,
            ps,
            tracer,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )

    def tendency(
        self,
        state: State,
        extra_params: Mapping[str, torch.Tensor | float | int] | None = None,
    ) -> State:
        columns = self.column_state(state, extra_params=extra_params)
        outputs = self.column_model(columns)
        if not isinstance(outputs, Mapping):
            raise TypeError("SCM column_model must return a mapping of tensor tendencies")
        return self.tendency_from_outputs(state, columns, outputs)

    def initial_physics_state(
        self,
        state: State,
        extra_params: Mapping[str, torch.Tensor | float | int] | None = None,
    ) -> dict[str, Any]:
        """Initial restart state for split-step SCM process coupling."""

        columns = self.column_state(state, extra_params=extra_params)
        initializer = getattr(self.column_model, "initial_physics_state", None)
        if callable(initializer):
            value = initializer(columns)
            return {} if value is None else dict(value)
        return {}

    def step_outputs(
        self,
        state: State,
        dt: float,
        physics_state=None,
        extra_params: Mapping[str, torch.Tensor | float | int] | None = None,
    ) -> tuple[Mapping[str, Any], Any, dict[str, Any], Mapping[str, Any]]:
        """Run one column-model process step and expose raw tendency outputs."""

        columns = self.column_state(state, extra_params=extra_params)
        step_columns = getattr(self.column_model, "step_columns", None)
        if callable(step_columns):
            raw = step_columns(columns, dt=float(dt), physics_state=physics_state)
            outputs, next_physics_state, diagnostics = _normalize_column_step_result(raw, physics_state)
            return outputs, next_physics_state, diagnostics, columns

        outputs = self.column_model(columns)
        if not isinstance(outputs, Mapping):
            raise TypeError("SCM column_model must return a mapping of tensor tendencies")
        return outputs, ({} if physics_state is None else physics_state), {}, columns

    def step(
        self,
        state: State,
        dt: float,
        physics_state=None,
        extra_params: Mapping[str, torch.Tensor | float | int] | None = None,
    ) -> PhysicsStepResult:
        """Advance SCM physics as a split process step.

        Column models with an explicit ``step_columns`` method own their process
        update and may return restart state.  Tendency-only column models retain
        the historical SCM-only SSPRK3 split-step behavior.
        """

        step_columns = getattr(self.column_model, "step_columns", None)
        if callable(step_columns):
            outputs, next_physics_state, diagnostics, columns = self.step_outputs(
                state,
                dt,
                physics_state=physics_state,
                extra_params=extra_params,
            )
            tendency = self.tendency_from_outputs(state, columns, outputs)
            return PhysicsStepResult(
                state + float(dt) * tendency,
                next_physics_state,
                diagnostics,
            )

        if extra_params is None:
            provider = self
        else:
            outer = self

            class _ExtraParamProvider:
                def tendency(self, inner_state: State) -> State:
                    return outer.tendency(inner_state, extra_params=extra_params)

                def __call__(self, inner_state: State) -> State:
                    return self.tendency(inner_state)

            provider = _ExtraParamProvider()
        next_state = self.process_integrator(provider, state, float(dt))
        return PhysicsStepResult(next_state, {} if physics_state is None else physics_state, {})

    def forward(self, state: State) -> State:
        return self.tendency(state)


def _normalize_column_step_result(
    raw: Any,
    fallback_physics_state: Any,
) -> tuple[Mapping[str, Any], Any, dict[str, Any]]:
    if isinstance(raw, tuple):
        if len(raw) == 2:
            outputs, next_physics_state = raw
            diagnostics = {}
        elif len(raw) == 3:
            outputs, next_physics_state, diagnostics = raw
        else:
            raise TypeError("SCM step_columns tuple results must be (outputs, state) or (outputs, state, diagnostics)")
    elif isinstance(raw, Mapping) and "outputs" in raw:
        outputs = raw["outputs"]
        next_physics_state = raw.get("physics_state", fallback_physics_state)
        diagnostics = raw.get("diagnostics", {})
    else:
        outputs = raw
        next_physics_state = fallback_physics_state
        diagnostics = {}
    if not isinstance(outputs, Mapping):
        raise TypeError("SCM step_columns must return a mapping of tensor tendencies or an {'outputs': ...} mapping")
    return outputs, ({} if next_physics_state is None else next_physics_state), dict(diagnostics or {})
