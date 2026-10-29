"""SCM process adapter for coupled atmosphere/ocean states."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..core.registry import register_component
from ..moisture import cloud_liquid_tracer_index, saturation_mixing_ratio, vapor_tracer_index
from ..process import PhysicsStepResult
from .ark_stage_sources import CoupledARKStageSourceResult
from ..surface import SurfaceBoundary, SurfaceExchange
from .state import CoupledState
from .surface_flux import AirSeaSurfaceFluxCoupler


def _first_tensor(outputs: Mapping[str, Any], names: tuple[str, ...]) -> torch.Tensor | None:
    for name in names:
        value = outputs.get(name)
        if value is not None:
            return value if torch.is_tensor(value) else torch.as_tensor(value)
    return None


def _scalar_mean(value: torch.Tensor) -> float:
    return float(value.detach().to(dtype=torch.float32).mean().cpu().item())


def _scalar_abs_max(value: torch.Tensor) -> float:
    return float(value.detach().abs().to(dtype=torch.float32).max().cpu().item())


def _scalar_value(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


_PROCESS_STATE_KEY = "_coupled_scm_process"
_FRESHWATER_ACCUMULATOR_KEY = "ocean_freshwater_thickness_accumulator"
_SURFACE_WATER_STORAGE_KEY = "unrouted_surface_water_storage"


@dataclass(frozen=True)
class CoupledSCMStageEvaluation:
    """Raw stateless SCM stage data shared with a coupled surface source.

    ``state_tendency`` contains the SCM atmospheric tendency and, unless
    explicitly disabled, its legacy routed ocean tendency.  A prognostic
    surface-system source disables that ocean route and applies its own
    fraction-aware ocean packet exactly once.
    """

    state_tendency: CoupledState
    outputs: Mapping[str, Any]
    surface_params: Mapping[str, torch.Tensor]
    diagnostics: dict[str, Any]


def _atmosphere_rh_diagnostics(atmosphere: Any, *, prefix: str) -> dict[str, float | int | bool]:
    q_all = atmosphere.tracers()
    index = vapor_tracer_index(atmosphere)
    if q_all is None or index is None:
        return {f"{prefix}_has_vapor": False}

    n_levels = int(atmosphere.column.n_levels)
    r_v = q_all[..., int(index)].reshape(-1, n_levels)
    temperature = atmosphere.temperature().reshape_as(r_v)
    pressure = atmosphere.full_pressure().reshape_as(r_v)
    r_sat = saturation_mixing_ratio(temperature, pressure)
    rh = r_v / r_sat.clamp_min(torch.finfo(r_v.dtype).tiny)
    supersaturated = rh > 1.0
    finite = torch.isfinite(rh)
    out: dict[str, float | int | bool] = {
        f"{prefix}_has_vapor": True,
        f"{prefix}_rough_relative_humidity_max": _scalar_value(torch.where(finite, rh, torch.full_like(rh, -float("inf"))).max()),
        f"{prefix}_supersaturated_cell_level_fraction": _scalar_value(
            supersaturated.to(dtype=torch.float32).mean()
        ),
    }
    if bool(finite.any().detach().cpu().item()):
        flat_index = int(torch.where(finite, rh, torch.full_like(rh, -float("inf"))).reshape(-1).argmax().detach().cpu().item())
        cell = flat_index // n_levels
        level = flat_index % n_levels
        out.update(
            {
                f"{prefix}_rough_relative_humidity_max_cell": int(cell),
                f"{prefix}_rough_relative_humidity_max_level": int(level),
                f"{prefix}_rough_relative_humidity_max_pressure_pa": _scalar_value(pressure[cell, level]),
                f"{prefix}_rough_relative_humidity_max_temperature_k": _scalar_value(temperature[cell, level]),
                f"{prefix}_rough_relative_humidity_max_vapor_mixing_ratio": _scalar_value(r_v[cell, level]),
                f"{prefix}_rough_relative_humidity_max_saturation_mixing_ratio": _scalar_value(r_sat[cell, level]),
            }
        )
    return out


class CoupledSCMProcess(nn.Module):
    """VFS-side adapter for running atmospheric SCM physics on ``CoupledState``.

    The wrapped ``SCMPhysics`` object still owns atmospheric column assembly and
    State-shaped tendency mapping. This process supplies ocean-aware surface
    context as per-column parameters, then applies any recognized ocean-side
    flux/tendency outputs back to ``OceanState``. It is intentionally an
    adapter contract, not a physical parameterization.
    """

    recipe_name = "coupled_scm_process"

    def __init__(
        self,
        scm_physics: Any,
        *,
        surface_context: AirSeaSurfaceFluxCoupler | None = None,
        surface_boundary: SurfaceBoundary | None = None,
        surface_exchange: SurfaceExchange | None = None,
        ocean_density: float = 1025.0,
        ocean_heat_capacity: float = 3990.0,
        reference_salinity: float = 35.0,
        ocean_temperature_offset: float = 273.15,
        freshwater_volume_mode: str = "virtual_salt",
        solar_forcing: Any | None = None,
    ) -> None:
        super().__init__()
        if surface_boundary is not None and surface_exchange is not None:
            raise ValueError("pass either surface_boundary or surface_exchange, not both")
        self.scm_physics = scm_physics
        self.surface_context = surface_context or AirSeaSurfaceFluxCoupler(
            ocean_density=ocean_density,
            ocean_heat_capacity=ocean_heat_capacity,
            reference_salinity=reference_salinity,
        )
        self.surface_exchange = surface_exchange or SurfaceExchange(surface_boundary)
        self.ocean_density = float(ocean_density)
        self.ocean_heat_capacity = float(ocean_heat_capacity)
        self.reference_salinity = float(reference_salinity)
        self.ocean_temperature_offset = float(ocean_temperature_offset)
        self.freshwater_volume_mode = str(freshwater_volume_mode)
        if self.freshwater_volume_mode not in ("virtual_salt", "thickness"):
            raise ValueError("freshwater_volume_mode must be 'virtual_salt' or 'thickness'")
        self.solar_forcing = solar_forcing
        self._surface_water_storage: torch.Tensor | None = None
        self.register_buffer("time_seconds", torch.as_tensor(0.0))
        for name in ("ocean_density", "ocean_heat_capacity"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def set_time_seconds(self, time_seconds: float | torch.Tensor) -> "CoupledSCMProcess":
        """Set the beginning-of-step clock used by time-dependent surface forcing."""

        value = torch.as_tensor(time_seconds, dtype=self.time_seconds.dtype, device=self.time_seconds.device)
        if value.ndim != 0:
            raise ValueError("time_seconds must be scalar")
        self.time_seconds.copy_(value)
        return self

    def required_state_requirement(self):
        method = getattr(self.scm_physics, "required_state_requirement", None)
        return method() if callable(method) else None

    @staticmethod
    def _cell_field(value: Any, like: torch.Tensor, *, name: str) -> torch.Tensor:
        tensor = value.to(dtype=like.dtype, device=like.device) if torch.is_tensor(value) else torch.as_tensor(
            value,
            dtype=like.dtype,
            device=like.device,
        )
        if tensor.ndim == 0:
            return torch.ones_like(like) * tensor
        if tensor.ndim == 1:
            tensor = tensor[:, None]
        if tensor.shape == like.shape:
            return tensor
        if tensor.ndim == 2 and int(tensor.shape[0]) == int(like.shape[0]) and int(tensor.shape[1]) == 1:
            return tensor
        raise ValueError(f"{name} must be scalar, [nCells], or [nCells, 1]; got {tuple(tensor.shape)}")

    @staticmethod
    def _edge_field(value: Any, like: torch.Tensor, *, name: str) -> torch.Tensor:
        tensor = value.to(dtype=like.dtype, device=like.device) if torch.is_tensor(value) else torch.as_tensor(
            value,
            dtype=like.dtype,
            device=like.device,
        )
        if tensor.ndim == 0:
            return torch.ones_like(like) * tensor
        if tensor.ndim == 1:
            tensor = tensor[:, None]
        if tensor.shape == like.shape:
            return tensor
        if tensor.ndim == 2 and int(tensor.shape[0]) == int(like.shape[0]) and int(tensor.shape[1]) == 1:
            return tensor
        raise ValueError(f"{name} must be scalar, [nEdges], or [nEdges, 1]; got {tuple(tensor.shape)}")

    def surface_params(
        self,
        state: CoupledState,
        *,
        time_seconds: float | None = None,
        surface_state: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return per-column ocean/wind context for an atmospheric SCM."""

        params = self.surface_exchange.surface_params(
            state,
            surface_context=self.surface_context,
            ocean_temperature_offset=float(self.ocean_temperature_offset),
            surface_state=surface_state,
        )
        if self.solar_forcing is not None:
            mesh = state.atmosphere.mesh
            ref = params["surface_temperature"]
            xyz = mesh.cell_xyz.to(dtype=ref.dtype, device=ref.device)
            lat = torch.asin(xyz[:, 2:3].clamp(-1.0, 1.0))
            lon = torch.atan2(xyz[:, 1:2], xyz[:, 0:1])
            forcing_time = (
                self.time_seconds
                if time_seconds is None
                else torch.as_tensor(time_seconds, dtype=self.time_seconds.dtype, device=self.time_seconds.device)
            )
            solar = self.solar_forcing(
                lat,
                lon,
                forcing_time.to(dtype=ref.dtype, device=ref.device),
                surface_albedo=params.get("surface_albedo"),
            )
            params.update(solar)
        return params

    def _require_ark_stage_source(self) -> None:
        """Reject split/stateful SCM paths from the derivative-stage contract."""

        column_model = self.scm_physics.column_model
        if callable(getattr(column_model, "step_columns", None)) and getattr(
            column_model,
            "ark_stage_contract",
            None,
        ) != "stateless_tendency":
            raise ValueError(
                "CoupledSCMProcess ARK stages require a tendency-form SCM column model; "
                "step_columns restart memory needs explicit memory derivatives (or an explicit "
                "ark_stage_contract='stateless_tendency' declaration)"
            )
        if self.freshwater_volume_mode != "virtual_salt":
            raise ValueError(
                "CoupledSCMProcess ARK stages currently require freshwater_volume_mode='virtual_salt'; "
                "thickness accumulators need an explicit ARK memory derivative"
            )

    def initial_ark_physics_state(self, state: CoupledState) -> dict[str, Any]:
        """Return source memory for a stateless tendency-form coupled SCM."""

        self._require_ark_stage_source()
        return {}

    def evaluate_ark_stage(
        self,
        state: CoupledState,
        physics_state: Mapping[str, Any] | None,
        *,
        time_seconds: float,
        surface_state: Any | None = None,
        include_ocean_outputs: bool = True,
    ) -> CoupledSCMStageEvaluation:
        """Evaluate one stateless SCM stage without advancing restart memory.

        This path is intentionally limited to column models that already own
        no restart update. A surface-system source may request the same
        atmospheric column tendency while replacing the legacy ocean-output
        route with its fraction-aware surface packet.
        """

        self._require_ark_stage_source()
        if physics_state:
            raise ValueError("stateless CoupledSCMProcess ARK source expects empty physics_state")
        surface_params = self.surface_params(
            state,
            time_seconds=float(time_seconds),
            surface_state=surface_state,
        )
        columns = self.scm_physics.column_state(state.atmosphere, extra_params=surface_params)
        outputs = self.scm_physics.column_model(columns)
        if not isinstance(outputs, Mapping):
            raise TypeError("SCM column_model must return a mapping of tensor tendencies")
        outputs = self._with_precipitation_freshwater_routing(outputs)
        atmosphere_rhs = self.scm_physics.tendency_from_outputs(state.atmosphere, columns, outputs)
        ocean_rhs = self._ocean_tendency_from_outputs(state.ocean, outputs) if include_ocean_outputs else state.ocean.zeros_like()
        diagnostics: dict[str, Any] = {
            "source_mode": "stateless_tendency_form",
            "time_seconds": float(time_seconds),
            "precipitation_routed_as_freshwater": bool("precipitation_routed_as_freshwater" in outputs),
        }
        for name in (
            "condensation_rate",
            "condensation_mixing_ratio_tendency",
            "evaporation_mixing_ratio_tendency",
            "autoconversion_mixing_ratio_tendency",
            "surface_precipitation_rate",
            "freshwater_flux_down",
        ):
            value = outputs.get(name)
            if torch.is_tensor(value):
                diagnostics[f"scm_{name}_max_abs"] = _scalar_abs_max(value)
        return CoupledSCMStageEvaluation(
            CoupledState(atmosphere_rhs, ocean_rhs, attrs=state._attrs_copy()),
            outputs,
            surface_params,
            diagnostics,
        )

    def ark_stage_tendency(
        self,
        state: CoupledState,
        physics_state: Mapping[str, Any] | None,
        *,
        time_seconds: float,
    ) -> CoupledARKStageSourceResult:
        """Return the normal stateless SCM coupled tendency at an ARK stage."""

        evaluation = self.evaluate_ark_stage(
            state,
            physics_state,
            time_seconds=time_seconds,
        )
        return CoupledARKStageSourceResult(evaluation.state_tendency, {}, evaluation.diagnostics)

    def initial_physics_state(self, state: CoupledState) -> dict[str, Any]:
        initializer = getattr(self.scm_physics, "initial_physics_state", None)
        out = (
            dict(initializer(state.atmosphere, extra_params=self.surface_params(state)) or {})
            if callable(initializer)
            else {}
        )
        out[_PROCESS_STATE_KEY] = self._initial_process_state(state)
        return out

    @staticmethod
    def _initial_process_state(state: CoupledState) -> dict[str, torch.Tensor]:
        return {
            _FRESHWATER_ACCUMULATOR_KEY: torch.zeros_like(state.ocean.h[..., :1]),
            _SURFACE_WATER_STORAGE_KEY: torch.zeros_like(state.ocean.h[..., :1]),
        }

    def _split_physics_state(
        self,
        physics_state: Mapping[str, Any] | None,
        state: CoupledState,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        inner = dict(physics_state or {})
        raw_process_state = inner.pop(_PROCESS_STATE_KEY, {})
        process_state = self._initial_process_state(state)
        if isinstance(raw_process_state, Mapping):
            value = raw_process_state.get(_FRESHWATER_ACCUMULATOR_KEY)
            if value is not None:
                process_state[_FRESHWATER_ACCUMULATOR_KEY] = self._cell_field(
                    value,
                    state.ocean.h[..., :1],
                    name=_FRESHWATER_ACCUMULATOR_KEY,
                )
            value = raw_process_state.get(_SURFACE_WATER_STORAGE_KEY)
            if value is not None:
                process_state[_SURFACE_WATER_STORAGE_KEY] = self._cell_field(
                    value,
                    state.ocean.h[..., :1],
                    name=_SURFACE_WATER_STORAGE_KEY,
                )
        self._surface_water_storage = process_state[_SURFACE_WATER_STORAGE_KEY]
        return inner, process_state

    def _ocean_tendency_from_outputs(self, ocean_state: Any, outputs: Mapping[str, Any]) -> Any:
        u_t = torch.zeros_like(ocean_state.u_e)
        h_t = torch.zeros_like(ocean_state.h)
        temp_h_t = torch.zeros_like(ocean_state.temp_h)
        salt_h_t = torch.zeros_like(ocean_state.salt_h)

        temp_h_full = _first_tensor(outputs, ("ocean_temp_h_dt", "temp_h_dt"))
        if temp_h_full is not None:
            temp_h_t = temp_h_t + self._cell_field(temp_h_full, temp_h_t, name="ocean_temp_h_dt")
        temp_h_surface = _first_tensor(outputs, ("ocean_surface_temp_h_dt", "surface_temp_h_dt"))
        if temp_h_surface is not None:
            temp_h_t[..., :1] = temp_h_t[..., :1] + self._cell_field(
                temp_h_surface,
                temp_h_t[..., :1],
                name="ocean_surface_temp_h_dt",
            )
        heat_down = _first_tensor(
            outputs,
            ("ocean_surface_heat_flux_down", "surface_heat_flux_down", "ocean_heat_flux_down"),
        )
        if heat_down is not None:
            temp_h_t[..., :1] = temp_h_t[..., :1] + self._cell_field(
                heat_down,
                temp_h_t[..., :1],
                name="ocean_surface_heat_flux_down",
            ) / (float(self.ocean_density) * float(self.ocean_heat_capacity))

        salt_h_full = _first_tensor(outputs, ("ocean_salt_h_dt", "salt_h_dt"))
        if salt_h_full is not None:
            salt_h_t = salt_h_t + self._cell_field(salt_h_full, salt_h_t, name="ocean_salt_h_dt")
        salt_h_surface = _first_tensor(outputs, ("ocean_surface_salt_h_dt", "surface_salt_h_dt"))
        if salt_h_surface is not None:
            salt_h_t[..., :1] = salt_h_t[..., :1] + self._cell_field(
                salt_h_surface,
                salt_h_t[..., :1],
                name="ocean_surface_salt_h_dt",
            )
        freshwater_down = _first_tensor(
            outputs,
            ("freshwater_flux_down", "surface_freshwater_flux_down", "ocean_freshwater_flux_down"),
        )
        if freshwater_down is not None and self.freshwater_volume_mode != "thickness":
            freshwater_cell = self._cell_field(
                freshwater_down,
                salt_h_t[..., :1],
                name="freshwater_flux_down",
            )
            # Legacy virtual-salt mode keeps ocean volume fixed and applies the
            # freshwater effect as a salt-content tendency. Physical thickness
            # increments are accumulated after non-freshwater ocean tendencies.
            salt_h_t[..., :1] = salt_h_t[..., :1] - float(self.reference_salinity) * freshwater_cell

        h_full = _first_tensor(outputs, ("ocean_h_dt", "h_dt"))
        if h_full is not None:
            h_t = h_t + self._cell_field(h_full, h_t, name="ocean_h_dt")
        h_surface = _first_tensor(outputs, ("ocean_surface_h_dt", "surface_h_dt"))
        if h_surface is not None:
            h_t[..., :1] = h_t[..., :1] + self._cell_field(h_surface, h_t[..., :1], name="ocean_surface_h_dt")

        u_full = _first_tensor(outputs, ("ocean_u_e_dt",))
        if u_full is not None:
            u_t = u_t + self._edge_field(u_full, u_t, name="ocean_u_e_dt")
        stress = _first_tensor(
            outputs,
            ("stress_on_ocean_edge", "surface_stress_on_ocean_edge", "surface_stress_edge"),
        )
        if stress is not None:
            tau = self._edge_field(stress, u_t[..., :1], name="surface_stress_on_ocean_edge")
            h_edge = ocean_state.mesh.cell_to_edge(ocean_state.h)[..., :1].to(dtype=tau.dtype, device=tau.device)
            u_t[..., :1] = u_t[..., :1] + tau / (
                float(self.ocean_density) * h_edge.clamp_min(torch.finfo(tau.dtype).tiny)
            )

        tracer_h_t = None if ocean_state.tracer_h is None else torch.zeros_like(ocean_state.tracer_h)
        return self.surface_exchange.route_ocean_tendency(
            ocean_state.with_tensors(u_t, h_t, temp_h_t, salt_h_t, tracer_h_t)
        )

    def _with_precipitation_freshwater_routing(self, outputs: Mapping[str, Any]) -> dict[str, Any]:
        """Supply a freshwater flux when a column model reports precipitation.

        Column physics conventionally reports surface precipitation as a mass
        flux (kg m-2 s-1), while the coupled ocean virtual-salt convention
        uses a positive-downward freshwater *volume* flux (m s-1). Native SCMs
        which already provide ``freshwater_flux_down`` own the complete surface
        budget and are left unchanged. Kessler-style tendency models only
        report precipitation, so without this bridge they remove water from
        the atmosphere without applying an ocean-side freshwater effect.
        """

        if _first_tensor(
            outputs,
            ("freshwater_flux_down", "surface_freshwater_flux_down", "ocean_freshwater_flux_down"),
        ) is not None:
            return dict(outputs)
        precipitation = _first_tensor(outputs, ("surface_precipitation_rate", "precipitation_rate", "precip_total"))
        if precipitation is None:
            return dict(outputs)
        routed = dict(outputs)
        routed["freshwater_flux_down"] = precipitation / float(self.surface_context.water_density)
        routed["precipitation_routed_as_freshwater"] = torch.ones_like(precipitation, dtype=torch.bool)
        return routed

    def _freshwater_thickness_delta(
        self,
        ocean_state: Any,
        outputs: Mapping[str, Any],
        surface_params: Mapping[str, torch.Tensor],
        *,
        dt: float,
    ) -> torch.Tensor:
        if self.freshwater_volume_mode != "thickness":
            return torch.zeros_like(ocean_state.h[..., :1])
        freshwater_down = _first_tensor(
            outputs,
            ("freshwater_flux_down", "surface_freshwater_flux_down", "ocean_freshwater_flux_down"),
        )
        if freshwater_down is None:
            return torch.zeros_like(ocean_state.h[..., :1])
        flux = self._cell_field(freshwater_down, ocean_state.h[..., :1], name="freshwater_flux_down")
        ocean_fraction = surface_params.get("ocean_fraction", surface_params.get("ocean_cell_mask"))
        if ocean_fraction is None:
            ocean_fraction_field = torch.ones_like(flux)
        else:
            ocean_fraction_field = self._cell_field(ocean_fraction, flux, name="ocean_fraction").clamp(0.0, 1.0)
        return flux * ocean_fraction_field * float(dt)

    def _surface_water_storage_delta(
        self,
        outputs: Mapping[str, Any],
        surface_params: Mapping[str, torch.Tensor],
        like: torch.Tensor,
        *,
        dt: float,
    ) -> torch.Tensor:
        precip = _first_tensor(outputs, ("surface_precipitation_rate", "precipitation_rate", "precip_total"))
        evap = _first_tensor(outputs, ("evaporation_mass_flux_up", "surface_evaporation_mass_flux_up"))
        if precip is None and evap is None:
            return torch.zeros_like(like)
        precip_cell = torch.zeros_like(like) if precip is None else self._cell_field(
            precip,
            like,
            name="surface_precipitation_rate",
        )
        evap_cell = torch.zeros_like(like) if evap is None else self._cell_field(
            evap,
            like,
            name="evaporation_mass_flux_up",
        )
        land_fraction = surface_params.get("land_fraction", surface_params.get("land_cell_mask"))
        if land_fraction is None:
            land = torch.zeros_like(like)
        else:
            land = self._cell_field(land_fraction, like, name="land_fraction").clamp(0.0, 1.0)
        return (precip_cell - evap_cell) * land * float(dt)

    @staticmethod
    def _integrate_cell_mass(mesh: Any, value: torch.Tensor) -> torch.Tensor:
        area = mesh.cell_area.to(dtype=value.dtype, device=value.device)[:, None]
        return (value * area).sum()

    def surface_water_storage_mass(self, mesh: Any, like: torch.Tensor) -> torch.Tensor:
        storage = self._surface_water_storage
        if storage is None:
            return torch.zeros((), dtype=like.dtype, device=like.device)
        storage = self._cell_field(storage, like.reshape(-1, 1), name=_SURFACE_WATER_STORAGE_KEY)
        return self._integrate_cell_mass(mesh, storage)

    @staticmethod
    def _freshwater_apply_threshold(h_top: torch.Tensor) -> torch.Tensor:
        if h_top.dtype == torch.float64:
            return torch.zeros_like(h_top)
        eps = torch.finfo(h_top.dtype).eps
        return h_top.detach().abs().clamp_min(1.0) * (8.0 * eps)

    def _apply_freshwater_accumulator(
        self,
        ocean_state: Any,
        process_state: Mapping[str, torch.Tensor],
        freshwater_delta: torch.Tensor,
    ) -> tuple[Any, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        pending0 = self._cell_field(
            process_state.get(_FRESHWATER_ACCUMULATOR_KEY, torch.zeros_like(ocean_state.h[..., :1])),
            ocean_state.h[..., :1],
            name=_FRESHWATER_ACCUMULATOR_KEY,
        )
        pending = pending0 + freshwater_delta.to(dtype=pending0.dtype, device=pending0.device)
        threshold = self._freshwater_apply_threshold(ocean_state.h[..., :1])
        applied = torch.where(pending.abs() >= threshold, pending, torch.zeros_like(pending))
        pending_next = pending - applied
        h = ocean_state.h.clone()
        h[..., :1] = h[..., :1] + applied
        next_state = ocean_state.with_tensors(
            ocean_state.u_e,
            h,
            ocean_state.temp_h,
            ocean_state.salt_h,
            ocean_state.tracer_h,
        )
        next_process_state = {_FRESHWATER_ACCUMULATOR_KEY: pending_next}
        diagnostics = {
            "freshwater_thickness_accumulator_applied": applied,
            "freshwater_thickness_accumulator_delta": pending_next - pending0,
            "freshwater_thickness_accumulator_pending": pending_next,
        }
        return next_state, next_process_state, diagnostics

    @staticmethod
    def _atmosphere_water_mass_delta(before: Any, after: Any) -> torch.Tensor:
        """Water-mass delta without subtracting large column totals."""

        if before.tracer_m is None or after.tracer_m is None:
            return torch.zeros((), dtype=before.ps.dtype, device=before.ps.device)
        n_cells = int(before.mesh.nCells)
        n_levels = int(before.column.n_levels)

        def water_tracer_mass(state: Any) -> torch.Tensor:
            tracer_m = state.tracer_m
            if tracer_m.ndim >= 2 and int(tracer_m.shape[-2]) == n_cells and int(tracer_m.shape[-1]) == n_levels:
                tracer_m = tracer_m.unsqueeze(-1)
            water = tracer_m.new_zeros(tracer_m.shape[:-1])
            for index in (vapor_tracer_index(state), cloud_liquid_tracer_index(state)):
                if index is not None and int(index) < int(tracer_m.shape[-1]):
                    water = water + tracer_m[..., int(index)]
            return water

        delta = water_tracer_mass(after) - water_tracer_mass(before)
        return before.mesh.integrate_cell(delta.sum(dim=-1, keepdim=True)) / before._gravity()

    @staticmethod
    def _ocean_water_mass_delta(before: Any, after: Any, *, density: float) -> torch.Tensor:
        """Ocean-water delta from local layer-thickness changes.

        Computing ``after.total_volume() - before.total_volume()`` loses the
        one-step freshwater signal in float32/MPS because the total ocean volume
        is many orders of magnitude larger than a single physics-step flux.
        """

        h_delta = after.h - before.h
        volume_delta = before.mesh.integrate_cell(h_delta.sum(dim=-1, keepdim=True))
        return volume_delta * float(density)

    @staticmethod
    def _water_scale(*values: torch.Tensor) -> torch.Tensor:
        ref = values[0]
        scale = ref.new_tensor(1.0)
        for value in values:
            scale = scale + value.detach().abs()
        return scale

    def _water_ledger_diagnostics(
        self,
        before: CoupledState,
        after: CoupledState,
        outputs: Mapping[str, Any],
        surface_params: Mapping[str, torch.Tensor],
        *,
        dt: float,
        freshwater_accumulator_delta: torch.Tensor | None = None,
        freshwater_accumulator_pending: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Detached one-step water ledger for SCM atmosphere/ocean exchange.

        This is an audit only: it checks whether the applied VFS state changes
        match SCM-reported precipitation/evaporation/freshwater transfers.
        """

        ref = before.atmosphere.ps
        area = before.atmosphere.mesh.cell_area.to(dtype=ref.dtype, device=ref.device)[:, None]
        total_area = area.sum().clamp_min(torch.finfo(ref.dtype).tiny)
        ocean_fraction = surface_params.get("ocean_fraction", surface_params.get("ocean_cell_mask"))
        if ocean_fraction is None:
            ocean_fraction = torch.ones_like(area)
        else:
            ocean_fraction = self._cell_field(ocean_fraction, area, name="ocean_fraction").to(
                dtype=ref.dtype,
                device=ref.device,
            )
        land_fraction = surface_params.get("land_fraction", surface_params.get("land_cell_mask"))
        if land_fraction is None:
            land_fraction = 1.0 - ocean_fraction
        else:
            land_fraction = self._cell_field(land_fraction, area, name="land_fraction").to(
                dtype=ref.dtype,
                device=ref.device,
            )
        ocean_area = (area * ocean_fraction).sum().clamp_min(torch.finfo(ref.dtype).tiny)
        dt_t = torch.as_tensor(float(dt), dtype=ref.dtype, device=ref.device)
        water_density = float(getattr(self.surface_context, "water_density", 1000.0))

        def optional_cell(names: tuple[str, ...]) -> torch.Tensor | None:
            value = _first_tensor(outputs, names)
            if value is None:
                return None
            return self._cell_field(value, area, name=names[0]).to(dtype=ref.dtype, device=ref.device)

        def integrate_rate(rate: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            active = torch.ones_like(area) if mask is None else mask
            return (rate * active * area).sum() * dt_t

        precip = optional_cell(("surface_precipitation_rate", "precipitation_rate", "precip_total"))
        evap = optional_cell(("evaporation_mass_flux_up", "surface_evaporation_mass_flux_up"))
        freshwater_down = optional_cell(
            ("freshwater_flux_down", "surface_freshwater_flux_down", "ocean_freshwater_flux_down")
        )

        atm_delta = self._atmosphere_water_mass_delta(before.atmosphere, after.atmosphere)
        ocean_delta = self._ocean_water_mass_delta(before.ocean, after.ocean, density=water_density)
        ocean_seawater_delta = self._ocean_water_mass_delta(
            before.ocean,
            after.ocean,
            density=float(self.ocean_density),
        )

        zero = ref.new_zeros(())
        precip_all = zero if precip is None else integrate_rate(precip)
        evap_all = zero if evap is None else integrate_rate(evap)
        precip_ocean = zero if precip is None else integrate_rate(precip, ocean_fraction)
        evap_ocean = zero if evap is None else integrate_rate(evap, ocean_fraction)
        precip_land = zero if precip is None else integrate_rate(precip, land_fraction)
        evap_land = zero if evap is None else integrate_rate(evap, land_fraction)
        freshwater_to_ocean = (
            zero
            if freshwater_down is None
            else integrate_rate(freshwater_down * water_density, ocean_fraction)
        )
        net_surface_to_atmosphere_all = evap_all - precip_all
        net_surface_to_atmosphere_ocean = evap_ocean - precip_ocean
        net_surface_to_land = precip_land - evap_land
        accumulator_delta = (
            zero
            if freshwater_accumulator_delta is None
            else (
                self._cell_field(
                    freshwater_accumulator_delta,
                    area,
                    name="freshwater_accumulator_delta",
                )
                * area
            ).sum()
            * water_density
        )
        accumulator_pending = (
            zero
            if freshwater_accumulator_pending is None
            else (
                self._cell_field(
                    freshwater_accumulator_pending,
                    area,
                    name="freshwater_accumulator_pending",
                )
                * area
            ).sum()
            * water_density
        )
        atmosphere_residual = atm_delta - net_surface_to_atmosphere_all
        ocean_residual = ocean_delta + accumulator_delta - freshwater_to_ocean
        flux_pair_residual = freshwater_to_ocean + net_surface_to_atmosphere_ocean
        total_delta = atm_delta + ocean_delta
        total_water_with_pending_delta = total_delta + accumulator_delta
        applied_total_residual = total_delta + net_surface_to_land
        total_residual = total_water_with_pending_delta + net_surface_to_land
        scale = self._water_scale(
            atm_delta,
            ocean_delta,
            accumulator_delta,
            net_surface_to_land,
            precip_all,
            evap_all,
            freshwater_to_ocean,
        )

        diagnostics = {
            "process_atmosphere_water_delta_kg": atm_delta,
            "process_ocean_water_delta_kg": ocean_delta,
            "process_ocean_seawater_mass_delta_kg": ocean_seawater_delta,
            "process_total_water_delta_kg": total_delta,
            "process_total_water_with_pending_delta_kg": total_water_with_pending_delta,
            "process_land_water_delta_kg": net_surface_to_land,
            "process_pending_ocean_freshwater_delta_kg": accumulator_delta,
            "process_pending_ocean_freshwater_mass_kg": accumulator_pending,
            "process_precipitation_transfer_kg": precip_all,
            "process_evaporation_transfer_kg": evap_all,
            "process_ocean_precipitation_transfer_kg": precip_ocean,
            "process_ocean_evaporation_transfer_kg": evap_ocean,
            "process_net_surface_water_to_atmosphere_kg": net_surface_to_atmosphere_all,
            "process_ocean_net_surface_water_to_atmosphere_kg": net_surface_to_atmosphere_ocean,
            "process_freshwater_transfer_to_ocean_kg": freshwater_to_ocean,
            "process_unrouted_land_net_surface_water_kg": net_surface_to_land,
            "process_atmosphere_water_residual_kg": atmosphere_residual,
            "process_ocean_water_residual_kg": ocean_residual,
            "process_surface_flux_pair_residual_kg": flux_pair_residual,
            "process_applied_total_water_residual_kg": applied_total_residual,
            "process_total_water_residual_kg": total_residual,
            "process_applied_total_water_relative_residual": applied_total_residual.abs()
            / scale.clamp_min(torch.finfo(ref.dtype).tiny),
            "process_total_water_relative_residual": total_residual.abs()
            / scale.clamp_min(torch.finfo(ref.dtype).tiny),
            "process_water_density_used_kg_m3": ref.new_tensor(water_density),
            "process_ocean_density_used_kg_m3": ref.new_tensor(float(self.ocean_density)),
        }
        per_area_keys = tuple(key for key in diagnostics if key.endswith("_kg"))
        for key in per_area_keys:
            diagnostics[f"{key}_m2"] = diagnostics[key] / total_area
        for key in (
            "process_ocean_water_delta_kg",
            "process_ocean_precipitation_transfer_kg",
            "process_ocean_evaporation_transfer_kg",
            "process_ocean_net_surface_water_to_atmosphere_kg",
            "process_freshwater_transfer_to_ocean_kg",
            "process_ocean_water_residual_kg",
            "process_surface_flux_pair_residual_kg",
        ):
            diagnostics[f"{key}_per_ocean_m2"] = diagnostics[key] / ocean_area
        return {key: _scalar_value(value) for key, value in diagnostics.items()}

    def surface_audit_outputs(
        self,
        state: CoupledState,
        *,
        physics_state=None,
        dt: float = 0.0,
    ) -> tuple[dict[str, torch.Tensor], Mapping[str, Any], Mapping[str, Any]]:
        """Return SCM surface params plus raw current-column outputs for audits."""

        surface_params = self.surface_params(state)
        outputs, _next_physics_state, _diagnostics, columns = self.scm_physics.step_outputs(
            state.atmosphere,
            float(dt),
            physics_state=physics_state,
            extra_params=surface_params,
        )
        return surface_params, outputs, columns

    def step(self, state: CoupledState, dt: float, physics_state=None) -> PhysicsStepResult:
        step_outputs = getattr(self.scm_physics, "step_outputs", None)
        if not callable(step_outputs):
            raise TypeError("CoupledSCMProcess requires an SCMPhysics-like object with step_outputs(...)")
        inner_physics_state, process_state = self._split_physics_state(physics_state, state)
        surface_params = self.surface_params(state)
        outputs, next_physics_state, diagnostics, columns = step_outputs(
            state.atmosphere,
            float(dt),
            physics_state=inner_physics_state,
            extra_params=surface_params,
        )
        outputs = self._with_precipitation_freshwater_routing(outputs)
        atmosphere_rhs = self.scm_physics.tendency_from_outputs(state.atmosphere, columns, outputs)
        ocean_rhs = self._ocean_tendency_from_outputs(state.ocean, outputs)
        ocean_state = state.ocean + float(dt) * ocean_rhs
        freshwater_delta = self._freshwater_thickness_delta(state.ocean, outputs, surface_params, dt=float(dt))
        ocean_state, next_process_state, process_accumulator_diag = self._apply_freshwater_accumulator(
            ocean_state,
            process_state,
            freshwater_delta,
        )
        surface_water_delta = self._surface_water_storage_delta(
            outputs,
            surface_params,
            state.ocean.h[..., :1],
            dt=float(dt),
        )
        surface_water_storage = process_state[_SURFACE_WATER_STORAGE_KEY] + surface_water_delta
        next_process_state[_SURFACE_WATER_STORAGE_KEY] = surface_water_storage
        self._surface_water_storage = surface_water_storage
        next_state = CoupledState(state.atmosphere + float(dt) * atmosphere_rhs, ocean_state, attrs=state._attrs_copy())
        next_physics_state = dict(next_physics_state or {})
        next_physics_state[_PROCESS_STATE_KEY] = next_process_state
        diag = dict(diagnostics or {})
        input_rh = _atmosphere_rh_diagnostics(state.atmosphere, prefix="process_input")
        output_rh = _atmosphere_rh_diagnostics(next_state.atmosphere, prefix="process_output")
        diag.update(
            {
                "surface_temperature_mean": _scalar_mean(surface_params["surface_temperature"]),
                "relative_wind_speed_mean": _scalar_mean(surface_params["relative_wind_speed_cell"]),
                "freshwater_volume_mode": self.freshwater_volume_mode,
                "freshwater_thickness_accumulator_applied_mean": _scalar_mean(
                    process_accumulator_diag["freshwater_thickness_accumulator_applied"]
                ),
                "freshwater_thickness_accumulator_pending_mean": _scalar_mean(
                    process_accumulator_diag["freshwater_thickness_accumulator_pending"]
                ),
                "freshwater_thickness_accumulator_pending_max_abs": _scalar_abs_max(
                    process_accumulator_diag["freshwater_thickness_accumulator_pending"]
                ),
                "surface_water_storage_delta_kg": _scalar_value(
                    self._integrate_cell_mass(state.atmosphere.mesh, surface_water_delta)
                ),
                "surface_water_storage_mass_kg": _scalar_value(
                    self._integrate_cell_mass(state.atmosphere.mesh, surface_water_storage)
                ),
            }
        )
        diag.update(self.surface_exchange.output_diagnostics(outputs, surface_params))
        diag.update(
            self._water_ledger_diagnostics(
                state,
                next_state,
                outputs,
                surface_params,
                dt=float(dt),
                freshwater_accumulator_delta=process_accumulator_diag["freshwater_thickness_accumulator_delta"],
                freshwater_accumulator_pending=process_accumulator_diag["freshwater_thickness_accumulator_pending"],
            )
        )
        diag.update(input_rh)
        diag.update(output_rh)
        input_rh_max = input_rh.get("process_input_rough_relative_humidity_max")
        output_rh_max = output_rh.get("process_output_rough_relative_humidity_max")
        if isinstance(input_rh_max, (float, int)) and isinstance(output_rh_max, (float, int)):
            diag["process_delta_rough_relative_humidity_max"] = float(output_rh_max) - float(input_rh_max)
        input_sup = input_rh.get("process_input_supersaturated_cell_level_fraction")
        output_sup = output_rh.get("process_output_supersaturated_cell_level_fraction")
        if isinstance(input_sup, (float, int)) and isinstance(output_sup, (float, int)):
            diag["process_delta_supersaturated_cell_level_fraction"] = float(output_sup) - float(input_sup)
        self.time_seconds.add_(
            torch.as_tensor(float(dt), dtype=self.time_seconds.dtype, device=self.time_seconds.device)
        )
        return PhysicsStepResult(next_state, next_physics_state, diag)


register_component("coupled_scm_process", CoupledSCMProcess)


__all__ = ["CoupledSCMProcess", "CoupledSCMStageEvaluation"]
