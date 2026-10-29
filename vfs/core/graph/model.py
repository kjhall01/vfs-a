"""Component-graph model prototype.

This module is the first narrow implementation of the component-graph
``ModelConfig`` direction documented in ``docs/component_graph_model_config.md``.
It intentionally supports only the dry Held-Suarez-sized slice for now while
leaving the existing ``VFSModel`` and ``dyn.tendency(state)`` fast paths intact.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vfs.field import Field

from .contracts import GraphComponentAPI, GraphDiagnosticAPI, GraphInitializerAPI, GraphObserverAPI, GraphState
from ..integrators import forward_euler, rk4, ssprk3
from .registry import (
    GraphComponentMetadata,
    _GRAPH_REGISTRY,
    _register_graph_metadata,
    graph_component_catalog as _graph_component_catalog,
    graph_component_metadata as _graph_component_metadata,
)
from ..state import State, StateSpec, TracerSpec


_INTEGRATORS = {
    "forward_euler": forward_euler,
    "rk4": rk4,
    "ssprk3": ssprk3,
}
_RUNTIME_DT_RE = re.compile(r"^runtime\.dt_seconds(?:\s*([*/])\s*([0-9]+(?:\.[0-9]*)?|\.[0-9]+))?$")

_GRAPH_STATE_SNAPSHOT_FORMAT = "vfs.component_graph_state"
_GRAPH_STATE_SNAPSHOT_VERSION = 1


def graph_component_metadata(kind: str | None = None) -> dict[str, GraphComponentMetadata]:
    """Return registered graph metadata.

    ``kind`` may be ``"component"``, ``"diagnostic"``, ``"initializer"``, or
    ``"observer"``. With no kind, the returned mapping is keyed as
    ``"kind:name"`` so names remain unambiguous.
    """

    return _graph_component_metadata(kind)


def graph_component_catalog(kind: str | None = None) -> dict[str, dict[str, Any]]:
    """Return JSON/Markdown-friendly graph registry metadata."""

    return _graph_component_catalog(kind)


class DryHydrostaticAtmosphere(GraphComponentAPI):
    """Graph adapter around ``DryHydrostaticDynamics``."""

    def __init__(self, *, column=None, dynamics=None, kwargs: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if dynamics is None:
            dynamics_kwargs = dict(kwargs or {})
            dynamics_type = str(dynamics_kwargs.pop("dynamics_type", "DryHydrostaticDynamics"))
            dynamics = _build_dry_dynamics(dynamics_type, dynamics_kwargs)
        self.dynamics = dynamics
        self.column = column

    def bind_state(self, state: State, *, mesh=None, imports: Mapping[str, Any] | None = None) -> State:
        bound_mesh = mesh if mesh is not None else state.mesh
        bound_column = self.column if self.column is not None else state.column
        phi_s = None if imports is None else imports.get("phi_s")
        if phi_s is not None:
            if state.column is not None and _same_phi_s(state.column.phi_s, phi_s, state=state, mesh=bound_mesh):
                bound_column = state.column
            else:
                bound_column = _column_with_phi_s(bound_column, phi_s, state=state, mesh=bound_mesh)
        if bound_mesh is state.mesh and bound_column is state.column:
            return state
        return State(fields=state.fields, mesh=bound_mesh, column=bound_column, attrs=dict(state.attrs or {}))

    def exports(self, state: State, *, mesh=None, imports: Mapping[str, Any] | None = None) -> dict[str, Any]:
        state = self.bind_state(state, mesh=mesh, imports=imports)
        out: dict[str, Any] = {
            "u_e": state.u_e,
            "mtheta_i": state.mtheta_i,
            "ps": state.ps,
            "phi_s": state._require_column().phi_s.to(dtype=state.ps.dtype, device=state.ps.device),
        }
        if state.tracer_m is not None:
            out["tracer_m"] = state.tracer_m
        try:
            pressure_fields = self.dynamics.pressure_fields(state)
            thermo_fields = self.dynamics.thermo_fields(state)
            wind_fields = self.dynamics.wind_fields(state)
            out.update(pressure_fields)
            out.update(thermo_fields)
            out.update(wind_fields)
            out.setdefault("pressure", pressure_fields["p_full"])
            out.setdefault("wind", wind_fields["cell_wind_vector"])
            out["mass"] = self.dynamics.total_mass(state)
            out["energy"] = self.dynamics.total_energy(state)
        except Exception:
            # Exports are best-effort in v1; stepping still uses the canonical
            # dynamics methods and should surface real computation errors there.
            pass
        return out

    def provide_terms(
        self,
        state: State,
        *,
        mesh=None,
        imports: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, dict[str, torch.Tensor]]], State]:
        state = self.bind_state(state, mesh=mesh, imports=imports)
        terms = self.dynamics.tendency_terms(state)
        return {"atmosphere": terms}, self.dynamics.terms_total(terms, like=state)


class HeldSuarezForcing(GraphComponentAPI):
    """Graph adapter around ``HeldSuarezPhysics``."""

    def __init__(self, *, physics=None, kwargs: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if physics is None:
            from vfs.physics import HeldSuarezPhysics

            physics = HeldSuarezPhysics(**dict(kwargs or {}))
        self.physics = physics

    def exports(self, state: State, *, mesh=None, imports: Mapping[str, Any] | None = None) -> dict[str, Any]:
        del state, mesh, imports
        return {}

    def provide_terms(
        self,
        state: State,
        *,
        mesh=None,
        imports: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, dict[str, torch.Tensor]]], State]:
        del mesh, imports
        rhs = self.physics.tendency(state)
        terms: dict[str, dict[str, torch.Tensor]] = {
            "u_e": {"total": rhs.u_e},
            "mtheta_i": {"total": rhs.mtheta_i},
            "ps": {"total": rhs.ps},
        }
        if rhs.tracer_m is not None:
            terms["tracer_m"] = {"total": rhs.tracer_m}
        return {"atmosphere": terms}, rhs


class SCMForcing(GraphComponentAPI):
    """Graph adapter around ``SCMPhysics`` source tendencies.

    Imports are passed to ``SCMPhysics.tendency(..., extra_params=imports)`` so
    ordinary graph wiring can provide surface/boundary parameters such as skin
    temperature, relative wind, or column-model rates.
    """

    def __init__(self, *, scm=None, kwargs: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if scm is None:
            from vfs.physics import SCMPhysics

            merged = dict(kwargs or {})
            column_model_spec = merged.pop("column_model", "VFSNativeSCM")
            column_model_kwargs = dict(merged.pop("column_model_kwargs", {}))
            scm_kwargs = dict(merged.pop("scm_kwargs", {}))
            if "params" in merged:
                scm_kwargs.setdefault("params", merged.pop("params"))
            if "validate_outputs" in merged:
                scm_kwargs.setdefault("validate_outputs", merged.pop("validate_outputs"))
            if merged:
                unknown = ", ".join(sorted(str(name) for name in merged))
                raise ValueError(f"unknown SCMForcing kwargs: {unknown}")
            scm = SCMPhysics(_build_scm_column_model(column_model_spec, column_model_kwargs), **scm_kwargs)
        self.scm = scm

    def exports(self, state: State, *, mesh=None, imports: Mapping[str, Any] | None = None) -> dict[str, Any]:
        del state, mesh, imports
        return {}

    def provide_terms(
        self,
        state: State,
        *,
        mesh=None,
        imports: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, dict[str, torch.Tensor]]], State]:
        del mesh
        rhs = self.scm.tendency(state, extra_params=dict(imports or {}))
        terms: dict[str, dict[str, torch.Tensor]] = {
            "u_e": {"total": rhs.u_e},
            "mtheta_i": {"total": rhs.mtheta_i},
            "ps": {"total": rhs.ps},
        }
        if rhs.tracer_m is not None:
            terms["tracer_m"] = {"total": rhs.tracer_m}
        return {"atmosphere": terms}, rhs


class OceanStateComponent(GraphComponentAPI):
    """Graph adapter for an owned ``OceanState`` slot."""

    def __init__(self, *, column=None, kwargs: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        del kwargs
        self.column = column

    def bind_state(self, state: Any, *, mesh=None, imports: Mapping[str, Any] | None = None):
        del imports
        from vfs.ocean import OceanState

        if not isinstance(state, OceanState):
            raise TypeError(f"OceanStateComponent requires OceanState, got {type(state).__name__}")
        bound_mesh = mesh if mesh is not None else state.mesh
        bound_column = self.column if self.column is not None else state.column
        if bound_mesh is state.mesh and bound_column is state.column:
            return state
        return OceanState(
            state.u_e,
            state.h,
            state.temp_h,
            state.salt_h,
            state.tracer_h,
            mesh=bound_mesh,
            column=bound_column,
            attrs=dict(state.attrs or {}),
        )

    def exports(self, state: Any, *, mesh=None, imports: Mapping[str, Any] | None = None) -> dict[str, Any]:
        del imports
        state = self.bind_state(state, mesh=mesh)
        temperature_c = state.temperature()
        surface_c = temperature_c[..., :1]
        out: dict[str, Any] = {
            "u_e": state.u_e,
            "h": state.h,
            "temp_h": state.temp_h,
            "salt_h": state.salt_h,
            "temperature_c": temperature_c,
            "surface_temperature_c": surface_c,
            "surface_temperature_kelvin": surface_c + 273.15,
            "salinity": state.salinity(),
            "eta": state.eta(),
            "volume": state.total_volume(),
            "heat_content": state.total_heat_content(),
            "salt_content": state.total_salt_content(),
        }
        if state.tracer_h is not None:
            out["tracer_h"] = state.tracer_h
        return out


class LandSurfaceComponent(GraphComponentAPI):
    """Graph adapter for an owned ``LandSurfaceState`` slot.

    This keeps the land model as an ordinary graph component: other
    components import diagnostic boundary properties from the land state, while
    land prognostic updates run through an explicit graph schedule operation.
    """

    def __init__(self, *, land=None, kwargs: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if land is None:
            from vfs.land import LandSurface

            land = LandSurface(**dict(kwargs or {}))
        self.land = land
        self.last_update_diagnostics: dict[str, torch.Tensor] = {}
        self.last_update_transfers: tuple[Any, ...] = ()

    def bind_state(self, state: Any, *, mesh=None, imports: Mapping[str, Any] | None = None):
        del mesh, imports
        from vfs.land import LandSurfaceState

        if not isinstance(state, LandSurfaceState):
            raise TypeError(f"LandSurfaceComponent requires LandSurfaceState, got {type(state).__name__}")
        return state

    def exports(self, state: Any, *, mesh=None, imports: Mapping[str, Any] | None = None) -> dict[str, Any]:
        del mesh, imports
        state = self.bind_state(state)
        boundary = self.land.boundary_properties(state)
        return {
            "soil_liquid_water": state.soil_liquid_water,
            "soil_temperature": state.soil_temperature,
            "snow_water_equivalent": state.snow_water_equivalent,
            "skin_temperature": boundary.skin_temperature,
            "soil_ice": state.soil_ice,
            "canopy_water": state.canopy_water,
            "land_use_type": boundary.land_use_type,
            "soil_water": state.soil_water,
            "top_soil_temperature": state.top_soil_temperature,
            "albedo": boundary.albedo,
            "roughness": boundary.roughness,
            "moisture_availability": boundary.moisture_availability,
            "snow_cover_fraction": boundary.snow_cover_fraction,
            "snow_insulation_factor": boundary.snow_insulation_factor,
            "surface_emissions_factor": boundary.surface_emissions_factor,
        }

    def update_state(self, state: Any, *, dt: float, mesh=None, imports: Mapping[str, Any] | None = None) -> Any:
        state = self.bind_state(state)
        imports = dict(imports or {})
        like = state.skin_temperature
        fractions = self._fractions_from_imports(imports, like)
        fluxes = self._fluxes_from_imports(imports, like)
        result = self.land.step(state, fractions, fluxes, float(dt), mesh=mesh)
        self.last_update_diagnostics = dict(result.diagnostics)
        self.last_update_transfers = tuple(result.transfers)
        return result.state

    def _fractions_from_imports(self, imports: Mapping[str, Any], like: torch.Tensor):
        from vfs.surface import SurfaceFractions

        land = _cell_column_import(imports, "land_fraction", like=like, default=1.0)
        ocean = (
            _cell_column_import(imports, "ocean_fraction", like=like, default=None)
            if "ocean_fraction" in imports
            else None
        )
        return SurfaceFractions.from_land_ocean(
            land,
            ocean,
            sea_ice_fraction=_cell_column_import(imports, "sea_ice_fraction", like=like, default=0.0),
            glacier_fraction=_cell_column_import(imports, "glacier_fraction", like=like, default=0.0),
            snow_fraction=_cell_column_import(imports, "snow_fraction", like=like, default=0.0),
            n_cells=int(like.shape[0]),
            dtype=like.dtype,
            device=like.device,
        )

    def _fluxes_from_imports(self, imports: Mapping[str, Any], like: torch.Tensor):
        from vfs.surface import AtmosSurfaceFluxPacket

        return AtmosSurfaceFluxPacket(
            _cell_column_import(imports, "net_shortwave", like=like, default=0.0),
            _cell_column_import(imports, "net_longwave", like=like, default=0.0),
            _cell_column_import(imports, "sensible_heat", like=like, default=0.0),
            _cell_column_import(imports, "latent_heat", like=like, default=0.0),
            _cell_column_import(imports, "precipitation_rain", like=like, default=0.0),
            _cell_column_import(imports, "precipitation_snow", like=like, default=0.0),
            _cell_column_import(imports, "wind_stress_x", like=like, default=0.0),
            _cell_column_import(imports, "wind_stress_y", like=like, default=0.0),
        )


class AirSeaHeatExchangeForcing(GraphComponentAPI):
    """Minimal graph source term for same-mesh atmosphere/ocean heat exchange.

    This is intentionally a small machinery component, not a mature surface
    physics package. It proves that a single component can import values from
    two state owners and provide separate tendencies back to each reservoir.
    """

    def __init__(self, *, kwargs: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        merged = dict(kwargs or {})
        self.coefficient = float(merged.pop("coefficient", 10.0))
        self.air_layer_index = int(merged.pop("air_layer_index", -1))
        self.ocean_layer_index = int(merged.pop("ocean_layer_index", 0))
        self.atmosphere_cp = float(merged.pop("atmosphere_cp", 1004.5))
        self.ocean_rho0 = float(merged.pop("ocean_rho0", 1025.0))
        self.ocean_cp = float(merged.pop("ocean_cp", 3990.0))
        self.gravity = float(merged.pop("gravity", 9.80665))
        if merged:
            unknown = ", ".join(sorted(str(name) for name in merged))
            raise ValueError(f"unknown AirSeaHeatExchangeForcing kwargs: {unknown}")

    def exports(self, state: Any | None = None, *, mesh=None, imports: Mapping[str, Any] | None = None) -> dict[str, Any]:
        del state, mesh
        if not imports:
            return {}
        flux = self._heat_flux_up(imports)
        return {
            "heat_flux_to_atmosphere": flux,
            "heat_flux_to_ocean": -flux,
        }

    def provide_terms(
        self,
        state: Any,
        *,
        mesh=None,
        imports: Mapping[str, Any] | None = None,
    ):
        del mesh
        from vfs.ocean import OceanState

        flux_up = self._heat_flux_up(imports or {})
        if isinstance(state, State):
            rhs = self._atmosphere_rhs(state, flux_up)
            return {"atmosphere": {"mtheta_i": {"air_sea_heat": rhs.mtheta_i}}}, rhs
        if isinstance(state, OceanState):
            rhs = self._ocean_rhs(state, -flux_up)
            return {"ocean": {"temp_h": {"air_sea_heat": rhs.temp_h}}}, rhs
        raise TypeError(f"AirSeaHeatExchangeForcing cannot provide terms for {type(state).__name__}")

    def _heat_flux_up(self, imports: Mapping[str, Any]) -> torch.Tensor:
        air_temperature = _require_import(imports, "air_temperature")
        ocean_temperature = _require_import(imports, "ocean_surface_temperature")
        air = _select_cell_level(_as_tensor(air_temperature), self.air_layer_index)
        ocean = _select_cell_level(_as_tensor(ocean_temperature), self.ocean_layer_index)
        return float(self.coefficient) * (ocean - air)

    def _atmosphere_rhs(self, state: State, heat_flux_up: torch.Tensor) -> State:
        rhs = state.zeros_like()
        layer = _normalize_level_index(self.air_layer_index, int(state.column.n_levels))
        dp = state.layer_dp()
        exner = state.exner_full() / state.column.c_p
        layer_dp = dp[..., layer : layer + 1].clamp_min(torch.finfo(dp.dtype).tiny)
        layer_exner = exner[..., layer : layer + 1].clamp_min(torch.finfo(exner.dtype).tiny)
        flux = heat_flux_up.to(dtype=dp.dtype, device=dp.device)
        dT_dt = float(self.gravity) * flux / (float(self.atmosphere_cp) * layer_dp)
        layer_tendency = torch.zeros_like(dp)
        layer_tendency[..., layer : layer + 1] = dp[..., layer : layer + 1] * (dT_dt / layer_exner)
        dmtheta = torch.zeros_like(state.mtheta_i)
        dmtheta[..., :-1] += 0.5 * layer_tendency
        dmtheta[..., 1:] += 0.5 * layer_tendency
        return State(
            rhs.u_e,
            dmtheta,
            rhs.ps,
            rhs.tracer_m,
            mesh=state.mesh,
            column=state.column,
            attrs=dict(state.attrs or {}),
        )

    def _ocean_rhs(self, state: Any, heat_flux_to_ocean: torch.Tensor):
        rhs = state.zeros_like()
        layer = _normalize_level_index(self.ocean_layer_index, int(state.column.n_layers))
        flux = heat_flux_to_ocean.to(dtype=state.temp_h.dtype, device=state.temp_h.device)
        dtemp_h = torch.zeros_like(state.temp_h)
        dtemp_h[..., layer : layer + 1] = flux / (float(self.ocean_rho0) * float(self.ocean_cp))
        return state.with_tensors(rhs.u_e, rhs.h, dtemp_h, rhs.salt_h, rhs.tracer_h)


class ExternalFields(GraphComponentAPI):
    """Component that exports fixed non-prognostic tensors or scalars."""

    def __init__(
        self,
        *,
        mesh=None,
        column=None,
        fields: Mapping[str, Any] | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        merged = dict(kwargs or {})
        specs = dict(fields or merged.get("fields", {}))
        self.field_names = tuple(str(name) for name in specs)
        self._field_buffers: dict[str, str] = {}
        for i, name in enumerate(self.field_names):
            buffer_name = f"_field_{i}_{_safe_key(name) or 'value'}"
            tensor = _external_tensor_from_spec(specs[name], mesh=mesh, column=column)
            self.register_buffer(buffer_name, tensor)
            self._field_buffers[name] = buffer_name

    def exports(self, state: State | None = None, *, mesh=None, imports: Mapping[str, Any] | None = None) -> dict[str, Any]:
        del state, mesh, imports
        return {name: getattr(self, buffer_name) for name, buffer_name in self._field_buffers.items()}


class HeldSuarezInitializer(GraphInitializerAPI):
    """Build a dry-atmosphere ``State`` from Held-Suarez-style IC choices."""

    def __init__(self, *, kwargs: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        merged = dict(kwargs or {})
        self.initial_condition = str(merged.pop("initial_condition", "isothermal_rest"))
        self.temperature = float(merged.pop("temperature", 288.0))
        self.ps_value = float(merged.pop("ps_value", 1.0e5))
        self.equilibrium_wind = str(merged.pop("equilibrium_wind", "small_noise"))
        self.wind_mode = merged.pop("wind_mode", None)
        self.initial_wind_noise = float(merged.pop("initial_wind_noise", merged.pop("wind_noise_std", 0.05)))
        self.seed = int(merged.pop("seed", 0))
        self.vapor_mixing_ratio = merged.pop("vapor_mixing_ratio", None)
        self.cloud_liquid_mixing_ratio = merged.pop("cloud_liquid_mixing_ratio", None)
        self.tracer_initial_values = dict(merged.pop("tracer_initial_values", {}))
        if merged:
            unknown = ", ".join(sorted(str(name) for name in merged))
            raise ValueError(f"unknown HeldSuarezInitializer kwargs: {unknown}")

    def build(self, model: "ComponentGraphModel", target: str, *, dtype=None, device=None) -> State:
        from vfs.initial_conditions.held_suarez import normalize_held_suarez_initial_condition
        from vfs.physics import HeldSuarezConstants, held_suarez_equilibrium_temperature

        mesh = _require_model_mesh(model)
        column = _component_column(model, target)
        dtype, device = _resolve_dtype_device(mesh=mesh, column=column, dtype=dtype, device=device)
        condition = normalize_held_suarez_initial_condition(self.initial_condition)
        constants = HeldSuarezConstants(p0=float(self.ps_value))
        ps = torch.full((int(mesh.nCells), 1), float(self.ps_value), dtype=dtype, device=device)

        if condition == "hs_equilibrium":
            p_i = column.interface_pressure(ps)
            lat = _cell_lat(mesh, dtype=dtype, device=device)
            temperature_i = held_suarez_equilibrium_temperature(column, constants, lat, p_i)
            wind_mode = self.equilibrium_wind
        else:
            temperature_i = torch.full(
                (int(mesh.nCells), int(column.n_levels) + 1),
                float(self.temperature),
                dtype=dtype,
                device=device,
            )
            wind_mode = str(self.wind_mode or "none")

        exner = (column.exner_interface(ps) / column.c_p).clamp_min(torch.finfo(dtype).tiny)
        theta_i = temperature_i / exner
        u_e = _initial_wind_for_mode(
            mesh,
            column,
            wind_mode=wind_mode,
            initial_wind_noise=self.initial_wind_noise,
            seed=self.seed,
            dtype=dtype,
            device=device,
        )
        q_tracers, spec, tracer_attrs = _initial_tracers_for_component(
            model,
            target,
            n_cells=int(mesh.nCells),
            n_levels=int(column.n_levels),
            dtype=dtype,
            device=device,
            vapor_mixing_ratio=self.vapor_mixing_ratio,
            cloud_liquid_mixing_ratio=self.cloud_liquid_mixing_ratio,
            tracer_initial_values=self.tracer_initial_values,
        )
        attrs = {
            "gravity": float(getattr(_component_dynamics(model, target), "gravity", getattr(column, "g", 9.80665))),
            "initial_condition": condition,
            "initializer": type(self).__name__,
            "equilibrium_wind": str(wind_mode),
        }
        attrs.update(tracer_attrs)
        return State.from_theta(
            u_e,
            theta_i,
            ps,
            mesh=mesh,
            column=column,
            q_tracers=q_tracers,
            attrs=attrs,
            spec=spec,
        )


class JablonskiWilliamsonInitializer(GraphInitializerAPI):
    """Build a dry-atmosphere ``State`` for JW-A/JW-B test cases."""

    _CASE_ALIASES = {
        "a": False,
        "jw_a": False,
        "jwa": False,
        "balanced_jet": False,
        "baroclinic_wave_unperturbed": False,
        "b": True,
        "jw_b": True,
        "jwb": True,
        "baroclinic_wave": True,
        "perturbed_baroclinic_wave": True,
    }

    def __init__(self, *, kwargs: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        merged = dict(kwargs or {})
        case = str(merged.pop("case", "jw_b")).lower()
        if case not in self._CASE_ALIASES:
            raise ValueError(f"unknown JW case {case!r}; choose jw_a or jw_b")
        self.case = case
        self.perturb = bool(merged.pop("perturb", self._CASE_ALIASES[case]))
        self.jw_ic_mode = str(merged.pop("jw_ic_mode", merged.pop("mode", "constant_ps_surface_phi")))
        self.theta_source = str(merged.pop("theta_source", "phi_derivative"))
        self.wind_source = str(merged.pop("wind_source", "U_jw"))
        self.theta_ic_mode = str(merged.pop("theta_ic_mode", "point"))
        self.theta_lsq_lambda = float(merged.pop("theta_lsq_lambda", 0.1))
        self.project_mass_flux = bool(merged.pop("project_mass_flux", False))
        self.phi_s_scale = float(merged.pop("phi_s_scale", 1.0))
        self.phi_full_mode = merged.pop("phi_full_mode", None)
        if merged:
            unknown = ", ".join(sorted(str(name) for name in merged))
            raise ValueError(f"unknown JablonskiWilliamsonInitializer kwargs: {unknown}")

    def build(self, model: "ComponentGraphModel", target: str, *, dtype=None, device=None) -> State:
        from vfs.initial_conditions.jablonski_williamson import JWInitialConditions

        mesh = _require_model_mesh(model)
        base_column = _component_column(model, target)
        dtype, device = _resolve_dtype_device(mesh=mesh, column=base_column, dtype=dtype, device=device)

        # JW construction has numpy-based formula paths and a few CPU linear
        # algebra helpers. Build an equivalent CPU geometry/column, then bind
        # the resulting prognostic tensors back to the live model geometry.
        cpu_mesh = copy.deepcopy(mesh).to(device="cpu", dtype=torch.float64)
        base_cpu_column = copy.deepcopy(base_column).to(device="cpu", dtype=torch.float64)
        if self.jw_ic_mode == "constant_ps_surface_phi":
            cpu_column = JWInitialConditions.make_column_for_mode(
                cpu_mesh,
                int(base_column.n_levels),
                base_cpu_column.A_interface.detach().cpu().numpy(),
                base_cpu_column.B_interface.detach().cpu().numpy(),
                mode=self.jw_ic_mode,
                phi_s_scale=self.phi_s_scale,
                phi_full_mode=str(self.phi_full_mode or base_column.phi_full_mode),
            )
        else:
            cpu_column = base_cpu_column
        jwic = JWInitialConditions(cpu_mesh, cpu_column, mode=self.jw_ic_mode)
        state, info = jwic.make_state(
            None,
            perturb=self.perturb,
            project_mass_flux=self.project_mass_flux,
            theta_source=self.theta_source,
            wind_source=self.wind_source,
            theta_ic_mode=self.theta_ic_mode,
            theta_lsq_lambda=self.theta_lsq_lambda,
            return_info=True,
        )
        live_column = copy.deepcopy(cpu_column).to(dtype=dtype, device=device)
        _set_component_column(model, target, live_column)
        attrs = dict(state.attrs or {})
        attrs.update(
            {
                "initializer": type(self).__name__,
                "initial_condition": self.case,
                "jw_ic_mode": self.jw_ic_mode,
                "jw_perturb": bool(self.perturb),
                "jw_u_max": float(info.u_max or 0.0),
            }
        )
        return State(
            fields=state.fields.to(dtype=dtype, device=device),
            mesh=mesh,
            column=live_column,
            attrs=attrs,
        )


class OceanRestInitializer(GraphInitializerAPI):
    """Build a same-mesh resting ``OceanState``."""

    def __init__(self, *, kwargs: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        merged = dict(kwargs or {})
        self.n_layers = merged.pop("n_layers", None)
        self.total_depth = float(merged.pop("total_depth", 1000.0))
        self.temperature_c = float(merged.pop("temperature_c", 15.0))
        self.salinity = float(merged.pop("salinity", 35.0))
        if merged:
            unknown = ", ".join(sorted(str(name) for name in merged))
            raise ValueError(f"unknown OceanRestInitializer kwargs: {unknown}")

    def build(self, model: "ComponentGraphModel", target: str, *, dtype=None, device=None):
        from vfs.ocean import OceanColumn, OceanState

        mesh = _require_model_mesh(model)
        column = _component_column(model, target, required=False)
        if column is None:
            if self.n_layers is None:
                raise ValueError("OceanRestInitializer requires an ocean column or kwargs.n_layers")
            column = OceanColumn(int(self.n_layers), total_depth=float(self.total_depth))
            _set_component_column(model, target, column)
        dtype, device = _resolve_dtype_device(mesh=mesh, column=column, dtype=dtype, device=device)
        column = column.to(dtype=dtype, device=device)
        _set_component_column(model, target, column)
        h = column.rest_thickness.to(dtype=dtype, device=device).reshape(1, -1).expand(int(mesh.nCells), int(column.n_layers)).clone()
        u = torch.zeros(int(mesh.nEdges), int(column.n_layers), dtype=dtype, device=device)
        temperature = torch.full_like(h, float(self.temperature_c))
        salinity = torch.full_like(h, float(self.salinity))
        return OceanState.from_temperature(
            u,
            h,
            temperature,
            salinity,
            mesh=mesh,
            column=column,
            attrs={"initializer": type(self).__name__, "initial_condition": "ocean_rest"},
        )


class LandSurfaceInitializer(GraphInitializerAPI):
    """Build a same-mesh ``LandSurfaceState`` from uniform land stores."""

    def __init__(self, *, kwargs: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        merged = dict(kwargs or {})
        self.soil_water = float(merged.pop("soil_water", 50.0))
        self.soil_temperature = float(merged.pop("soil_temperature", 280.0))
        self.snow_water_equivalent = float(merged.pop("snow_water_equivalent", 0.0))
        self.skin_temperature = float(merged.pop("skin_temperature", 280.0))
        self.canopy_water = float(merged.pop("canopy_water", 0.0))
        self.land_use_type = merged.pop("land_use_type", 0)
        if merged:
            unknown = ", ".join(sorted(str(name) for name in merged))
            raise ValueError(f"unknown LandSurfaceInitializer kwargs: {unknown}")

    def build(self, model: "ComponentGraphModel", target: str, *, dtype=None, device=None):
        mesh = _require_model_mesh(model)
        dtype, device = _resolve_dtype_device(mesh=mesh, dtype=dtype, device=device)
        component = model.components[str(target)]
        land = getattr(component, "land", None)
        if land is None or not hasattr(land, "initial_state"):
            raise ValueError(f"component {target!r} does not expose a LandSurface initial_state(...) method")
        return land.initial_state(
            int(mesh.nCells),
            soil_water=float(self.soil_water),
            soil_temperature=float(self.soil_temperature),
            snow_water_equivalent=float(self.snow_water_equivalent),
            skin_temperature=float(self.skin_temperature),
            canopy_water=float(self.canopy_water),
            land_use_type=self.land_use_type,
            dtype=dtype,
            device=device,
        )


class WeatherBench2ERA5DryInitializer(GraphInitializerAPI):
    """Build a dry atmospheric ``State`` from a local or remote WeatherBench2/ERA5 snapshot."""

    def __init__(self, *, kwargs: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        merged = dict(kwargs or {})
        self.uri = str(merged.pop("uri", merged.pop("path", "")))
        self.time = merged.pop("time", None)
        self.carry_vapor = bool(merged.pop("carry_vapor", False))
        self.chunks = merged.pop("chunks", "auto")
        self.consolidated = bool(merged.pop("consolidated", True))
        self.storage_options = dict(merged.pop("storage_options", {}))
        if merged:
            unknown = ", ".join(sorted(str(name) for name in merged))
            raise ValueError(f"unknown WeatherBench2ERA5DryInitializer kwargs: {unknown}")
        if not self.uri:
            raise ValueError("WeatherBench2ERA5DryInitializer requires kwargs.uri or kwargs.path")

    def build(self, model: "ComponentGraphModel", target: str, *, dtype=None, device=None) -> State:
        from vfs.era5 import (
            latlon_grid_from_xarray,
            open_weatherbench2_era5,
            weatherbench2_dry_state_from_dataset,
        )
        from vfs.geometry.regrid import LatLonToMeshBilinear

        mesh = _require_model_mesh(model)
        column = _component_column(model, target)
        dtype, device = _resolve_dtype_device(mesh=mesh, column=column, dtype=dtype, device=device)
        cpu_mesh = copy.deepcopy(mesh).to(device="cpu", dtype=torch.float64)
        cpu_column = copy.deepcopy(column).to(device="cpu", dtype=torch.float64)
        ds = open_weatherbench2_era5(
            self.uri,
            chunks=self.chunks,
            consolidated=self.consolidated,
            storage_options=self.storage_options,
        )
        grid = latlon_grid_from_xarray(ds)
        grid_to_mesh = LatLonToMeshBilinear.from_mesh(grid, cpu_mesh)
        state = weatherbench2_dry_state_from_dataset(
            ds,
            mesh=cpu_mesh,
            column=cpu_column,
            grid_to_mesh=grid_to_mesh,
            time=self.time,
            dyn=_component_dynamics(model, target),
            carry_vapor=self.carry_vapor,
        )
        live_column = copy.deepcopy(cpu_column).to(dtype=dtype, device=device)
        _set_component_column(model, target, live_column)
        attrs = dict(state.attrs or {})
        attrs.update({"initializer": type(self).__name__, "weatherbench2_uri": self.uri})
        return State(fields=state.fields.to(dtype=dtype, device=device), mesh=mesh, column=live_column, attrs=attrs)


class DryAtmosphereHealthDiagnostics(GraphDiagnosticAPI):
    """Read-only dry-atmosphere diagnostic component for graph models."""

    def __init__(
        self,
        *,
        target: str = "atmosphere",
        include_term_stats: bool = True,
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        merged = dict(kwargs or {})
        self.target = str(merged.get("target", target))
        self.include_term_stats = bool(merged.get("include_term_stats", include_term_stats))

    def row(
        self,
        model: "ComponentGraphModel",
        state0: State | None,
        state: State,
        *,
        t: float | None = None,
        dt: float | None = None,
        step: int | None = None,
        day: float | None = None,
    ) -> dict[str, Any]:
        component = model.components[self.target]
        dynamics = getattr(component, "dynamics", None)
        diagnostics = getattr(dynamics, "diagnostics", None)
        if diagnostics is None:
            raise ValueError(f"diagnostic target {self.target!r} does not expose dry diagnostics")
        bound = model.bind_state(self.target, state)
        bound0 = None if state0 is None else model.bind_state(self.target, state0)
        if bound0 is None:
            row = diagnostics.health_row(bound, t=t, dt=dt, step=step, day=day)
        else:
            row = diagnostics.health_row(bound0, bound, t=t, dt=dt, step=step, day=day)
        if self.include_term_stats:
            row.update(_term_stats(model.last_terms, prefix="graph_term"))
        return row


class NativeExportObserver(GraphObserverAPI):
    """Read-only observer that returns selected component exports unchanged."""

    def __init__(
        self,
        *,
        target: str = "atmosphere",
        fields: tuple[str, ...] | list[str] | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        merged = dict(kwargs or {})
        self.target = str(merged.get("target", target))
        selected = merged.get("fields", fields)
        self.fields = None if selected is None else tuple(str(name) for name in selected)

    def observe(self, model: "ComponentGraphModel", state: Any) -> dict[str, Any]:
        exports = model.component_exports(self.target, state)
        fields = tuple(exports) if self.fields is None else self.fields
        return {name: _require_export(exports, self.target, name) for name in fields}


class MeshToLatLonObserver(GraphObserverAPI):
    """Read-only observer that regrids selected cell-centered exports."""

    def __init__(
        self,
        *,
        mesh,
        target: str = "atmosphere",
        fields: tuple[str, ...] | list[str] | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if mesh is None:
            raise ValueError("MeshToLatLonObserver requires model mesh")
        merged = dict(kwargs or {})
        self.target = str(merged.get("target", target))
        selected = merged.get("fields", fields)
        self.fields = tuple(str(name) for name in (selected or ("ps", "temperature")))
        from vfs.geometry.regrid import LatLonGrid, MeshToLatLonKNN

        grid_spec = dict(merged.get("grid", {}))
        nlat = int(merged.get("nlat", grid_spec.get("nlat", 181)))
        nlon = int(merged.get("nlon", grid_spec.get("nlon", 360)))
        grid = LatLonGrid.regular(
            nlat,
            nlon,
            lat_min=float(merged.get("lat_min", grid_spec.get("lat_min", -90.0))),
            lat_max=float(merged.get("lat_max", grid_spec.get("lat_max", 90.0))),
            lon_min=float(merged.get("lon_min", grid_spec.get("lon_min", 0.0))),
            lon_max=float(merged.get("lon_max", grid_spec.get("lon_max", 360.0))),
        )
        self.regridder = MeshToLatLonKNN.from_mesh(
            mesh,
            grid,
            k=int(merged.get("k", 8)),
            power=float(merged.get("power", 2.0)),
            eps=float(merged.get("eps", 1.0e-12)),
        )

    def observe(self, model: "ComponentGraphModel", state: Any) -> dict[str, Any]:
        exports = model.component_exports(self.target, state)
        ref_state = model.bind_state(self.target, state)
        ref = _state_reference_tensor(ref_state)
        out: dict[str, Any] = {
            "lat_deg": self.regridder.grid.lat_deg.to(device=ref.device, dtype=ref.dtype),
            "lon_deg": self.regridder.grid.lon_deg.to(device=ref.device, dtype=ref.dtype),
            "attrs": {
                "observer": type(self).__name__,
                "component": self.target,
                "regrid_method": self.regridder.method,
                "conservative": False,
                "differentiable_wrt_fields": True,
            },
        }
        n_cells = int(model.mesh.nCells)
        for name in self.fields:
            value = _require_export(exports, self.target, name)
            if isinstance(value, Field):
                value = value.values
            if not torch.is_tensor(value):
                raise ValueError(f"cannot lat/lon observe non-tensor export {self.target!r}.{name!r}")
            if value.ndim < 1 or int(value.shape[0]) != n_cells:
                raise ValueError(
                    f"lat/lon observer expected cell-centered export {self.target!r}.{name!r} "
                    f"with leading nCells={n_cells}, got shape {tuple(value.shape)}"
                )
            out[name] = _squeeze_last_singleton(self.regridder(value))
        return out


for _metadata in (
    GraphComponentMetadata(
        name="AirSeaHeatExchangeForcing",
        cls=AirSeaHeatExchangeForcing,
        kind="component",
        status="experimental",
        default_exports=("heat_flux_to_atmosphere", "heat_flux_to_ocean"),
        default_provides={"atmosphere": ("mtheta_i",), "ocean": ("temp_h",)},
        description="Minimal same-mesh atmosphere/ocean heat-exchange source used for graph machinery smoke tests.",
    ),
    GraphComponentMetadata(
        name="DryHydrostaticAtmosphere",
        cls=DryHydrostaticAtmosphere,
        kind="component",
        status="provisional",
        state_type="State",
        default_owns=("u_e", "mtheta_i", "ps", "tracer_m"),
        default_exports=("u_e", "mtheta_i", "ps", "tracer_m", "pressure", "temperature", "wind", "mass", "energy", "phi_s"),
        default_provides={"atmosphere": ("u_e", "mtheta_i", "ps", "tracer_m")},
        requires_column=True,
        description="Graph adapter around dry hydrostatic atmospheric dynamics.",
    ),
    GraphComponentMetadata(
        name="ExternalFields",
        cls=ExternalFields,
        kind="component",
        status="provisional",
        requires_mesh=True,
        requires_column=True,
        description="Fixed non-prognostic scalar/tensor field provider such as phi_s or solar constants.",
    ),
    GraphComponentMetadata(
        name="HeldSuarezForcing",
        cls=HeldSuarezForcing,
        kind="component",
        status="provisional",
        default_provides={"atmosphere": ("u_e", "mtheta_i", "ps", "tracer_m")},
        description="Graph adapter around Held-Suarez Newtonian cooling and low-level Rayleigh drag.",
    ),
    GraphComponentMetadata(
        name="HeldSuarezInitializer",
        cls=HeldSuarezInitializer,
        kind="initializer",
        status="provisional",
        state_type="State",
        requires_mesh=True,
        requires_column=True,
        description="Build a dry atmosphere State from isothermal-rest or Held-Suarez-equilibrium initial conditions.",
    ),
    GraphComponentMetadata(
        name="JablonskiWilliamsonInitializer",
        cls=JablonskiWilliamsonInitializer,
        kind="initializer",
        status="provisional",
        state_type="State",
        requires_mesh=True,
        requires_column=True,
        description="Build JW-A/JW-B dry atmosphere initial states from analytic Jablonowski-Williamson formulas.",
    ),
    GraphComponentMetadata(
        name="LandSurfaceComponent",
        cls=LandSurfaceComponent,
        kind="component",
        status="provisional",
        state_type="LandSurfaceState",
        default_owns=(
            "soil_liquid_water",
            "soil_temperature",
            "snow_water_equivalent",
            "skin_temperature",
            "soil_ice",
            "canopy_water",
            "land_use_type",
        ),
        default_exports=(
            "soil_liquid_water",
            "soil_temperature",
            "snow_water_equivalent",
            "skin_temperature",
            "soil_ice",
            "canopy_water",
            "land_use_type",
            "soil_water",
            "top_soil_temperature",
            "albedo",
            "roughness",
            "moisture_availability",
            "snow_cover_fraction",
            "snow_insulation_factor",
            "surface_emissions_factor",
        ),
        description="Graph adapter for an owned vfs.land LandSurfaceState and diagnostic land boundary exports.",
    ),
    GraphComponentMetadata(
        name="LandSurfaceInitializer",
        cls=LandSurfaceInitializer,
        kind="initializer",
        status="provisional",
        state_type="LandSurfaceState",
        requires_mesh=True,
        description="Build a uniform LandSurfaceState through the configured LandSurface component.",
    ),
    GraphComponentMetadata(
        name="OceanStateComponent",
        cls=OceanStateComponent,
        kind="component",
        status="provisional",
        state_type="OceanState",
        default_owns=("u_e", "h", "temp_h", "salt_h", "tracer_h"),
        default_exports=(
            "u_e",
            "h",
            "temp_h",
            "salt_h",
            "tracer_h",
            "temperature_c",
            "surface_temperature_c",
            "surface_temperature_kelvin",
            "salinity",
            "eta",
            "volume",
            "heat_content",
            "salt_content",
        ),
        requires_column=True,
        description="Graph adapter for an owned OceanState slot.",
    ),
    GraphComponentMetadata(
        name="OceanRestInitializer",
        cls=OceanRestInitializer,
        kind="initializer",
        status="provisional",
        state_type="OceanState",
        requires_mesh=True,
        requires_column=True,
        description="Build a same-mesh resting OceanState from layer thickness, temperature, and salinity.",
    ),
    GraphComponentMetadata(
        name="SCMForcing",
        cls=SCMForcing,
        kind="component",
        status="experimental",
        default_provides={"atmosphere": ("u_e", "mtheta_i", "ps", "tracer_m")},
        description="Graph adapter around SCMPhysics source tendencies.",
    ),
    GraphComponentMetadata(
        name="WeatherBench2ERA5DryInitializer",
        cls=WeatherBench2ERA5DryInitializer,
        kind="initializer",
        status="experimental",
        state_type="State",
        requires_mesh=True,
        requires_column=True,
        description="Build a dry atmosphere State from a WeatherBench2/ERA5 pressure-level snapshot.",
    ),
    GraphComponentMetadata(
        name="DryAtmosphereHealthDiagnostics",
        cls=DryAtmosphereHealthDiagnostics,
        kind="diagnostic",
        status="provisional",
        description="Read-only diagnostic adapter around DryHydrostaticDiagnostics.health_row.",
    ),
    GraphComponentMetadata(
        name="NativeExportObserver",
        cls=NativeExportObserver,
        kind="observer",
        status="provisional",
        description="Read-only observer returning selected component exports unchanged.",
    ),
    GraphComponentMetadata(
        name="MeshToLatLonObserver",
        cls=MeshToLatLonObserver,
        kind="observer",
        status="provisional",
        requires_mesh=True,
        description="Read-only observer regridding selected cell-centered exports to a regular lat/lon grid.",
    ),
):
    _register_graph_metadata(_metadata)
del _metadata


class _GraphTendencyProvider:
    def __init__(
        self,
        model: "ComponentGraphModel",
        target: str,
        providers: tuple[str, ...],
        graph_state: GraphState,
    ) -> None:
        self.model = model
        self.target = str(target)
        self.providers = tuple(str(name) for name in providers)
        self.graph_state = graph_state

    def tendency(self, state: Any) -> Any:
        graph_state = self.graph_state.with_state(self.target, state)
        target_state = self.model.bind_state(self.target, graph_state)
        graph_state = graph_state.with_state(self.target, target_state)
        total: Any | None = None
        by_provider: dict[str, dict[str, dict[str, dict[str, torch.Tensor]]]] = {}
        merged: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
        for provider_name in self.providers:
            component = self.model.components[provider_name]
            imports = self.model.component_imports(provider_name, graph_state)
            terms, rhs = component.provide_terms(target_state, mesh=self.model.mesh, imports=imports)
            by_provider[provider_name] = terms
            _merge_terms(merged, provider_name, terms)
            total = rhs if total is None else total + rhs
        if total is None:
            raise ValueError("integrate step requires at least one terms_from provider")
        self.model.last_terms_by_provider = by_provider
        self.model.last_terms = merged
        return total


class ComponentGraphModel(nn.Module):
    """Narrow graph-configured model prototype for dry Held-Suarez stepping."""

    def __init__(self, config: Mapping[str, Any], *, mesh=None, column=None, columns: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.config = _as_config_dict(config)
        self.mesh = mesh
        self.runtime = dict(self.config.get("runtime", {}))
        self.last_terms_by_provider: dict[str, Any] = {}
        self.last_terms: dict[str, Any] = {}
        self.last_evaluation_outputs: dict[str, Any] = {}

        columns_by_name = dict(columns or {})
        if column is not None:
            columns_by_name.setdefault("atmosphere", column)
        self.columns_by_name = columns_by_name

        specs = dict(self.config.get("components", {}))
        if not specs:
            raise ValueError("ComponentGraphModel config requires at least one component")
        self.component_specs = {str(name): _as_config_dict(spec) for name, spec in specs.items()}
        self._validate_component_specs()

        modules: dict[str, nn.Module] = {}
        for name, spec in self.component_specs.items():
            modules[name] = self._build_component(name, spec, mesh=self.mesh, column=columns_by_name.get(name, column))
        self.components = nn.ModuleDict(modules)
        self._validate_wiring()
        self.diagnostic_specs = {
            str(name): _as_config_dict(spec) for name, spec in dict(self.config.get("diagnostics", {})).items()
        }
        self.diagnostic_components = nn.ModuleDict(self._build_diagnostics())
        self._validate_diagnostics()
        self.observer_specs = {
            str(name): _as_config_dict(spec) for name, spec in dict(self.config.get("observers", {})).items()
        }
        self.observer_components = nn.ModuleDict(self._build_observers())
        self._validate_observers()
        self.schedule = self._compile_schedule(self.config.get("schedule", {}))
        self.initialization_spec = _as_config_dict(self.config.get("initialization", {}))
        self._validate_initialization_spec()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, mesh=None, column=None, columns: Mapping[str, Any] | None = None) -> "ComponentGraphModel":
        return cls(config, mesh=mesh, column=column, columns=columns)

    @classmethod
    def from_yaml(cls, path: str | Path, *, mesh=None, column=None, columns: Mapping[str, Any] | None = None) -> "ComponentGraphModel":
        import yaml

        with Path(path).open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        if not isinstance(config, Mapping):
            raise ValueError(f"YAML model config must contain a mapping, got {type(config).__name__}")
        return cls.from_config(config, mesh=mesh, column=column, columns=columns)

    def _build_component(self, name: str, spec: Mapping[str, Any], *, mesh=None, column=None) -> nn.Module:
        type_name = str(spec.get("type", ""))
        metadata = _graph_metadata("component", type_name)
        if metadata is None:
            raise ValueError(f"unknown graph component type {type_name!r} for component {name!r}")
        kwargs = dict(spec.get("kwargs", {}))
        return _construct_graph_module(metadata, mesh=mesh, column=column, kwargs=kwargs)

    def _build_diagnostics(self) -> dict[str, nn.Module]:
        modules: dict[str, nn.Module] = {}
        for name, spec in self.diagnostic_specs.items():
            type_name = str(spec.get("type", ""))
            metadata = _graph_metadata("diagnostic", type_name)
            if metadata is None:
                raise ValueError(f"unknown graph diagnostic type {type_name!r} for diagnostic {name!r}")
            observes = _as_config_dict(spec.get("observes", {}))
            kwargs = dict(spec.get("kwargs", {}))
            if "component" in observes:
                kwargs.setdefault("target", observes["component"])
            modules[name] = _construct_graph_module(metadata, kwargs=kwargs)
        return modules

    def _build_observers(self) -> dict[str, nn.Module]:
        modules: dict[str, nn.Module] = {}
        for name, spec in self.observer_specs.items():
            type_name = str(spec.get("type", ""))
            metadata = _graph_metadata("observer", type_name)
            if metadata is None:
                raise ValueError(f"unknown graph observer type {type_name!r} for observer {name!r}")
            observes = _as_config_dict(spec.get("observes", {}))
            kwargs = dict(spec.get("kwargs", {}))
            if "component" in observes:
                kwargs.setdefault("target", observes["component"])
            if "exports" in observes:
                kwargs.setdefault("fields", observes["exports"])
            modules[name] = _construct_graph_module(metadata, mesh=self.mesh, kwargs=kwargs)
        return modules

    def _build_initializer(self, spec: Mapping[str, Any]) -> nn.Module:
        type_name = str(spec.get("type", ""))
        metadata = _graph_metadata("initializer", type_name)
        if metadata is None:
            raise ValueError(f"unknown graph initializer type {type_name!r}")
        return metadata.cls(kwargs=dict(spec.get("kwargs", {})))

    def _validate_component_specs(self) -> None:
        for name, spec in self.component_specs.items():
            type_name = str(spec.get("type", ""))
            if not type_name:
                raise ValueError(f"component {name!r} requires a non-empty type")
            if _graph_metadata("component", type_name) is None:
                raise ValueError(f"unknown graph component type {spec.get('type')!r} for component {name!r}")
            declared_state_type = _configured_state_type(spec)
            default_state_type = _default_state_type_for_component(type_name)
            if declared_state_type is not None and default_state_type is not None and declared_state_type != default_state_type:
                raise ValueError(
                    f"component {name!r} declares state type {declared_state_type!r}, "
                    f"but {type_name!r} expects {default_state_type!r}"
                )

    def _validate_wiring(self) -> None:
        names = set(self.component_specs)
        for name, spec in self.component_specs.items():
            for local, import_spec in dict(spec.get("imports", {})).items():
                source_component = str(dict(import_spec).get("component", ""))
                source_export = str(dict(import_spec).get("export", ""))
                if source_component not in names:
                    raise ValueError(f"component {name!r} import {local!r} references unknown component {source_component!r}")
                if source_export not in self._available_exports(source_component):
                    raise ValueError(
                        f"component {name!r} import {local!r} references missing export "
                        f"{source_component!r}.{source_export!r}"
                    )

            provides = dict(spec.get("provides", {}))
            for target_component, by_variable in provides.items():
                target = str(target_component)
                if target not in names:
                    raise ValueError(f"component {name!r} provides terms for unknown component {target!r}")
                target_owns = set(dict(self.component_specs[target].get("owns", {})))
                for variable in dict(by_variable):
                    if str(variable) not in target_owns:
                        raise ValueError(
                            f"component {name!r} provides terms for {target!r}.{variable!s}, "
                            f"but {target!r} does not own that variable"
                        )
        self._validate_import_cycles()

    def _validate_import_cycles(self) -> None:
        graph = {name: set() for name in self.component_specs}
        for name, spec in self.component_specs.items():
            for import_spec in dict(spec.get("imports", {})).values():
                graph[name].add(str(dict(import_spec).get("component", "")))
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str, path: tuple[str, ...]) -> None:
            if name in visiting:
                raise ValueError(f"component import cycle detected: {' -> '.join(path + (name,))}")
            if name in visited:
                return
            visiting.add(name)
            for other in graph[name]:
                if other in graph:
                    visit(other, path + (name,))
            visiting.remove(name)
            visited.add(name)

        for name in graph:
            visit(name, ())

    def _validate_diagnostics(self) -> None:
        for name, spec in self.diagnostic_specs.items():
            if not str(spec.get("type", "")):
                raise ValueError(f"diagnostic {name!r} requires a non-empty type")
            if _graph_metadata("diagnostic", str(spec.get("type"))) is None:
                raise ValueError(f"unknown graph diagnostic type {spec.get('type')!r} for diagnostic {name!r}")
            target = str(getattr(self.diagnostic_components[name], "target", ""))
            if target not in self.components:
                raise ValueError(f"diagnostic {name!r} observes unknown component {target!r}")

    def _validate_observers(self) -> None:
        for name, spec in self.observer_specs.items():
            if not str(spec.get("type", "")):
                raise ValueError(f"observer {name!r} requires a non-empty type")
            if _graph_metadata("observer", str(spec.get("type"))) is None:
                raise ValueError(f"unknown graph observer type {spec.get('type')!r} for observer {name!r}")
            observer = self.observer_components[name]
            target = str(getattr(observer, "target", ""))
            if target not in self.components:
                raise ValueError(f"observer {name!r} observes unknown component {target!r}")
            fields = getattr(observer, "fields", None)
            if fields is None:
                continue
            available = self._available_exports(target)
            for field in fields:
                if str(field) not in available:
                    raise ValueError(f"observer {name!r} observes missing export {target!r}.{field!s}")

    def _validate_initialization_spec(self) -> None:
        if not self.initialization_spec:
            return
        if "states" in self.initialization_spec:
            states = _as_config_dict(self.initialization_spec.get("states"))
            for target, state_spec in states.items():
                self._validate_one_initialization_spec(str(target), _as_config_dict(state_spec))
            return
        target = str(self.initialization_spec.get("target", ""))
        if not target:
            target = self._single_state_component()
        self._validate_one_initialization_spec(target, self.initialization_spec)

    def _validate_one_initialization_spec(self, target: str, spec: Mapping[str, Any]) -> None:
        if target not in self.component_specs:
            raise ValueError(f"initializer target references unknown component {target!r}")
        if not self._component_owns_state(target):
            raise ValueError(f"initializer target {target!r} does not own state variables")
        type_name = str(spec.get("type", ""))
        if not type_name:
            raise ValueError(f"initializer for target {target!r} requires a non-empty type")
        metadata = _graph_metadata("initializer", type_name)
        if metadata is None:
            raise ValueError(f"unknown graph initializer type {type_name!r} for target {target!r}")
        expected = self._component_state_type(target)
        if expected is not None and metadata.state_type is not None and expected != metadata.state_type:
            raise ValueError(
                f"initializer {type_name!r} builds {metadata.state_type!r}, "
                f"but component {target!r} expects {expected!r}"
            )

    def _compile_schedule(self, schedule_spec: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        forward = list(dict(schedule_spec or {}).get("forward", ()))
        if not forward:
            raise ValueError("ComponentGraphModel config requires schedule.forward")
        return self._compile_schedule_items(forward, path="schedule.forward")

    def _compile_schedule_items(self, items: list[Any] | tuple[Any, ...], *, path: str) -> tuple[dict[str, Any], ...]:
        compiled = []
        for index, raw_item in enumerate(items):
            item_path = f"{path}[{index}]"
            item = _as_config_dict(raw_item)
            op = str(item.get("op", ""))
            if op == "integrate":
                compiled.append(self._compile_integrate_step(item, path=item_path))
            elif op == "update":
                compiled.append(self._compile_update_step(item, path=item_path))
            elif op == "repeat":
                compiled.append(self._compile_repeat_step(item, path=item_path))
            elif op == "evaluate":
                compiled.append(self._compile_evaluate_step(item, path=item_path))
            else:
                raise ValueError(
                    f"unsupported schedule op {op!r} at {item_path}; "
                    "supported ops are 'integrate', 'update', 'repeat', and 'evaluate'"
                )
        return tuple(compiled)

    def _compile_integrate_step(self, item: Mapping[str, Any], *, path: str) -> dict[str, Any]:
        target = str(item.get("component", ""))
        if target not in self.components:
            raise ValueError(f"integrate step at {path} references unknown component {target!r}")
        if not self._component_owns_state(target):
            raise ValueError(f"integrate step at {path} target {target!r} does not own state variables")
        method = str(item.get("method", "ssprk3"))
        integrator = _INTEGRATORS.get(method)
        if integrator is None:
            raise ValueError(f"unknown graph integrator {method!r} at {path}")
        providers = tuple(str(name) for name in item.get("terms_from", ()))
        if not providers:
            raise ValueError(f"integrate step at {path} requires non-empty terms_from")
        for provider in providers:
            if provider not in self.components:
                raise ValueError(f"integrate step at {path} references unknown terms_from provider {provider!r}")
            provides = dict(self.component_specs[provider].get("provides", {}))
            if target not in provides:
                raise ValueError(f"provider {provider!r} does not provide terms for target component {target!r}")
        return {
            "op": "integrate",
            "component": target,
            "integrator": integrator,
            "providers": providers,
            "dt": item.get("dt", self.runtime.get("dt_seconds")),
        }

    def _compile_update_step(self, item: Mapping[str, Any], *, path: str) -> dict[str, Any]:
        target = str(item.get("component", item.get("target", "")))
        if target not in self.components:
            raise ValueError(f"update step at {path} references unknown component {target!r}")
        if not self._component_owns_state(target):
            raise ValueError(f"update step at {path} target {target!r} does not own state variables")
        component = self.components[target]
        if not _overrides_method(component, GraphComponentAPI, "update_state"):
            raise ValueError(f"update step at {path} target {target!r} does not implement update_state(...)")
        return {
            "op": "update",
            "component": target,
            "dt": item.get("dt", self.runtime.get("dt_seconds")),
        }

    def _compile_repeat_step(self, item: Mapping[str, Any], *, path: str) -> dict[str, Any]:
        raw_count = item.get("n", 0)
        if isinstance(raw_count, bool) or (isinstance(raw_count, float) and not raw_count.is_integer()):
            raise ValueError(f"repeat step at {path} requires integer n")
        try:
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"repeat step at {path} requires integer n") from exc
        if count < 1:
            raise ValueError(f"repeat step at {path} requires n >= 1")
        steps = list(item.get("steps", ()))
        if not steps:
            raise ValueError(f"repeat step at {path} requires non-empty steps")
        return {
            "op": "repeat",
            "n": count,
            "steps": self._compile_schedule_items(steps, path=f"{path}.steps"),
        }

    def _compile_evaluate_step(self, item: Mapping[str, Any], *, path: str) -> dict[str, Any]:
        component = str(item.get("component", item.get("target", "")))
        if not component:
            raise ValueError(f"evaluate step at {path} requires component")
        output_name = str(item.get("name", component))
        if not output_name:
            raise ValueError(f"evaluate step at {path} requires non-empty name")
        if component in self.components:
            kind = "component"
        elif component in self.diagnostic_components:
            kind = "diagnostic"
        elif component in self.observer_components:
            kind = "observer"
        else:
            raise ValueError(f"evaluate step at {path} references unknown component/diagnostic/observer {component!r}")
        return {"op": "evaluate", "name": output_name, "component": component, "kind": kind}

    def _component_owns_state(self, component: str) -> bool:
        return bool(dict(self.component_specs[str(component)].get("owns", {})))

    def _available_exports(self, component: str) -> set[str]:
        name = str(component)
        spec = self.component_specs[name]
        available = set(dict(spec.get("exports", {})))
        available.update(dict(spec.get("owns", {})))
        metadata = _graph_metadata("component", str(spec.get("type", "")))
        if metadata is not None:
            available.update(metadata.default_exports)
        return available

    def _state_component_names(self) -> tuple[str, ...]:
        return tuple(name for name in self.component_specs if self._component_owns_state(name))

    def _single_state_component(self) -> str:
        names = self._state_component_names()
        if len(names) != 1:
            raise ValueError(
                "plain State input is only valid for single-state graphs; "
                f"this graph owns state for {names}"
            )
        return names[0]

    def _as_graph_state(self, state: Any) -> tuple[GraphState, bool]:
        if isinstance(state, GraphState):
            self._validate_graph_state_slots(state)
            return state, False
        graph_state = GraphState({self._single_state_component(): state})
        self._validate_graph_state_slots(graph_state)
        return graph_state, True

    def initial_state(self, *, dtype=None, device=None) -> Any:
        """Build the YAML-declared initial state.

        Single-state graphs return the component state directly. Multistate
        graphs return ``GraphState`` with one slot per state-owning component.
        """

        if not self.initialization_spec:
            raise ValueError("ComponentGraphModel config has no initialization section")
        if "states" in self.initialization_spec:
            states = {
                str(target): self._build_initial_state_for_target(str(target), _as_config_dict(spec), dtype=dtype, device=device)
                for target, spec in _as_config_dict(self.initialization_spec.get("states")).items()
            }
            graph_state = GraphState(states)
            self.validate_state(graph_state)
            return graph_state

        target = str(self.initialization_spec.get("target", ""))
        if not target:
            target = self._single_state_component()
        if len(self._state_component_names()) != 1:
            raise ValueError("multistate graph initialization must use initialization.states")
        state = self._build_initial_state_for_target(target, self.initialization_spec, dtype=dtype, device=device)
        self.validate_state(state)
        return state

    def _build_initial_state_for_target(self, target: str, spec: Mapping[str, Any], *, dtype=None, device=None) -> Any:
        initializer = self._build_initializer(spec)
        if not hasattr(initializer, "build"):
            raise ValueError(f"initializer {type(initializer).__name__} does not expose build(...)")
        state = initializer.build(self, str(target), dtype=dtype, device=device)
        self._validate_component_state(str(target), state)
        return state

    def save_state_snapshot(
        self,
        path: str | Path,
        state: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Write a typed graph-state restart snapshot.

        The snapshot stores only component-owned state tensors and lightweight
        metadata. Geometry still comes from the receiving ``ComponentGraphModel``
        when the snapshot is loaded, matching ``State.load_netcdf(...)``.
        """

        graph_state = self.validate_state(state)
        payload = {
            "format": _GRAPH_STATE_SNAPSHOT_FORMAT,
            "version": _GRAPH_STATE_SNAPSHOT_VERSION,
            "metadata": _snapshot_clean_mapping(metadata or {}),
            "state_slots": list(self._state_component_names()),
            "states": {
                name: self._serialize_component_state(name, graph_state[name])
                for name in self._state_component_names()
            },
        }
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out_path)
        return str(out_path)

    def load_state_snapshot(self, path: str | Path, *, dtype=None, device=None) -> Any:
        """Load a typed graph-state restart snapshot for this model."""

        payload = _load_graph_state_snapshot_payload(path)
        snapshot_slots = set(str(name) for name in payload["states"])
        expected_slots = set(self._state_component_names())
        missing = sorted(expected_slots - snapshot_slots)
        extra = sorted(snapshot_slots - expected_slots)
        if missing:
            raise ValueError(f"graph state snapshot is missing state slot(s): {missing}")
        if extra:
            raise ValueError(f"graph state snapshot has unexpected state slot(s): {extra}")
        graph_state = GraphState(
            {
                name: self._deserialize_component_state(name, payload["states"][name], dtype=dtype, device=device)
                for name in self._state_component_names()
            }
        )
        self.validate_state(graph_state)
        return self._unwrap_graph_state(graph_state, len(self._state_component_names()) == 1)

    @staticmethod
    def state_snapshot_metadata(path: str | Path) -> dict[str, Any]:
        """Return metadata from a typed graph-state restart snapshot."""

        payload = _load_graph_state_snapshot_payload(path)
        return dict(payload.get("metadata", {}))

    def _serialize_component_state(self, component: str, state: Any) -> dict[str, Any]:
        state_type = self._component_state_type(component)
        if state_type == "State":
            if not isinstance(state, State):
                raise TypeError(f"component {component!r} expected State, got {type(state).__name__}")
            return {
                "state_type": "State",
                "attrs": _snapshot_clean_mapping(state.attrs or {}),
                "tensors": {
                    "u_e": _snapshot_tensor(state.u_e),
                    "mtheta_i": _snapshot_tensor(state.mtheta_i),
                    "ps": _snapshot_tensor(state.ps),
                    "tracer_m": None if state.tracer_m is None else _snapshot_tensor(state.tracer_m),
                },
            }
        if state_type == "OceanState":
            from vfs.ocean import OceanState

            if not isinstance(state, OceanState):
                raise TypeError(f"component {component!r} expected OceanState, got {type(state).__name__}")
            return {
                "state_type": "OceanState",
                "attrs": _snapshot_clean_mapping(state.attrs or {}),
                "tensors": {
                    "u_e": _snapshot_tensor(state.u_e),
                    "h": _snapshot_tensor(state.h),
                    "temp_h": _snapshot_tensor(state.temp_h),
                    "salt_h": _snapshot_tensor(state.salt_h),
                    "tracer_h": None if state.tracer_h is None else _snapshot_tensor(state.tracer_h),
                },
            }
        if state_type == "LandSurfaceState":
            from vfs.land import LandSurfaceState

            if not isinstance(state, LandSurfaceState):
                raise TypeError(f"component {component!r} expected LandSurfaceState, got {type(state).__name__}")
            return {
                "state_type": "LandSurfaceState",
                "attrs": {},
                "tensors": {
                    "soil_liquid_water": _snapshot_tensor(state.soil_liquid_water),
                    "soil_temperature": _snapshot_tensor(state.soil_temperature),
                    "snow_water_equivalent": _snapshot_tensor(state.snow_water_equivalent),
                    "skin_temperature": _snapshot_tensor(state.skin_temperature),
                    "soil_ice": _snapshot_tensor(state.soil_ice),
                    "canopy_water": _snapshot_tensor(state.canopy_water),
                    "land_use_type": _snapshot_tensor(state.land_use_type),
                },
            }
        raise ValueError(f"component {component!r} has unsupported graph snapshot state type {state_type!r}")

    def _deserialize_component_state(
        self,
        component: str,
        payload: Mapping[str, Any],
        *,
        dtype=None,
        device=None,
    ) -> Any:
        expected_type = self._component_state_type(component)
        actual_type = str(payload.get("state_type", ""))
        if actual_type != expected_type:
            raise ValueError(
                f"graph state snapshot slot {component!r} has state_type={actual_type!r}; "
                f"expected {expected_type!r}"
            )
        tensors = _as_config_dict(payload.get("tensors"))
        attrs = dict(payload.get("attrs", {}))
        component_column = _component_column(self, component, required=expected_type in ("State", "OceanState"))
        target_dtype, target_device = _resolve_dtype_device(
            mesh=self.mesh,
            column=component_column,
            dtype=dtype,
            device=device,
        )
        if expected_type == "State":
            u_e = _restore_float_tensor(tensors["u_e"], dtype=target_dtype, device=target_device)
            mtheta_i = _restore_float_tensor(tensors["mtheta_i"], dtype=target_dtype, device=target_device)
            ps = _restore_float_tensor(tensors["ps"], dtype=target_dtype, device=target_device)
            tracer_m = None if tensors.get("tracer_m") is None else _restore_float_tensor(tensors["tracer_m"], dtype=target_dtype, device=target_device)
            return State(
                u_e,
                mtheta_i,
                ps,
                tracer_m,
                mesh=self.mesh,
                column=component_column,
                attrs=attrs,
            )
        if expected_type == "OceanState":
            from vfs.ocean import OceanState

            return OceanState(
                _restore_float_tensor(tensors["u_e"], dtype=target_dtype, device=target_device),
                _restore_float_tensor(tensors["h"], dtype=target_dtype, device=target_device),
                _restore_float_tensor(tensors["temp_h"], dtype=target_dtype, device=target_device),
                _restore_float_tensor(tensors["salt_h"], dtype=target_dtype, device=target_device),
                None if tensors.get("tracer_h") is None else _restore_float_tensor(tensors["tracer_h"], dtype=target_dtype, device=target_device),
                mesh=self.mesh,
                column=component_column,
                attrs=attrs,
            )
        if expected_type == "LandSurfaceState":
            from vfs.land import LandSurfaceState

            return LandSurfaceState(
                _restore_float_tensor(tensors["soil_liquid_water"], dtype=target_dtype, device=target_device),
                _restore_float_tensor(tensors["soil_temperature"], dtype=target_dtype, device=target_device),
                _restore_float_tensor(tensors["snow_water_equivalent"], dtype=target_dtype, device=target_device),
                _restore_float_tensor(tensors["skin_temperature"], dtype=target_dtype, device=target_device),
                soil_ice=_restore_float_tensor(tensors["soil_ice"], dtype=target_dtype, device=target_device),
                canopy_water=_restore_float_tensor(tensors["canopy_water"], dtype=target_dtype, device=target_device),
                land_use_type=_restore_integral_tensor(tensors["land_use_type"], device=target_device),
            )
        raise ValueError(f"component {component!r} has unsupported graph snapshot state type {expected_type!r}")

    def validate_state(self, state: Any) -> GraphState:
        """Validate state slots and owned fields, returning normalized graph state."""

        graph_state, _ = self._as_graph_state(state)
        for name in self._state_component_names():
            self._validate_component_state(name, graph_state[name])
        return graph_state

    def _validate_graph_state_slots(self, state: GraphState) -> None:
        expected = set(self._state_component_names())
        actual = set(str(name) for name in state.keys())
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            raise ValueError(f"GraphState is missing state slot(s): {missing}")
        if extra:
            raise ValueError(f"GraphState has unexpected state slot(s): {extra}")

    def _unwrap_graph_state(self, state: GraphState, unwrap: bool) -> Any:
        if not unwrap:
            return state
        return state[self._single_state_component()]

    def _state_for_component(self, component: str, state: Any) -> Any:
        name = str(component)
        if isinstance(state, GraphState):
            if name not in state:
                raise ValueError(f"GraphState is missing state for component {name!r}")
            component_state = state[name]
            self._validate_component_state(name, component_state)
            return component_state
        if name != self._single_state_component():
            raise ValueError(f"plain State input cannot supply component {name!r}")
        self._validate_component_state(name, state)
        return state

    def _validate_component_state(self, component: str, state: Any) -> None:
        name = str(component)
        if not self._component_owns_state(name):
            return
        spec = self.component_specs[name]
        expected_state_type = self._component_state_type(name)
        if expected_state_type == "State":
            if not isinstance(state, State):
                raise TypeError(f"component {name!r} expected State, got {type(state).__name__}")
            state.validate_schema()
        elif expected_state_type == "OceanState":
            from vfs.ocean import OceanState

            if not isinstance(state, OceanState):
                raise TypeError(f"component {name!r} expected OceanState, got {type(state).__name__}")
            state.validate()
        elif expected_state_type == "LandSurfaceState":
            from vfs.land import LandSurfaceState

            if not isinstance(state, LandSurfaceState):
                raise TypeError(f"component {name!r} expected LandSurfaceState, got {type(state).__name__}")
        for field_name, field_spec in dict(spec.get("owns", {})).items():
            self._validate_owned_field(name, state, str(field_name), _as_config_dict(field_spec))

    def _component_state_type(self, component: str) -> str | None:
        spec = self.component_specs[str(component)]
        configured = _configured_state_type(spec)
        if configured is not None:
            return configured
        return _default_state_type_for_component(str(spec.get("type", "")))

    def _validate_owned_field(self, component: str, state: Any, field_name: str, field_spec: Mapping[str, Any]) -> None:
        optional = bool(field_spec.get("optional", False))
        if not hasattr(state, field_name):
            if optional:
                return
            raise ValueError(f"component {component!r} requires owned field {field_name!r}")
        value = getattr(state, field_name)
        if value is None:
            if optional:
                return
            raise ValueError(f"component {component!r} requires owned field {field_name!r}")
        if isinstance(value, Field):
            value = value.values
        if not torch.is_tensor(value):
            raise TypeError(
                f"component {component!r} owned field {field_name!r} must be a tensor, "
                f"got {type(value).__name__}"
            )
        self._validate_owned_field_shape(component, state, field_name, value)
        if field_name in ("tracer_m", "tracer_h"):
            self._validate_owned_tracer_metadata(component, state, field_name, field_spec)

    def _validate_owned_field_shape(self, component: str, state: Any, field_name: str, value: torch.Tensor) -> None:
        state_type = self._component_state_type(component)
        if state_type == "State":
            mesh = state._require_mesh()
            column = state._require_column()
            expected = {
                "u_e": (int(mesh.nEdges), int(column.n_levels)),
                "mtheta_i": (int(mesh.nCells), int(column.n_levels) + 1),
                "ps": (int(mesh.nCells), 1),
            }.get(field_name)
            if expected is not None:
                _require_trailing_shape(f"{component}.{field_name}", value, expected)
                return
            if field_name == "tracer_m":
                _require_tracer_shape(f"{component}.{field_name}", value, int(mesh.nCells), int(column.n_levels))
        elif state_type == "OceanState":
            mesh = state._require_mesh()
            column = state._require_column()
            expected = {
                "u_e": (int(mesh.nEdges), int(column.n_layers)),
                "h": (int(mesh.nCells), int(column.n_layers)),
                "temp_h": (int(mesh.nCells), int(column.n_layers)),
                "salt_h": (int(mesh.nCells), int(column.n_layers)),
            }.get(field_name)
            if expected is not None:
                _require_trailing_shape(f"{component}.{field_name}", value, expected)
                return
            if field_name == "tracer_h":
                _require_tracer_shape(f"{component}.{field_name}", value, int(mesh.nCells), int(column.n_layers))
        elif state_type == "LandSurfaceState":
            n_cells = int(self.mesh.nCells) if self.mesh is not None and hasattr(self.mesh, "nCells") else None
            n_layers = int(state.n_soil_layers)
            if n_cells is None:
                return
            expected = {
                "soil_liquid_water": (n_cells, n_layers),
                "soil_temperature": (n_cells, n_layers),
                "soil_ice": (n_cells, n_layers),
                "snow_water_equivalent": (n_cells, 1),
                "skin_temperature": (n_cells, 1),
                "canopy_water": (n_cells, 1),
                "land_use_type": (n_cells, 1),
            }.get(field_name)
            if expected is not None:
                _require_trailing_shape(f"{component}.{field_name}", value, expected)

    def _validate_owned_tracer_metadata(
        self,
        component: str,
        state: Any,
        field_name: str,
        field_spec: Mapping[str, Any],
    ) -> None:
        if field_name != "tracer_m":
            return
        requirements = _tracer_requirements_from_spec(field_spec)
        if not requirements:
            return
        spec = state.spec
        for requirement in requirements:
            tracer = spec.tracer(str(requirement["name"]))
            if tracer is None:
                raise ValueError(f"component {component!r} requires named tracer {requirement['name']!r}")
            for key in ("index", "quantity", "role", "units"):
                if key in requirement and requirement[key] is not None:
                    actual = getattr(tracer, key)
                    expected = requirement[key]
                    if key == "index":
                        actual = int(actual)
                        expected = int(expected)
                    else:
                        actual = str(actual)
                        expected = str(expected)
                    if actual != expected:
                        raise ValueError(
                            f"component {component!r} requires tracer {requirement['name']!r} "
                            f"{key}={expected!r}, got {actual!r}"
                        )

    def bind_state(self, component: str, state: Any) -> Any:
        name = str(component)
        target = self.components[name]
        component_state = self._state_for_component(name, state)
        if hasattr(target, "bind_state"):
            imports = self.component_imports(name, state)
            return target.bind_state(component_state, mesh=self.mesh, imports=imports)
        return component_state

    def component_imports(self, component: str, state: Any) -> dict[str, Any]:
        name = str(component)
        if name not in self.component_specs:
            raise ValueError(f"unknown component {name!r}")
        imports: dict[str, Any] = {}
        for local_name, import_spec in dict(self.component_specs[name].get("imports", {})).items():
            spec = dict(import_spec)
            source_component = str(spec.get("component", ""))
            source_export = str(spec.get("export", ""))
            source_exports = self.component_exports(source_component, state)
            imports[str(local_name)] = _require_export(source_exports, source_component, source_export)
        return imports

    def component_exports(self, component: str, state: Any) -> dict[str, Any]:
        name = str(component)
        target = self.components[name]
        if not hasattr(target, "exports"):
            raise ValueError(f"component {name!r} does not expose exports")
        imports = self.component_imports(name, state)
        component_state = self._state_for_component(name, state) if self._component_owns_state(name) else None
        bound = (
            target.bind_state(component_state, mesh=self.mesh, imports=imports)
            if component_state is not None and hasattr(target, "bind_state")
            else component_state
        )
        return target.exports(bound, mesh=self.mesh, imports=imports)

    def step(self, state: Any, dt: float | None = None) -> Any:
        current, unwrap = self._as_graph_state(state)
        self.validate_state(current)
        self.last_evaluation_outputs = {}
        current = self._execute_schedule(current, self.schedule, dt_override=dt)
        return self._unwrap_graph_state(current, unwrap)

    def _execute_schedule(
        self,
        current: GraphState,
        schedule: tuple[dict[str, Any], ...],
        *,
        dt_override: float | None = None,
    ) -> GraphState:
        for item in schedule:
            op = str(item["op"])
            if op == "integrate":
                current = self._execute_integrate_step(current, item, dt_override=dt_override)
            elif op == "update":
                current = self._execute_update_step(current, item, dt_override=dt_override)
            elif op == "repeat":
                for _ in range(int(item["n"])):
                    current = self._execute_schedule(current, item["steps"], dt_override=dt_override)
            elif op == "evaluate":
                self._execute_evaluate_step(current, item)
            else:  # pragma: no cover - compile-time validation should prevent this.
                raise ValueError(f"unknown compiled schedule op {op!r}")
        return current

    def _execute_integrate_step(
        self,
        current: GraphState,
        item: Mapping[str, Any],
        *,
        dt_override: float | None = None,
    ) -> GraphState:
        step_dt = _resolve_dt(item["dt"], self.runtime, override=dt_override)
        target = str(item["component"])
        target_state = self.bind_state(target, current)
        current = current.with_state(target, target_state)
        provider = _GraphTendencyProvider(self, target, tuple(item["providers"]), current)
        next_state = item["integrator"](provider, target_state, step_dt)
        current = current.with_state(target, self.bind_state(target, current.with_state(target, next_state)))
        self._validate_component_state(target, current[target])
        return current

    def _execute_update_step(
        self,
        current: GraphState,
        item: Mapping[str, Any],
        *,
        dt_override: float | None = None,
    ) -> GraphState:
        step_dt = _resolve_dt(item["dt"], self.runtime, override=dt_override)
        target = str(item["component"])
        component = self.components[target]
        target_state = self.bind_state(target, current)
        current = current.with_state(target, target_state)
        imports = self.component_imports(target, current)
        next_state = component.update_state(target_state, dt=step_dt, mesh=self.mesh, imports=imports)
        current = current.with_state(target, self.bind_state(target, current.with_state(target, next_state)))
        self._validate_component_state(target, current[target])
        return current

    def _execute_evaluate_step(self, current: GraphState, item: Mapping[str, Any]) -> None:
        output_name = str(item["name"])
        component = str(item["component"])
        kind = str(item["kind"])
        if kind == "component":
            result = self.components[component].evaluate(self, current)
            if result is None:
                result = self.component_exports(component, current)
        elif kind == "diagnostic":
            result = self.diagnostic_components[component].row(self, None, current)
        elif kind == "observer":
            result = self.observer_components[component].observe(self, current)
        else:  # pragma: no cover - compile-time validation should prevent this.
            raise ValueError(f"unknown evaluate step kind {kind!r}")
        self.last_evaluation_outputs[output_name] = result

    def forward(self, state: Any, dt: float | None = None) -> Any:
        return self.step(state, dt=dt)

    def diagnostics_row(
        self,
        state0: Any,
        state: Any | None = None,
        *,
        t: float | None = None,
        dt: float | None = None,
        step: int | None = None,
        day: float | None = None,
        diagnostics: tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, Any]:
        if state is None:
            state = state0
            state0_for_row = None
        else:
            state0_for_row = state0
        names = tuple(self.diagnostic_components) if diagnostics is None else tuple(str(name) for name in diagnostics)
        row: dict[str, Any] = {}
        for name in names:
            if name not in self.diagnostic_components:
                raise ValueError(f"unknown graph diagnostic {name!r}")
            payload = self.diagnostic_components[name].row(
                self,
                state0_for_row,
                state,
                t=t,
                dt=dt,
                step=step,
                day=day,
            )
            row.update(payload)
        return row

    def observe(self, state: Any, observer: str | None = None) -> dict[str, Any]:
        if not self.observer_components:
            raise ValueError("ComponentGraphModel has no observers configured")
        if observer is not None:
            name = str(observer)
            if name not in self.observer_components:
                raise ValueError(f"unknown graph observer {name!r}")
            return self.observer_components[name].observe(self, state)
        if len(self.observer_components) == 1:
            return next(iter(self.observer_components.values())).observe(self, state)
        return {name: component.observe(self, state) for name, component in self.observer_components.items()}


def _as_config_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"expected mapping config, got {type(value).__name__}")


def _snapshot_tensor(value: torch.Tensor | Field) -> torch.Tensor:
    tensor = value.values if isinstance(value, Field) else value
    return tensor.detach().to(device="cpu").clone()


def _snapshot_clean_value(value: Any) -> Any:
    if isinstance(value, Field):
        return _snapshot_tensor(value)
    if torch.is_tensor(value):
        return _snapshot_tensor(value)
    if isinstance(value, Mapping):
        return {str(name): _snapshot_clean_value(child) for name, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot_clean_value(child) for child in value]
    return value


def _snapshot_clean_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(name): _snapshot_clean_value(child) for name, child in dict(value).items()}


def _load_graph_state_snapshot_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), weights_only=False, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"graph state snapshot must contain a mapping payload, got {type(payload).__name__}")
    payload = dict(payload)
    if payload.get("format") != _GRAPH_STATE_SNAPSHOT_FORMAT:
        raise ValueError(
            f"unsupported graph state snapshot format {payload.get('format')!r}; "
            f"expected {_GRAPH_STATE_SNAPSHOT_FORMAT!r}"
        )
    if int(payload.get("version", -1)) != _GRAPH_STATE_SNAPSHOT_VERSION:
        raise ValueError(
            f"unsupported graph state snapshot version {payload.get('version')!r}; "
            f"expected {_GRAPH_STATE_SNAPSHOT_VERSION}"
        )
    if not isinstance(payload.get("states"), Mapping):
        raise ValueError("graph state snapshot is missing mapping field 'states'")
    return payload


def _resolve_snapshot_dtype(dtype: Any, fallback: torch.dtype) -> torch.dtype:
    if dtype is None:
        return fallback
    if isinstance(dtype, torch.dtype):
        return dtype
    return getattr(torch, str(dtype).replace("torch.", ""))


def _resolve_snapshot_device(device: Any) -> torch.device | None:
    if device is None:
        return None
    return torch.device(device)


def _restore_float_tensor(value: Any, *, dtype=None, device=None) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float64)
    target_dtype = _resolve_snapshot_dtype(dtype, tensor.dtype)
    target_device = _resolve_snapshot_device(device)
    kwargs: dict[str, Any] = {"dtype": target_dtype}
    if target_device is not None:
        kwargs["device"] = target_device
    return tensor.to(**kwargs)


def _restore_integral_tensor(value: Any, *, device=None) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    target_device = _resolve_snapshot_device(device)
    if target_device is not None:
        tensor = tensor.to(device=target_device)
    return tensor.to(dtype=torch.long)


def _graph_metadata(kind: str, name: str) -> GraphComponentMetadata | None:
    return _GRAPH_REGISTRY[str(kind)].get(str(name))


def _construct_graph_module(
    metadata: GraphComponentMetadata,
    *,
    mesh=None,
    column=None,
    kwargs: Mapping[str, Any] | None = None,
) -> nn.Module:
    constructor = metadata.cls
    merged = dict(kwargs or {})
    if metadata.requires_mesh and metadata.requires_column:
        return constructor(mesh=mesh, column=column, kwargs=merged)
    if metadata.requires_mesh:
        return constructor(mesh=mesh, kwargs=merged)
    if metadata.requires_column:
        return constructor(column=column, kwargs=merged)
    return constructor(kwargs=merged)


def _overrides_method(module: nn.Module, base: type, method_name: str) -> bool:
    return getattr(type(module), str(method_name), None) is not getattr(base, str(method_name), None)


def _configured_state_type(spec: Mapping[str, Any]) -> str | None:
    if "state_type" in spec:
        return str(spec["state_type"])
    state_spec = spec.get("state")
    if isinstance(state_spec, Mapping) and "type" in state_spec:
        return str(state_spec["type"])
    return None


def _default_state_type_for_component(type_name: str) -> str | None:
    metadata = _graph_metadata("component", str(type_name))
    return None if metadata is None else metadata.state_type


def _runtime_dt_seconds(runtime: Mapping[str, Any], *, override: float | None = None) -> float:
    if override is not None:
        return float(override)
    if "dt_seconds" not in runtime:
        raise ValueError("step dt was not provided and runtime.dt_seconds is absent")
    return float(runtime["dt_seconds"])


def _resolve_dt(value: Any, runtime: Mapping[str, Any], *, override: float | None = None) -> float:
    if value is None:
        return _runtime_dt_seconds(runtime, override=override)
    if isinstance(value, str):
        expr = value.strip()
        match = _RUNTIME_DT_RE.match(expr)
        if match is not None:
            dt = _runtime_dt_seconds(runtime, override=override)
            op, factor_raw = match.groups()
            if op is None:
                return dt
            factor = float(factor_raw)
            if factor == 0.0:
                raise ValueError(f"dt expression {value!r} divides or multiplies by zero")
            return dt / factor if op == "/" else dt * factor
        try:
            return float(expr)
        except ValueError as exc:
            raise ValueError(f"unsupported dt expression {value!r}") from exc
    return float(value)


def _build_dry_dynamics(type_name: str, kwargs: Mapping[str, Any]):
    from vfs.atmosphere import DryHydrostaticDynamics, DryHydrostaticVirtualTemperatureDynamics

    constructors = {
        "DryHydrostaticDynamics": DryHydrostaticDynamics,
        "DryHydrostaticVirtualTemperatureDynamics": DryHydrostaticVirtualTemperatureDynamics,
    }
    constructor = constructors.get(str(type_name))
    if constructor is None:
        raise ValueError(f"unknown DryHydrostaticAtmosphere dynamics_type {type_name!r}")
    return constructor(**dict(kwargs))


def _build_scm_column_model(spec: Any, kwargs: Mapping[str, Any]) -> nn.Module:
    if isinstance(spec, nn.Module):
        if kwargs:
            raise ValueError("SCMForcing column_model kwargs are invalid when column_model is already a module")
        return spec
    if isinstance(spec, Mapping):
        nested = dict(spec)
        type_name = str(nested.pop("type", nested.pop("name", "")))
        merged_kwargs = dict(nested.pop("kwargs", {}))
        merged_kwargs.update(kwargs)
        if nested:
            unknown = ", ".join(sorted(str(name) for name in nested))
            raise ValueError(f"unknown SCMForcing column_model spec keys: {unknown}")
    else:
        type_name = str(spec)
        merged_kwargs = dict(kwargs)

    from vfs.physics import KesslerWarmRainSCM, ToyCondensationSCM, VFSNativeSCM

    constructors = {
        "KesslerWarmRainSCM": KesslerWarmRainSCM,
        "ToyCondensationSCM": ToyCondensationSCM,
        "VFSNativeSCM": VFSNativeSCM,
    }
    constructor = constructors.get(type_name)
    if constructor is None:
        raise ValueError(f"unknown SCMForcing column_model {type_name!r}")
    return constructor(**merged_kwargs)


def _require_model_mesh(model: ComponentGraphModel):
    if model.mesh is None:
        raise ValueError("graph initialization requires model mesh")
    return model.mesh


def _component_column(model: ComponentGraphModel, component: str, *, required: bool = True):
    name = str(component)
    target = model.components[name]
    column = getattr(target, "column", None)
    if column is None:
        column = model.columns_by_name.get(name)
    if column is None and required:
        raise ValueError(f"component {name!r} has no column for initialization")
    return column


def _set_component_column(model: ComponentGraphModel, component: str, column: Any) -> None:
    name = str(component)
    target = model.components[name]
    setattr(target, "column", column)
    model.columns_by_name[name] = column


def _component_dynamics(model: ComponentGraphModel, component: str):
    return getattr(model.components[str(component)], "dynamics", None)


def _resolve_dtype_device(*, mesh=None, column=None, dtype=None, device=None) -> tuple[torch.dtype, torch.device]:
    ref = None
    for owner in (column, mesh):
        if owner is None:
            continue
        for value in getattr(owner, "buffers", lambda: ())():
            ref = value
            break
        if ref is not None:
            break
    if dtype is None:
        dtype = ref.dtype if torch.is_tensor(ref) and ref.is_floating_point() else torch.float64
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if device is None:
        device = ref.device if torch.is_tensor(ref) else torch.device("cpu")
    else:
        device = torch.device(device)
    return dtype, device


def _cell_lat(mesh, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    xyz = mesh.cell_xyz.to(dtype=dtype, device=device)
    return torch.asin(torch.clamp(xyz[:, 2:3], -1.0, 1.0))


def _initial_wind_for_mode(
    mesh,
    column,
    *,
    wind_mode: str,
    initial_wind_noise: float,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    mode = str(wind_mode)
    shape = (int(mesh.nEdges), int(column.n_levels))
    if mode == "none":
        return torch.zeros(shape, dtype=dtype, device=device)
    if mode == "small_noise":
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        return (float(initial_wind_noise) * torch.randn(shape, generator=gen, dtype=torch.float64)).to(dtype=dtype, device=device)
    if mode == "thermal_wind":
        raise NotImplementedError("equilibrium_wind='thermal_wind' is not implemented; choose none or small_noise")
    raise ValueError("wind_mode/equilibrium_wind must be none, small_noise, or thermal_wind")


def _initial_tracers_for_component(
    model: ComponentGraphModel,
    component: str,
    *,
    n_cells: int,
    n_levels: int,
    dtype: torch.dtype,
    device: torch.device,
    vapor_mixing_ratio: Any,
    cloud_liquid_mixing_ratio: Any,
    tracer_initial_values: Mapping[str, Any],
) -> tuple[torch.Tensor | None, StateSpec | None, dict[str, Any]]:
    from vfs.moisture import (
        CLOUD_LIQUID_QUANTITY_ATTR,
        CLOUD_LIQUID_QUANTITY_DRY_MIXING_RATIO,
        CLOUD_LIQUID_TRACER_INDEX_ATTR,
        VAPOR_QUANTITY_ATTR,
        VAPOR_QUANTITY_DRY_MIXING_RATIO,
        VAPOR_TRACER_INDEX_ATTR,
    )

    field_spec = _as_config_dict(dict(model.component_specs[str(component)].get("owns", {})).get("tracer_m", {}))
    requirements = list(_tracer_requirements_from_spec(field_spec))
    if not requirements and vapor_mixing_ratio is None and cloud_liquid_mixing_ratio is None and not tracer_initial_values:
        return None, None, {}
    if not requirements:
        if vapor_mixing_ratio is not None:
            requirements.append(
                {
                    "name": "vapor",
                    "index": 0,
                    "role": "passive",
                    "quantity": VAPOR_QUANTITY_DRY_MIXING_RATIO,
                    "units": "kg kg-1 dry_air",
                }
            )
        if cloud_liquid_mixing_ratio is not None:
            requirements.append(
                {
                    "name": "cloud_liquid",
                    "index": 1 if requirements else 0,
                    "role": "passive",
                    "quantity": CLOUD_LIQUID_QUANTITY_DRY_MIXING_RATIO,
                    "units": "kg kg-1 dry_air",
                }
            )
    specs = tuple(
        TracerSpec(
            name=str(item["name"]),
            index=int(item["index"]),
            role=str(item.get("role", "passive")),
            quantity=str(item.get("quantity", VAPOR_QUANTITY_DRY_MIXING_RATIO)),
            units=str(item.get("units", "kg kg-1 dry_air")),
        )
        for item in requirements
    )
    n_tracers = max((spec.index for spec in specs), default=-1) + 1
    if n_tracers <= 0:
        return None, None, {}
    q = torch.zeros((int(n_cells), int(n_levels), int(n_tracers)), dtype=dtype, device=device)
    attrs: dict[str, Any] = {}
    values = {str(k): v for k, v in dict(tracer_initial_values or {}).items()}
    for spec in specs:
        name = str(spec.name)
        value = values.get(name, 0.0)
        if name == "vapor":
            value = 0.0 if vapor_mixing_ratio is None else vapor_mixing_ratio
            attrs[VAPOR_TRACER_INDEX_ATTR] = int(spec.index)
            attrs[VAPOR_QUANTITY_ATTR] = str(spec.quantity)
            attrs["vapor_units"] = spec.units
        elif name in ("cloud_liquid", "cloud_water"):
            value = 0.0 if cloud_liquid_mixing_ratio is None else cloud_liquid_mixing_ratio
            attrs[CLOUD_LIQUID_TRACER_INDEX_ATTR] = int(spec.index)
            attrs[CLOUD_LIQUID_QUANTITY_ATTR] = str(spec.quantity)
            attrs["cloud_liquid_units"] = spec.units
        q[..., int(spec.index)] = torch.as_tensor(value, dtype=dtype, device=device)
    return q, StateSpec(tracers=specs), attrs


def _column_with_phi_s(base_column: Any, phi_s: Any, *, state: State, mesh=None):
    if base_column is None:
        base_column = state.column
    if base_column is None:
        raise ValueError("cannot bind imported phi_s without a column")

    requested = _as_tensor(phi_s)
    requested = requested.to(dtype=state.ps.dtype, device=state.ps.device)
    if requested.ndim == 2 and int(requested.shape[-1]) == 1:
        requested = requested.squeeze(-1)
    elif requested.ndim > 1:
        requested = requested.reshape(-1)

    n_cells = None if mesh is None or not hasattr(mesh, "nCells") else int(mesh.nCells)
    if requested.ndim > 0:
        requested = requested.reshape(-1)
        if n_cells is not None and int(requested.numel()) != n_cells:
            raise ValueError(f"imported phi_s must have nCells={n_cells} values; got shape {tuple(requested.shape)}")

    if _same_phi_s(base_column.phi_s, requested, state=state, mesh=mesh):
        return base_column

    # CharneyPhillipsColumn currently canonicalizes constructor buffers through
    # float64 CPU tensors; build there, then move back to the active state
    # device/dtype. This keeps MPS float32 graph configs usable.
    from vfs.geometry.column import CharneyPhillipsColumn

    requested_cpu = requested.detach().to(device="cpu", dtype=torch.float64)
    column = CharneyPhillipsColumn(
        int(base_column.n_levels),
        base_column.A_interface.detach().to(device="cpu", dtype=torch.float64),
        base_column.B_interface.detach().to(device="cpu", dtype=torch.float64),
        phi_s=requested_cpu,
        phi_full_mode=base_column.phi_full_mode,
        R_d=base_column.R_d,
        c_p=base_column.c_p,
        p0=base_column.p0,
        g=base_column.g,
    )
    return column.to(device=state.ps.device, dtype=state.ps.dtype)


def _same_phi_s(existing: torch.Tensor, requested: torch.Tensor, *, state: State, mesh=None) -> bool:
    lhs = existing.to(dtype=state.ps.dtype, device=state.ps.device)
    rhs = requested.to(dtype=state.ps.dtype, device=state.ps.device)
    if lhs.ndim == 2 and int(lhs.shape[-1]) == 1:
        lhs = lhs.squeeze(-1)
    if rhs.ndim == 2 and int(rhs.shape[-1]) == 1:
        rhs = rhs.squeeze(-1)
    n_cells = None if mesh is None or not hasattr(mesh, "nCells") else int(mesh.nCells)
    if lhs.ndim == 0 and rhs.ndim > 0 and n_cells is not None:
        lhs = lhs.expand(n_cells)
    if rhs.ndim == 0 and lhs.ndim > 0 and n_cells is not None:
        rhs = rhs.expand(n_cells)
    return bool(torch.equal(lhs.reshape(-1), rhs.reshape(-1)))


def _external_tensor_from_spec(spec: Any, *, mesh=None, column=None) -> torch.Tensor:
    if isinstance(spec, Mapping):
        value = spec.get("value", spec.get("values", 0.0))
        shape_spec = spec.get("shape")
        dtype_name = spec.get("dtype")
    else:
        value = spec
        shape_spec = None
        dtype_name = None
    dtype = getattr(torch, str(dtype_name)) if dtype_name is not None else torch.float64
    tensor = torch.as_tensor(value, dtype=dtype)
    if shape_spec is None:
        return tensor
    shape = _resolve_external_shape(shape_spec, mesh=mesh, column=column)
    if tensor.numel() == 1:
        return torch.full(shape, tensor.reshape(()).item(), dtype=dtype)
    try:
        return torch.broadcast_to(tensor, shape).clone()
    except RuntimeError as exc:
        raise ValueError(f"external field with shape {tuple(tensor.shape)} cannot broadcast to {shape}") from exc


def _resolve_external_shape(shape_spec: Any, *, mesh=None, column=None) -> tuple[int, ...]:
    if isinstance(shape_spec, str):
        shape_items = (shape_spec,)
    else:
        shape_items = tuple(shape_spec)
    context: dict[str, int] = {}
    if mesh is not None:
        for name in ("nCells", "nEdges", "nVertices"):
            if hasattr(mesh, name):
                context[name] = int(getattr(mesh, name))
    if column is not None:
        if hasattr(column, "n_levels"):
            context["nLevels"] = int(column.n_levels)
            context["nInterfaces"] = int(column.n_levels) + 1
        if hasattr(column, "n_layers"):
            context["nLayers"] = int(column.n_layers)
            context.setdefault("nInterfaces", int(column.n_layers) + 1)
    resolved: list[int] = []
    for item in shape_items:
        if isinstance(item, str):
            if item not in context:
                raise ValueError(f"unknown external field shape token {item!r}")
            resolved.append(context[item])
        else:
            resolved.append(int(item))
    return tuple(resolved)


def _require_trailing_shape(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if value.ndim < len(shape) or tuple(int(dim) for dim in value.shape[-len(shape) :]) != tuple(shape):
        raise ValueError(f"{name} must have trailing shape {shape}; got {tuple(value.shape)}")


def _require_tracer_shape(name: str, value: torch.Tensor, n_cells: int, n_levels: int) -> None:
    if value.ndim >= 2 and int(value.shape[-2]) == int(n_cells) and int(value.shape[-1]) == int(n_levels):
        return
    if value.ndim >= 3 and int(value.shape[-3]) == int(n_cells) and int(value.shape[-2]) == int(n_levels):
        return
    raise ValueError(
        f"{name} must have trailing shape (nCells={int(n_cells)}, nLevels={int(n_levels)}) "
        f"or (nCells, nLevels, nTracers); got {tuple(value.shape)}"
    )


def _tracer_requirements_from_spec(field_spec: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = field_spec.get("tracers", field_spec.get("required_tracers"))
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        out = []
        for name, spec in raw.items():
            item = dict(spec or {})
            item.setdefault("name", str(name))
            out.append(item)
        return tuple(out)
    out = []
    for spec in raw:
        item = dict(spec or {})
        if "name" not in item:
            raise ValueError("graph tracer requirements require a name")
        out.append(item)
    return tuple(out)


def _state_reference_tensor(state: Any) -> torch.Tensor:
    for name in ("ps", "h", "skin_temperature", "soil_liquid_water", "u_e"):
        value = getattr(state, name, None)
        if isinstance(value, Field):
            return value.values
        if torch.is_tensor(value):
            return value
    raise ValueError(f"could not find a tensor reference on {type(state).__name__}")


def _require_import(imports: Mapping[str, Any], name: str) -> Any:
    if name not in imports:
        raise ValueError(f"required graph import {name!r} is absent")
    return imports[name]


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, Field):
        return value.values
    return value if torch.is_tensor(value) else torch.as_tensor(value)


def _cell_column_import(
    imports: Mapping[str, Any],
    name: str,
    *,
    like: torch.Tensor,
    default: Any,
) -> torch.Tensor | None:
    from vfs.surface.utils import as_cell_column

    if str(name) in imports:
        value = imports[str(name)]
    elif default is None:
        return None
    else:
        value = default
    return as_cell_column(value, like=like, name=str(name))


def _normalize_level_index(index: int, size: int) -> int:
    idx = int(index)
    if idx < 0:
        idx += int(size)
    if idx < 0 or idx >= int(size):
        raise IndexError(f"level index {index} out of bounds for size {size}")
    return idx


def _select_cell_level(value: torch.Tensor, index: int) -> torch.Tensor:
    if value.ndim == 0:
        return value.reshape(1, 1)
    if value.ndim == 1:
        return value[:, None]
    if int(value.shape[-1]) == 1:
        return value
    idx = _normalize_level_index(index, int(value.shape[-1]))
    return value[..., idx : idx + 1]


def _merge_terms(
    merged: dict[str, dict[str, dict[str, torch.Tensor]]],
    provider_name: str,
    terms: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
) -> None:
    for target, by_variable in terms.items():
        target_terms = merged.setdefault(str(target), {})
        for variable, by_term in by_variable.items():
            variable_terms = target_terms.setdefault(str(variable), {})
            for term_name, tensor in by_term.items():
                variable_terms[f"{provider_name}.{term_name}"] = tensor


def _require_export(exports: Mapping[str, Any], component: str, name: str) -> Any:
    if name not in exports:
        raise ValueError(f"component {component!r} did not produce export {name!r}")
    return exports[name]


def _squeeze_last_singleton(value: torch.Tensor) -> torch.Tensor:
    if value.ndim > 0 and int(value.shape[-1]) == 1:
        return value.squeeze(-1)
    return value


def _safe_key(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _term_stats(
    terms: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    *,
    prefix: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for target, by_variable in terms.items():
        for variable, by_term in by_variable.items():
            for term_name, tensor in by_term.items():
                if isinstance(tensor, Field):
                    tensor = tensor.values
                if not torch.is_tensor(tensor):
                    continue
                detached = tensor.detach()
                key = f"{prefix}_{_safe_key(target)}_{_safe_key(variable)}_{_safe_key(term_name)}"
                out[f"{key}_rms"] = float(detached.square().mean().sqrt().cpu().item())
                out[f"{key}_max_abs"] = float(detached.abs().max().cpu().item())
    return out


__all__ = [
    "AirSeaHeatExchangeForcing",
    "ComponentGraphModel",
    "DryAtmosphereHealthDiagnostics",
    "DryHydrostaticAtmosphere",
    "ExternalFields",
    "GraphComponentAPI",
    "GraphDiagnosticAPI",
    "GraphInitializerAPI",
    "GraphObserverAPI",
    "GraphState",
    "HeldSuarezForcing",
    "HeldSuarezInitializer",
    "JablonskiWilliamsonInitializer",
    "MeshToLatLonObserver",
    "NativeExportObserver",
    "OceanRestInitializer",
    "OceanStateComponent",
    "SCMForcing",
    "WeatherBench2ERA5DryInitializer",
    "graph_component_catalog",
    "graph_component_metadata",
]
