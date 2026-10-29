"""Optional adapters from external PyTorch SCMs into VFS column physics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
from typing import Any

import torch
from torch import nn

from .moisture import (
    CLOUD_LIQUID_QUANTITY_DRY_MIXING_RATIO,
    VAPOR_QUANTITY_DRY_MIXING_RATIO,
    cloud_liquid_tracer_index,
    mixing_ratio_from_specific_humidity,
    specific_humidity_from_mixing_ratio,
    vapor_tracer_index,
)
from .core.state import DEFAULT_GRAVITY, StateRequirement, TracerRequirement


_CLOUD_OPTICAL_STATE_NAMES = (
    "cloud_fraction",
    "cloud_sw_tau_layer",
    "cloud_lw_tau_layer",
)

_REPORTED_EXTERNAL_PARAM_NAMES = (
    "cloud_rh_min",
    "cloud_qc_ref",
    "cloud_cf_qc_min",
    "cloud_cf_rh_power",
    "cloud_cf_qc_power",
    "cloud_cf_max",
    "cloud_qc_max",
    "cloud_liquid_temp",
    "cloud_ice_temp",
    "cloud_k_liq_sw",
    "cloud_k_ice_sw",
    "cloud_k_liq_lw",
    "cloud_k_ice_lw",
    "cloud_sw_scattering_efficiency",
    "cloud_sw_absorption_fraction",
    "cloud_autoconv_qc_thresh",
    "cloud_autoconv_qc_scale",
    "cloud_autoconv_tau",
)


@dataclass(frozen=True)
class ExternalSCMContract:
    """VFS-side semantic contract for an external single-column adapter."""

    vapor_quantity: str = VAPOR_QUANTITY_DRY_MIXING_RATIO
    pressure_mode: str = "external_sigma"
    requires_cloud_liquid: bool = False

    def state_requirement(self) -> StateRequirement:
        tracers = [
            TracerRequirement(name="vapor", quantity=self.vapor_quantity),
        ]
        if self.requires_cloud_liquid:
            tracers.append(
                TracerRequirement(
                    name="cloud_liquid",
                    quantity=CLOUD_LIQUID_QUANTITY_DRY_MIXING_RATIO,
                )
            )
        return StateRequirement(tracers=tuple(tracers))


class WellmeyerGCMColumnAdapter(nn.Module):
    """Adapter for Evan Wellmeyer's optional PyTorch SCM package.

    The external package is not vendored or imported at VFS import time.  Put
    the external repository on ``PYTHONPATH`` or install it in the environment,
    then wrap this adapter with ``SCMPhysics``.

    This first bridge converts one external ``physics_step`` over ``dt`` into a
    VFS tendency-equivalent dictionary.  It is useful for adapter development
    and smoke tests, but it is still an explicit split-step approximation.  The
    external package may own a different sigma grid; the adapter maps between
    VFS pressure interfaces and external pressure interfaces at the boundary.
    Scalar external parameters are passed as Python scalars for compatibility
    with the current external code; per-column parameters remain tensors.
    """

    def __init__(
        self,
        *,
        dt: float,
        params: Mapping[str, Any] | None = None,
        import_package: str = "scm",
        pressure_mode: str = "external_sigma",
        include_cloud_liquid: bool = False,
        use_slab_ocean: bool = False,
        external_nlevels: int | None = None,
        radiation_every_steps: int | None = None,
        vertical_remap: str = "conservative_pressure",
        config_path: str | None = None,
    ) -> None:
        super().__init__()
        self.import_package = str(import_package)
        self.pressure_mode = str(pressure_mode)
        if self.pressure_mode != "external_sigma":
            raise ValueError(
                "WellmeyerGCMColumnAdapter currently supports only "
                "pressure_mode='external_sigma'"
            )
        self.contract = ExternalSCMContract(
            pressure_mode=self.pressure_mode,
            requires_cloud_liquid=bool(include_cloud_liquid),
        )
        self.external_nlevels = None if external_nlevels is None else int(external_nlevels)
        if self.external_nlevels is not None and self.external_nlevels <= 0:
            raise ValueError("external_nlevels must be positive when provided")
        self.radiation_every_steps = None if radiation_every_steps is None else int(radiation_every_steps)
        if self.radiation_every_steps is not None and self.radiation_every_steps <= 0:
            raise ValueError("radiation_every_steps must be positive when provided")
        self.vertical_remap = str(vertical_remap)
        if self.vertical_remap not in {"conservative_pressure", "linear_sigma", "identity", "dycore_native"}:
            raise ValueError(
                "vertical_remap must be 'conservative_pressure', 'linear_sigma', "
                "'identity', or 'dycore_native'"
            )
        self.config_path = _normalize_config_path(config_path)
        self._config_param_overrides: dict[str, Any] | None = None
        self.static_params: dict[str, Any] = {"use_slab_ocean": bool(use_slab_ocean)}
        self._param_names: list[tuple[str, str]] = []
        self._external_modules: tuple[Any, Any, Any] | None = None
        self.register_buffer("external_scm_dt", torch.as_tensor(float(dt)))
        for raw_name, value in dict(params or {}).items():
            name = str(raw_name)
            if _is_numeric_param(value):
                safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
                if not safe or safe[0].isdigit():
                    safe = f"p_{safe}"
                buffer_name = f"external_scm_param_{safe}"
                if hasattr(self, buffer_name):
                    raise ValueError(f"duplicate or conflicting external SCM parameter name {name!r}")
                self.register_buffer(buffer_name, torch.as_tensor(value))
                self._param_names.append((name, buffer_name))
            else:
                self.static_params[name] = value
        self.last_diagnostics: Any = None
        self.last_radiation_cache: Any = None
        self.last_vfs_level_count: int | None = None
        self.last_external_level_count: int | None = None

    def required_state_requirement(self) -> StateRequirement:
        return self.contract.state_requirement()

    def _import_modules(self):
        if self._external_modules is not None:
            return self._external_modules
        try:
            thermo = importlib.import_module(f"{self.import_package}.thermo")
            column_model = importlib.import_module(f"{self.import_package}.column_model")
            ensemble = importlib.import_module(f"{self.import_package}.ensemble")
        except ImportError as exc:
            raise ImportError(
                "WellmeyerGCMColumnAdapter requires the external SCM package. "
                "Install https://github.com/evanwellmeyer/GCM or put its parent "
                "directory on PYTHONPATH so modules like 'scm.column_model' import."
            ) from exc
        self._external_modules = (thermo, column_model, ensemble)
        return self._external_modules

    def _configured_param_overrides(self) -> dict[str, Any]:
        """Return external-package run-config overrides, if explicitly enabled."""

        if self.config_path is None:
            return {}
        if self._config_param_overrides is not None:
            return dict(self._config_param_overrides)
        try:
            configuration = importlib.import_module(f"{self.import_package}.configuration")
        except ImportError as exc:
            raise ImportError(
                "WellmeyerGCMColumnAdapter config_path requires "
                f"{self.import_package}.configuration from the external SCM package."
            ) from exc
        path = None if self.config_path == "default" else self.config_path
        config = configuration.load_run_config(path)
        self._config_param_overrides = dict(configuration.extract_param_overrides(config))
        return dict(self._config_param_overrides)

    def _params_for_columns(self, default_params, ref: torch.Tensor, n_cells: int) -> dict[str, Any]:
        try:
            params = dict(default_params(device=ref.device))
        except TypeError:
            params = dict(default_params(ref.device))
        params.update(self._configured_param_overrides())
        params.update(self.static_params)
        params["dt"] = _python_scalar(self.external_scm_dt)
        # The external package accepts this as an optional hint; it keeps
        # surface/slab bookkeeping on the same dtype/device path as VFS.
        params["dtype"] = ref.dtype
        for name, buffer_name in self._param_names:
            params[name] = _external_param(getattr(self, buffer_name), ref, n_cells)
        return params

    def initial_physics_state(self, columns: Mapping[str, Any]) -> dict[str, Any]:
        """Initial restart payload for split-step external SCM coupling."""

        thermo, column_model, ensemble = self._import_modules()
        vfs_temperature = _require_column_tensor(columns, "temperature_full")
        n_cells = int(vfs_temperature.shape[0])
        ref = vfs_temperature
        grid = self._make_external_grid(thermo, columns, ref)
        temperature = self._to_external_levels(columns, vfs_temperature, grid, fill="nearest")
        params = self._params_for_columns(ensemble.default_params, ref, n_cells)
        surface_temperature = self._surface_temperature(columns, params, temperature)
        slab_dtype = _slab_dtype(ref)
        payload = {
            "external_state": {
                "ts": surface_temperature,
                "slab_ts_ref": surface_temperature.to(dtype=slab_dtype),
                "slab_energy": torch.zeros_like(surface_temperature, dtype=slab_dtype),
                # Evan's SCM computes radiation before cloud microphysics; the
                # optical cloud fields produced at the end of one step must be
                # available to radiation at the start of the next split step.
                "cloud_fraction": torch.zeros_like(temperature),
                "cloud_sw_tau_layer": torch.zeros_like(temperature),
                "cloud_lw_tau_layer": torch.zeros_like(temperature),
            },
            "rad_cache": None,
        }
        if self.radiation_every_steps is not None:
            payload["radiation_step_count"] = 0
        return payload

    def forward(self, columns: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        outputs, _physics_state, _diagnostics = self._step_columns_impl(
            columns,
            dt=self.external_scm_dt,
            physics_state=None,
        )
        return outputs

    def step_columns(
        self,
        columns: Mapping[str, Any],
        *,
        dt: float,
        physics_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one explicit external SCM physics step and return restart state."""

        outputs, next_physics_state, diagnostics = self._step_columns_impl(
            columns,
            dt=dt,
            physics_state=physics_state,
        )
        return {
            "outputs": outputs,
            "physics_state": next_physics_state,
            "diagnostics": diagnostics,
        }

    def _step_columns_impl(
        self,
        columns: Mapping[str, Any],
        *,
        dt: Any,
        physics_state: Mapping[str, Any] | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        thermo, column_model, ensemble = self._import_modules()
        vfs_temperature = _require_column_tensor(columns, "temperature_full")
        vfs_q_old = _require_column_tensor(columns, "specific_humidity").clamp_min(0.0)
        ps_old = _require_column_tensor(columns, "surface_pressure").reshape(-1)
        n_cells = int(vfs_temperature.shape[0])
        ref = vfs_temperature
        vfs_i_old = _vfs_pressure_interface(columns, ref)
        p_top_old = vfs_i_old[:, :1].reshape(n_cells)
        dycore_native_grid = self.vertical_remap == "dycore_native"
        ps_external_old = (
            ps_old
            if dycore_native_grid
            else (ps_old - p_top_old).clamp_min(torch.finfo(ref.dtype).tiny)
        )

        grid = self._make_external_grid(thermo, columns, ref)
        temperature = self._to_external_levels(columns, vfs_temperature, grid, fill="nearest")
        q_old = self._to_external_levels(columns, vfs_q_old, grid, fill="zero").clamp_min(0.0)
        params = self._params_for_columns(ensemble.default_params, ref, n_cells)
        _merge_coupled_column_params(params, columns)
        params["dt"] = _python_scalar(_scalar_like(dt, ref))
        dt_tensor = _scalar_like(params["dt"], ref)
        ext_state = column_model.initial_state(
            batch=n_cells,
            grid=grid,
            params=_safe_initial_state_params(params),
            device=ref.device,
            ncol=n_cells,
            nmember=1,
        )
        ext_state = dict(ext_state)
        ext_state["t"] = temperature
        ext_state["q"] = q_old
        # Legacy external sigma mode uses a column with no top-pressure offset.
        # VFS uses an active pressure interval p_top..ps, so give legacy grids
        # only the active column mass and label interfaces on the VFS interval
        # when remapping. Native dycore-grid mode receives absolute pressure
        # interfaces, so it uses the real VFS surface pressure directly.
        ext_state["ps"] = ps_external_old
        ext_state["u"] = torch.zeros_like(temperature)
        ext_state["v"] = torch.zeros_like(temperature)
        surface_temperature = self._surface_temperature(columns, params, temperature)
        carried_state = _physics_state_get(physics_state, "external_state", {})
        slab_dtype = _slab_dtype(ref)
        ocean_surface_mask = _surface_ocean_mask(columns, surface_temperature)
        ext_state["ts"] = _blend_surface_owner(
            surface_temperature,
            _carried_tensor(carried_state, "ts", surface_temperature),
            ocean_surface_mask,
        )
        ext_state["slab_ts_ref"] = _carried_tensor(
            carried_state,
            "slab_ts_ref",
            surface_temperature.to(dtype=slab_dtype),
        )
        ext_state["slab_ts_ref"] = _blend_surface_owner(
            surface_temperature.to(dtype=slab_dtype),
            ext_state["slab_ts_ref"],
            ocean_surface_mask,
        )
        ext_state["slab_energy"] = _carried_tensor(
            carried_state,
            "slab_energy",
            torch.zeros_like(surface_temperature, dtype=slab_dtype),
        )
        ext_state["slab_energy"] = _blend_surface_owner(
            torch.zeros_like(surface_temperature, dtype=slab_dtype),
            ext_state["slab_energy"],
            ocean_surface_mask,
        )
        if self.contract.requires_cloud_liquid:
            q_l_old = self._to_external_levels(
                columns,
                self._cloud_liquid_specific(columns, vfs_q_old),
                grid,
                fill="zero",
            )
            ext_state["qc"] = q_l_old
        else:
            q_l_old = None
            if "qc" in ext_state:
                ext_state["qc"] = torch.zeros_like(temperature)
        for name in _CLOUD_OPTICAL_STATE_NAMES:
            ext_state[name] = _carried_tensor(
                carried_state,
                name,
                _optional_state_tensor(ext_state, name, default=torch.zeros_like(temperature)),
            )

        step_count = _physics_state_step_count(_physics_state_get(physics_state, "radiation_step_count", 0))
        rad_cache_in = _physics_state_get(physics_state, "rad_cache", None)
        if self.radiation_every_steps is not None and step_count % int(self.radiation_every_steps) == 0:
            # The external package's long-run driver periodically clears its
            # radiation cache; VFS owns that cadence when calling physics_step
            # directly from coupled process stepping.
            rad_cache_in = None
        radiation_recomputed = rad_cache_in is None
        new_state, diagnostics, rad_cache = column_model.physics_step(
            ext_state,
            grid,
            params,
            rad_cache=rad_cache_in,
            ls_forcing=None,
        )
        self.last_diagnostics = diagnostics
        self.last_radiation_cache = rad_cache

        t_new = _state_tensor(new_state, "t", ref=temperature)
        q_new = _state_tensor(new_state, "q", ref=q_old).clamp(0.0, 1.0 - torch.finfo(ref.dtype).eps)
        ps_external_new = _state_tensor(new_state, "ps", ref=ps_external_old).reshape(n_cells, 1)
        ps_new = ps_external_new if dycore_native_grid else ps_external_new + p_top_old.reshape(n_cells, 1)
        state_ref = columns.get("state")
        if not hasattr(state_ref, "tracer_m") or state_ref.tracer_m is None:
            raise ValueError("WellmeyerGCMColumnAdapter requires State.tracer_m for vapor target-state updates")
        target_dp = state_ref.column.layer_thickness(ps_new).to(dtype=ref.dtype, device=ref.device)

        dT_dt = (t_new - temperature) / dt_tensor
        dr_v_dt = (
            mixing_ratio_from_specific_humidity(q_new)
            - mixing_ratio_from_specific_humidity(q_old)
        ) / dt_tensor
        dT_dt_vfs = self._from_external_levels(columns, dT_dt, grid, target_ref=vfs_temperature, fill="zero")
        r_v_new_external = mixing_ratio_from_specific_humidity(q_new)
        r_v_target_raw = pressure_overlap_remap(
            r_v_new_external,
            _external_pressure_interface(columns, grid, r_v_new_external, surface_pressure=ps_new),
            _vfs_pressure_interface(columns, vfs_q_old, surface_pressure=ps_new),
            fill="zero",
        )
        r_v_target, vapor_repair = _conservative_nonnegative_layer_mean(r_v_target_raw, target_dp)
        vfs_rv_old = _require_column_tensor(columns, "r_v")
        vapor_index = vapor_tracer_index(state_ref)
        if vapor_index is None:
            raise ValueError("WellmeyerGCMColumnAdapter requires a named vapor tracer")
        if state_ref.tracer_m.ndim == 3:
            old_vfs_vapor_content = state_ref.tracer_m[..., int(vapor_index)]
        else:
            old_vfs_vapor_content = state_ref.tracer_m

        # CoupledSCMProcess recognizes these flux names and routes them to the
        # ocean/land reservoirs; they are diagnostics for atmosphere-only use.
        outputs: dict[str, torch.Tensor] = {}
        surface_total_flux = _mapping_tensor(diagnostics, "surface_total_flux", ref=ps_old)
        if surface_total_flux is not None:
            outputs["ocean_surface_heat_flux_down"] = surface_total_flux.reshape(n_cells, 1)
        latent_heat_flux = _mapping_tensor(diagnostics, "lhf", ref=ps_old)
        evaporation = None
        if latent_heat_flux is not None:
            evaporation = latent_heat_flux / float(getattr(thermo, "Lv", 2.5e6))
            outputs["evaporation_mass_flux_up"] = evaporation.reshape(n_cells, 1)
            outputs["latent_heat_flux_up"] = latent_heat_flux.reshape(n_cells, 1)
        sensible_heat_flux = _mapping_tensor(diagnostics, "shf", ref=ps_old)
        if sensible_heat_flux is not None:
            outputs["sensible_heat_flux_up"] = sensible_heat_flux.reshape(n_cells, 1)
        precipitation = _mapping_tensor(diagnostics, "precip_total", ref=ps_old)
        if precipitation is not None:
            outputs["surface_precipitation_rate"] = precipitation.reshape(n_cells, 1)
            if evaporation is None:
                evaporation = torch.zeros_like(precipitation)
            rho_water = float(getattr(thermo, "rho_water", 1000.0))
            outputs["freshwater_flux_down"] = (
                precipitation.reshape(n_cells, 1) / rho_water - evaporation.reshape(n_cells, 1) / rho_water
            )

        r_l_target = None
        r_l_old = None
        if self.contract.requires_cloud_liquid and q_l_old is not None and "qc" in new_state:
            q_l_new = _state_tensor(new_state, "qc", ref=q_l_old).clamp_min(0.0)
            r_l_new_external = _dry_mixing_ratio_from_specific_cloud(q_new, q_l_new)
            r_l_target_raw = pressure_overlap_remap(
                r_l_new_external,
                _external_pressure_interface(columns, grid, r_l_new_external, surface_pressure=ps_new),
                _vfs_pressure_interface(columns, vfs_q_old, surface_pressure=ps_new),
                fill="zero",
            )
            r_l_target, cloud_repair = _conservative_nonnegative_layer_mean(r_l_target_raw, target_dp)
            r_l_old = _require_column_tensor(columns, "r_l")
            cloud_index = cloud_liquid_tracer_index(state_ref)
            old_vfs_cloud_content = (
                None
                if cloud_index is None or state_ref.tracer_m.ndim != 3
                else state_ref.tracer_m[..., int(cloud_index)]
            )
        else:
            cloud_repair = None
            old_vfs_cloud_content = None

        water_budget_projection = None
        if precipitation is not None and evaporation is not None:
            surface_water_to_atmosphere = evaporation.reshape(n_cells, 1) - precipitation.reshape(n_cells, 1)
            r_v_target, r_l_target, water_budget_projection = _project_moisture_targets_to_surface_budget(
                vapor_target=r_v_target,
                target_dp=target_dp,
                old_vapor_content=old_vfs_vapor_content,
                dt=dt_tensor,
                surface_water_to_atmosphere=surface_water_to_atmosphere,
                gravity=float((state_ref.attrs or {}).get("gravity", DEFAULT_GRAVITY)),
                cloud_target=r_l_target,
                old_cloud_content=old_vfs_cloud_content,
            )

        dr_v_dt_vfs = (r_v_target - vfs_rv_old) / dt_tensor
        tracer_m_dt = _tracer_target_tendency(
            state_ref=state_ref,
            target_dp=target_dp,
            dt=dt_tensor,
            tracer_index=vapor_index,
            target_mixing_ratio=r_v_target,
        )
        outputs.update(
            {
                "dT_dt": dT_dt_vfs,
                "dr_v_dt": dr_v_dt_vfs,
                "tracer_m_dt": tracer_m_dt,
                "ps_dt": ((ps_new - ps_old.reshape(n_cells, 1)) / dt_tensor).reshape_as(
                    columns["surface_pressure"]
                ),
            }
        )
        if r_l_target is not None and r_l_old is not None:
            outputs["dr_l_dt"] = (r_l_target - r_l_old) / dt_tensor
            outputs["tracer_m_dt"] = _tracer_target_tendency(
                state_ref=state_ref,
                target_dp=target_dp,
                dt=dt_tensor,
                tracer_index=cloud_liquid_tracer_index(state_ref),
                target_mixing_ratio=r_l_target,
                tracer_m_dt=outputs["tracer_m_dt"],
            )

        water_budget_diagnostics: dict[str, torch.Tensor] = {}
        gravity = float((state_ref.attrs or {}).get("gravity", DEFAULT_GRAVITY))
        if vapor_index is not None:
            tracer_m_dt_for_vapor = outputs["tracer_m_dt"]
            if tracer_m_dt_for_vapor.ndim == 3:
                tracer_m_dt_for_vapor = tracer_m_dt_for_vapor[..., int(vapor_index)]
            vfs_vapor_water_tendency = tracer_m_dt_for_vapor.sum(dim=-1, keepdim=True) / gravity
            external_dp_new = torch.diff(
                _external_pressure_interface(columns, grid, r_v_new_external, surface_pressure=ps_new),
                dim=-1,
            ).abs()
            external_dp_old = torch.diff(
                _external_pressure_interface(
                    columns,
                    grid,
                    r_v_new_external,
                    surface_pressure=ps_old.reshape(n_cells, 1),
                ),
                dim=-1,
            ).abs()
            external_vapor_water_tendency = (
                (r_v_new_external * external_dp_new).sum(dim=-1, keepdim=True)
                - (mixing_ratio_from_specific_humidity(q_old) * external_dp_old).sum(dim=-1, keepdim=True)
            ) / (gravity * dt_tensor)
            old_vfs_vapor_column_water = old_vfs_vapor_content.sum(dim=-1, keepdim=True) / gravity
            target_vfs_vapor_column_water = (r_v_target * target_dp).sum(dim=-1, keepdim=True) / gravity
            external_old_vapor_column_water = (
                mixing_ratio_from_specific_humidity(q_old) * external_dp_old
            ).sum(dim=-1, keepdim=True) / gravity
            external_new_vapor_column_water = (
                r_v_new_external * external_dp_new
            ).sum(dim=-1, keepdim=True) / gravity
            precipitation_out = outputs.get("surface_precipitation_rate")
            evaporation_out = outputs.get("evaporation_mass_flux_up")
            water_budget_diagnostics["vfs_target_vapor_water_tendency_kg_m2_s"] = vfs_vapor_water_tendency
            water_budget_diagnostics["external_vapor_water_tendency_kg_m2_s"] = external_vapor_water_tendency
            water_budget_diagnostics["vfs_minus_external_vapor_water_tendency_kg_m2_s"] = (
                vfs_vapor_water_tendency - external_vapor_water_tendency
            )
            water_budget_diagnostics["vfs_old_vapor_column_water_kg_m2"] = old_vfs_vapor_column_water
            water_budget_diagnostics["vfs_target_vapor_column_water_kg_m2"] = target_vfs_vapor_column_water
            water_budget_diagnostics["external_old_vapor_column_water_kg_m2"] = external_old_vapor_column_water
            water_budget_diagnostics["external_new_vapor_column_water_kg_m2"] = external_new_vapor_column_water
            water_budget_diagnostics["external_minus_vfs_old_vapor_column_water_kg_m2"] = (
                external_old_vapor_column_water - old_vfs_vapor_column_water
            )
            water_budget_diagnostics["external_minus_vfs_target_vapor_column_water_kg_m2"] = (
                external_new_vapor_column_water - target_vfs_vapor_column_water
            )
            if precipitation_out is not None:
                water_budget_diagnostics["scm_surface_precipitation_kg_m2_s"] = precipitation_out.reshape(n_cells, 1)
            if evaporation_out is not None:
                water_budget_diagnostics["scm_surface_evaporation_kg_m2_s"] = evaporation_out.reshape(n_cells, 1)
            if precipitation_out is not None and evaporation_out is not None:
                surface_water_to_atmosphere = evaporation_out.reshape(n_cells, 1) - precipitation_out.reshape(n_cells, 1)
                water_budget_diagnostics["scm_surface_water_to_atmosphere_kg_m2_s"] = surface_water_to_atmosphere
                water_budget_diagnostics["vfs_target_vapor_water_budget_residual_kg_m2_s"] = (
                    vfs_vapor_water_tendency - surface_water_to_atmosphere
                )
            if water_budget_projection is not None:
                water_budget_diagnostics.update(water_budget_projection)
        if "qc" in new_state:
            q_l_new_diag = _state_tensor(new_state, "qc", ref=q_new).clamp_min(0.0)
            q_l_old_diag = q_l_old if q_l_old is not None else torch.zeros_like(q_l_new_diag)
            external_dp_new = torch.diff(
                _external_pressure_interface(columns, grid, q_l_new_diag, surface_pressure=ps_new),
                dim=-1,
            ).abs()
            external_dp_old = torch.diff(
                _external_pressure_interface(
                    columns,
                    grid,
                    q_l_old_diag,
                    surface_pressure=ps_old.reshape(n_cells, 1),
                ),
                dim=-1,
            ).abs()
            qc_column_new = (q_l_new_diag * external_dp_new).sum(dim=-1, keepdim=True) / gravity
            qc_column_old = (q_l_old_diag * external_dp_old).sum(dim=-1, keepdim=True) / gravity
            water_budget_diagnostics["external_qc_column_water_kg_m2"] = qc_column_new
            water_budget_diagnostics["external_qc_column_water_tendency_kg_m2_s"] = (
                qc_column_new - qc_column_old
            ) / dt_tensor
        next_physics_state = {
            "external_state": {
                # VFS owns ocean-cell SST; external slab state only persists
                # over non-ocean cells supplied by SurfaceExchange.
                "ts": _blend_surface_owner(
                    surface_temperature,
                    _optional_state_tensor(new_state, "ts", default=ext_state["ts"]),
                    ocean_surface_mask,
                ),
                "slab_ts_ref": _blend_surface_owner(
                    surface_temperature.to(dtype=slab_dtype),
                    _optional_state_tensor(new_state, "slab_ts_ref", default=ext_state["slab_ts_ref"]),
                    ocean_surface_mask,
                ),
                "slab_energy": _blend_surface_owner(
                    torch.zeros_like(surface_temperature, dtype=slab_dtype),
                    _optional_state_tensor(new_state, "slab_energy", default=ext_state["slab_energy"]),
                    ocean_surface_mask,
                ),
                "cloud_fraction": _optional_state_tensor(
                    new_state,
                    "cloud_fraction",
                    default=ext_state["cloud_fraction"],
                ),
                "cloud_sw_tau_layer": _optional_state_tensor(
                    new_state,
                    "cloud_sw_tau_layer",
                    default=ext_state["cloud_sw_tau_layer"],
                ),
                "cloud_lw_tau_layer": _optional_state_tensor(
                    new_state,
                    "cloud_lw_tau_layer",
                    default=ext_state["cloud_lw_tau_layer"],
                ),
            },
            "rad_cache": rad_cache,
        }
        diagnostics_out = dict(diagnostics or {})
        if isinstance(rad_cache, Mapping):
            for key in (
                "toa_insolation",
                "clear_sky_olr",
                "clear_sky_asr",
                "clear_sky_toa_net",
                "cloud_lw_cre",
                "cloud_sw_cre",
                "cloud_toa_cre",
            ):
                if key not in diagnostics_out and key in rad_cache:
                    diagnostics_out[key] = rad_cache[key]
        projected_rv = _project_vapor_after_tracer_tendency(
            state_ref=state_ref,
            target_dp=target_dp,
            dt=dt_tensor,
            tracer_m_dt=outputs["tracer_m_dt"],
        )
        diagnostics_out.update(
            {
                "external_scm_config_path": "legacy" if self.config_path is None else self.config_path,
                "external_scm_vertical_remap": str(self.vertical_remap),
                "external_scm_uses_dycore_grid": bool(dycore_native_grid),
                "radiation_scheme": str(params.get("radiation_scheme", "semi_gray")),
                "radiation_mode": str(params.get("radiation_mode", "auto")),
                "radiation_clear_sky_diagnostics": bool(params.get("radiation_clear_sky_diagnostics", False)),
                "cloud_microphysics_enabled": bool(params.get("cloud_microphysics_enabled", False)),
                "cloud_radiative_effects_enabled": bool(params.get("cloud_radiative_effects_enabled", False)),
                "cloud_optics_scheme": str(params.get("cloud_optics_scheme", "auto")),
                "external_cloud_fraction_input_max": _tensor_max(ext_state["cloud_fraction"]),
                "external_cloud_sw_tau_input_column_max": _tensor_max(ext_state["cloud_sw_tau_layer"].sum(dim=-1)),
                "external_cloud_lw_tau_input_column_max": _tensor_max(ext_state["cloud_lw_tau_layer"].sum(dim=-1)),
                # Detached scalar bound audits distinguish external-column q
                # behavior from VFS target-state/remap application behavior.
                "vfs_specific_humidity_input_min": _tensor_min(vfs_q_old),
                "vfs_specific_humidity_input_max": _tensor_max(vfs_q_old),
                "external_q_old_min": _tensor_min(q_old),
                "external_q_old_max": _tensor_max(q_old),
                "external_q_new_min": _tensor_min(q_new),
                "external_q_new_max": _tensor_max(q_new),
                "external_active_surface_pressure_min": _tensor_min(ps_external_new),
                "external_active_surface_pressure_max": _tensor_max(ps_external_new),
                "vfs_top_pressure_min": _tensor_min(p_top_old),
                "vfs_top_pressure_max": _tensor_max(p_top_old),
                "external_dr_v_dt_min": _tensor_min(dr_v_dt),
                "external_dr_v_dt_max": _tensor_max(dr_v_dt),
                "vfs_r_v_target_raw_min": _tensor_min(r_v_target_raw),
                "vfs_r_v_target_raw_max": _tensor_max(r_v_target_raw),
                "vfs_r_v_target_raw_negative_count": _negative_count(r_v_target_raw),
                "vfs_r_v_target_repaired_min": _tensor_min(r_v_target),
                "vfs_r_v_target_repaired_max": _tensor_max(r_v_target),
                "vfs_r_v_target_repaired_negative_count": _negative_count(r_v_target),
                "vfs_r_v_target_repair_deficit_pressure_integral_max": _tensor_max(
                    vapor_repair["negative_content_pressure_integral"]
                ),
                "vfs_r_v_target_repair_adjustment_pressure_integral_max": _tensor_max(
                    vapor_repair["adjustment_abs_pressure_integral"]
                ),
                "vfs_r_v_target_repair_residual_pressure_integral_max": _tensor_max(
                    vapor_repair["residual_pressure_integral"]
                ),
                "vfs_dr_v_dt_target_state_min": _tensor_min(dr_v_dt_vfs),
                "vfs_dr_v_dt_target_state_max": _tensor_max(dr_v_dt_vfs),
                "vfs_dr_v_dt_remapped_min": _tensor_min(dr_v_dt_vfs),
                "vfs_dr_v_dt_remapped_max": _tensor_max(dr_v_dt_vfs),
                "vfs_projected_r_v_after_scm_min": _tensor_min(projected_rv),
                "vfs_projected_r_v_after_scm_max": _tensor_max(projected_rv),
                "vfs_projected_negative_vapor_count": _negative_count(projected_rv),
            }
        )
        for name in _REPORTED_EXTERNAL_PARAM_NAMES:
            if name in params:
                diagnostics_out[f"{name}_param"] = _diagnostic_param_value(params[name])
        if "albedo" in params:
            diagnostics_out["radiation_albedo_param"] = params["albedo"]
        diagnostics_out.update(water_budget_diagnostics)
        if cloud_repair is not None:
            diagnostics_out.update(
                {
                    "vfs_r_l_target_repair_deficit_pressure_integral_max": _tensor_max(
                        cloud_repair["negative_content_pressure_integral"]
                    ),
                    "vfs_r_l_target_repair_adjustment_pressure_integral_max": _tensor_max(
                        cloud_repair["adjustment_abs_pressure_integral"]
                    ),
                    "vfs_r_l_target_repair_residual_pressure_integral_max": _tensor_max(
                        cloud_repair["residual_pressure_integral"]
                    ),
                }
            )
        if self.radiation_every_steps is not None:
            next_physics_state["radiation_step_count"] = step_count + 1
            diagnostics_out.update(
                {
                    "radiation_every_steps": int(self.radiation_every_steps),
                    "radiation_step_count": int(step_count),
                    "radiation_cache_recomputed": bool(radiation_recomputed),
                }
            )
        return outputs, next_physics_state, diagnostics_out

    def _make_external_grid(self, thermo, columns: Mapping[str, Any], ref: torch.Tensor) -> dict[str, Any]:
        vfs_nlevels = int(_require_column_tensor(columns, "temperature_full").shape[1])
        if self.vertical_remap == "dycore_native":
            if self.external_nlevels is not None and int(self.external_nlevels) != vfs_nlevels:
                raise ValueError(
                    "vertical_remap='dycore_native' runs the external SCM on VFS levels; "
                    f"external_nlevels must be auto or {vfs_nlevels}, got {int(self.external_nlevels)}"
                )
            if not hasattr(thermo, "grid_from_pressure_interfaces"):
                raise ValueError(
                    "vertical_remap='dycore_native' requires an external SCM with "
                    "scm.thermo.grid_from_pressure_interfaces"
                )
            p_interface = _vfs_pressure_interface(columns, ref)
            grid = dict(thermo.grid_from_pressure_interfaces(p_interface.to(dtype=ref.dtype, device=ref.device)))
            grid["_vfs_dycore_native"] = True
            actual_nlevels = _external_level_count(grid)
            if actual_nlevels != vfs_nlevels:
                raise ValueError(
                    "external SCM dycore-native grid level count does not match VFS; "
                    f"got external={actual_nlevels}, VFS={vfs_nlevels}"
                )
            grid["nlevels"] = actual_nlevels
            self.last_vfs_level_count = vfs_nlevels
            self.last_external_level_count = actual_nlevels
            return grid

        requested_nlevels = self.external_nlevels if self.external_nlevels is not None else vfs_nlevels
        grid = dict(thermo.make_grid(nlevels=int(requested_nlevels), device=ref.device))
        actual_nlevels = _external_level_count(grid)
        if actual_nlevels <= 0:
            raise ValueError("external SCM grid must expose at least one full level")
        grid["nlevels"] = actual_nlevels
        self.last_vfs_level_count = vfs_nlevels
        self.last_external_level_count = actual_nlevels
        return grid

    def _to_external_levels(
        self,
        columns: Mapping[str, Any],
        value: torch.Tensor,
        grid: Mapping[str, Any],
        *,
        fill: str = "nearest",
    ) -> torch.Tensor:
        if self.vertical_remap in {"identity", "dycore_native"}:
            external_nlevels = _external_level_count(grid)
            if int(value.shape[-1]) != int(external_nlevels):
                raise ValueError(
                    f"vertical_remap={self.vertical_remap!r} requires matching VFS and external level counts; "
                    f"got VFS={int(value.shape[-1])}, external={int(external_nlevels)}"
                )
            return value
        if self.vertical_remap == "conservative_pressure":
            return pressure_overlap_remap(
                value,
                _vfs_pressure_interface(columns, value),
                _external_pressure_interface(columns, grid, value),
                fill=fill,
            )
        external_sigma = _external_sigma_full(grid, value)
        source_sigma = _vfs_sigma_full(columns, value)
        return _linear_interp_columns(source_sigma, value, external_sigma)

    def _from_external_levels(
        self,
        columns: Mapping[str, Any],
        value: torch.Tensor,
        grid: Mapping[str, Any],
        *,
        target_ref: torch.Tensor,
        fill: str = "zero",
    ) -> torch.Tensor:
        if self.vertical_remap in {"identity", "dycore_native"}:
            return value
        if self.vertical_remap == "conservative_pressure":
            return pressure_overlap_remap(
                value,
                _external_pressure_interface(columns, grid, value),
                _vfs_pressure_interface(columns, target_ref),
                fill=fill,
            )
        external_sigma = _external_sigma_full(grid, value)
        target_sigma = _vfs_sigma_full(columns, target_ref)
        return _linear_interp_columns(external_sigma, value, target_sigma)

    def _surface_temperature(
        self,
        columns: Mapping[str, Any],
        params: Mapping[str, Any],
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(columns.get("params"), Mapping) and "surface_temperature" in columns["params"]:
            return _batch_vector("surface_temperature", columns["params"]["surface_temperature"], temperature)
        if "surface_temperature" in params:
            return _batch_vector("surface_temperature", params["surface_temperature"], temperature)
        if "ts_init" in params:
            return _batch_vector("ts_init", params["ts_init"], temperature)
        return temperature[..., -1]

    def _cloud_liquid_specific(self, columns: Mapping[str, Any], q_old: torch.Tensor) -> torch.Tensor:
        r_l = columns.get("cloud_liquid_mixing_ratio", columns.get("r_l"))
        if r_l is None:
            raise ValueError("include_cloud_liquid=True requires columns['cloud_liquid_mixing_ratio']")
        r_l = torch.as_tensor(r_l, dtype=q_old.dtype, device=q_old.device)
        denom = (1.0 + mixing_ratio_from_specific_humidity(q_old) + r_l).clamp_min(torch.finfo(q_old.dtype).tiny)
        return r_l / denom


def _is_numeric_param(value: Any) -> bool:
    return torch.is_tensor(value) or (isinstance(value, (int, float)) and not isinstance(value, bool))


def _normalize_config_path(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"legacy", "none", "off", "false"}:
        return None
    if lowered in {"default", "package_default", "package-default"}:
        return "default"
    return text


def _require_column_tensor(columns: Mapping[str, Any], name: str) -> torch.Tensor:
    value = columns.get(name)
    if value is None:
        raise ValueError(f"external SCM adapter requires columns[{name!r}]")
    tensor = torch.as_tensor(value)
    if tensor.ndim < 2:
        raise ValueError(f"columns[{name!r}] must have at least [nCells, nLevels] shape")
    return tensor


def _external_param(value: torch.Tensor, ref: torch.Tensor, n_cells: int) -> torch.Tensor | float | int:
    tensor = value.to(device=ref.device)
    if tensor.is_floating_point():
        tensor = tensor.to(dtype=ref.dtype)
    if tensor.ndim == 0:
        return _python_scalar(tensor)
    if tensor.ndim == 1 and int(tensor.shape[0]) == int(n_cells):
        return tensor
    if tensor.ndim >= 2 and int(tensor.shape[-2]) == int(n_cells):
        return tensor
    raise ValueError(
        "external SCM parameter must be scalar, [nCells], or have nCells on the "
        f"next-to-last axis; got {tuple(tensor.shape)} for nCells={int(n_cells)}"
    )


def _scalar_like(value: Any, ref: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=ref.device)
    if tensor.is_floating_point():
        tensor = tensor.to(dtype=ref.dtype)
    if tensor.ndim != 0:
        raise ValueError(f"expected scalar external SCM parameter, got shape {tuple(tensor.shape)}")
    return tensor


def _python_scalar(value: Any) -> float | int:
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim != 0:
        raise ValueError(f"expected scalar external SCM parameter, got shape {tuple(tensor.shape)}")
    item = tensor.item()
    return int(item) if not tensor.is_floating_point() else float(item)


def _diagnostic_param_value(value: Any) -> Any:
    """Return a CSV/report-friendly param value without changing SCM behavior."""

    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        return value
    if torch.is_tensor(value) and value.ndim == 0:
        return _python_scalar(value)
    return value


def _safe_initial_state_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Keep external initialization helpers away from vector fill values."""

    out = dict(params)
    for name in ("ps0", "ts_init"):
        value = out.get(name)
        if torch.is_tensor(value) and value.ndim > 0:
            out[name] = _python_scalar(value.reshape(-1)[0])
    return out


def _merge_coupled_column_params(params: dict[str, Any], columns: Mapping[str, Any]) -> None:
    """Pass VFS coupled surface context through to the external SCM."""

    column_params = columns.get("params")
    if not isinstance(column_params, Mapping):
        return

    def external_value(value: Any) -> Any:
        if not torch.is_tensor(value):
            return value
        tensor = value
        if tensor.ndim == 2 and int(tensor.shape[-1]) == 1:
            return tensor[:, 0]
        return tensor

    direct_fields = (
        "relative_wind_speed_cell",
        "relative_wind_speed",
        "surface_wind_speed",
        "wind_speed",
        "solar_constant",
        "zenith_factor",
        "solar_zenith_factor",
        "toa_insolation",
        "solar_toa_insolation",
        "surface_temperature",
        "surface_type",
        "land_fraction",
        "ocean_fraction",
        "sea_ice_fraction",
        "glacier_fraction",
        "land_use_type",
        "soil_type",
        "topography",
        "soil_moisture",
        "soil_temperature",
        "snow_water_equivalent",
        "sea_ice_thickness",
        "surface_albedo",
        "albedo",
        "roughness_length",
        "exchange_coefficient_heat",
        "exchange_coefficient_moisture",
        "surface_moisture_availability",
        "surface_emissions",
        "co2",
        "co2_ppm",
        "co2_ref",
        "ch4",
        "ch4_ppb",
        "n2o",
        "n2o_ppb",
        "o3_lw_tau",
        "ozone_lw_tau",
        "aerosol_sw_tau",
        "dry_deposition_velocity",
        "wet_deposition_rate",
    )
    for name in direct_fields:
        if name in column_params:
            target_name = "zenith_factor" if name == "solar_zenith_factor" else name
            params[target_name] = external_value(column_params[name])

    aliases = (
        ("land_cell_mask", "land_fraction"),
        ("ocean_cell_mask", "ocean_fraction"),
        ("surface_roughness", "roughness_length"),
        ("surface_exchange_coefficient_heat", "exchange_coefficient_heat"),
        ("surface_exchange_coefficient_moisture", "exchange_coefficient_moisture"),
        ("topography_m", "topography"),
    )
    for source_name, target_name in aliases:
        if target_name not in params and source_name in column_params:
            params[target_name] = external_value(column_params[source_name])

    if "cd" not in params and "surface_exchange_coefficient_heat" in column_params:
        params["cd"] = external_value(column_params["surface_exchange_coefficient_heat"])
    if "cd" not in params and "exchange_coefficient_heat" in column_params:
        params["cd"] = external_value(column_params["exchange_coefficient_heat"])
    if "albedo" not in params and "surface_albedo" in column_params:
        params["albedo"] = external_value(column_params["surface_albedo"])


def _surface_ocean_mask(columns: Mapping[str, Any], ref: torch.Tensor) -> torch.Tensor | None:
    column_params = columns.get("params")
    if not isinstance(column_params, Mapping):
        return None
    if "ocean_fraction" in column_params:
        return _batch_vector("ocean_fraction", column_params["ocean_fraction"], ref).clamp(0.0, 1.0)
    if "ocean_cell_mask" in column_params:
        return _batch_vector("ocean_cell_mask", column_params["ocean_cell_mask"], ref).clamp(0.0, 1.0)
    if "land_fraction" in column_params:
        return 1.0 - _batch_vector("land_fraction", column_params["land_fraction"], ref).clamp(0.0, 1.0)
    if "land_cell_mask" in column_params:
        return 1.0 - _batch_vector("land_cell_mask", column_params["land_cell_mask"], ref).clamp(0.0, 1.0)
    return None


def _blend_surface_owner(
    vfs_owned_value: torch.Tensor,
    external_owned_value: torch.Tensor,
    ocean_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Use VFS surface state over ocean and external carried state elsewhere."""

    if ocean_mask is None:
        return external_owned_value
    mask = ocean_mask.to(dtype=external_owned_value.dtype, device=external_owned_value.device).reshape_as(
        external_owned_value
    )
    vfs_value = torch.as_tensor(
        vfs_owned_value,
        dtype=external_owned_value.dtype,
        device=external_owned_value.device,
    ).reshape_as(external_owned_value)
    return mask * vfs_value + (1.0 - mask) * external_owned_value


def _batch_vector(name: str, value: Any, ref: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
    n_cells = int(ref.shape[0])
    if tensor.ndim == 0:
        return torch.broadcast_to(tensor, (n_cells,))
    tensor = tensor.reshape(-1)
    if int(tensor.numel()) != n_cells:
        raise ValueError(f"{name} must be scalar or length nCells={n_cells}, got {tuple(tensor.shape)}")
    return tensor


def _external_level_count(grid: Mapping[str, Any]) -> int:
    sigma_full = grid.get("sigma_full")
    if sigma_full is not None:
        return int(torch.as_tensor(sigma_full).shape[-1])
    return int(grid.get("nlevels", -1))


def _external_sigma_full(grid: Mapping[str, Any], ref: torch.Tensor) -> torch.Tensor:
    sigma_full = grid.get("sigma_full")
    if sigma_full is None:
        raise ValueError("external SCM grid must expose sigma_full for sigma remapping")
    sigma = torch.as_tensor(sigma_full, dtype=ref.dtype, device=ref.device)
    if sigma.ndim == 1:
        sigma = sigma.reshape(-1)
    elif sigma.ndim == 2:
        if int(sigma.shape[0]) == int(ref.shape[0]):
            pass
        elif int(sigma.shape[0]) == 1:
            sigma = sigma.expand(int(ref.shape[0]), -1)
        else:
            raise ValueError(
                f"external SCM grid sigma_full has incompatible shape {tuple(sigma.shape)} "
                f"for nCells={int(ref.shape[0])}"
            )
    else:
        raise ValueError(f"external SCM grid sigma_full must be 1-D or 2-D, got {tuple(sigma.shape)}")
    if int(sigma.shape[-1]) <= 0:
        raise ValueError("external SCM grid sigma_full cannot be empty")
    return sigma.clamp(0.0, 1.0)


def _external_sigma_interface(grid: Mapping[str, Any], ref: torch.Tensor) -> torch.Tensor:
    sigma_half = grid.get("sigma_half")
    if sigma_half is None:
        raise ValueError("external SCM grid must expose sigma_half for conservative pressure remapping")
    sigma = torch.as_tensor(sigma_half, dtype=ref.dtype, device=ref.device)
    if sigma.ndim == 1:
        sigma = sigma.reshape(-1)
    elif sigma.ndim == 2:
        if int(sigma.shape[0]) == int(ref.shape[0]):
            pass
        elif int(sigma.shape[0]) == 1:
            sigma = sigma.expand(int(ref.shape[0]), -1)
        else:
            raise ValueError(
                f"external SCM grid sigma_half has incompatible shape {tuple(sigma.shape)} "
                f"for nCells={int(ref.shape[0])}"
            )
    else:
        raise ValueError(f"external SCM grid sigma_half must be 1-D or 2-D, got {tuple(sigma.shape)}")
    if int(sigma.shape[-1]) <= 1:
        raise ValueError("external SCM grid sigma_half must contain at least two interfaces")
    return sigma.clamp(0.0, 1.0)


def _vfs_sigma_full(columns: Mapping[str, Any], ref: torch.Tensor) -> torch.Tensor:
    p_full = _require_column_tensor(columns, "p_full").to(dtype=ref.dtype, device=ref.device)
    ps = _require_column_tensor(columns, "surface_pressure").to(dtype=ref.dtype, device=ref.device)
    ps = ps.reshape(ps.shape[0], -1)
    if int(ps.shape[-1]) != 1:
        raise ValueError("surface_pressure must have one value per column for sigma remapping")
    return (p_full / ps.clamp_min(torch.finfo(ref.dtype).tiny)).clamp(0.0, 1.0)


def _vfs_pressure_interface(
    columns: Mapping[str, Any],
    ref: torch.Tensor,
    *,
    surface_pressure: torch.Tensor | None = None,
) -> torch.Tensor:
    if surface_pressure is None:
        return _require_column_tensor(columns, "p_interface").to(dtype=ref.dtype, device=ref.device)
    state = columns.get("state")
    if state is None or getattr(state, "column", None) is None:
        raise ValueError("VFS target pressure remap with updated ps requires columns['state'].column")
    return state.column.interface_pressure(surface_pressure).to(dtype=ref.dtype, device=ref.device)


def _external_pressure_interface(
    columns: Mapping[str, Any],
    grid: Mapping[str, Any],
    ref: torch.Tensor,
    *,
    surface_pressure: torch.Tensor | None = None,
) -> torch.Tensor:
    if bool(grid.get("_vfs_dycore_native", False)):
        if surface_pressure is not None:
            return _vfs_pressure_interface(columns, ref, surface_pressure=surface_pressure)
        p_interface = grid.get("p_interface")
        if p_interface is None:
            raise ValueError("dycore-native external SCM grid must expose p_interface")
        return torch.as_tensor(p_interface, dtype=ref.dtype, device=ref.device)
    sigma_half = _external_sigma_interface(grid, ref)
    vfs_i = _vfs_pressure_interface(columns, ref, surface_pressure=surface_pressure)
    p_top = vfs_i[:, :1]
    p_bottom = vfs_i[:, -1:]
    active_dp = (p_bottom - p_top).clamp_min(torch.finfo(ref.dtype).tiny)
    if sigma_half.ndim == 1:
        sigma_half = sigma_half.reshape(1, -1)
    return p_top + sigma_half * active_dp


def pressure_overlap_remap(
    source_layer_mean: torch.Tensor,
    source_p_interface: torch.Tensor,
    target_p_interface: torch.Tensor,
    *,
    fill: str = "zero",
) -> torch.Tensor:
    """Pressure-overlap remap of layer-mean quantities.

    The remap conserves the pressure-thickness integral of
    ``source_layer_mean`` over the overlapping source/target pressure domain.
    ``fill='zero'`` gives uncovered target pressure thickness zero content.
    ``fill='nearest'`` fills uncovered target thickness with nearest source
    layer values, which is useful for primitive thermodynamic fields when the
    external column extends beyond the VFS model top.
    """

    if fill not in {"zero", "nearest"}:
        raise ValueError("fill must be 'zero' or 'nearest'")
    values = torch.as_tensor(source_layer_mean)
    if values.ndim != 2:
        raise ValueError(f"pressure remap expects [nCells, nLevels] fields, got {tuple(values.shape)}")
    source_i = torch.as_tensor(source_p_interface, dtype=values.dtype, device=values.device)
    target_i = torch.as_tensor(target_p_interface, dtype=values.dtype, device=values.device)
    if source_i.shape != values.shape[:-1] + (values.shape[-1] + 1,):
        raise ValueError(
            "source pressure interfaces must have one more vertical level than source values; "
            f"got interfaces={tuple(source_i.shape)} values={tuple(values.shape)}"
        )
    if target_i.ndim != 2 or int(target_i.shape[0]) != int(values.shape[0]) or int(target_i.shape[-1]) < 2:
        raise ValueError(
            "target pressure interfaces must have shape [nCells, nTargetLevels + 1]; "
            f"got {tuple(target_i.shape)} for nCells={int(values.shape[0])}"
        )

    source_i, values = _increasing_interfaces_and_layers(source_i, values)
    target_i, _ = _increasing_interfaces_and_layers(target_i, None)
    source_top = source_i[:, :-1]
    source_bottom = source_i[:, 1:]
    target_top = target_i[:, :-1]
    target_bottom = target_i[:, 1:]
    target_dp = (target_bottom - target_top).clamp_min(torch.finfo(values.dtype).tiny)
    overlap_top = torch.maximum(target_top[:, :, None], source_top[:, None, :])
    overlap_bottom = torch.minimum(target_bottom[:, :, None], source_bottom[:, None, :])
    overlap = (overlap_bottom - overlap_top).clamp_min(0.0)
    weighted = (overlap * values[:, None, :]).sum(dim=-1)
    covered = overlap.sum(dim=-1)
    if fill == "nearest":
        source_mid = 0.5 * (source_top + source_bottom)
        target_mid = 0.5 * (target_top + target_bottom)
        nearest = _linear_interp_columns(source_mid, values, target_mid)
        weighted = weighted + nearest * (target_dp - covered).clamp_min(0.0)
    return weighted / target_dp


def _increasing_interfaces_and_layers(
    interfaces: torch.Tensor,
    layers: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if bool((interfaces[:, -1] < interfaces[:, 0]).all().detach().cpu().item()):
        interfaces = torch.flip(interfaces, dims=(-1,))
        if layers is not None:
            layers = torch.flip(layers, dims=(-1,))
    return interfaces, layers


def _linear_interp_columns(
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    target_x: torch.Tensor,
) -> torch.Tensor:
    """Linearly interpolate column fields along the last dimension.

    Coordinates are expected to be ordered vertically.  Values outside the
    source coordinate range are clamped to the nearest source level.
    """

    if source_y.ndim != 2:
        raise ValueError(f"vertical remap expects [nCells, nLevels] fields, got {tuple(source_y.shape)}")
    n_cells, n_source = int(source_y.shape[0]), int(source_y.shape[1])
    if n_source <= 0:
        raise ValueError("vertical remap requires at least one source level")
    sx = torch.as_tensor(source_x, dtype=source_y.dtype, device=source_y.device)
    tx = torch.as_tensor(target_x, dtype=source_y.dtype, device=source_y.device)
    if sx.ndim == 1:
        sx = torch.broadcast_to(sx.reshape(1, -1), (n_cells, int(sx.numel())))
    if tx.ndim == 1:
        tx = torch.broadcast_to(tx.reshape(1, -1), (n_cells, int(tx.numel())))
    if sx.shape != source_y.shape:
        raise ValueError(f"source coordinates shape {tuple(sx.shape)} must match field shape {tuple(source_y.shape)}")
    if int(tx.shape[0]) != n_cells:
        raise ValueError(f"target coordinates must have nCells={n_cells}, got {tuple(tx.shape)}")
    if n_source == 1:
        return torch.broadcast_to(source_y[:, :1], tx.shape)

    if bool((sx[:, -1] < sx[:, 0]).all().detach().cpu().item()):
        sx = torch.flip(sx, dims=(-1,))
        source_y = torch.flip(source_y, dims=(-1,))
    tiny = torch.finfo(source_y.dtype).tiny
    tx = tx.clamp(min=sx[:, :1], max=sx[:, -1:])
    idx_hi = torch.searchsorted(sx.contiguous(), tx.contiguous(), right=False)
    idx_hi = idx_hi.clamp(1, n_source - 1)
    idx_lo = idx_hi - 1
    x0 = torch.gather(sx, -1, idx_lo)
    x1 = torch.gather(sx, -1, idx_hi)
    y0 = torch.gather(source_y, -1, idx_lo)
    y1 = torch.gather(source_y, -1, idx_hi)
    weight = (tx - x0) / (x1 - x0).abs().clamp_min(tiny)
    return y0 + weight * (y1 - y0)


def _conservative_nonnegative_layer_mean(
    layer_mean: torch.Tensor,
    layer_dp: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return nonnegative layer means preserving each column pressure integral.

    The pressure-overlap remap should preserve nonnegativity for nonnegative
    inputs.  This repair is a final invariant guard for roundoff or future
    remap variants: negative layer content is clipped, then the positive
    content in the same column is scaled down so the original column integral
    is preserved whenever a nonnegative solution is feasible.
    """

    values = torch.as_tensor(layer_mean)
    weights = torch.as_tensor(layer_dp, dtype=values.dtype, device=values.device).clamp_min(
        torch.finfo(values.dtype).tiny
    )
    raw_content = values * weights
    desired_total = raw_content.sum(dim=-1, keepdim=True)
    feasible_total = desired_total.clamp_min(0.0)
    clipped_content = raw_content.clamp_min(0.0)
    clipped_total = clipped_content.sum(dim=-1, keepdim=True)
    scale = torch.where(
        clipped_total > torch.finfo(values.dtype).tiny,
        feasible_total / clipped_total.clamp_min(torch.finfo(values.dtype).tiny),
        torch.zeros_like(clipped_total),
    )
    repaired_content = clipped_content * scale
    repaired = repaired_content / weights
    residual = (repaired_content.sum(dim=-1, keepdim=True) - feasible_total).abs()
    diagnostics = {
        "negative_content_pressure_integral": (-raw_content).clamp_min(0.0).sum(dim=-1, keepdim=True),
        "adjustment_abs_pressure_integral": (repaired_content - raw_content).abs().sum(dim=-1, keepdim=True),
        "residual_pressure_integral": residual,
        "infeasible_negative_total_pressure_integral": (-desired_total).clamp_min(0.0),
    }
    return repaired, diagnostics


def _project_moisture_targets_to_surface_budget(
    *,
    vapor_target: torch.Tensor,
    target_dp: torch.Tensor,
    old_vapor_content: torch.Tensor,
    dt: torch.Tensor,
    surface_water_to_atmosphere: torch.Tensor,
    gravity: float,
    cloud_target: torch.Tensor | None = None,
    old_cloud_content: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    """Project target moisture to the SCM-reported surface water exchange.

    The external SCM returns both a post-physics humidity profile and separate
    precipitation/evaporation diagnostics.  For coupled VFS runs the latter is
    the conservative transfer routed to the surface reservoir, so the returned
    VFS target water column must have the same integral.  This preserves the
    SCM vertical target shape by applying a column-wise scale factor; if the
    target profile has no positive water but the budget demands water, the
    fallback is a uniform vapor distribution so conservation still wins.
    """

    vapor = torch.as_tensor(vapor_target)
    weights = torch.as_tensor(target_dp, dtype=vapor.dtype, device=vapor.device).clamp_min(
        torch.finfo(vapor.dtype).tiny
    )
    old_vapor = torch.as_tensor(old_vapor_content, dtype=vapor.dtype, device=vapor.device)
    dt_t = dt.to(dtype=vapor.dtype, device=vapor.device)
    surface = torch.as_tensor(surface_water_to_atmosphere, dtype=vapor.dtype, device=vapor.device).reshape(
        vapor.shape[0], 1
    )
    g_t = torch.as_tensor(float(gravity), dtype=vapor.dtype, device=vapor.device)

    raw_vapor_content = vapor * weights
    raw_cloud_content = None if cloud_target is None else torch.as_tensor(
        cloud_target,
        dtype=vapor.dtype,
        device=vapor.device,
    ) * weights
    old_cloud = (
        torch.zeros_like(old_vapor)
        if old_cloud_content is None
        else torch.as_tensor(old_cloud_content, dtype=vapor.dtype, device=vapor.device)
    )
    raw_total_pressure = raw_vapor_content.sum(dim=-1, keepdim=True)
    if raw_cloud_content is not None:
        raw_total_pressure = raw_total_pressure + raw_cloud_content.sum(dim=-1, keepdim=True)
    old_total_pressure = old_vapor.sum(dim=-1, keepdim=True) + old_cloud.sum(dim=-1, keepdim=True)
    requested_surface_pressure = surface * g_t * dt_t
    desired_total_pressure = old_total_pressure + requested_surface_pressure
    feasible_total_pressure = desired_total_pressure.clamp_min(0.0)

    tiny = torch.finfo(vapor.dtype).tiny
    scale = torch.where(
        raw_total_pressure > tiny,
        feasible_total_pressure / raw_total_pressure.clamp_min(tiny),
        torch.zeros_like(raw_total_pressure),
    )
    projected_vapor_content = raw_vapor_content * scale
    projected_cloud_content = None if raw_cloud_content is None else raw_cloud_content * scale

    use_uniform_vapor = (raw_total_pressure <= tiny) & (feasible_total_pressure > tiny)
    weight_sum = weights.sum(dim=-1, keepdim=True).clamp_min(tiny)
    uniform_vapor_content = weights / weight_sum * feasible_total_pressure
    projected_vapor_content = torch.where(use_uniform_vapor, uniform_vapor_content, projected_vapor_content)
    if projected_cloud_content is not None:
        projected_cloud_content = torch.where(
            use_uniform_vapor,
            torch.zeros_like(projected_cloud_content),
            projected_cloud_content,
        )

    projected_total_pressure = projected_vapor_content.sum(dim=-1, keepdim=True)
    if projected_cloud_content is not None:
        projected_total_pressure = projected_total_pressure + projected_cloud_content.sum(dim=-1, keepdim=True)
    projected_vapor = projected_vapor_content / weights
    projected_cloud = None if projected_cloud_content is None else projected_cloud_content / weights
    correction_pressure = projected_total_pressure - raw_total_pressure
    residual_pressure = projected_total_pressure - feasible_total_pressure
    diagnostics = {
        "vfs_water_budget_projection_surface_water_to_atmosphere_kg_m2_s": surface,
        "vfs_water_budget_projection_raw_total_column_water_kg_m2": raw_total_pressure / g_t,
        "vfs_water_budget_projection_desired_total_column_water_kg_m2": desired_total_pressure / g_t,
        "vfs_water_budget_projection_final_total_column_water_kg_m2": projected_total_pressure / g_t,
        "vfs_water_budget_projection_correction_kg_m2": correction_pressure / g_t,
        "vfs_water_budget_projection_abs_correction_kg_m2": correction_pressure.abs() / g_t,
        "vfs_water_budget_projection_residual_kg_m2": residual_pressure / g_t,
        "vfs_water_budget_projection_abs_residual_kg_m2": residual_pressure.abs() / g_t,
        "vfs_water_budget_projection_scale_factor": scale,
        "vfs_water_budget_projection_infeasible_deficit_kg_m2": (-desired_total_pressure).clamp_min(0.0) / g_t,
        "vfs_water_budget_projection_uniform_fallback_count": use_uniform_vapor.to(dtype=vapor.dtype),
    }
    return projected_vapor, projected_cloud, diagnostics


def _tracer_target_tendency(
    *,
    state_ref,
    target_dp: torch.Tensor,
    dt: torch.Tensor,
    tracer_index: int | None,
    target_mixing_ratio: torch.Tensor,
    tracer_m_dt: torch.Tensor | None = None,
) -> torch.Tensor:
    if tracer_index is None:
        raise ValueError("target-state external SCM update requires a named tracer index")
    old = state_ref.tracer_m
    if old is None:
        raise ValueError("target-state external SCM update requires State.tracer_m")
    out = torch.zeros_like(old) if tracer_m_dt is None else tracer_m_dt
    target_content = target_dp.to(dtype=old.dtype, device=old.device) * target_mixing_ratio.to(
        dtype=old.dtype,
        device=old.device,
    )
    dt_old = dt.to(dtype=old.dtype, device=old.device)
    if old.ndim == 2:
        if int(tracer_index) != 0:
            raise ValueError("2D tracer_m can only address tracer index 0")
        out[...] = (target_content - old) / dt_old
    else:
        out[..., int(tracer_index)] = (target_content - old[..., int(tracer_index)]) / dt_old
    return out


def _project_vapor_after_tracer_tendency(
    *,
    state_ref,
    target_dp: torch.Tensor,
    dt: torch.Tensor,
    tracer_m_dt: torch.Tensor,
) -> torch.Tensor:
    index = vapor_tracer_index(state_ref)
    if index is None or state_ref.tracer_m is None:
        raise ValueError("projected vapor diagnostics require named vapor tracer_m")
    old = state_ref.tracer_m
    dt_old = dt.to(dtype=old.dtype, device=old.device)
    if old.ndim == 2:
        content = old + dt_old * tracer_m_dt
    else:
        content = old[..., int(index)] + dt_old * tracer_m_dt[..., int(index)]
    dp = target_dp.to(dtype=content.dtype, device=content.device).clamp_min(torch.finfo(content.dtype).tiny)
    return content / dp


def _state_tensor(state: Mapping[str, Any], name: str, *, ref: torch.Tensor) -> torch.Tensor:
    if name not in state:
        raise ValueError(f"external SCM returned no state[{name!r}]")
    return torch.as_tensor(state[name], dtype=ref.dtype, device=ref.device)


def _tensor_min(value: torch.Tensor) -> float:
    detached = value.detach()
    finite = detached[torch.isfinite(detached)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.to(dtype=torch.float32).min().cpu().item())


def _tensor_max(value: torch.Tensor) -> float:
    detached = value.detach()
    finite = detached[torch.isfinite(detached)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.to(dtype=torch.float32).max().cpu().item())


def _negative_count(value: torch.Tensor) -> int:
    return int((value.detach() < 0.0).to(dtype=torch.int64).sum().cpu().item())


def _mapping_tensor(mapping: Mapping[str, Any], name: str, *, ref: torch.Tensor) -> torch.Tensor | None:
    if name not in mapping:
        return None
    return torch.as_tensor(mapping[name], dtype=ref.dtype, device=ref.device).reshape(-1)


def _optional_state_tensor(state: Mapping[str, Any], name: str, *, default: torch.Tensor) -> torch.Tensor:
    if name not in state:
        return default
    return torch.as_tensor(state[name], dtype=default.dtype, device=default.device).reshape_as(default)


def _physics_state_get(physics_state: Mapping[str, Any] | None, name: str, default: Any) -> Any:
    if not isinstance(physics_state, Mapping):
        return default
    return physics_state.get(name, default)


def _physics_state_step_count(value: Any) -> int:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.ndim != 0:
            raise ValueError("radiation_step_count must be scalar")
        return int(tensor.item())
    return int(value)


def _carried_tensor(value: Any, name: str, default: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, Mapping) or name not in value:
        return default
    return torch.as_tensor(value[name], dtype=default.dtype, device=default.device).reshape_as(default)


def _slab_dtype(ref: torch.Tensor) -> torch.dtype:
    return ref.dtype if ref.device.type == "mps" else torch.float64


def _dry_mixing_ratio_from_specific_cloud(q_v: torch.Tensor, q_l: torch.Tensor) -> torch.Tensor:
    denom = (1.0 - q_v - q_l).clamp_min(torch.finfo(q_v.dtype).tiny)
    return q_l / denom


__all__ = [
    "ExternalSCMContract",
    "WellmeyerGCMColumnAdapter",
    "pressure_overlap_remap",
]
