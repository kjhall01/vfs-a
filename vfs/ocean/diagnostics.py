"""Detached diagnostics and status helpers for VFS-O."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch




def scalar(value: Any) -> float | int | str | bool | None:
    """Return a detached Python scalar when possible."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return float(value)
    if torch.is_tensor(value):
        t = value.detach()
        if t.numel() == 0:
            return None
        if t.numel() == 1:
            item = t.reshape(()).cpu().item()
        else:
            item = t.cpu().to(dtype=torch.float64).mean().item()
        if isinstance(item, bool):
            return bool(item)
        if isinstance(item, int):
            return int(item)
        return float(item)
    try:
        return float(value)
    except Exception:
        return None

def ocean_maybe_call_field(component: Any, name: str, state: Any) -> torch.Tensor | None:
    method = getattr(component, name, None)
    if method is None:
        return None
    return method(state)


def ocean_rms(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return scalar(value.detach().square().mean().sqrt())


def ocean_max_abs(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return scalar(value.detach().abs().max())


def ocean_abs_quantile(value: torch.Tensor | None, quantile: float) -> float | None:
    if value is None:
        return None
    flat = value.detach().abs().reshape(-1).cpu()
    if int(flat.numel()) == 0:
        return None
    return float(torch.quantile(flat.to(dtype=torch.float64), float(quantile)).item())


def ocean_safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if abs(float(denominator)) <= 1.0e-300:
        return None
    return float(numerator) / float(denominator)


def _finite_row_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def ocean_pressure_work_summary_from_rows(
    rows: Sequence[dict[str, Any]],
    *,
    key_prefix: str = "",
    summary_prefix: str = "",
) -> dict[str, float | None]:
    """Return compact max/latest ocean pressure-work diagnostics from rows."""

    def value(row: dict[str, Any], key: str) -> float | None:
        return _finite_row_float(row.get(f"{key_prefix}{key}"))

    def max_abs(key: str) -> float | None:
        values = [abs(item) for row in rows if (item := value(row, key)) is not None]
        return max(values) if values else None

    latest = None
    for row in rows:
        if value(row, "pressure_work_total") is not None:
            latest = row
    latest_values = {
        f"{summary_prefix}latest_pressure_work_total": None if latest is None else value(latest, "pressure_work_total"),
        f"{summary_prefix}latest_closed_pressure_work_residual_relative": None
        if latest is None
        else value(latest, "closed_pressure_work_residual_relative"),
        f"{summary_prefix}latest_total_pressure_energy_residual_relative": None
        if latest is None
        else value(latest, "total_pressure_energy_residual_relative"),
    }
    return {
        f"{summary_prefix}max_free_surface_work_residual_relative": max_abs("free_surface_work_residual_relative"),
        f"{summary_prefix}max_baroclinic_work_residual_relative": max_abs("baroclinic_work_residual_relative"),
        f"{summary_prefix}max_closed_pressure_work_residual_relative": max_abs("closed_pressure_work_residual_relative"),
        f"{summary_prefix}max_total_pressure_energy_residual_relative": max_abs("total_pressure_energy_residual_relative"),
        f"{summary_prefix}max_total_pressure_energy_residual_abs": max_abs("total_pressure_energy_residual_abs"),
        f"{summary_prefix}max_baroclinic_metric_correction_work_abs": max_abs("baroclinic_metric_correction_work"),
        **latest_values,
    }


def ocean_integrated_cell_total(mesh, value: torch.Tensor) -> torch.Tensor:
    return mesh.integrate_cell(value.sum(dim=-1, keepdim=True)).sum()


def ocean_all_finite(state: Any) -> bool:
    tensors = [state.u_e, state.h, state.temp_h, state.salt_h]
    if state.tracer_h is not None:
        tensors.append(state.tracer_h)
    return all(bool(torch.isfinite(value).all().detach().cpu().item()) for value in tensors if value is not None)






class OceanDiagnostics:
    """Detached summary facade for VFS-O dynamics and optional physics."""

    def __init__(self, dynamics, *, physics=None, provider=None) -> None:
        self.dynamics = dynamics
        self.physics = physics
        self.provider = provider if provider is not None else dynamics

    @property
    def domain(self):
        return getattr(self.dynamics, "domain", None)

    def _total_volume(self, state) -> torch.Tensor:
        if self.domain is None:
            return state.total_volume().sum()
        return self.domain.total_volume(state).sum()

    def _total_heat_content(self, state) -> torch.Tensor:
        if self.domain is None:
            return state.total_heat_content().sum()
        return self.domain.total_heat_content(state).sum()

    def _total_salt_content(self, state) -> torch.Tensor:
        if self.domain is None:
            return state.total_salt_content().sum()
        return self.domain.total_salt_content(state).sum()

    def domain_summary(self, state) -> dict[str, Any]:
        mesh = state.mesh
        if self.domain is None:
            return {
                "wet_cell_count": int(mesh.nCells),
                "active_edge_count": int(mesh.nEdges),
                "coast_edge_count": 0,
                "wet_area_fraction": 1.0,
            }
        wet_cell_count = int(self.domain.wet_cell.sum().detach().cpu().item())
        active_edge_count = int(self.domain.active_edge.sum().detach().cpu().item())
        coast_edge_count = int(self.domain.coast_edge.sum().detach().cpu().item())
        wet = self.domain.wet_cell.to(dtype=mesh.cell_area.dtype, device=mesh.cell_area.device)
        wet_area = (mesh.cell_area * wet).sum()
        wet_area_fraction = wet_area / mesh.cell_area.sum().clamp_min(torch.finfo(mesh.cell_area.dtype).tiny)
        return {
            "wet_cell_count": wet_cell_count,
            "active_edge_count": active_edge_count,
            "coast_edge_count": coast_edge_count,
            "wet_area_fraction": scalar(wet_area_fraction),
        }

    def state_summary(self, state) -> dict[str, Any]:
        eta = state.eta()
        density = self.dynamics.density(state)
        out = {
            "finite": ocean_all_finite(state),
            "h_min": scalar(state.h.min()),
            "h_max": scalar(state.h.max()),
            "eta_min": scalar(eta.min()),
            "eta_max": scalar(eta.max()),
            "eta_rms": scalar(eta.square().mean().sqrt()),
            "u_max": scalar(state.u_e.abs().max()),
            "u_rms": scalar(state.u_e.square().mean().sqrt()),
            "top_u_max": scalar(state.u_e[..., :1].abs().max()),
            "bottom_u_max": scalar(state.u_e[..., -1:].abs().max()),
            "temperature_min": scalar(state.temperature().min()),
            "temperature_max": scalar(state.temperature().max()),
            "top_temperature_min": scalar(state.temperature()[..., :1].min()),
            "top_temperature_max": scalar(state.temperature()[..., :1].max()),
            "salinity_min": scalar(state.salinity().min()),
            "salinity_max": scalar(state.salinity().max()),
            "top_salinity_min": scalar(state.salinity()[..., :1].min()),
            "top_salinity_max": scalar(state.salinity()[..., :1].max()),
            "density_min": scalar(density.min()),
            "density_max": scalar(density.max()),
        }
        out.update(self.domain_summary(state))
        return out

    def conservation_summary(self, state0, state) -> dict[str, Any]:
        volume0 = self._total_volume(state0)
        heat0 = self._total_heat_content(state0)
        salt0 = self._total_salt_content(state0)
        volume = self._total_volume(state)
        heat = self._total_heat_content(state)
        salt = self._total_salt_content(state)
        return {
            "volume": scalar(volume),
            "volume_rel_to_initial": scalar((volume - volume0) / volume0.abs().clamp_min(1.0e-300)),
            "heat_rel_to_initial": scalar((heat - heat0) / heat0.abs().clamp_min(1.0e-300)),
            "salt_rel_to_initial": scalar((salt - salt0) / salt0.abs().clamp_min(1.0e-300)),
        }

    def energy_summary(self, state) -> dict[str, Any]:
        return {
            "kinetic_energy": scalar(self.dynamics.kinetic_energy(state).sum()),
            "free_surface_potential_energy": scalar(self.dynamics.free_surface_potential_energy(state).sum()),
            "total_energy": scalar(self.dynamics.total_energy(state).sum()),
        }

    def tendency_summary(self, state, *, dt: float | None = None) -> dict[str, Any]:
        rhs = self.provider.tendency(state)
        h_tend_scale = state.h.abs().clamp_min(1.0).max()
        return {
            "rhs_u_rms": scalar(rhs.u_e.square().mean().sqrt()),
            "rhs_h_rms": scalar(rhs.h.square().mean().sqrt()),
            "h_step_fraction_max": None if dt is None else scalar((float(dt) * rhs.h.abs() / h_tend_scale).max()),
            "vorticity_rms": scalar(self.dynamics.vorticity(state).square().mean().sqrt()),
        }

    def tendency_budget_summary(self, state) -> dict[str, Any]:
        method = getattr(self.dynamics, "tendency_budget_terms", None)
        if method is None:
            return {
                "volume_tendency_integral": None,
                "heat_tendency_integral": None,
                "salt_tendency_integral": None,
                "tracer_tendency_integral_max_abs": None,
            }
        budget = method(state)
        tracer = budget.get("tracer_tendency")
        return {
            "volume_tendency_integral": scalar(budget["volume_tendency"]),
            "heat_tendency_integral": scalar(budget["heat_tendency"]),
            "salt_tendency_integral": scalar(budget["salt_tendency"]),
            "tracer_tendency_integral_max_abs": None if tracer is None else scalar(tracer.abs().max()),
        }

    def coriolis_work_summary(self, state) -> dict[str, Any]:
        method = getattr(self.dynamics, "coriolis_work_terms", None)
        if method is None:
            return {
                "coriolis_work": None,
                "coriolis_work_abs_scale": None,
                "coriolis_work_relative": None,
            }
        work = method(state)
        return {
            "coriolis_work": scalar(work["work"]),
            "coriolis_work_abs_scale": scalar(work["work_abs_scale"]),
            "coriolis_work_relative": scalar(work["work_relative"]),
        }

    def pressure_work_summary(self, state) -> dict[str, Any]:
        method = getattr(self.dynamics, "pressure_work_terms", None)
        if method is None:
            return {
                "pressure_work_total": None,
                "free_surface_pressure_work": None,
                "baroclinic_pressure_work": None,
                "baroclinic_gradient_pressure_work": None,
                "baroclinic_metric_correction_work": None,
                "free_surface_energy_rate": None,
                "baroclinic_pressure_potential_energy_rate": None,
                "closed_pressure_work": None,
                "closed_pressure_energy_rate": None,
                "free_surface_work_residual": None,
                "free_surface_work_residual_abs": None,
                "free_surface_work_residual_relative": None,
                "baroclinic_work_residual": None,
                "baroclinic_work_residual_abs": None,
                "baroclinic_work_residual_relative": None,
                "closed_pressure_work_residual": None,
                "closed_pressure_work_residual_abs": None,
                "closed_pressure_work_residual_relative": None,
                "total_pressure_energy_residual": None,
                "total_pressure_energy_residual_abs": None,
                "total_pressure_energy_residual_relative": None,
            }
        work = method(state)
        return {
            "pressure_work_total": scalar(work["pressure_work_total"]),
            "free_surface_pressure_work": scalar(work["free_surface_pressure_work"]),
            "baroclinic_pressure_work": scalar(work["baroclinic_pressure_work"]),
            "baroclinic_gradient_pressure_work": scalar(work["baroclinic_gradient_pressure_work"]),
            "baroclinic_metric_correction_work": scalar(work["baroclinic_metric_correction_work"]),
            "free_surface_energy_rate": scalar(work["free_surface_energy_rate"]),
            "baroclinic_pressure_potential_energy_rate": scalar(work["baroclinic_pressure_potential_energy_rate"]),
            "closed_pressure_work": scalar(work["closed_pressure_work"]),
            "closed_pressure_energy_rate": scalar(work["closed_pressure_energy_rate"]),
            "free_surface_work_residual": scalar(work["free_surface_work_residual"]),
            "free_surface_work_residual_abs": scalar(work["free_surface_work_residual_abs"]),
            "free_surface_work_residual_relative": scalar(work["free_surface_work_residual_relative"]),
            "baroclinic_work_residual": scalar(work["baroclinic_work_residual"]),
            "baroclinic_work_residual_abs": scalar(work["baroclinic_work_residual_abs"]),
            "baroclinic_work_residual_relative": scalar(work["baroclinic_work_residual_relative"]),
            "closed_pressure_work_residual": scalar(work["closed_pressure_work_residual"]),
            "closed_pressure_work_residual_abs": scalar(work["closed_pressure_work_residual_abs"]),
            "closed_pressure_work_residual_relative": scalar(work["closed_pressure_work_residual_relative"]),
            "total_pressure_energy_residual": scalar(work["total_pressure_energy_residual"]),
            "total_pressure_energy_residual_abs": scalar(work["total_pressure_energy_residual_abs"]),
            "total_pressure_energy_residual_relative": scalar(work["total_pressure_energy_residual_relative"]),
        }

    def forcing_summary(self, state) -> dict[str, Any]:
        if self.physics is None:
            return {
                "target_surface_temperature_min": None,
                "target_surface_temperature_max": None,
                "target_surface_salinity_min": None,
                "target_surface_salinity_max": None,
                "surface_heat_flux_min": None,
                "surface_heat_flux_max": None,
                "surface_heat_flux_global_integral": None,
                "freshwater_flux_min": None,
                "freshwater_flux_max": None,
                "freshwater_flux_global_integral": None,
                "wind_stress_edge_max_abs": None,
                "forcing_u_rms": None,
                "forcing_temp_h_rms": None,
                "forcing_salt_h_rms": None,
                "forcing_volume_tendency_integral": None,
                "forcing_heat_tendency_integral": None,
                "forcing_salt_tendency_integral": None,
                "mixed_layer_depth_min": None,
                "mixed_layer_depth_mean": None,
                "mixed_layer_depth_max": None,
                "mixed_layer_scalar_diffusivity_max": None,
                "mixed_layer_momentum_viscosity_max": None,
                "convective_interface_fraction": None,
                "unstable_column_fraction": None,
                "convective_adjustment_depth_mean": None,
                "convective_adjustment_depth_max": None,
                "surface_stress_effective_depth_mean": None,
                "surface_stress_effective_depth_max": None,
                "surface_stress_redistribution_u_rms": None,
            }
        forcing_rhs = self.physics.tendency(state)
        target_temp = ocean_maybe_call_field(self.physics, "surface_temperature_target", state)
        target_salt = ocean_maybe_call_field(self.physics, "surface_salinity_target", state)
        tau_edge = ocean_maybe_call_field(self.physics, "wind_stress_edge", state)
        heat_flux = ocean_maybe_call_field(self.physics, "surface_heat_flux", state)
        freshwater_flux = ocean_maybe_call_field(self.physics, "freshwater_flux", state)
        mixed_layer_method = getattr(self.physics, "mixed_layer_diagnostics", None)
        mixed_layer = {} if mixed_layer_method is None else mixed_layer_method(state)
        mixed_layer_depth = mixed_layer.get("mixed_layer_depth")
        scalar_kappa = mixed_layer.get("scalar_interface_diffusivity")
        momentum_viscosity = mixed_layer.get("momentum_interface_viscosity")
        convective_mask = mixed_layer.get("convective_interface_mask")
        unstable_column_mask = mixed_layer.get("unstable_column_mask")
        convective_adjustment_depth = mixed_layer.get("convective_adjustment_depth")
        stress_effective_depth = mixed_layer.get("surface_stress_effective_depth")
        stress_redistribution = mixed_layer.get("surface_stress_redistribution_tendency")
        mesh = state.mesh
        if self.domain is not None:
            wet = self.domain.wet_cell.to(device=state.h.device)
            active = self.domain.active_edge.to(device=state.u_e.device)
            mixed_layer_depth = None if mixed_layer_depth is None else mixed_layer_depth[wet]
            scalar_kappa = None if scalar_kappa is None else scalar_kappa[wet]
            convective_mask = None if convective_mask is None else convective_mask[wet]
            unstable_column_mask = None if unstable_column_mask is None else unstable_column_mask[wet]
            convective_adjustment_depth = None if convective_adjustment_depth is None else convective_adjustment_depth[wet]
            momentum_viscosity = None if momentum_viscosity is None else momentum_viscosity[active]
            stress_effective_depth = None if stress_effective_depth is None else stress_effective_depth[active]
            stress_redistribution = None if stress_redistribution is None else stress_redistribution[active]
        return {
            "target_surface_temperature_min": None if target_temp is None else scalar(target_temp.min()),
            "target_surface_temperature_max": None if target_temp is None else scalar(target_temp.max()),
            "target_surface_salinity_min": None if target_salt is None else scalar(target_salt.min()),
            "target_surface_salinity_max": None if target_salt is None else scalar(target_salt.max()),
            "surface_heat_flux_min": None if heat_flux is None else scalar(heat_flux.min()),
            "surface_heat_flux_max": None if heat_flux is None else scalar(heat_flux.max()),
            "surface_heat_flux_global_integral": None if heat_flux is None else scalar(mesh.integrate_cell(heat_flux).sum()),
            "freshwater_flux_min": None if freshwater_flux is None else scalar(freshwater_flux.min()),
            "freshwater_flux_max": None if freshwater_flux is None else scalar(freshwater_flux.max()),
            "freshwater_flux_global_integral": None if freshwater_flux is None else scalar(mesh.integrate_cell(freshwater_flux).sum()),
            "wind_stress_edge_max_abs": None if tau_edge is None else scalar(tau_edge.abs().max()),
            "forcing_u_rms": scalar(forcing_rhs.u_e.square().mean().sqrt()),
            "forcing_temp_h_rms": scalar(forcing_rhs.temp_h.square().mean().sqrt()),
            "forcing_salt_h_rms": scalar(forcing_rhs.salt_h.square().mean().sqrt()),
            "forcing_volume_tendency_integral": scalar(ocean_integrated_cell_total(mesh, forcing_rhs.h)),
            "forcing_heat_tendency_integral": scalar(ocean_integrated_cell_total(mesh, forcing_rhs.temp_h)),
            "forcing_salt_tendency_integral": scalar(ocean_integrated_cell_total(mesh, forcing_rhs.salt_h)),
            "mixed_layer_depth_min": None if mixed_layer_depth is None else scalar(mixed_layer_depth.min()),
            "mixed_layer_depth_mean": None if mixed_layer_depth is None else scalar(mixed_layer_depth.mean()),
            "mixed_layer_depth_max": None if mixed_layer_depth is None else scalar(mixed_layer_depth.max()),
            "mixed_layer_scalar_diffusivity_max": None if scalar_kappa is None else scalar(scalar_kappa.max()),
            "mixed_layer_momentum_viscosity_max": None
            if momentum_viscosity is None
            else scalar(momentum_viscosity.max()),
            "convective_interface_fraction": None if convective_mask is None else scalar(convective_mask.mean()),
            "unstable_column_fraction": None if unstable_column_mask is None else scalar(unstable_column_mask.mean()),
            "convective_adjustment_depth_mean": None
            if convective_adjustment_depth is None
            else scalar(convective_adjustment_depth.mean()),
            "convective_adjustment_depth_max": None
            if convective_adjustment_depth is None
            else scalar(convective_adjustment_depth.max()),
            "surface_stress_effective_depth_mean": None
            if stress_effective_depth is None
            else scalar(stress_effective_depth.mean()),
            "surface_stress_effective_depth_max": None
            if stress_effective_depth is None
            else scalar(stress_effective_depth.max()),
            "surface_stress_redistribution_u_rms": None
            if stress_redistribution is None
            else scalar(stress_redistribution.square().mean().sqrt()),
        }

    def bathymetry_metrics(self, state) -> dict[str, Any]:
        depth = state.column.total_depth_like(state.h)
        depth_gradient = state.mesh.grad(depth)
        pressure_gradient = -self.dynamics.pressure_gradient(state)
        density_anomaly = self.dynamics.density_anomaly(state)
        topographic_acceleration_scale = (
            float(self.dynamics.gravity)
            / float(self.dynamics.rho0)
            * state.mesh.cell_to_edge(density_anomaly.detach().abs())
            * depth_gradient.detach().abs()
        )
        if self.domain is not None:
            depth_gradient = self.domain.apply_edge(depth_gradient)
            pressure_gradient = self.domain.apply_edge(pressure_gradient)
            topographic_acceleration_scale = self.domain.apply_edge(topographic_acceleration_scale)
        pressure_gradient_rms = ocean_rms(pressure_gradient)
        pressure_gradient_p99 = ocean_abs_quantile(pressure_gradient, 0.99)
        pressure_gradient_max = ocean_max_abs(pressure_gradient)
        topo_scale_rms = ocean_rms(topographic_acceleration_scale)
        topo_scale_p99 = ocean_abs_quantile(topographic_acceleration_scale, 0.99)
        topo_scale_max = ocean_max_abs(topographic_acceleration_scale)
        return {
            "bathymetry_depth_min": scalar(depth.min()),
            "bathymetry_depth_max": scalar(depth.max()),
            "bathymetry_depth_relief": scalar(depth.max() - depth.min()),
            "bathymetry_depth_slope_rms": ocean_rms(depth_gradient),
            "bathymetry_depth_slope_abs_p99": ocean_abs_quantile(depth_gradient, 0.99),
            "bathymetry_depth_slope_max_abs": ocean_max_abs(depth_gradient),
            "pressure_gradient_rms": pressure_gradient_rms,
            "pressure_gradient_abs_p99": pressure_gradient_p99,
            "pressure_gradient_max_abs": pressure_gradient_max,
            "topographic_acceleration_scale_rms": topo_scale_rms,
            "topographic_acceleration_scale_abs_p99": topo_scale_p99,
            "topographic_acceleration_scale_max_abs": topo_scale_max,
            "topographic_normalized_pressure_gradient_rms": ocean_safe_ratio(pressure_gradient_rms, topo_scale_rms),
            "topographic_normalized_pressure_gradient_p99": ocean_safe_ratio(pressure_gradient_p99, topo_scale_p99),
            "topographic_normalized_pressure_gradient_max": ocean_safe_ratio(pressure_gradient_max, topo_scale_max),
        }

    def bathymetry_reference_metrics(self, reference, *, seconds: float = 86400.0) -> dict[str, Any]:
        rhs = self.dynamics.tendency(reference)
        metrics = self.bathymetry_metrics(reference)
        rhs_max = ocean_max_abs(rhs.u_e)
        rhs_p99 = ocean_abs_quantile(rhs.u_e, 0.99)
        metrics.update(
            {
                "bathymetry_reference_rhs_u_rms": ocean_rms(rhs.u_e),
                "bathymetry_reference_one_day_velocity": None if rhs_max is None else float(seconds) * float(rhs_max),
                "bathymetry_reference_one_day_velocity_p99": None if rhs_p99 is None else float(seconds) * float(rhs_p99),
            }
        )
        return metrics

    def health_row(
        self,
        state0,
        state=None,
        *,
        step: int | None = None,
        day: float | None = None,
        t: float | None = None,
        dt: float | None = None,
        case: str = "ocean_planet",
        mode: str | None = None,
    ) -> dict[str, Any]:
        if state is None:
            state = state0
            state0 = None
        row = self.state_summary(state)
        if step is not None:
            row["step"] = int(step)
        if day is not None:
            row["day"] = float(day)
        if t is not None:
            row["time"] = float(t)
        if dt is not None:
            row["dt"] = float(dt)
        if state0 is not None:
            row.update(self.conservation_summary(state0, state))
        row.update(self.energy_summary(state))
        row.update(self.tendency_summary(state, dt=dt))
        row.update(self.tendency_budget_summary(state))
        row.update(self.coriolis_work_summary(state))
        row.update(self.pressure_work_summary(state))
        row.update(self.forcing_summary(state))
        row.update(self.bathymetry_metrics(state))
        row["bathymetry_shape"] = str(state.attrs.get("bathymetry_shape", "flat"))
        row["case"] = str(case)
        row["mode"] = str(mode or getattr(self.provider, "recipe_name", getattr(self.dynamics, "recipe_name", "ocean")))
        return row


__all__ = [
    "OceanDiagnostics",
    "ocean_abs_quantile",
    "ocean_all_finite",
    "ocean_bathymetry_reference_warnings",
    "ocean_health_status_from_rows",
    "ocean_integrated_cell_total",
    "ocean_max_abs",
    "ocean_maybe_call_field",
    "ocean_pressure_work_summary_from_rows",
    "ocean_rms",
    "ocean_safe_ratio",
]
