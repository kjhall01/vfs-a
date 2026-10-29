"""Reporting facade for dry hydrostatic dynamics diagnostics.

The methods here turn canonical tensor diagnostics on `DryHydrostaticDynamics`
into flat Python dictionaries suitable for CSV, JSON, and Markdown reports.
Pure calculations remain on the dynamics, mesh, and column objects.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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


def stats(value: torch.Tensor | None, prefix: str) -> dict[str, float | None]:
    """Return finite min/max/mean/rms stats for a tensor."""
    if value is None:
        return {
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_mean": None,
            f"{prefix}_rms": None,
        }
    t = value.detach().cpu().to(dtype=torch.float64)
    if t.numel() == 0:
        return {
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_mean": None,
            f"{prefix}_rms": None,
        }
    finite = torch.isfinite(t)
    if not bool(finite.any().item()):
        return {
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_mean": None,
            f"{prefix}_rms": None,
        }
    f = t[finite]
    return {
        f"{prefix}_min": float(f.min().item()),
        f"{prefix}_max": float(f.max().item()),
        f"{prefix}_mean": float(f.mean().item()),
        f"{prefix}_rms": float(f.square().mean().sqrt().item()),
    }


def flatten_tensor_dict(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dictionaries, converting tensor leaves to scalars."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten_tensor_dict(value, name))
        else:
            out[name] = scalar(value)
    return out


def relative_error(value: Any, reference: Any, *, floor: float = 1.0) -> float | None:
    """Absolute relative error with a denominator floor."""
    v = scalar(value)
    r = scalar(reference)
    if v is None or r is None:
        return None
    return abs(float(v)) / max(abs(float(r)), float(floor))


class DryHydrostaticDiagnostics:
    """Detached summary facade for `DryHydrostaticDynamics`."""

    def __init__(self, dyn) -> None:
        self.dyn = dyn

    def state_summary(self, state) -> dict[str, Any]:
        theta = state.theta()
        dp = state.layer_dp()
        out: dict[str, Any] = {
            "finite": bool(
                torch.isfinite(state.u_e).all()
                and torch.isfinite(state.mtheta_i).all()
                and torch.isfinite(state.ps).all()
                and torch.isfinite(theta).all()
            ),
            "u_max": float(state.u_e.detach().abs().max().item()),
            "theta_min": float(theta.detach().min().item()),
            "theta_max": float(theta.detach().max().item()),
            "ps_min": float(state.ps.detach().min().item()),
            "ps_max": float(state.ps.detach().max().item()),
            "dp_min": float(dp.detach().min().item()),
        }
        if state.tracer_m is not None:
            out["tracer_finite"] = bool(torch.isfinite(state.tracer_m).all())
        return out

    def mass_summary(self, state0, state=None) -> dict[str, Any]:
        if state is None:
            state = state0
            state0 = None
        mass = state.total_mass()
        out = {"mass": scalar(mass)}
        if state0 is not None:
            mass0 = state0.total_mass()
            out["mass0"] = scalar(mass0)
            out["mass_drift"] = scalar(mass - mass0)
            out["mass_drift_rel"] = scalar((mass - mass0) / mass0.abs().clamp_min(1.0))
        return out

    def pressure_summary(self, state) -> dict[str, Any]:
        fields = self.dyn.pressure_fields(state)
        out = {}
        out.update(stats(fields["ps"], "ps"))
        out.update(stats(fields["p_interface"], "p_interface"))
        out.update(stats(fields["p_full"], "p_full"))
        out.update(stats(fields["dp"], "dp"))
        return out

    def thermo_summary(self, state) -> dict[str, Any]:
        fields = self.dyn.thermo_fields(state)
        out = {}
        out.update(stats(fields["theta"], "theta"))
        out.update(stats(fields["theta_full"], "theta_full"))
        out.update(stats(fields["temperature"], "temperature"))
        out.update(stats(fields["geopotential"], "geopotential"))
        if "virtual_temperature" in fields:
            out.update(stats(fields["virtual_temperature"], "virtual_temperature"))
        if "theta_virtual_full" in fields:
            out.update(stats(fields["theta_virtual_full"], "theta_virtual_full"))
        if "specific_volume" in fields:
            out.update(stats(fields["specific_volume"], "specific_volume"))
        return out

    def wind_summary(self, state) -> dict[str, Any]:
        fields = self.dyn.wind_fields(state)
        out = {}
        out.update(stats(fields["u_e"], "u_e"))
        out["u_max"] = float(fields["u_e"].detach().abs().max().item())
        if "cell_wind_vector" in fields:
            speed = fields["cell_wind_vector"].detach().square().sum(dim=-1).sqrt()
            out.update(stats(speed, "cell_wind_speed"))
        return out

    def energy_summary(self, state0, state=None) -> dict[str, Any]:
        if state is None:
            state = state0
            state0 = None
        terms = self.dyn.energy_terms(state)
        out = {
            "kinetic_energy": scalar(terms["kinetic"]),
            "potential_energy": scalar(terms["potential"]),
            "internal_energy": scalar(terms["internal"]),
            "total_energy": scalar(terms["total"]),
        }
        if state0 is not None:
            e0 = self.dyn.total_energy(state0)
            e = terms["total"]
            out["total_energy0"] = scalar(e0)
            out["total_energy_drift"] = scalar(e - e0)
            out["total_energy_drift_rel"] = scalar((e - e0) / e0.abs().clamp_min(1.0))
            out["energy_rel"] = out["total_energy_drift_rel"]
        return out

    def tracer_summary(self, state0, state=None) -> dict[str, Any]:
        if state is None:
            state = state0
            state0 = None
        terms = self.dyn.tracer_terms(state)
        if not terms:
            return {}
        q = terms["q"]
        out: dict[str, Any] = {
            "n_tracers": int(q.shape[-1]),
            "total_tracer_mass": scalar(terms["total_tracer_mass"]),
        }
        out.update(stats(q, "q_tracer"))
        if state0 is not None and state0.tracer_m is not None:
            m0 = state0.total_tracer_mass()
            m = terms["total_tracer_mass"]
            if m0 is not None:
                rel = (m - m0) / m0.abs().clamp_min(1.0)
                out["tracer_mass_drift_rel_max_abs"] = float(rel.detach().abs().max().item())
        return out

    def static_stability_summary(self, state) -> dict[str, Any]:
        stability = state.static_stability()
        dtheta_down = -stability
        unstable = dtheta_down > 1.0e-8
        penalty = self.dyn.static_instability_penalty(state)
        worst_flat = int(torch.argmax(dtheta_down.clamp_min(0.0)).item()) if dtheta_down.numel() else 0
        n_levels = state.column.n_levels
        worst_cell = (worst_flat // n_levels) % state.mesh.nCells if n_levels else 0
        worst_level = worst_flat % n_levels if n_levels else 0
        lat = torch.asin(state.mesh.cell_xyz[:, 2].clamp(-1.0, 1.0))
        return {
            "static_stability_metric": "theta_i[k] - theta_i[k+1]; negative values indicate unstable downward theta increase",
            "static_instability_penalty": scalar(penalty),
            "max_positive_dtheta_down": float(dtheta_down.detach().clamp_min(0.0).max().item()),
            "fraction_unstable_layers": float(unstable.detach().cpu().to(torch.float64).mean().item()),
            "unstable_layer_count": int(unstable.sum().item()),
            "total_layer_count": int(unstable.numel()),
            "worst_unstable_lat_deg": float(torch.rad2deg(lat[worst_cell]).item()) if unstable.numel() else None,
            "worst_unstable_level": int(worst_level),
        }

    def vertical_structure_summary(self, state) -> dict[str, Any]:
        return flatten_tensor_dict(self.dyn.vertical_structure_terms(state))

    def tendency_summary(self, state, terms=None) -> dict[str, Any]:
        if terms is None:
            terms = self.dyn.tendency_terms(state)
        total = self.dyn.terms_total(terms, like=state)
        out = {
            "u_tend_rms": float(total.u_e.detach().square().mean().sqrt().item()),
            "u_tend_max": float(total.u_e.detach().abs().max().item()),
            "ps_tend_rms": float(total.ps.detach().square().mean().sqrt().item()),
            "ps_tend_max": float(total.ps.detach().abs().max().item()),
            "mtheta_tend_rms": float(total.mtheta_i.detach().square().mean().sqrt().item()),
            "mtheta_tend_max": float(total.mtheta_i.detach().abs().max().item()),
        }
        if total.tracer_m is not None:
            out["tracer_tend_rms"] = float(total.tracer_m.detach().square().mean().sqrt().item())
            out["tracer_tend_max"] = float(total.tracer_m.detach().abs().max().item())
        return out

    def t0_summary(self, state, *, dt: float = 180.0) -> dict[str, Any]:
        out = self.state_summary(state)
        tend = self.dyn.tendency(state)
        out.update(
            {
                "u_tend_rms": float(tend.u_e.detach().square().mean().sqrt().item()),
                "u_tend_max": float(tend.u_e.detach().abs().max().item()),
                "u_step_max": float(float(dt) * tend.u_e.detach().abs().max().item()),
                "finite": bool(
                    out["finite"]
                    and torch.isfinite(tend.u_e).all()
                    and torch.isfinite(tend.mtheta_i).all()
                    and torch.isfinite(tend.ps).all()
                ),
            }
        )
        if tend.tracer_m is not None:
            out["tracer_tendency_finite"] = bool(torch.isfinite(tend.tracer_m).all())
        return out

    def force_balance_summary(self, state, terms=None, primitives=None) -> dict[str, Any]:
        return flatten_tensor_dict(self.dyn.force_balance_terms(state, terms=terms, primitives=primitives))

    def pgf_correction_summary(self, state, terms=None, primitives=None) -> dict[str, Any]:
        if terms is None:
            primitives = self.dyn._transport_primitives(state) if primitives is None else primitives
            terms = self.dyn.tendency_terms(state, primitives=primitives)
        u_terms = terms["u_e"] if "u_e" in terms else terms
        correction = u_terms.get("pgf_correction")
        base = u_terms.get("pgf_base", u_terms.get("pgf"))
        if correction is None or base is None:
            return {
                "pgf_correction_rms": 0.0,
                "pgf_correction_ratio_rms": 0.0,
                "pgf_correction_ratio_max": 0.0,
                "pgf_correction_max_abs": 0.0,
                "pgf_correction_curl_proxy_rms": 0.0,
            }
        correction_t = correction.detach()
        base_t = base.detach()
        pgf_config = getattr(getattr(self.dyn, "pressure_gradient", None), "config", None)
        scale_floor = float(getattr(pgf_config, "scale_floor", 1.0e-8))
        ratio = correction_t.abs() / (base_t.abs() + scale_floor)
        curl_proxy = state.curl(correction_t)
        return {
            "pgf_correction_rms": float(correction_t.square().mean().sqrt().item()),
            "pgf_base_rms": float(base_t.square().mean().sqrt().item()),
            "pgf_correction_ratio_rms": float(ratio.square().mean().sqrt().item()),
            "pgf_correction_ratio_max": float(ratio.max().item()),
            "pgf_correction_max_abs": float(correction_t.abs().max().item()),
            "pgf_correction_curl_proxy_rms": float(curl_proxy.square().mean().sqrt().item()),
        }

    def force_summary_rows(self, state, *, case: str, dt: float = 180.0, terms=None) -> list[dict[str, Any]]:
        primitives = self.dyn._transport_primitives(state)
        if terms is None:
            terms = self.dyn.tendency_terms(state, primitives=primitives)
        force = self.dyn.force_terms(state, terms=terms, primitives=primitives)
        metric = force["metric"]
        ke0 = 0.5 * (state.u_e.square() * metric).sum().clamp_min(1.0e-300)
        rows: list[dict[str, Any]] = []
        for name in ("hydro", "rotation", "minus_grad_ke", "uvert", "total"):
            field = force[name]
            work = (state.u_e * field * metric).sum()
            rows.append(
                {
                    "case": case,
                    "term": name,
                    "weighted_rms": float(((field.square() * metric).sum() / metric.sum().clamp_min(1.0e-300)).sqrt().detach().item()),
                    "max": float(field.detach().abs().max().item()),
                    "dt_max": float(dt) * float(field.detach().abs().max().item()),
                    "ke_work_rate_over_ke": float((work / ke0).detach().item()),
                }
            )
        return rows

    def rotation_work_row(self, state, label: str) -> dict[str, Any]:
        terms = self.dyn.rotation_work_terms(state)
        return {
            "case": label,
            "rotation_rms": scalar(terms["rotation_rms"]),
            "work": scalar(terms["work"]),
            "work_relative": scalar(terms["work_relative"]),
        }

    def energy_exchange_summary(self, state, terms=None, primitives=None) -> dict[str, Any]:
        """Flattened closed-form energy-rate table for named dry RHS terms."""
        if terms is None:
            terms = self.dyn.tendency_terms(state, primitives=primitives)
        exchange = self.dyn.energy_exchange_terms(state, terms=terms, primitives=primitives)
        row = flatten_tensor_dict(exchange, prefix="energy_exchange")
        row["work_rate_units"] = "J s-1"
        return row

    def pressure_work_local_summary(self, state, terms=None, primitives=None) -> dict[str, Any]:
        """Detached summaries of local PGF / ps kinetic-mass work fields."""
        if terms is None:
            terms = self.dyn.tendency_terms(state, primitives=primitives)
        local = self.dyn.pressure_work_local_terms(state, terms=terms, primitives=primitives)
        col = local["cell_column"]
        layer = local["cell_layer"]
        out: dict[str, Any] = {"work_rate_units": "J s-1"}
        out.update(stats(col["pgf_kinetic"], "pressure_work_local_column_pgf_kinetic"))
        out.update(stats(col["ps_kinetic_mass"], "pressure_work_local_column_ps_kinetic_mass"))
        out.update(stats(col["pgf_plus_ps_kinetic_mass"], "pressure_work_local_column_pgf_plus_ps_kinetic_mass"))
        out.update(stats(col["pgf_plus_ps_kinetic_mass_relative"], "pressure_work_local_column_pgf_plus_ps_kinetic_mass_relative"))
        out.update(stats(layer["pgf_plus_ps_kinetic_mass"], "pressure_work_local_layer_pgf_plus_ps_kinetic_mass"))
        pgf = col["pgf_kinetic"].sum()
        ps_ke = col["ps_kinetic_mass"].sum()
        residual = col["pgf_plus_ps_kinetic_mass"].sum()
        scale = torch.maximum(pgf.abs(), ps_ke.abs()).clamp_min(1.0)
        out.update(
            {
                "pressure_work_local_global_pgf_kinetic": scalar(pgf),
                "pressure_work_local_global_ps_kinetic_mass": scalar(ps_ke),
                "pressure_work_local_global_pgf_plus_ps_kinetic_mass": scalar(residual),
                "pressure_work_local_global_pgf_plus_ps_kinetic_mass_relative": scalar(residual.abs() / scale),
                "pressure_work_local_global_ps_total": scalar(col["ps_total"].sum()),
                "pressure_work_local_global_pgf_plus_ps_total": scalar(col["pgf_plus_ps_total"].sum()),
            }
        )
        return out

    def pressure_work_local_attribution_summary(self, state, terms=None, primitives=None) -> dict[str, Any]:
        """Compare edge-to-cell attribution choices for local pressure-work maps."""
        if terms is None:
            terms = self.dyn.tendency_terms(state, primitives=primitives)
        comparison = self.dyn.pressure_work_local_attribution_terms(
            state,
            terms=terms,
            primitives=primitives,
        )
        out = flatten_tensor_dict(comparison, prefix="pressure_work_attribution")
        out["work_rate_units"] = "J s-1"
        return out

    def pressure_work_edge_exchange_summary(self, state, terms=None, primitives=None) -> dict[str, Any]:
        """Detached summary of the edge-native pressure-work exchange audit."""
        if terms is None:
            terms = self.dyn.tendency_terms(state, primitives=primitives)
        edge = self.dyn.pressure_work_edge_exchange_terms(state, terms=terms, primitives=primitives)
        out = flatten_tensor_dict({"global": edge["global"]}, prefix="pressure_work_edge_exchange")
        out.update(stats(edge["edge_layer"]["pgf_plus_ps_kinetic_mass_flux"], "pressure_work_edge_exchange_residual"))
        out.update(stats(edge["edge_layer"]["pgf_normal_plus_ps_kinetic_mass_flux"], "pressure_work_edge_exchange_normal_residual"))
        out.update(stats(edge["edge_layer"]["pgf_kinetic"], "pressure_work_edge_exchange_pgf_kinetic"))
        out.update(stats(edge["edge_layer"]["pgf_kinetic_normal"], "pressure_work_edge_exchange_pgf_kinetic_normal"))
        out.update(stats(edge["edge_layer"]["pgf_kinetic_tangential"], "pressure_work_edge_exchange_pgf_kinetic_tangential"))
        out.update(stats(edge["edge_layer"]["ps_kinetic_mass_flux"], "pressure_work_edge_exchange_ps_kinetic_mass_flux"))
        out["work_rate_units"] = "J s-1"
        return out

    def pressure_work_tangential_stencil_summary(self, state, terms=None, primitives=None) -> dict[str, Any]:
        """Detached summary of stencil-aware tangential pressure-work exchange."""
        if terms is None:
            terms = self.dyn.tendency_terms(state, primitives=primitives)
        stencil = self.dyn.pressure_work_tangential_stencil_terms(state, terms=terms, primitives=primitives)
        out = flatten_tensor_dict({"global": stencil["global"]}, prefix="pressure_work_tangential_stencil")
        out.update(stats(stencil["edge_layer"]["tangential_source"], "pressure_work_tangential_source"))
        out.update(stats(stencil["edge_layer"]["tangential_receiver"], "pressure_work_tangential_receiver"))
        out.update(stats(stencil["edge_layer"]["source_receiver_difference"], "pressure_work_tangential_source_receiver_difference"))
        out.update(stats(stencil["edge_layer"]["source_residual"], "pressure_work_tangential_source_residual"))
        out.update(stats(stencil["edge_layer"]["receiver_residual"], "pressure_work_tangential_receiver_residual"))
        out["work_rate_units"] = "J s-1"
        return out

    def pressure_work_residual_structure_summary(
        self,
        state,
        terms=None,
        *,
        max_zonal_wavenumber: int = 6,
        max_spherical_degree: int = 3,
        primitives=None,
    ) -> dict[str, Any]:
        """Detached spatial-structure summary of the local pressure-work residual."""
        if terms is None:
            terms = self.dyn.tendency_terms(state, primitives=primitives)
        structure = self.dyn.pressure_work_residual_structure_terms(
            state,
            terms=terms,
            max_zonal_wavenumber=max_zonal_wavenumber,
            max_spherical_degree=max_spherical_degree,
            primitives=primitives,
        )
        out = flatten_tensor_dict(
            {
                "global": structure["global"],
                "zonal_wavenumber": structure["zonal_wavenumber"],
                "spherical_harmonic": structure["spherical_harmonic"],
            },
            prefix="pressure_work_residual_structure",
        )
        out.update(stats(structure["fields"]["residual_density"], "pressure_work_residual_density"))
        out.update(stats(structure["fields"]["residual_density_anomaly"], "pressure_work_residual_density_anomaly"))
        out.update(stats(structure["fields"]["residual_density_rough"], "pressure_work_residual_density_rough"))
        out["work_density_units"] = "J s-1 m-2"
        return out

    def pressure_work_summary(
        self,
        state,
        terms=None,
        *,
        eps: float = 1.0,
        primitives=None,
    ) -> dict[str, Any]:
        if terms is None:
            terms = self.dyn.tendency_terms(state, primitives=primitives)
        work = self.dyn.pressure_work_terms(state, terms=terms, eps=eps, primitives=primitives)
        force = self.dyn.force_terms(state, terms=terms, primitives=primitives)
        metric = force["metric"]
        rot_work = (state.u_e * force["rotation"] * metric).sum() / float(self.dyn.gravity)
        rot_norm = ((state.u_e.abs() * force["rotation"].abs() * metric).sum() / float(self.dyn.gravity)).clamp_min(1.0e-300)
        residual_rel = float(work["residual_rel_current"].detach().item())
        status = "PASS" if residual_rel < 1.0e-2 else ("WARN" if residual_rel < 1.0e-1 else "WARN")
        pgf_abs = work["pgf_work"].abs().clamp_min(1.0)
        scalar_abs = work["scalar_work"].abs().clamp_min(1.0)
        row = {
            "work_rate_units": "J s-1",
            "pressure_work_pgf": scalar(work["pgf_work"]),
            "pressure_work_pgf_simple": scalar(work["pgf_work_simple"]),
            "pressure_work_pgf_kinetic": scalar(work["pgf_work_kinetic"]),
            "pressure_work_scalar": scalar(work["scalar_work"]),
            "pressure_work_scalar_closed": scalar(work["scalar_work_closed"]),
            "pressure_work_coord": scalar(work["coord_work"]),
            "pressure_work_residual_abs": scalar(work["residual_abs"]),
            "pressure_work_residual_rel_current": residual_rel,
            "pressure_work_residual_kinetic_abs": scalar(work["residual_kinetic_abs"]),
            "pressure_work_residual_kinetic_rel_current": scalar(work["residual_kinetic_rel_current"]),
            "pressure_work_residual_closed_abs": scalar(work["residual_closed_abs"]),
            "pressure_work_residual_closed_rel_current": scalar(work["residual_closed_rel_current"]),
            "pressure_work_residual_kinetic_closed_abs": scalar(work["residual_kinetic_closed_abs"]),
            "pressure_work_residual_kinetic_closed_rel_current": scalar(work["residual_kinetic_closed_rel_current"]),
            "hydro_pgf_work_J_per_s": scalar(work["pgf_work"]),
            "hydro_pgf_kinetic_work_J_per_s": scalar(work["pgf_work_kinetic"]),
            "scalar_coordinate_energy_rate_J_per_s": scalar(work["scalar_work"]),
            "scalar_coordinate_energy_rate_closed_J_per_s": scalar(work["scalar_work_closed"]),
            "coordinate_work_energy_rate_J_per_s": scalar(work["coord_work"]),
            "pressure_work_residual_J_per_s": scalar(work["residual"]),
            "pressure_work_residual_abs_J_per_s": scalar(work["residual_abs"]),
            "pressure_work_residual_kinetic_J_per_s": scalar(work["residual_kinetic"]),
            "pressure_work_residual_kinetic_abs_J_per_s": scalar(work["residual_kinetic_abs"]),
            "pressure_work_residual_kinetic_closed_J_per_s": scalar(work["residual_kinetic_closed"]),
            "pressure_work_residual_kinetic_closed_abs_J_per_s": scalar(work["residual_kinetic_closed_abs"]),
            "wind_work_kinetic_rotation_J_per_s": scalar(work.get("wind_work_kinetic_rotation")),
            "wind_work_kinetic_pgf_J_per_s": scalar(work.get("wind_work_kinetic_pgf")),
            "wind_work_kinetic_bernoulli_J_per_s": scalar(work.get("wind_work_kinetic_bernoulli")),
            "wind_work_kinetic_vertical_J_per_s": scalar(work.get("wind_work_kinetic_vertical")),
            "wind_work_kinetic_total_J_per_s": scalar(work.get("wind_work_kinetic_total")),
            "scalar_work_closed_mtheta_horizontal_J_per_s": scalar(work.get("scalar_work_closed_mtheta_horizontal")),
            "scalar_work_closed_mtheta_vertical_J_per_s": scalar(work.get("scalar_work_closed_mtheta_vertical")),
            "scalar_work_closed_mtheta_coord_work_J_per_s": scalar(work.get("scalar_work_closed_mtheta_coord_work")),
            "scalar_work_closed_mtheta_total_J_per_s": scalar(work.get("scalar_work_closed_mtheta_total")),
            "scalar_work_closed_ps_mass_flux_J_per_s": scalar(work.get("scalar_work_closed_ps_mass_flux")),
            "scalar_work_closed_ps_total_J_per_s": scalar(work.get("scalar_work_closed_ps_total")),
            "total_energy_rate_closed_J_per_s": scalar(work.get("total_energy_rate_closed")),
            "residual_over_abs_hydro": scalar(work["residual_abs"] / pgf_abs),
            "residual_over_abs_coordinate": scalar(work["residual_abs"] / scalar_abs),
            "residual_over_max_pair_scale": residual_rel,
            "residual_kinetic_over_max_pair_scale": scalar(work["residual_kinetic_rel_current"]),
            "residual_kinetic_closed_over_max_pair_scale": scalar(work["residual_kinetic_closed_rel_current"]),
            "rotation_work_relative": scalar(rot_work.abs() / rot_norm),
            "total_energy_directional_derivative_J_per_s": scalar(work["total_energy_directional_derivative"]),
            "split_sum_residual_J_per_s": None,
            "status": status,
            # Legacy names retained for old rows/scripts.
            "hydro_pgf_work": scalar(work["pgf_work"]),
            "hydro_pgf_kinetic_work": scalar(work["pgf_work_kinetic"]),
            "scalar_coordinate_energy_rate": scalar(work["scalar_work"]),
            "scalar_coordinate_energy_rate_closed": scalar(work["scalar_work_closed"]),
            "pressure_work_residual": scalar(work["residual"]),
            "pressure_work_residual_kinetic": scalar(work["residual_kinetic"]),
            "pressure_work_residual_kinetic_closed": scalar(work["residual_kinetic_closed"]),
            "pressure_work_residual_normalized_current_scale": residual_rel,
            "pressure_work_residual_kinetic_normalized_current_scale": scalar(work["residual_kinetic_rel_current"]),
            "pressure_work_residual_kinetic_closed_normalized_current_scale": scalar(work["residual_kinetic_closed_rel_current"]),
            "total_energy_directional_derivative": scalar(work["total_energy_directional_derivative"]),
            "total_energy_rate_closed": scalar(work.get("total_energy_rate_closed")),
            "split_sum_residual": None,
        }
        return row

    def tracer_no_feedback_row(self, dry_state, tracer_state) -> dict[str, Any]:
        terms = self.dyn.tracer_no_feedback_terms(dry_state, tracer_state)
        max_diff = float(terms["max_abs_diff"].detach().item())
        if dry_state.u_e.dtype == torch.float32:
            pass_below = 1.0e-6
            warn_below = 1.0e-4
        else:
            pass_below = 1.0e-13
            warn_below = 1.0e-11
        status = "PASS" if max_diff < pass_below else ("WARN" if max_diff < warn_below else "FAIL")
        return {
            "dry_tendency_u_max_abs_diff": scalar(terms["u_max_abs_diff"]),
            "dry_tendency_mtheta_max_abs_diff": scalar(terms["mtheta_max_abs_diff"]),
            "dry_tendency_ps_max_abs_diff": scalar(terms["ps_max_abs_diff"]),
            "dry_tendency_no_feedback_status": status,
        }

    def tracer_sample_row(
        self,
        state,
        *,
        day: float,
        step: int,
        dt: float,
        mass0: torch.Tensor,
        dry_mass0: torch.Tensor,
        q_min0: float,
        q_max0: float,
        case: str,
    ) -> dict[str, Any]:
        tracer = self.dyn.tracer_terms(state)
        if not tracer:
            raise ValueError("tracer_sample_row requires State.tracer_m")
        q = tracer["q"]
        tracer_mass = tracer["total_tracer_mass"]
        rel = (tracer_mass - mass0) / mass0.abs().clamp_min(1.0e-300)
        primitives = self.dyn._transport_primitives(state)
        cfl = self.dyn.cfl_terms(state, dt, primitives=primitives)
        terms = self.dyn.tendency_terms(state, primitives=primitives)
        q_const_err = (q - float(q_min0)).abs() if case == "constant" else torch.zeros_like(q)
        return {
            "day": float(day),
            "step": int(step),
            "dry_mass_rel": scalar((state.total_mass() - dry_mass0) / dry_mass0.abs().clamp_min(1.0e-300)),
            "tracer_mass_rel_error_max_abs": float(rel.detach().abs().max().item()),
            "q_min": float(q.detach().min().item()),
            "q_max": float(q.detach().max().item()),
            "q_overshoot_above_initial_max": max(float(q.detach().max().item()) - float(q_max0), 0.0),
            "q_undershoot_below_initial_min": max(float(q_min0) - float(q.detach().min().item()), 0.0),
            "q_constant_max_abs_error": float(q_const_err.detach().max().item()) if case == "constant" else 0.0,
            "u_max": float(state.u_e.detach().abs().max().item()),
            "vertical_cfl_max": scalar(cfl["vertical_max"]),
            "ps_step_frac_max": float((float(dt) * terms["ps"]["total"].detach().abs() / state.ps.detach().abs().clamp_min(1.0)).max().item()),
        }

    def cfl_summary(self, state, dt: float, primitives=None) -> dict[str, Any]:
        terms = self.dyn.cfl_terms(state, dt, primitives=primitives)
        return {
            "horizontal_cfl_max": scalar(terms["horizontal_max"]),
            "vertical_cfl_max": scalar(terms["vertical_max"]),
            "vertical_interface_cfl_max": scalar(terms["vertical_interface_max"]),
        }

    def divergence_vorticity_summary(self, state, primitives=None) -> dict[str, Any]:
        out = {}
        out.update(stats(self.dyn.divergence(state, primitives=primitives), "divergence"))
        out.update(stats(self.dyn.vorticity(state), "vorticity"))
        return out

    def omega_summary(self, state, terms=None, primitives=None) -> dict[str, Any]:
        return stats(self.dyn.omega(state, terms=terms, primitives=primitives), "omega")

    def tendency_scale_summary(self, state, terms=None) -> dict[str, Any]:
        return flatten_tensor_dict(self.dyn.tendency_scale_terms(state, terms=terms))

    def adjustment_shock_summary(self, state, terms=None, dt: float | None = None, primitives=None) -> dict[str, Any]:
        return flatten_tensor_dict(self.dyn.adjustment_shock_terms(state, terms=terms, dt=dt, primitives=primitives))

    def mesh_summary(self, mesh) -> dict[str, Any]:
        out = flatten_tensor_dict(self.dyn.mesh_health_terms(mesh))
        out["mesh_operator_version"] = int(getattr(mesh, "mesh_operator_version", 2))
        out["exact_thuburn_version"] = int(getattr(mesh, "exact_thuburn_version", 2))
        out["thuburn_corner_convention"] = str(getattr(mesh, "thuburn_corner_convention", ""))
        return out

    def operator_summary(self, mesh) -> dict[str, Any]:
        return flatten_tensor_dict(self.dyn.operator_identity_terms(mesh))

    def health_row(
        self,
        state0,
        state=None,
        *,
        t: float | None = None,
        dt: float | None = None,
        step: int | None = None,
        day: float | None = None,
    ) -> dict[str, Any]:
        if state is None:
            state = state0
            state0 = None
        primitives = self.dyn._transport_primitives(state)
        terms = self.dyn.tendency_terms(state, primitives=primitives)
        row = self.state_summary(state)
        row.update(self.tendency_summary(state, terms=terms))
        if step is not None:
            row["step"] = int(step)
        if day is not None:
            row["day"] = float(day)
        if t is not None:
            row["time"] = float(t)
        if state0 is not None:
            row.update(self.mass_summary(state0, state))
            row.update(self.energy_summary(state0, state))
            row.update(self.tracer_summary(state0, state))
        row.update(self.force_balance_summary(state, terms=terms, primitives=primitives))
        row.update(self.pgf_correction_summary(state, terms=terms, primitives=primitives))
        row.update(self.pressure_work_summary(state, terms=terms, primitives=primitives))
        row.update(self.static_stability_summary(state))
        row.update(self.divergence_vorticity_summary(state, primitives=primitives))
        row.update(self.omega_summary(state, terms=terms, primitives=primitives))
        row.update(self.adjustment_shock_summary(state, terms=terms, dt=dt, primitives=primitives))
        if dt is not None:
            row.update(self.cfl_summary(state, dt, primitives=primitives))
            row["ps_step_frac_max"] = float((float(dt) * terms["ps"]["total"].detach().abs() / state.ps.detach().abs().clamp_min(1.0)).max().item())
        return row

    def transition_summary(self, state0, state, *, t: float | None = None, dt: float | None = None) -> dict[str, Any]:
        return self.health_row(state0, state, t=t, dt=dt)


__all__ = [
    "DryHydrostaticDiagnostics",
    "flatten_tensor_dict",
    "relative_error",
    "scalar",
    "stats",
]
