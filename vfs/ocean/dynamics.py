"""VFS-O multilayer and hydrostatic-Boussinesq dynamics."""

from __future__ import annotations

import torch
import torch.nn as nn

from vfs.core.registry import register_component
from vfs.ocean_diagnostics import OceanDiagnostics

from .constants import (
    OCEAN_ALPHA_T,
    OCEAN_BETA_S,
    OCEAN_EOS_MODES,
    OCEAN_GRAVITY,
    OCEAN_PRESSURE_GRADIENT_MODES,
    OCEAN_RHO0,
    OCEAN_ROTATION_RATE,
    OCEAN_S0,
    OCEAN_T0,
)
from .domain import OceanDomain
from .eos import jmd95_seawater_density, linear_seawater_density
from .helpers import _cell_midlayer_spacing, _normalize_tracer_tensor, _ocean_vertical_momentum_stress_tendency
from .state import OceanState, OceanStateRequirement


class MultilayerOceanDynamics(nn.Module):
    """Experimental C-grid multilayer Boussinesq ocean dynamics."""

    recipe_name = "multilayer_ocean"

    def __init__(
        self,
        *,
        domain: OceanDomain | None = None,
        omega: float = OCEAN_ROTATION_RATE,
        gravity: float = OCEAN_GRAVITY,
        rho0: float = OCEAN_RHO0,
        alpha: float = OCEAN_ALPHA_T,
        beta: float = OCEAN_BETA_S,
        T0: float = OCEAN_T0,
        S0: float = OCEAN_S0,
        eos: str = "linear",
        pressure_gradient_mode: str = "montgomery",
        validate_state: bool = True,
        min_thickness: float = 1.0e-3,
    ) -> None:
        super().__init__()
        pressure_gradient_mode = str(pressure_gradient_mode)
        if pressure_gradient_mode not in OCEAN_PRESSURE_GRADIENT_MODES:
            raise ValueError(
                "pressure_gradient_mode must be one of "
                f"{OCEAN_PRESSURE_GRADIENT_MODES}, got {pressure_gradient_mode!r}"
            )
        eos = str(eos)
        if eos not in OCEAN_EOS_MODES:
            raise ValueError(f"eos must be one of {OCEAN_EOS_MODES}, got {eos!r}")
        self.rotation_rate = float(omega)
        self.gravity = float(gravity)
        self.rho0 = float(rho0)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.T0 = float(T0)
        self.S0 = float(S0)
        self.eos = eos
        self.pressure_gradient_mode = pressure_gradient_mode
        self.validate_state = bool(validate_state)
        self.min_thickness = float(min_thickness)
        self.domain = domain

    def check_state_compatible(self, state: OceanState) -> None:
        if state.mesh is None:
            raise ValueError("MultilayerOceanDynamics requires OceanState.mesh")
        if state.column is None:
            raise ValueError("MultilayerOceanDynamics requires OceanState.column")
        if self.domain is not None:
            self.domain.check_compatible(state)

    def required_state_requirement(self) -> OceanStateRequirement:
        return OceanStateRequirement()

    def _validate_state(self, state: OceanState) -> None:
        if not self.validate_state:
            return
        state.validate()
        state.column.validate_thickness(state.h, min_thickness=self.min_thickness)
        for name, value in (("u_e", state.u_e), ("temp_h", state.temp_h), ("salt_h", state.salt_h)):
            if not torch.isfinite(value).all().item():
                raise FloatingPointError(f"{name} is not finite")
        if state.tracer_h is not None and not torch.isfinite(state.tracer_h).all().item():
            raise FloatingPointError("tracer_h is not finite")

    def eos_pressure_dbar(self, state: OceanState) -> torch.Tensor:
        """Approximate pressure at layer midpoints in dbar for nonlinear EOS."""

        mid_depth = state.column.mid_depth(state.h)
        return float(self.rho0) * float(self.gravity) * mid_depth / 1.0e4

    def density(self, state: OceanState) -> torch.Tensor:
        temp = state.temperature()
        salt = state.salinity()
        if self.eos == "linear":
            return linear_seawater_density(temp, salt, rho0=self.rho0, alpha=self.alpha, beta=self.beta, T0=self.T0, S0=self.S0)
        if self.eos == "jmd95":
            return jmd95_seawater_density(salt, temp, self.eos_pressure_dbar(state))
        raise AssertionError(f"unsupported ocean eos {self.eos!r}")

    def reference_density(self, state: OceanState) -> torch.Tensor:
        """Return reference density used for Boussinesq baroclinic anomalies."""

        if self.eos == "linear":
            return torch.full_like(state.h, float(self.rho0))
        if self.eos == "jmd95":
            pressure = self.eos_pressure_dbar(state)
            salt = torch.full_like(state.h, float(self.S0))
            temp = torch.full_like(state.h, float(self.T0))
            return jmd95_seawater_density(salt, temp, pressure)
        raise AssertionError(f"unsupported ocean eos {self.eos!r}")

    def density_anomaly(self, state: OceanState) -> torch.Tensor:
        """Return density anomaly used by baroclinic pressure-gradient terms."""

        return self.density(state) - self.reference_density(state)

    def montgomery_potential(self, state: OceanState) -> torch.Tensor:
        """Layer pressure potential per unit mass at cell centers."""

        return float(self.gravity) * state.eta() + self._baroclinic_potential(state)

    def _baroclinic_potential(self, state: OceanState) -> torch.Tensor:
        """Density-anomaly hydrostatic pressure potential at layer midpoints."""

        h = state.h
        density_anomaly = self.density_anomaly(state)
        anomaly_thickness = density_anomaly * h
        overlying_to_midpoint = torch.cumsum(anomaly_thickness, dim=-1) - 0.5 * anomaly_thickness
        return (float(self.gravity) / float(self.rho0)) * overlying_to_midpoint

    def sigma_depth_pressure_gradient(self, state: OceanState) -> torch.Tensor:
        """Experimental sigma-coordinate pressure gradient at edges.

        The operator applies a discrete chain-rule correction to the
        baroclinic Montgomery potential so a horizontally uniform density
        profile over variable depth does not directly accelerate a resting
        state just because layer midpoints lie at different physical depths.
        """

        mesh = state.mesh
        column = state.column
        h = state.h
        eta_potential = float(self.gravity) * state.eta()
        baroclinic = self._baroclinic_potential(state)
        mid_depth = column.mid_depth(h)
        density_anomaly = self.density_anomaly(state)
        dbaroclinic_dd = (float(self.gravity) / float(self.rho0)) * density_anomaly
        return (
            mesh.grad(eta_potential)
            + mesh.grad(baroclinic)
            - mesh.cell_to_edge(dbaroclinic_dd) * mesh.grad(mid_depth)
        )

    def pressure_gradient(self, state: OceanState) -> torch.Tensor:
        """Layer pressure-gradient acceleration term before the leading minus sign."""

        if self.pressure_gradient_mode == "montgomery":
            return state.mesh.grad(self.montgomery_potential(state))
        if self.pressure_gradient_mode == "sigma_depth":
            return self.sigma_depth_pressure_gradient(state)
        raise AssertionError(f"unsupported pressure_gradient_mode {self.pressure_gradient_mode!r}")

    def _trisk_rotation(self, state: OceanState, u_e: torch.Tensor | None = None) -> torch.Tensor:
        mesh = state.mesh
        if u_e is None:
            u_e = state.u_e
        return mesh.trisk_pv_flux_rotation(u_e, self.rotation_rate)

    def _edge_upwind(self, state: OceanState, cell_field: torch.Tensor, u_e: torch.Tensor) -> torch.Tensor:
        mesh = state.mesh
        c0 = mesh.cellsOnEdge[:, 0]
        c1 = mesh.cellsOnEdge[:, 1]
        left = cell_field[..., c0, :]
        right = cell_field[..., c1, :]
        return torch.where(u_e >= 0.0, left, right)

    def _apply_cell_domain(self, value: torch.Tensor) -> torch.Tensor:
        return value if self.domain is None else self.domain.apply_cell(value)

    def _apply_edge_domain(self, value: torch.Tensor) -> torch.Tensor:
        return value if self.domain is None else self.domain.apply_edge(value)

    def _active_edge_velocity(self, state: OceanState) -> torch.Tensor:
        return state.u_e if self.domain is None else self.domain.apply_edge(state.u_e)

    def _horizontal_content_transport(
        self,
        state: OceanState,
        concentration: torch.Tensor,
        *,
        u_e: torch.Tensor,
        h_edge: torch.Tensor,
    ) -> torch.Tensor:
        """Return conservative horizontal transport for a layer concentration."""

        mesh = state.mesh
        flux = u_e * h_edge * self._edge_upwind(state, concentration, u_e)
        return self._apply_cell_domain(-mesh.div(flux))

    def layer_thickness_tendency_terms(
        self,
        state: OceanState,
        *,
        u_e: torch.Tensor | None = None,
        h_edge: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return layer-thickness continuity terms owned by volume transport."""

        mesh = state.mesh
        velocity = self._active_edge_velocity(state) if u_e is None else u_e
        thickness_edge = mesh.cell_to_edge(state.h) if h_edge is None else h_edge
        if h_edge is None and self.domain is not None:
            thickness_edge = self.domain.apply_edge(thickness_edge)
        volume_flux = self._apply_cell_domain(-mesh.div(velocity * thickness_edge))
        return {
            "volume_flux": volume_flux,
            "total": volume_flux,
        }

    def temperature_content_tendency_terms(
        self,
        state: OceanState,
        *,
        u_e: torch.Tensor | None = None,
        h_edge: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return conservative temperature-content terms owned by transport."""

        velocity = self._active_edge_velocity(state) if u_e is None else u_e
        thickness_edge = state.mesh.cell_to_edge(state.h) if h_edge is None else h_edge
        if h_edge is None and self.domain is not None:
            thickness_edge = self.domain.apply_edge(thickness_edge)
        transport = self._horizontal_content_transport(
            state,
            state.temperature(),
            u_e=velocity,
            h_edge=thickness_edge,
        )
        return {
            "horizontal_transport": transport,
            "total": transport,
        }

    def salinity_content_tendency_terms(
        self,
        state: OceanState,
        *,
        u_e: torch.Tensor | None = None,
        h_edge: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return conservative salinity-content terms owned by transport."""

        velocity = self._active_edge_velocity(state) if u_e is None else u_e
        thickness_edge = state.mesh.cell_to_edge(state.h) if h_edge is None else h_edge
        if h_edge is None and self.domain is not None:
            thickness_edge = self.domain.apply_edge(thickness_edge)
        transport = self._horizontal_content_transport(
            state,
            state.salinity(),
            u_e=velocity,
            h_edge=thickness_edge,
        )
        return {
            "horizontal_transport": transport,
            "total": transport,
        }

    def tracer_content_tendency_terms(
        self,
        state: OceanState,
        *,
        u_e: torch.Tensor | None = None,
        h_edge: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Return conservative passive-tracer-content terms owned by transport."""

        if state.tracer_h is None:
            return None
        velocity = self._active_edge_velocity(state) if u_e is None else u_e
        thickness_edge = state.mesh.cell_to_edge(state.h) if h_edge is None else h_edge
        if h_edge is None and self.domain is not None:
            thickness_edge = self.domain.apply_edge(thickness_edge)
        tracer = state.tracers()
        fields = []
        for tracer_index in range(int(tracer.shape[-1])):
            fields.append(
                self._horizontal_content_transport(
                    state,
                    tracer[..., tracer_index],
                    u_e=velocity,
                    h_edge=thickness_edge,
                )
            )
        transport = torch.stack(fields, dim=-1)
        return {
            "horizontal_transport": transport,
            "total": transport,
        }

    def momentum_tendency_terms(
        self,
        state: OceanState,
        *,
        u_e: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return ocean momentum-equation terms owned by the C-grid dynamics."""

        if u_e is None:
            u_e = self._active_edge_velocity(state)
        mesh = state.mesh
        rotation = self._apply_edge_domain(self._trisk_rotation(state, u_e))
        bernoulli = self._apply_edge_domain(-mesh.grad(mesh.ke_cell(u_e)))
        pressure = self._apply_edge_domain(-self.pressure_gradient(state))
        return {
            "rotation": rotation,
            "bernoulli": bernoulli,
            "pressure_gradient": pressure,
            "total": rotation + bernoulli + pressure,
        }

    def edge_layer_volume_metric(
        self,
        state: OceanState,
        *,
        h_edge: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the edge/layer volume metric used by ocean work diagnostics."""

        mesh = state.mesh
        thickness_edge = mesh.cell_to_edge(state.h) if h_edge is None else h_edge
        if h_edge is None and self.domain is not None:
            thickness_edge = self.domain.apply_edge(thickness_edge)
        return mesh.dc_edge[:, None] * mesh.dv_edge[:, None] * thickness_edge

    def free_surface_pressure_acceleration(self, state: OceanState) -> torch.Tensor:
        """Return the free-surface pressure-gradient acceleration term."""

        acceleration = -state.mesh.grad(float(self.gravity) * state.eta())
        return self._apply_edge_domain(acceleration)

    def baroclinic_pressure_potential(self, state: OceanState) -> torch.Tensor:
        """Return the baroclinic pressure potential used by the PGF."""

        return self._baroclinic_potential(state)

    def baroclinic_gradient_pressure_acceleration(self, state: OceanState) -> torch.Tensor:
        """Return the gradient-form baroclinic pressure acceleration."""

        acceleration = -state.mesh.grad(self.baroclinic_pressure_potential(state))
        return self._apply_edge_domain(acceleration)

    def tendency_terms(self, state: OceanState) -> dict[str, dict[str, torch.Tensor]]:
        """Return a nested tensor decomposition whose totals match `tendency`."""
        self.check_state_compatible(state)
        self._validate_state(state)
        u_e = self._active_edge_velocity(state)
        h_edge = state.mesh.cell_to_edge(state.h)
        if self.domain is not None:
            h_edge = self.domain.apply_edge(h_edge)

        out: dict[str, dict[str, torch.Tensor]] = {
            "u_e": self.momentum_tendency_terms(state, u_e=u_e),
            "h": self.layer_thickness_tendency_terms(state, u_e=u_e, h_edge=h_edge),
            "temp_h": self.temperature_content_tendency_terms(state, u_e=u_e, h_edge=h_edge),
            "salt_h": self.salinity_content_tendency_terms(state, u_e=u_e, h_edge=h_edge),
        }
        tracer_terms = self.tracer_content_tendency_terms(state, u_e=u_e, h_edge=h_edge)
        if tracer_terms is not None:
            out["tracer_h"] = tracer_terms
        return out

    def terms_total(
        self,
        terms: dict[str, dict[str, torch.Tensor]],
        *,
        like: OceanState | None = None,
    ) -> OceanState:
        """Collapse `tendency_terms` output to an OceanState-shaped RHS."""
        tracer_total = terms["tracer_h"]["total"] if "tracer_h" in terms else None
        return OceanState(
            terms["u_e"]["total"],
            terms["h"]["total"],
            terms["temp_h"]["total"],
            terms["salt_h"]["total"],
            tracer_total,
            mesh=None if like is None else like.mesh,
            column=None if like is None else like.column,
            attrs={} if like is None else like._attrs_copy(),
        )

    def pressure_work_terms(
        self,
        state: OceanState,
        terms: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return pressure and pressure-potential work-exchange diagnostics.

        Free-surface and gradient-form baroclinic pressure work are compared
        against the pressure-potential energy rates implied by the horizontal
        layer-thickness tendency. ``sigma_depth`` metric-correction work is
        reported separately because it is not a pure ``-grad(phi)`` exchange.
        """

        self.check_state_compatible(state)
        if terms is None:
            terms = self.tendency_terms(state)

        u_e = self._active_edge_velocity(state)
        h_edge = state.mesh.cell_to_edge(state.h)
        if self.domain is not None:
            h_edge = self.domain.apply_edge(h_edge)
        metric = self.edge_layer_volume_metric(state, h_edge=h_edge)

        total_pressure_acceleration = terms["u_e"]["pressure_gradient"]
        free_surface_acceleration = self.free_surface_pressure_acceleration(state)
        baroclinic_gradient_acceleration = self.baroclinic_gradient_pressure_acceleration(state)
        baroclinic_pressure_acceleration = total_pressure_acceleration - free_surface_acceleration
        baroclinic_metric_correction_acceleration = baroclinic_pressure_acceleration - baroclinic_gradient_acceleration
        total_pressure_work = float(self.rho0) * (u_e * total_pressure_acceleration * metric).sum()
        free_surface_pressure_work = float(self.rho0) * (u_e * free_surface_acceleration * metric).sum()
        baroclinic_pressure_work = total_pressure_work - free_surface_pressure_work
        baroclinic_gradient_pressure_work = float(self.rho0) * (u_e * baroclinic_gradient_acceleration * metric).sum()
        baroclinic_metric_correction_work = float(self.rho0) * (u_e * baroclinic_metric_correction_acceleration * metric).sum()

        h_t = terms["h"].get("volume_flux", terms["h"]["total"])
        eta_t = h_t.sum(dim=-1, keepdim=True)
        free_surface_energy_rate = (
            float(self.rho0)
            * float(self.gravity)
            * state.mesh.integrate_cell(state.eta() * eta_t).sum()
        )
        baroclinic_potential = self.baroclinic_pressure_potential(state)
        baroclinic_energy_rate = (
            float(self.rho0)
            * state.mesh.integrate_cell((baroclinic_potential * h_t).sum(dim=-1, keepdim=True)).sum()
        )

        free_surface_residual = free_surface_pressure_work + free_surface_energy_rate
        free_surface_scale = torch.maximum(
            torch.maximum(free_surface_pressure_work.abs(), free_surface_energy_rate.abs()),
            torch.ones_like(free_surface_residual),
        )
        baroclinic_residual = baroclinic_gradient_pressure_work + baroclinic_energy_rate
        baroclinic_scale = torch.maximum(
            torch.maximum(baroclinic_gradient_pressure_work.abs(), baroclinic_energy_rate.abs()),
            torch.ones_like(baroclinic_residual),
        )
        closed_pressure_work = free_surface_pressure_work + baroclinic_gradient_pressure_work
        closed_pressure_energy_rate = free_surface_energy_rate + baroclinic_energy_rate
        closed_pressure_residual = closed_pressure_work + closed_pressure_energy_rate
        closed_pressure_scale = torch.maximum(
            torch.maximum(closed_pressure_work.abs(), closed_pressure_energy_rate.abs()),
            torch.ones_like(closed_pressure_residual),
        )
        total_pressure_energy_residual = total_pressure_work + closed_pressure_energy_rate
        total_pressure_scale = torch.maximum(
            torch.maximum(total_pressure_work.abs(), closed_pressure_energy_rate.abs()),
            torch.ones_like(total_pressure_energy_residual),
        )
        return {
            "pressure_work_total": total_pressure_work,
            "free_surface_pressure_work": free_surface_pressure_work,
            "baroclinic_pressure_work": baroclinic_pressure_work,
            "baroclinic_gradient_pressure_work": baroclinic_gradient_pressure_work,
            "baroclinic_metric_correction_work": baroclinic_metric_correction_work,
            "free_surface_energy_rate": free_surface_energy_rate,
            "baroclinic_pressure_potential_energy_rate": baroclinic_energy_rate,
            "closed_pressure_work": closed_pressure_work,
            "closed_pressure_energy_rate": closed_pressure_energy_rate,
            "free_surface_work_residual": free_surface_residual,
            "free_surface_work_residual_abs": free_surface_residual.abs(),
            "free_surface_work_residual_relative": free_surface_residual.abs() / free_surface_scale,
            "baroclinic_work_residual": baroclinic_residual,
            "baroclinic_work_residual_abs": baroclinic_residual.abs(),
            "baroclinic_work_residual_relative": baroclinic_residual.abs() / baroclinic_scale,
            "closed_pressure_work_residual": closed_pressure_residual,
            "closed_pressure_work_residual_abs": closed_pressure_residual.abs(),
            "closed_pressure_work_residual_relative": closed_pressure_residual.abs() / closed_pressure_scale,
            "total_pressure_energy_residual": total_pressure_energy_residual,
            "total_pressure_energy_residual_abs": total_pressure_energy_residual.abs(),
            "total_pressure_energy_residual_relative": total_pressure_energy_residual.abs() / total_pressure_scale,
        }

    def conservation_terms(self, state: OceanState) -> dict[str, torch.Tensor]:
        """Return integrated conservative ocean contents as pure tensors."""
        self.check_state_compatible(state)
        if self.domain is None:
            volume = state.total_volume().sum()
            heat = state.total_heat_content().sum()
            salt = state.total_salt_content().sum()
            tracer = state.total_tracer_content()
        else:
            volume = self.domain.total_volume(state).sum()
            heat = self.domain.total_heat_content(state).sum()
            salt = self.domain.total_salt_content(state).sum()
            tracer = self.domain.total_tracer_content(state)
        out = {
            "volume": volume,
            "heat": heat,
            "salt": salt,
        }
        if tracer is not None:
            out["tracer"] = tracer
        return out

    def tendency_budget_terms(
        self,
        state: OceanState,
        terms: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return integrated tendencies for conservative ocean contents."""
        if terms is None:
            terms = self.tendency_terms(state)
        rhs = self.terms_total(terms, like=state)
        mesh = state.mesh
        out = {
            "volume_tendency": mesh.integrate_cell(rhs.h.sum(dim=-1, keepdim=True)).sum(),
            "heat_tendency": mesh.integrate_cell(rhs.temp_h.sum(dim=-1, keepdim=True)).sum(),
            "salt_tendency": mesh.integrate_cell(rhs.salt_h.sum(dim=-1, keepdim=True)).sum(),
        }
        if rhs.tracer_h is not None:
            tracer_h = _normalize_tracer_tensor(rhs.tracer_h, state.mesh.nCells, state.column.n_layers)
            out["tracer_tendency"] = mesh.integrate_cell(tracer_h.sum(dim=-2))
        return out

    def coriolis_work_terms(
        self,
        state: OceanState,
        terms: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return pure tensor skew-work diagnostics for ocean rotation."""
        if terms is None:
            terms = self.tendency_terms(state)
        u_e = self._active_edge_velocity(state)
        rotation = terms["u_e"]["rotation"]
        metric = state.mesh.dc_edge[:, None] * state.mesh.dv_edge[:, None]
        work_density = u_e * rotation * metric
        work = work_density.sum()
        scale = work_density.abs().sum().clamp_min(torch.finfo(work.dtype).tiny)
        return {
            "rotation": rotation,
            "work": work,
            "work_abs_scale": scale,
            "work_relative": work.abs() / scale,
        }

    def tendency(self, state: OceanState) -> OceanState:
        self.check_state_compatible(state)
        self._validate_state(state)
        mesh = state.mesh
        domain = self.domain
        u_e = state.u_e if domain is None else domain.apply_edge(state.u_e)
        h = state.h

        h_edge = mesh.cell_to_edge(h)
        if domain is not None:
            h_edge = domain.apply_edge(h_edge)
        h_t = -mesh.div(u_e * h_edge)

        temp_flux = u_e * h_edge * self._edge_upwind(state, state.temperature(), u_e)
        salt_flux = u_e * h_edge * self._edge_upwind(state, state.salinity(), u_e)
        temp_h_t = -mesh.div(temp_flux)
        salt_h_t = -mesh.div(salt_flux)

        tracer_h_t = None
        if state.tracer_h is not None:
            tracer = state.tracers()
            fields = []
            for tracer_index in range(int(tracer.shape[-1])):
                flux = u_e * h_edge * self._edge_upwind(state, tracer[..., tracer_index], u_e)
                fields.append(-mesh.div(flux))
            tracer_h_t = torch.stack(fields, dim=-1)

        rotation = self._trisk_rotation(state, u_e)
        u_t = rotation - mesh.grad(mesh.ke_cell(u_e)) - self.pressure_gradient(state)
        if domain is not None:
            u_t = domain.apply_edge(u_t)
            h_t = domain.apply_cell(h_t)
            temp_h_t = domain.apply_cell(temp_h_t)
            salt_h_t = domain.apply_cell(salt_h_t)
            if tracer_h_t is not None:
                tracer_h_t = domain.apply_cell(tracer_h_t)

        return OceanState(
            u_t,
            h_t,
            temp_h_t,
            salt_h_t,
            tracer_h_t,
            mesh=state.mesh,
            column=state.column,
            attrs=state._attrs_copy(),
        )

    def kinetic_energy(self, state: OceanState) -> torch.Tensor:
        return float(self.rho0) * state.mesh.integrate_cell((state.ke_cell() * state.h).sum(dim=-1, keepdim=True))

    def free_surface_potential_energy(self, state: OceanState) -> torch.Tensor:
        eta = state.eta()
        return 0.5 * float(self.rho0) * float(self.gravity) * state.mesh.integrate_cell(eta.square())

    def total_energy(self, state: OceanState) -> torch.Tensor:
        return self.kinetic_energy(state) + self.free_surface_potential_energy(state)

    def vorticity(self, state: OceanState) -> torch.Tensor:
        return state.mesh.curl(state.u_e)

    @property
    def diagnostics(self) -> OceanDiagnostics:
        """Detached reporting facade for VFS-O diagnostics."""

        if not hasattr(self, "_diagnostics"):
            self._diagnostics = OceanDiagnostics(self)
        return self._diagnostics


class HydrostaticBoussinesqOceanDynamics(MultilayerOceanDynamics):
    """PE-oriented extension of the multilayer ocean core.

    This keeps the horizontal vector-invariant C-grid dynamics from
    ``MultilayerOceanDynamics`` and adds the first hydrostatic-Boussinesq
    primitive-equation pieces: diagnostic interface vertical velocity from
    continuity plus optional vertical scalar diffusion and momentum viscosity.
    """

    recipe_name = "hydrostatic_boussinesq_ocean"

    def __init__(
        self,
        *,
        scalar_vertical_diffusivity: float = 0.0,
        momentum_vertical_viscosity: float = 0.0,
        horizontal_momentum_viscosity: float = 0.0,
        horizontal_momentum_viscosity_scheme: str = "constant",
        horizontal_momentum_viscosity_cfl: float = 0.0,
        horizontal_momentum_viscosity_dt: float | None = None,
        horizontal_momentum_biharmonic_viscosity: float = 0.0,
        horizontal_momentum_biharmonic_cfl: float = 0.0,
        vertical_remap_mode: str = "none",
        vertical_remap_timescale: float = 86400.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.scalar_vertical_diffusivity = float(scalar_vertical_diffusivity)
        self.momentum_vertical_viscosity = float(momentum_vertical_viscosity)
        self.horizontal_momentum_viscosity = float(horizontal_momentum_viscosity)
        self.horizontal_momentum_viscosity_scheme = str(horizontal_momentum_viscosity_scheme)
        self.horizontal_momentum_viscosity_cfl = float(horizontal_momentum_viscosity_cfl)
        self.horizontal_momentum_viscosity_dt = None if horizontal_momentum_viscosity_dt is None else float(horizontal_momentum_viscosity_dt)
        self.horizontal_momentum_biharmonic_viscosity = float(horizontal_momentum_biharmonic_viscosity)
        self.horizontal_momentum_biharmonic_cfl = float(horizontal_momentum_biharmonic_cfl)
        self.vertical_remap_mode = str(vertical_remap_mode)
        self.vertical_remap_timescale = float(vertical_remap_timescale)
        if self.horizontal_momentum_viscosity_scheme not in ("constant", "mesh_cfl_scaled", "biharmonic"):
            raise ValueError(
                "horizontal_momentum_viscosity_scheme must be one of "
                "'constant', 'mesh_cfl_scaled', or 'biharmonic'; "
                f"got {self.horizontal_momentum_viscosity_scheme!r}"
            )
        if self.horizontal_momentum_viscosity_scheme == "mesh_cfl_scaled":
            if self.horizontal_momentum_viscosity_dt is None or self.horizontal_momentum_viscosity_dt <= 0.0:
                raise ValueError("horizontal_momentum_viscosity_dt must be positive for mesh_cfl_scaled viscosity")
            if self.horizontal_momentum_viscosity_cfl <= 0.0:
                raise ValueError("horizontal_momentum_viscosity_cfl must be positive for mesh_cfl_scaled viscosity")
        if self.horizontal_momentum_viscosity_scheme == "biharmonic":
            if self.horizontal_momentum_biharmonic_viscosity == 0.0 and self.horizontal_momentum_biharmonic_cfl <= 0.0:
                raise ValueError(
                    "biharmonic viscosity requires horizontal_momentum_biharmonic_viscosity "
                    "or horizontal_momentum_biharmonic_cfl"
                )
            if self.horizontal_momentum_biharmonic_cfl > 0.0:
                if self.horizontal_momentum_viscosity_dt is None or self.horizontal_momentum_viscosity_dt <= 0.0:
                    raise ValueError("horizontal_momentum_viscosity_dt must be positive for CFL-scaled biharmonic viscosity")
        if self.vertical_remap_mode not in ("none", "target_thickness_relaxation"):
            raise ValueError(
                "vertical_remap_mode must be 'none' or 'target_thickness_relaxation', "
                f"got {self.vertical_remap_mode!r}"
            )
        if self.vertical_remap_mode != "none" and self.vertical_remap_timescale <= 0.0:
            raise ValueError("vertical_remap_timescale must be positive when vertical remap is enabled")

    def layer_volume_divergence(
        self,
        state: OceanState,
        *,
        u_e: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Horizontal divergence of layer volume transport ``div(h u)``."""

        mesh = state.mesh
        u = state.u_e if u_e is None else u_e
        thickness = state.h if h is None else h
        if self.domain is not None:
            u = self.domain.apply_edge(u)
        h_edge = mesh.cell_to_edge(thickness)
        if self.domain is not None:
            h_edge = self.domain.apply_edge(h_edge)
        div_hu = mesh.div(u * h_edge)
        return div_hu if self.domain is None else self.domain.apply_cell(div_hu)

    def interface_vertical_velocity(
        self,
        state: OceanState,
        *,
        u_e: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
        top_velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Diagnostic interface velocity from layer continuity.

        The vertical coordinate is depth-positive-downward. Returned values
        are positive downward at layer interfaces with shape
        ``[..., nCells, nLayers + 1]``. The diagnostic chooses ``w_top = 0``
        by default and enforces

        ``div(h u)_k + w_{k+1/2} - w_{k-1/2} = 0``

        exactly for each layer. The bottom value is therefore the column
        integral of horizontal divergence; for a free-surface run it should be
        interpreted as the barotropic column convergence diagnostic, not as an
        impermeable-bottom boundary condition.
        """

        div_hu = self.layer_volume_divergence(state, u_e=u_e, h=h)
        if top_velocity is None:
            top = torch.zeros_like(div_hu[..., :1])
        else:
            top = top_velocity.to(dtype=div_hu.dtype, device=div_hu.device)
            if top.ndim == div_hu.ndim - 1:
                top = top.unsqueeze(-1)
            top = top.expand(div_hu.shape[:-1] + (1,))
        interior_and_bottom = top - torch.cumsum(div_hu, dim=-1)
        return torch.cat([top, interior_and_bottom], dim=-1)

    def continuity_residual(
        self,
        state: OceanState,
        *,
        u_e: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``div(h u) + delta_z(w)`` for the diagnostic interface velocity."""

        div_hu = self.layer_volume_divergence(state, u_e=u_e, h=h)
        w_i = self.interface_vertical_velocity(state, u_e=u_e, h=h)
        return div_hu + w_i[..., 1:] - w_i[..., :-1]

    def closed_interface_velocity(
        self,
        state: OceanState,
        interface_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Return a closed-boundary interface transport velocity.

        ``interface_velocity`` may carry only internal interfaces with trailing
        shape ``[..., nCells, nLayers - 1]`` or all interfaces with trailing
        shape ``[..., nCells, nLayers + 1]``. In both cases the returned tensor
        has zero top/bottom boundary fluxes and shape
        ``[..., nCells, nLayers + 1]``.
        """

        n_cells = int(state.mesh.nCells)
        n_layers = int(state.column.n_layers)
        value = interface_velocity.to(dtype=state.h.dtype, device=state.h.device)
        if value.ndim < 2 or int(value.shape[-2]) != n_cells:
            raise ValueError(
                "interface_velocity must have trailing shape "
                f"[nCells, nLayers - 1] or [nCells, nLayers + 1]; got {tuple(value.shape)}"
            )
        if int(value.shape[-1]) == n_layers + 1:
            internal = value[..., 1:-1]
        elif int(value.shape[-1]) == max(n_layers - 1, 0):
            internal = value
        else:
            raise ValueError(
                "interface_velocity must have trailing shape "
                f"[nCells, {max(n_layers - 1, 0)}] or [nCells, {n_layers + 1}]; got {tuple(value.shape)}"
            )
        boundary = torch.zeros(value.shape[:-1] + (1,), dtype=value.dtype, device=value.device)
        return torch.cat([boundary, internal, boundary], dim=-1)

    def vertical_volume_transport_tendency(
        self,
        state: OceanState,
        interface_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Conservative layer-thickness tendency from vertical remapping.

        Interface velocity is positive downward. Top and bottom fluxes are
        closed by construction through ``closed_interface_velocity``.
        """

        w_i = self.closed_interface_velocity(state, interface_velocity)
        out = torch.zeros_like(state.h)
        if int(state.column.n_layers) < 2:
            return out
        flux = w_i[..., 1:-1]
        out[..., :-1] -= flux
        out[..., 1:] += flux
        return out

    def vertical_content_transport_tendency(
        self,
        state: OceanState,
        content: torch.Tensor,
        interface_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Conservative upwind vertical transport for layer contents.

        ``content`` is a conservative cell/layer quantity such as ``temp_h``,
        ``salt_h``, or ``tracer_h``. The same volume flux should be applied to
        ``h`` when the intended behavior is vertical remapping; then a uniform
        concentration remains uniform.
        """

        w_i = self.closed_interface_velocity(state, interface_velocity)
        if int(state.column.n_layers) < 2:
            return torch.zeros_like(content)
        h = state.h.clamp_min(torch.finfo(state.h.dtype).tiny)
        if content.ndim == h.ndim:
            q = content / h
            w = w_i[..., 1:-1]
            flux = w * torch.where(w >= 0.0, q[..., :-1], q[..., 1:])
            out = torch.zeros_like(content)
            out[..., :-1] -= flux
            out[..., 1:] += flux
            return out
        if content.ndim == h.ndim + 1:
            q = content / h[..., None]
            w = w_i[..., 1:-1, None]
            flux = w * torch.where(w >= 0.0, q[..., :-1, :], q[..., 1:, :])
            out = torch.zeros_like(content)
            out[..., :-1, :] -= flux
            out[..., 1:, :] += flux
            return out
        raise ValueError(
            "vertical content transport expects content with shape "
            "[..., nCells, nLayers] or [..., nCells, nLayers, nTracers]"
        )

    def conservative_vertical_transport_tendency(
        self,
        state: OceanState,
        interface_velocity: torch.Tensor,
        *,
        advect_thickness: bool = True,
    ) -> OceanState:
        """Return a closed-column conservative vertical-remap tendency."""

        h_t = self.vertical_volume_transport_tendency(state, interface_velocity) if advect_thickness else torch.zeros_like(state.h)
        temp_h_t = self.vertical_content_transport_tendency(state, state.temp_h, interface_velocity)
        salt_h_t = self.vertical_content_transport_tendency(state, state.salt_h, interface_velocity)
        tracer_h_t = None
        if state.tracer_h is not None:
            tracer_h_t = self.vertical_content_transport_tendency(
                state,
                _normalize_tracer_tensor(state.tracer_h, state.mesh.nCells, state.column.n_layers),
                interface_velocity,
            )
        return state.with_tensors(torch.zeros_like(state.u_e), h_t, temp_h_t, salt_h_t, tracer_h_t)

    def target_thickness(self, state: OceanState) -> torch.Tensor:
        """Layer thickness target that preserves each column's total depth."""

        rest = state.column.rest_thickness_like(state.h)
        rest_total = rest.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(state.h.dtype).tiny)
        column_total = state.h.sum(dim=-1, keepdim=True)
        return rest * (column_total / rest_total)

    def target_thickness_relaxation_tendency(self, state: OceanState) -> torch.Tensor:
        """Closed-column relaxation of layer thickness toward rest-shape fractions."""

        target = self.target_thickness(state)
        tendency = (target - state.h) / float(self.vertical_remap_timescale)
        return tendency - tendency.mean(dim=-1, keepdim=True)

    def target_thickness_relaxation_interface_velocity(self, state: OceanState) -> torch.Tensor:
        """Internal interface velocity that realizes target-thickness relaxation."""

        if int(state.column.n_layers) < 2:
            return torch.zeros(state.h.shape[:-1] + (0,), dtype=state.h.dtype, device=state.h.device)
        h_t = self.target_thickness_relaxation_tendency(state)
        return -torch.cumsum(h_t[..., :-1], dim=-1)

    def scalar_vertical_diffusion_tendency(
        self,
        state: OceanState,
        content: torch.Tensor,
        *,
        diffusivity: float | None = None,
    ) -> torch.Tensor:
        """Conservative vertical diffusion tendency for cell/layer contents."""

        kappa = self.scalar_vertical_diffusivity if diffusivity is None else float(diffusivity)
        if kappa == 0.0 or int(state.column.n_layers) < 2:
            return torch.zeros_like(content)
        h = state.h
        spacing = _cell_midlayer_spacing(state)
        if content.ndim == h.ndim:
            q = content / h.clamp_min(torch.finfo(h.dtype).tiny)
            flux = float(kappa) * (q[..., 1:] - q[..., :-1]) / spacing
            out = torch.zeros_like(content)
            out[..., :-1] += flux
            out[..., 1:] -= flux
            return out
        if content.ndim == h.ndim + 1:
            h_b = h[..., None].clamp_min(torch.finfo(h.dtype).tiny)
            q = content / h_b
            flux = float(kappa) * (q[..., 1:, :] - q[..., :-1, :]) / spacing[..., None]
            out = torch.zeros_like(content)
            out[..., :-1, :] += flux
            out[..., 1:, :] -= flux
            return out
        raise ValueError(
            "vertical scalar diffusion expects content with shape "
            "[..., nCells, nLayers] or [..., nCells, nLayers, nTracers]"
        )

    def velocity_vertical_diffusion_tendency(
        self,
        state: OceanState,
        *,
        viscosity: float | None = None,
    ) -> torch.Tensor:
        """Vertical viscosity tendency for edge/layer velocity."""

        nu = self.momentum_vertical_viscosity if viscosity is None else float(viscosity)
        return _ocean_vertical_momentum_stress_tendency(
            state,
            viscosity=float(nu),
            rho0=self.rho0,
        )

    def velocity_horizontal_diffusion_tendency(
        self,
        state: OceanState,
        *,
        viscosity: float | None = None,
    ) -> torch.Tensor:
        """Dissipative mesh-owned horizontal momentum diffusion."""

        if viscosity is not None:
            nu = float(viscosity)
            if nu == 0.0:
                return torch.zeros_like(state.u_e)
            out = nu * state.mesh.vector_laplacian_edge(state.u_e)
            return out if self.domain is None else self.domain.apply_edge(out)

        scheme = self.horizontal_momentum_viscosity_scheme
        if scheme == "constant":
            nu = self.horizontal_momentum_viscosity
            if nu == 0.0:
                return torch.zeros_like(state.u_e)
            out = float(nu) * state.mesh.vector_laplacian_edge(state.u_e)
            return out if self.domain is None else self.domain.apply_edge(out)

        if scheme == "mesh_cfl_scaled":
            nu_eff = self.effective_horizontal_momentum_viscosity(state)
            out = nu_eff * state.mesh.vector_laplacian_edge(state.u_e)
            return out if self.domain is None else self.domain.apply_edge(out)

        if scheme == "biharmonic":
            nu4 = self.effective_horizontal_biharmonic_viscosity(state)
            lap_u = state.mesh.vector_laplacian_edge(state.u_e)
            out = -nu4 * state.mesh.vector_laplacian_edge(lap_u)
            return out if self.domain is None else self.domain.apply_edge(out)

        raise AssertionError(f"unsupported horizontal viscosity scheme {scheme!r}")

    def horizontal_viscosity_length_scale(self, state: OceanState) -> torch.Tensor:
        """Area-equivalent wet-cell mesh length used by scaled viscosity schemes."""

        area = state.mesh.cell_area.to(dtype=state.h.dtype, device=state.h.device)
        if self.domain is None:
            area_mean = area.mean()
        else:
            wet = self.domain.wet_cell.to(dtype=area.dtype, device=area.device)
            area_mean = (area * wet).sum() / wet.sum().clamp_min(1.0)
        return torch.sqrt(area_mean.clamp_min(torch.finfo(area.dtype).tiny))

    def effective_horizontal_momentum_viscosity(self, state: OceanState) -> torch.Tensor:
        """Return the Laplacian viscosity currently implied by the selected scheme."""

        if self.horizontal_momentum_viscosity_scheme == "constant":
            return torch.as_tensor(self.horizontal_momentum_viscosity, dtype=state.u_e.dtype, device=state.u_e.device)
        if self.horizontal_momentum_viscosity_scheme == "mesh_cfl_scaled":
            dx = self.horizontal_viscosity_length_scale(state).to(dtype=state.u_e.dtype, device=state.u_e.device)
            return float(self.horizontal_momentum_viscosity_cfl) * dx.square() / float(self.horizontal_momentum_viscosity_dt)
        return torch.zeros((), dtype=state.u_e.dtype, device=state.u_e.device)

    def effective_horizontal_biharmonic_viscosity(self, state: OceanState) -> torch.Tensor:
        """Return the biharmonic viscosity implied by the selected scheme."""

        if self.horizontal_momentum_viscosity_scheme != "biharmonic":
            return torch.zeros((), dtype=state.u_e.dtype, device=state.u_e.device)
        if self.horizontal_momentum_biharmonic_cfl > 0.0:
            dx = self.horizontal_viscosity_length_scale(state).to(dtype=state.u_e.dtype, device=state.u_e.device)
            return float(self.horizontal_momentum_biharmonic_cfl) * dx.pow(4) / float(self.horizontal_momentum_viscosity_dt)
        return torch.as_tensor(self.horizontal_momentum_biharmonic_viscosity, dtype=state.u_e.dtype, device=state.u_e.device)

    def has_horizontal_momentum_diffusion(self) -> bool:
        """Return whether the configured horizontal momentum diffusion is active."""

        scheme = self.horizontal_momentum_viscosity_scheme
        if scheme == "constant":
            return self.horizontal_momentum_viscosity != 0.0
        if scheme == "mesh_cfl_scaled":
            return self.horizontal_momentum_viscosity_cfl != 0.0
        if scheme == "biharmonic":
            return self.horizontal_momentum_biharmonic_viscosity != 0.0 or self.horizontal_momentum_biharmonic_cfl != 0.0
        return False

    def horizontal_momentum_diffusion_metadata(self, state: OceanState) -> dict[str, torch.Tensor | str]:
        """Return tensor-friendly metadata for reporting selected diffusion strength."""

        scheme = self.horizontal_momentum_viscosity_scheme
        if scheme == "biharmonic":
            return {
                "scheme": scheme,
                "length_scale_m": self.horizontal_viscosity_length_scale(state),
                "effective_biharmonic_viscosity_m4_s": self.effective_horizontal_biharmonic_viscosity(state),
            }
        if scheme == "mesh_cfl_scaled":
            return {
                "scheme": scheme,
                "length_scale_m": self.horizontal_viscosity_length_scale(state),
                "effective_viscosity_m2_s": self.effective_horizontal_momentum_viscosity(state),
            }
        return {
            "scheme": scheme,
            "effective_viscosity_m2_s": self.effective_horizontal_momentum_viscosity(state),
        }

    def _horizontal_viscosity_term_is_active(self) -> bool:
        """Compatibility helper for process-term bookkeeping."""

        return self.has_horizontal_momentum_diffusion()

    def velocity_horizontal_laplacian_tendency(
        self,
        state: OceanState,
        *,
        viscosity: float,
    ) -> torch.Tensor:
        """Explicit constant-Laplacian helper retained for focused diagnostics."""

        if viscosity == 0.0:
            return torch.zeros_like(state.u_e)
        out = float(viscosity) * state.mesh.vector_laplacian_edge(state.u_e)
        return out if self.domain is None else self.domain.apply_edge(out)

    def momentum_process_tendency_terms(self, state: OceanState) -> dict[str, torch.Tensor]:
        """Return optional momentum process terms owned by ocean viscosity."""

        out: dict[str, torch.Tensor] = {}
        if self.momentum_vertical_viscosity != 0.0:
            out["vertical_viscosity"] = self.velocity_vertical_diffusion_tendency(state)
        if self._horizontal_viscosity_term_is_active():
            out["horizontal_viscosity"] = self.velocity_horizontal_diffusion_tendency(state)
        return out

    def scalar_diffusion_tendency_terms(
        self,
        state: OceanState,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Return optional scalar process terms owned by vertical diffusion."""

        if self.scalar_vertical_diffusivity == 0.0:
            return {}
        out = {
            "temp_h": {
                "vertical_diffusion": self.scalar_vertical_diffusion_tendency(state, state.temp_h),
            },
            "salt_h": {
                "vertical_diffusion": self.scalar_vertical_diffusion_tendency(state, state.salt_h),
            },
        }
        if state.tracer_h is not None:
            tracer_h = _normalize_tracer_tensor(
                state.tracer_h,
                state.mesh.nCells,
                state.column.n_layers,
            )
            out["tracer_h"] = {
                "vertical_diffusion": self.scalar_vertical_diffusion_tendency(state, tracer_h),
            }
        return out

    def vertical_remap_tendency_terms(
        self,
        state: OceanState,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Return optional closed-column vertical remap process terms."""

        if self.vertical_remap_mode != "target_thickness_relaxation":
            return {}
        interface_velocity = self.target_thickness_relaxation_interface_velocity(state)
        remap = self.conservative_vertical_transport_tendency(state, interface_velocity)
        out = {
            "h": {"vertical_remap": remap.h},
            "temp_h": {"vertical_remap": remap.temp_h},
            "salt_h": {"vertical_remap": remap.salt_h},
        }
        if remap.tracer_h is not None:
            out["tracer_h"] = {"vertical_remap": remap.tracer_h}
        return out

    def _add_process_terms(
        self,
        out: dict[str, dict[str, torch.Tensor]],
        field: str,
        field_terms: dict[str, torch.Tensor],
    ) -> None:
        for name, value in field_terms.items():
            if field == "u_e":
                value = self._apply_edge_domain(value)
            else:
                value = self._apply_cell_domain(value)
            out[field][name] = value
            out[field]["total"] = out[field]["total"] + value

    def tendency_terms(self, state: OceanState) -> dict[str, dict[str, torch.Tensor]]:
        """Return hydrostatic-ocean terms whose totals match `tendency`."""
        out = {
            field: dict(field_terms)
            for field, field_terms in super().tendency_terms(state).items()
        }

        self._add_process_terms(out, "u_e", self.momentum_process_tendency_terms(state))
        if self.scalar_vertical_diffusivity != 0.0:
            for field, field_terms in self.scalar_diffusion_tendency_terms(state).items():
                self._add_process_terms(out, field, field_terms)
        if self.vertical_remap_mode != "none":
            for field, field_terms in self.vertical_remap_tendency_terms(state).items():
                self._add_process_terms(out, field, field_terms)
        return out

    def tendency(self, state: OceanState) -> OceanState:
        rhs = super().tendency(state)
        u_t = rhs.u_e + self.velocity_vertical_diffusion_tendency(state) + self.velocity_horizontal_diffusion_tendency(state)
        h_t = rhs.h
        temp_h_t = rhs.temp_h + self.scalar_vertical_diffusion_tendency(state, state.temp_h)
        salt_h_t = rhs.salt_h + self.scalar_vertical_diffusion_tendency(state, state.salt_h)
        tracer_h_t = rhs.tracer_h
        if state.tracer_h is not None:
            tracer_diffusion = self.scalar_vertical_diffusion_tendency(state, _normalize_tracer_tensor(state.tracer_h, state.mesh.nCells, state.column.n_layers))
            tracer_h_t = tracer_diffusion if tracer_h_t is None else tracer_h_t + tracer_diffusion
        if self.vertical_remap_mode == "target_thickness_relaxation":
            interface_velocity = self.target_thickness_relaxation_interface_velocity(state)
            remap = self.conservative_vertical_transport_tendency(state, interface_velocity)
            h_t = h_t + remap.h
            temp_h_t = temp_h_t + remap.temp_h
            salt_h_t = salt_h_t + remap.salt_h
            if remap.tracer_h is not None:
                tracer_h_t = remap.tracer_h if tracer_h_t is None else tracer_h_t + remap.tracer_h
        if self.domain is not None:
            u_t = self.domain.apply_edge(u_t)
            h_t = self.domain.apply_cell(h_t)
            temp_h_t = self.domain.apply_cell(temp_h_t)
            salt_h_t = self.domain.apply_cell(salt_h_t)
            if tracer_h_t is not None:
                tracer_h_t = self.domain.apply_cell(tracer_h_t)
        return rhs.with_tensors(u_t, h_t, temp_h_t, salt_h_t, tracer_h_t)


register_component("multilayer_ocean", MultilayerOceanDynamics)
register_component("hydrostatic_boussinesq_ocean", HydrostaticBoussinesqOceanDynamics)
