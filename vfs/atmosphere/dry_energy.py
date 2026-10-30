"""Dry atmospheric energy and analytic derivatives at fixed prognostic content.

Methods are defined here normally, not installed from external function tables.
The conservative core supplies its diagonal kinetic metric and geometry checks.
There is no strategy selection and this class owns no state or geometry.
"""
from __future__ import annotations
import torch
from ..core.state import interface_mass
from ..core.state import State


class DryEnergy:
    """Thermal energy, pressure-content derivatives and tensor budget diagnostics."""
    def total_mass(self, state: State) -> torch.Tensor:
        """
        Total atmospheric mass per unit area, integrated horizontally.

        mass = (1/g) * Σ_k integrate_cell(dp_k)   [Pa * m² / (m/s²)] = [kg]

        Returns a scalar (or [...] tensor for batched states).
        """
        self.check_state_compatible(state)
        return state.total_mass()

    def kinetic_energy_gradient_u(self, state: State) -> torch.Tensor:
        """
        Closed-form ``d(KE)/d(u_e)`` for the active mesh kinetic energy.

        The conservative core supplies its diagonal edge metric: edge area
        times pressure thickness divided by gravity, multiplied by edge wind.
        This covector pairs with edge/full acceleration to give watts.
        """
        self.check_state_compatible(state)
        return self.kinetic_energy_gradient_u_terms(state)["total"]

    def scalar_energy_gradients(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Closed-form dry-energy gradients for scalar prognostic fields.

        Returns ``(dE/dmtheta_i, dE/dps)`` for the dry energy functional used by
        ``total_energy``.  The pressure gradient is taken at fixed prognostic
        ``mtheta_i``, so it includes both explicit pressure dependence and the
        induced ``theta = mtheta/dm(ps)`` dependence.
        """
        self.check_state_compatible(state)
        theta = state.theta()
        dp = state.layer_dp()
        dm = interface_mass(dp)
        grad_mtheta = self._dry_energy_gradient_mtheta_for_coord_work(state, theta, dp=dp, dm=dm)
        grad_ps = self._dry_energy_gradient_ps(state, theta, dp=dp, dm=dm, grad_mtheta=grad_mtheta)
        return grad_mtheta, grad_ps

    def scalar_pressure_energy_gradient_terms(self, state: State) -> dict[str, torch.Tensor]:
        """
        Decompose ``dE/dps`` into physically interpretable closed-form pieces.

        The terms sum to the pressure component returned by
        ``scalar_energy_gradients``.  They are diagnostic inspection fields:
        changing ``ps`` at fixed prognostic ``mtheta`` changes layer mass,
        geopotential/internal-energy placement, and recovered theta.
        """
        self.check_state_compatible(state)
        theta = state.theta()
        dp = state.layer_dp()
        dm = interface_mass(dp)
        grad_mtheta = self._dry_energy_gradient_mtheta_for_coord_work(state, theta, dp=dp, dm=dm)
        return self._dry_energy_gradient_ps_terms(state, theta, dp=dp, dm=dm, grad_mtheta=grad_mtheta)

    def scalar_pressure_energy_gradient_local_terms(self, state: State) -> dict[str, dict[str, torch.Tensor]]:
        """
        Local ``dE/dps`` decomposition before column summation.

        Returns ``layer`` pieces for kinetic-mass, potential, and internal
        energy, ``interface`` pieces for theta/mass coupling, and ``column``
        pieces whose values sum to ``scalar_pressure_energy_gradient_terms``.
        """
        self.check_state_compatible(state)
        theta = state.theta()
        dp = state.layer_dp()
        dm = interface_mass(dp)
        grad_mtheta = self._dry_energy_gradient_mtheta_for_coord_work(state, theta, dp=dp, dm=dm)
        return self._dry_energy_gradient_ps_local_terms(state, theta, dp=dp, dm=dm, grad_mtheta=grad_mtheta)

    def scalar_energy_rate_closed(
        self,
        state: State,
        mtheta_tendency: torch.Tensor,
        ps_tendency: torch.Tensor,
    ) -> torch.Tensor:
        """Exact dry-energy rate from scalar tendencies using closed-form gradients."""
        grad_mtheta, grad_ps = self.scalar_energy_gradients(state)
        return (grad_mtheta * mtheta_tendency).sum() + (grad_ps * ps_tendency).sum()

    def scalar_energy_rate_terms_closed(
        self,
        state: State,
        mtheta_terms: dict[str, torch.Tensor],
        ps_terms: dict[str, torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        """
        Closed-form dry-energy rates for scalar tendency components.

        This is an inspection helper for pressure-work debugging.  It uses the
        same analytic ``dE/dmtheta`` and ``dE/dps`` as ``total_energy`` and
        returns rates for each named scalar term plus their totals.
        """
        grad_mtheta, grad_ps = self.scalar_energy_gradients(state)
        mtheta_rates = {name: (grad_mtheta * value).sum() for name, value in mtheta_terms.items()}
        ps_rates = {name: (grad_ps * value).sum() for name, value in ps_terms.items()}
        zero = torch.zeros((), dtype=state.ps.dtype, device=state.ps.device)
        mtheta_total = mtheta_rates.get(
            "total",
            sum((v for k, v in mtheta_rates.items() if k != "total"), zero),
        )
        ps_total = ps_rates.get(
            "total",
            sum((v for k, v in ps_rates.items() if k != "total"), zero),
        )
        return {
            "mtheta_i": mtheta_rates,
            "ps": ps_rates,
            "mtheta_total": mtheta_total,
            "ps_total": ps_total,
            "total": mtheta_total + ps_total,
        }

    def scalar_pressure_energy_rate_terms_closed(
        self,
        state: State,
        ps_terms: dict[str, torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Closed-form ps-energy-rate decomposition by ps-gradient piece."""
        grad_terms = self.scalar_pressure_energy_gradient_terms(state)
        return {
            component: {
                name: (grad * value).sum()
                for name, value in ps_terms.items()
            }
            for component, grad in grad_terms.items()
        }

    def wind_energy_rate_terms_closed(
        self,
        state: State,
        u_terms: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Closed-form kinetic-energy rates for named wind tendency terms."""
        grad_u = self.kinetic_energy_gradient_u(state)
        return {name: (grad_u * value).sum() for name, value in u_terms.items()}

    def energy_rate_closed(self, state: State, tendency: State) -> torch.Tensor:
        """Closed-form dry-energy directional derivative for a State-shaped RHS."""
        grad_u = self.kinetic_energy_gradient_u(state)
        grad_mtheta, grad_ps = self.scalar_energy_gradients(state)
        return (
            (grad_u * tendency.u_e).sum()
            + (grad_mtheta * tendency.mtheta_i).sum()
            + (grad_ps * tendency.ps).sum()
        )

    def total_energy(self, state: State) -> torch.Tensor:
        """
        Total dry energy: KE + PE + IE.

        Energy is in joules. The semidiscrete closed budget includes the
        fixed-pressure lid reservoir separately (boundary_total_energy).
        Finite timesteps can introduce an energy residual.
        """
        return (
            self.total_kinetic_energy(state)
            + self.total_potential_energy(state)
            + self.total_internal_energy(state)
        )

    def total_potential_energy(self, state: State) -> torch.Tensor:
        """
        Total potential energy (geopotential) integrated over the domain.

        PE = (1/g) * Σ_k integrate_cell(phi_full_k * dp_k)

        Returns a scalar (or [...] tensor).
        """
        self.check_state_compatible(state)
        dp       = state.layer_dp()
        phi_full = state.geopotential()
        return state.mesh.integrate_cell(
            (phi_full * dp).sum(dim=-1, keepdim=True)
        ) / self.gravity

    def total_internal_energy(self, state: State) -> torch.Tensor:
        """
        Total internal energy integrated over the domain.

        IE = (c_v / g) * Σ_k integrate_cell(T_full_k * dp_k)
           = (c_v / c_p / g) * Σ_k integrate_cell(theta_bar_k * pi_full_k * dp_k)

        For a dry ideal gas: c_v = c_p - R_d.

        Returns a scalar (or [...] tensor).
        """
        self.check_state_compatible(state)
        c_v   = state.column.c_p - state.column.R_d
        dp    = state.layer_dp()
        T     = state.temperature()
        return state.mesh.integrate_cell(
            (c_v * T * dp).sum(dim=-1, keepdim=True)
        ) / self.gravity

    def _cell_full_energy_weight(self, state: State, dp: torch.Tensor) -> torch.Tensor:
        """Cell-area/full-layer energy weight area*dp/g."""

        shape = [1] * dp.ndim
        shape[-2] = state.mesh.nCells
        shape[-1] = 1
        area = state.mesh.cell_area.reshape(shape).to(dtype=dp.dtype, device=dp.device)
        return area * dp / float(self.gravity)

    def _dry_energy_gradient_mtheta_for_coord_work(
        self,
        state: State,
        theta_i: torch.Tensor,
        *,
        dp: torch.Tensor | None = None,
        dm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return dE/dmtheta for the current thermodynamic state.

        Closed-form adjoint of the thermodynamic energy with ps fixed.  KE has
        no mtheta dependence.  The result is tied to the active full-level
        placement mode used by total_potential_energy/total_internal_energy.
        """

        ps = state.ps
        column = state.column
        dp = state.layer_dp() if dp is None else dp
        dm = interface_mass(dp) if dm is None else dm
        w_full = self._cell_full_energy_weight(state, dp)
        grad_theta = torch.zeros_like(theta_i)

        mode = column.phi_full_mode
        pi_i = column.exner_interface(ps)

        # Potential-energy adjoint.
        if mode == "interface_mean":
            psi_i = torch.zeros_like(theta_i)
            psi_i[..., :-1] += 0.5 * w_full
            psi_i[..., 1:] += 0.5 * w_full
            grad_theta = grad_theta + column.hydrostatic_adjoint_rhs(psi_i, ps)
        elif mode in ("linear_pfull_adjoint", "direct_u_linear_adjoint"):
            pi_full = column.exner_full_pmid(ps)
            partial = pi_i[..., 1:] - pi_full
            psi_i = torch.zeros_like(theta_i)
            psi_i[..., 1:] += w_full
            grad_theta = grad_theta + column.hydrostatic_adjoint_rhs(psi_i, ps)
            grad_theta[..., :-1] += 0.5 * w_full * partial
            grad_theta[..., 1:] += 0.5 * w_full * partial
        elif mode == "direct_u_gl2_adjoint":
            pi_u = column.exner_full_pmid(ps)
            delta = (pi_i[..., 1:] - pi_i[..., :-1]).clamp_min(torch.finfo(theta_i.dtype).tiny)
            s = (pi_u - pi_i[..., :-1]) / delta
            coef_top = 0.5 * delta * (1.0 - s).square()
            coef_bot = 0.5 * delta * (1.0 - s.square())
            psi_i = torch.zeros_like(theta_i)
            psi_i[..., 1:] += w_full
            grad_theta = grad_theta + column.hydrostatic_adjoint_rhs(psi_i, ps)
            grad_theta[..., :-1] += w_full * coef_top
            grad_theta[..., 1:] += w_full * coef_bot
        else:
            raise RuntimeError(f"unsupported phi_full_mode={mode!r}")

        # Internal-energy adjoint.
        c_v = column.c_p - column.R_d
        if mode == "direct_u_gl2_adjoint":
            pi_full = column.exner_full_pmid(ps)
            delta = (pi_i[..., 1:] - pi_i[..., :-1]).clamp_min(torch.finfo(theta_i.dtype).tiny)
            s = (pi_full - pi_i[..., :-1]) / delta
            scale = w_full * c_v * (pi_full / column.c_p)
            grad_theta[..., :-1] += scale * (1.0 - s)
            grad_theta[..., 1:] += scale * s
        else:
            if mode in ("linear_pfull_adjoint", "direct_u_linear_adjoint"):
                pi_full = column.exner_full_pmid(ps)
            else:
                pi_full = column.exner_full(ps)
            scale = 0.5 * w_full * c_v * (pi_full / column.c_p)
            grad_theta[..., :-1] += scale
            grad_theta[..., 1:] += scale

        return grad_theta / dm.clamp_min(torch.finfo(dm.dtype).tiny)

    def _dry_energy_gradient_ps_local_terms(
        self,
        state: State,
        theta_i: torch.Tensor,
        *,
        dp: torch.Tensor | None = None,
        dm: torch.Tensor | None = None,
        grad_mtheta: torch.Tensor | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Local closed-form ``dE/dps`` pieces at fixed prognostic ``mtheta``.

        The dry energy depends on ``ps`` directly through layer thicknesses and
        Exner-pressure placement, and indirectly because
        ``theta = mtheta / dm(dp(ps))``.  The indirect part is

            -sum_i (dE/dmtheta_i) * theta_i * d(dm_i)/dps.

        The column-summed terms mirror ``total_kinetic_energy``,
        ``total_potential_energy``, and ``total_internal_energy``:

        - ``kinetic_mass``: pressure-thickness weighting of existing KE;
        - ``potential``: explicit ``ps`` dependence of layer mass and phi;
        - ``internal``: explicit ``ps`` dependence of layer mass and T;
        - ``theta_mass_coupling``: induced theta change at fixed mtheta.
        """

        ps = state.ps
        column = state.column
        dp = state.layer_dp() if dp is None else dp
        dm = interface_mass(dp) if dm is None else dm
        grad_mtheta = (
            self._dry_energy_gradient_mtheta_for_coord_work(state, theta_i, dp=dp, dm=dm)
            if grad_mtheta is None
            else grad_mtheta
        )

        dtype = ps.dtype
        device = ps.device
        n_levels = int(column.n_levels)
        layer_shape = [1] * dp.ndim
        layer_shape[-1] = n_levels
        interface_shape = [1] * theta_i.ndim
        interface_shape[-1] = n_levels + 1

        B_i = column.B_interface.to(dtype=dtype, device=device).reshape(interface_shape)
        dB_layer = (
            (column.B_interface[1:] - column.B_interface[:-1])
            .to(dtype=dtype, device=device)
            .reshape(layer_shape)
        )
        dB_layer = dB_layer.expand_as(dp)
        dmdps = interface_mass(dB_layer)

        area_shape = [1] * dp.ndim
        area_shape[-2] = state.mesh.nCells
        area_shape[-1] = 1
        area_over_g = state.mesh.cell_area.reshape(area_shape).to(dtype=dtype, device=device) / float(self.gravity)

        p_i = column.interface_pressure(ps)
        pi_i = column.exner_interface(ps)
        dpi_i = column.kappa * pi_i / p_i.clamp_min(torch.finfo(dtype).tiny) * B_i
        delta_pi = pi_i[..., 1:] - pi_i[..., :-1]
        ddelta_pi = dpi_i[..., 1:] - dpi_i[..., :-1]

        theta_top = theta_i[..., :-1]
        theta_bot = theta_i[..., 1:]
        theta_avg = 0.5 * (theta_top + theta_bot)
        d_integrand = theta_avg * ddelta_pi
        d_suffix = torch.flip(torch.cumsum(torch.flip(d_integrand, dims=[-1]), dim=-1), dims=[-1])
        dphi_i = torch.cat([d_suffix, torch.zeros_like(d_suffix[..., :1])], dim=-1)

        p_full_mid = column.full_pressure(ps)
        pi_mid = column.exner_full_pmid(ps)
        B_mid = 0.5 * (B_i[..., :-1] + B_i[..., 1:])
        dpi_mid = column.kappa * pi_mid / p_full_mid.clamp_min(torch.finfo(dtype).tiny) * B_mid

        mode = column.phi_full_mode
        if mode == "interface_mean":
            dphi_full = 0.5 * (dphi_i[..., :-1] + dphi_i[..., 1:])
            dpi_full = 0.5 * (dpi_i[..., :-1] + dpi_i[..., 1:])
            theta_full = theta_avg
            dtheta_full = torch.zeros_like(theta_full)
        elif mode in ("linear_pfull_adjoint", "direct_u_linear_adjoint"):
            dphi_full = dphi_i[..., 1:] + theta_avg * (dpi_i[..., 1:] - dpi_mid)
            dpi_full = dpi_mid
            theta_full = theta_avg
            dtheta_full = torch.zeros_like(theta_full)
        elif mode == "direct_u_gl2_adjoint":
            a = pi_i[..., :-1]
            b = pi_i[..., 1:]
            u = pi_mid
            da = dpi_i[..., :-1]
            db = dpi_i[..., 1:]
            du = dpi_mid
            delta = (b - a).clamp_min(torch.finfo(dtype).tiny)
            ddelta = db - da
            s = (u - a) / delta
            ds = ((du - da) * delta - (u - a) * ddelta) / delta.square().clamp_min(
                torch.finfo(dtype).tiny
            )
            coef_top = 0.5 * delta * (1.0 - s).square()
            coef_bot = 0.5 * delta * (1.0 - s.square())
            dcoef_top = 0.5 * ddelta * (1.0 - s).square() - delta * (1.0 - s) * ds
            dcoef_bot = 0.5 * ddelta * (1.0 - s.square()) - delta * s * ds
            dpartial = dcoef_top * theta_top + dcoef_bot * theta_bot
            dphi_full = dphi_i[..., 1:] + dpartial
            dpi_full = dpi_mid
            theta_full = theta_top + s * (theta_bot - theta_top)
            dtheta_full = ds * (theta_bot - theta_top)
        else:
            raise RuntimeError(f"unsupported phi_full_mode={mode!r}")

        phi_full = column.geopotential_full(theta_i, ps)
        temperature = column.temperature_full(theta_i, ps)
        pi_temperature = column.exner_full(ps) if mode == "interface_mean" else pi_mid
        dtemperature = (dtheta_full * pi_temperature + theta_full * dpi_full) / column.c_p
        if mode == "direct_u_gl2_adjoint":
            dtemperature = (dtheta_full * pi_mid + theta_full * dpi_mid) / column.c_p

        c_v = column.c_p - column.R_d
        ke = state.ke_cell()
        kinetic_mass_layer = area_over_g * dB_layer * ke
        potential_layer = area_over_g * (dB_layer * phi_full + dp * dphi_full)
        internal_layer = area_over_g * (dB_layer * c_v * temperature + dp * c_v * dtemperature)
        theta_mass_coupling_interface = -(grad_mtheta * theta_i * dmdps)

        kinetic_mass = kinetic_mass_layer.sum(dim=-1, keepdim=True)
        potential = potential_layer.sum(dim=-1, keepdim=True)
        internal = internal_layer.sum(dim=-1, keepdim=True)
        theta_mass_coupling = theta_mass_coupling_interface.sum(dim=-1, keepdim=True)
        total = kinetic_mass + potential + internal + theta_mass_coupling
        return {
            "layer": {
                "kinetic_mass": kinetic_mass_layer,
                "potential": potential_layer,
                "internal": internal_layer,
            },
            "interface": {
                "theta_mass_coupling": theta_mass_coupling_interface,
            },
            "column": {
                "kinetic_mass": kinetic_mass,
                "potential": potential,
                "internal": internal,
                "theta_mass_coupling": theta_mass_coupling,
                "total": total,
            },
        }

    def _dry_energy_gradient_ps_terms(
        self,
        state: State,
        theta_i: torch.Tensor,
        *,
        dp: torch.Tensor | None = None,
        dm: torch.Tensor | None = None,
        grad_mtheta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Column-summed closed-form ``dE/dps`` pieces."""

        local = self._dry_energy_gradient_ps_local_terms(
            state,
            theta_i,
            dp=dp,
            dm=dm,
            grad_mtheta=grad_mtheta,
        )
        return {
            name: value
            for name, value in local["column"].items()
        }

    def _dry_energy_gradient_ps(
        self,
        state: State,
        theta_i: torch.Tensor,
        *,
        dp: torch.Tensor | None = None,
        dm: torch.Tensor | None = None,
        grad_mtheta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Closed-form ``dE/dps`` at fixed prognostic ``mtheta``."""

        terms = self._dry_energy_gradient_ps_terms(
            state,
            theta_i,
            dp=dp,
            dm=dm,
            grad_mtheta=grad_mtheta,
        )
        return terms["total"]

    def pressure_fields(self, state: State) -> dict[str, torch.Tensor]:
        """Return pressure fields used by the active column."""
        self.check_state_compatible(state)
        return {
            "ps": state.ps,
            "p_interface": state.interface_pressure(),
            "p_full": state.full_pressure(),
            "dp": state.layer_dp(),
        }

    def wind_fields(self, state: State) -> dict[str, torch.Tensor]:
        """Return wind fields and available reconstructions."""
        self.check_state_compatible(state)
        out = {"u_e": state.u_e}
        out["cell_wind_vector"] = state.cell_wind_vector()
        return out

    def energy_terms(self, state: State) -> dict[str, torch.Tensor]:
        """Return integrated dry energy components."""
        kinetic = self.total_kinetic_energy(state)
        potential = self.total_potential_energy(state)
        internal = self.total_internal_energy(state)
        return {
            "kinetic": kinetic,
            "potential": potential,
            "internal": internal,
            "total": kinetic + potential + internal,
        }

    def edge_metric(
        self,
        state: State,
        *,
        primitives: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Energy/force edge metric `dc_edge * dv_edge * cell_to_edge(dp)`."""
        self.check_state_compatible(state)
        if primitives is not None:
            dp_e = primitives.get("dp_e")
            if dp_e is None:
                dp_e = state.cell_to_edge(primitives["dp"])
        else:
            dp_e = state.cell_to_edge(state.layer_dp())
        return state.mesh.dc_edge[:, None] * state.mesh.dv_edge[:, None] * dp_e

    def force_terms(
        self,
        state: State,
        terms: dict[str, dict[str, torch.Tensor]] | None = None,
        *,
        primitives: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return matched momentum-force tensors and aliases used by reports."""
        if terms is None:
            terms = self.tendency_terms(state, primitives=primitives)
        u_terms = terms["u_e"]
        total = u_terms["total"]
        ctx = self._transport_primitives(state) if primitives is None else primitives
        metric = self.edge_metric(state, primitives=ctx)
        diffusion = u_terms.get("momentum_diffusion", u_terms.get("hyperdiffusion", torch.zeros_like(total)))
        out = {
            "rotation": u_terms["rotation"],
            "pgf": u_terms["pgf"],
            "hydro": u_terms["pgf"],
            "bernoulli": u_terms["bernoulli"],
            "minus_grad_ke": u_terms["bernoulli"],
            "vertical": u_terms["vertical"],
            "uvert": u_terms["vertical"],
            "hyperdiffusion": u_terms.get("hyperdiffusion", torch.zeros_like(total)),
            "divergence_damping": u_terms.get("divergence_damping", torch.zeros_like(total)),
            "momentum_diffusion": diffusion,
            "diffusion": diffusion,
            "total": total,
            "metric": metric,
            "dp": ctx["dp"],
            "dps_dt": ctx["dps_dt"],
            "layer_div": ctx["layer_div"],
            "mu": ctx["mu"],
            "omega": ctx["omega"],
        }
        if "top_sponge" in u_terms:
            out["top_sponge"] = u_terms["top_sponge"]
        return out

    def force_balance_terms(
        self,
        state: State,
        terms: dict[str, dict[str, torch.Tensor]] | None = None,
        *,
        primitives: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return weighted force-balance norms and cancellation ratio."""
        force = self.force_terms(state, terms=terms, primitives=primitives)
        metric = force["metric"].clamp_min(torch.finfo(state.u_e.dtype).tiny)

        def norm(name: str) -> torch.Tensor:
            return (force[name].square() * metric).sum().sqrt()

        rotation_norm = norm("rotation")
        pgf_norm = norm("pgf")
        bernoulli_norm = norm("bernoulli")
        vertical_norm = norm("vertical")
        total_norm = norm("total")
        denom = (rotation_norm + pgf_norm + bernoulli_norm + vertical_norm).clamp_min(torch.finfo(total_norm.dtype).tiny)
        return {
            "rotation_norm": rotation_norm,
            "pgf_norm": pgf_norm,
            "bernoulli_norm": bernoulli_norm,
            "vertical_norm": vertical_norm,
            "total_norm": total_norm,
            "cancellation_ratio": total_norm / denom,
        }

    def cfl_terms(
        self,
        state: State,
        dt: float,
        *,
        primitives: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return CFL fields and maxima for the current state."""
        ctx = self._transport_primitives(state) if primitives is None else primitives
        dp = ctx["dp"]
        dm = ctx["dm"]
        omega = ctx["omega"]
        mu = ctx["mu"]
        vertical = float(dt) * omega.abs() / dp.clamp_min(1.0)
        vertical_interface = float(dt) * mu.abs() / dm.clamp_min(torch.finfo(dm.dtype).tiny)
        horizontal = float(dt) * state.u_e.abs() / state.mesh.dc_edge[:, None].clamp_min(1.0)
        return {
            "horizontal": horizontal,
            "vertical": vertical,
            "vertical_interface": vertical_interface,
            "horizontal_max": horizontal.max(),
            "vertical_max": vertical.max(),
            "vertical_interface_max": vertical_interface.max(),
        }

    def divergence(
        self,
        state: State,
        *,
        primitives: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Mass-flux divergence per full level."""
        self.check_state_compatible(state)
        if primitives is not None:
            return primitives["layer_div"]
        return self._layer_divergence(state, state.u_e, state.layer_dp())

    def vorticity(self, state: State) -> torch.Tensor:
        """Vertex relative vorticity per full level."""
        self.check_state_compatible(state)
        return state.curl(state.u_e)

    def omega(
        self,
        state: State,
        terms: dict[str, dict[str, torch.Tensor]] | None = None,
        *,
        primitives: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Full-level vertical mass flux.

        The optional ``terms`` argument is accepted for compatibility but does
        not carry hidden metadata; omega is recomputed explicitly from state.
        """
        ctx = self._transport_primitives(state) if primitives is None else primitives
        return ctx["omega"]

    def thermo_fields(self, state: State) -> dict[str, torch.Tensor]:
        self.check_state_compatible(state)
        theta = state.theta()
        return {
            "theta": theta,
            "theta_full": 0.5 * (theta[..., :-1] + theta[..., 1:]),
            "temperature": state.temperature(),
            "geopotential_interface": state.column.geopotential_interface(
                theta, state.ps
            ),
            "geopotential": state.geopotential(),
            "exner_interface": state.exner_interface(),
            "exner_full": state.exner_full(),
        }
