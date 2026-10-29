"""Matched dry hydrostatic dynamics with diagonal edge kinetic energy.

The closed semidiscrete invariant is atmospheric energy plus the fixed-pressure
lid reservoir. Time integrators need not preserve it exactly. This opt-in scheme
uses centered scalar transport, without positivity or monotonicity guarantees.
See docs/conservative_moist_design.md for scope and the moisture migration plan.
"""
import torch
from ..core.registry import register_component
from ..core.state import State
from .constants import GRAVITY, OMEGA
from .dry_dynamics import DryHydrostaticDynamics


def v(x):
    """Unwrap a Field while retaining the computation graph."""
    return x.values if hasattr(x, "location") else x


@register_component("dry_hydrostatic_conservative")
class DryHydrostaticConservativeDynamics(DryHydrostaticDynamics):
    """Unfitted diagonal KE, matched pressure/rotation/momentum transport.

    Currently unbatched, dry, direct_u_gl2_adjoint columns with a fixed positive
    pressure top. No runtime autodiff or iterative kinetic-metric solve.
    Strategy substitution is intentionally unavailable: the operators form a pair.
    """
    recipe_name = "conservative_diagonal"

    def __init__(self, omega=OMEGA, gravity=GRAVITY, validate_state=True):
        super().__init__(omega=omega, gravity=gravity, validate_state=validate_state)

    def check_state_compatible(self, state):
        super().check_state_compatible(state)
        if state.tracer_m is not None:
            raise ValueError("conservative_diagonal currently supports dry states only")
        if v(state.u_e).ndim != 2:
            raise ValueError("conservative_diagonal requires an unbatched state")
        if state.column.phi_full_mode != "direct_u_gl2_adjoint":
            raise ValueError("conservative_diagonal requires direct_u_gl2_adjoint")
        if float(state.column.B_interface[0]) != 0 or float(state.column.A_interface[0]) <= 0:
            raise ValueError("conservative_diagonal requires a fixed positive pressure top")

    def _mtheta_tendency(
        self,
        state: State,
        u_e: torch.Tensor,
        theta_i: torch.Tensor,
        dp: torch.Tensor,
        *,
        dp_e: torch.Tensor | None = None,
        mass_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mesh = state.mesh
        if mass_flux is None:
            dp_e = mesh.cell_to_edge(dp) if dp_e is None else dp_e
            mass_flux = dp_e * u_e

        c0 = mesh.cellsOnEdge[:, 0]
        c1 = mesh.cellsOnEdge[:, 1]
        theta_upper_e = 0.5 * (theta_i[..., c0, :-1] + theta_i[..., c1, :-1])
        theta_lower_e = 0.5 * (theta_i[..., c0, 1:] + theta_i[..., c1, 1:])

        div_upper = mesh.div(0.5 * mass_flux * theta_upper_e)
        div_lower = mesh.div(0.5 * mass_flux * theta_lower_e)

        dmtheta_dt = torch.zeros_like(theta_i)
        dmtheta_dt[..., :-1] -= div_upper
        dmtheta_dt[..., 1:] -= div_lower
        return dmtheta_dt

    def boundary_reservoir(self, state):
        """Fixed-pressure lid reservoir in joules."""
        top_phi = v(state.column.geopotential_interface(state.theta(), state.ps))[..., 0]
        return state.column.A_interface[0] / self.gravity * (state.mesh.cell_area * top_phi).sum()

    def boundary_total_energy(self, state):
        """Atmospheric energy plus lid reservoir, in joules."""
        return self.total_energy(state) + self.boundary_reservoir(state)

    def metric_action(self, state, z, thickness):
        mesh = state.mesh
        return (mesh.dc_edge[:, None] * mesh.dv_edge[:, None]
                * v(mesh.cell_to_edge(thickness)) * z / self.gravity)

    def solve_kinetic_metric(self, state, covector):
        diagonal = self.metric_action(state, torch.ones_like(v(state.u_e)), v(state.layer_dp()))
        return covector / diagonal

    def cell_kinetic_energy(self, state):
        mesh, u = state.mesh, v(state.u_e)
        edge_area = mesh.dc_edge * mesh.dv_edge
        weights = mesh.edge_side_area / mesh.edge_side_area.sum(-1, keepdim=True)
        contribution = 0.5 * edge_area[:, None, None] * weights[:, :, None] * u[:, None, :].square()
        indices = mesh.cellsOnEdge.reshape(-1, 1).expand(-1, u.shape[-1])
        cell_total = torch.zeros(mesh.nCells, u.shape[-1], dtype=u.dtype, device=u.device).scatter_add(
            0, indices, contribution.reshape(-1, u.shape[-1]),
        )
        return cell_total / mesh.cell_area[:, None]

    def total_kinetic_energy(self, state):
        return (0.5 * v(state.u_e) * self.metric_action(state, v(state.u_e), v(state.layer_dp()))).sum()

    def kinetic_energy_gradient_u_terms(self, state):
        gradient = self.metric_action(state, v(state.u_e), v(state.layer_dp()))
        return {"normal": gradient, "tangential": torch.zeros_like(gradient), "total": gradient}

    def _dry_energy_gradient_ps_local_terms(self, state, theta_i, **kwargs):
        terms = super()._dry_energy_gradient_ps_local_terms(state, theta_i, **kwargs)
        delta_B = state.column.B_interface[1:] - state.column.B_interface[:-1]
        layer = state.mesh.cell_area[:, None] * delta_B * self.cell_kinetic_energy(state) / self.gravity
        change = layer.sum(-1, keepdim=True) - v(terms["column"]["kinetic_mass"])
        terms["layer"]["kinetic_mass"] = layer
        terms["column"]["kinetic_mass"] = layer.sum(-1, keepdim=True)
        terms["column"]["total"] = terms["column"]["total"] + change
        return terms

    def thermal_gradients(self, state):
        """Gradients of IE+PE+Btop with respect to mtheta_i and ps.

        The pressure derivative holds prognostic mtheta fixed, so theta's
        dependence on pressure-weighted interface mass is included.
        """
        column = state.column
        theta, dp = v(state.theta()), v(state.layer_dp())
        dm = self._interface_mass(dp)
        gm = v(self._dry_energy_gradient_mtheta_for_coord_work(state, state.theta(), dp=dp, dm=dm))
        parts = self._dry_energy_gradient_ps_terms(state, state.theta(), dp=dp, dm=dm, grad_mtheta=gm)
        gp = v(parts["potential"] + parts["internal"] + parts["theta_mass_coupling"])
        pi, pressure = v(column.exner_interface(state.ps)), v(column.interface_pressure(state.ps))
        factor = state.mesh.cell_area[:, None] * column.A_interface[0] / self.gravity
        delta = pi[:, 1:] - pi[:, :-1]
        lid_gradient = torch.cat([
            0.5 * delta[:, :1], 0.5 * (delta[:, :-1] + delta[:, 1:]), 0.5 * delta[:, -1:],
        ], dim=-1) * factor / dm
        dpi = column.kappa * pi / pressure * column.B_interface
        dmdp = self._interface_mass((column.B_interface[1:] - column.B_interface[:-1]).expand_as(dp))
        gp = gp + factor * (0.5 * (theta[:, :-1] + theta[:, 1:])
                            * (dpi[:, 1:] - dpi[:, :-1])).sum(-1, keepdim=True)
        gp = gp - (lid_gradient * theta * dmdp).sum(-1, keepdim=True)
        return gm + lid_gradient, gp

    def tendency_terms(self, state, *, primitives=None):
        return self._assemble(state, primitives=primitives, combine=False)

    def tendency(self, state):
        return self.terms_total(self._assemble(state, combine=True), like=state)

    def _assemble(self, state, *, primitives=None, combine=False):
        self.check_state_compatible(state)
        ctx = self._transport_primitives(state) if primitives is None else primitives
        mesh = state.mesh
        u, theta = v(state.u_e), v(state.theta())
        dp, dp_edge = v(ctx["dp"]), v(ctx["dp_e"])
        mu, omega = v(ctx["mu"]), v(ctx["omega"])
        horizontal = v(self._mtheta_tendency(
            state, state.u_e, ctx["theta"], ctx["dp"],
            dp_e=ctx["dp_e"], mass_flux=ctx["mass_flux"],
        ))
        gm, gp = self.thermal_gradients(state)
        edge_area = mesh.dc_edge[:, None] * mesh.dv_edge[:, None]

        def divergence_transpose(covector):
            # Euclidean transpose; cell areas belong inside the gradient.
            return -edge_area * v(mesh.grad(covector / mesh.cell_area[:, None]))

        # Transpose centered horizontal theta transport.
        c0, c1 = mesh.cellsOnEdge.T
        theta_edge = 0.5 * (theta[c0] + theta[c1])
        horizontal_adjoint = -0.5 * (
            theta_edge[:, :-1] * divergence_transpose(gm[:, :-1])
            + theta_edge[:, 1:] * divergence_transpose(gm[:, 1:])
        )
        # Transpose vertical theta flux, omega average, mu prefix sum,
        # and ps_dot=-sum(div). These are all Euclidean covectors.
        grad_omega = (gm[:, 1:] - gm[:, :-1]) * 0.5 * (theta[:, :-1] + theta[:, 1:])
        grad_mu = torch.cat([
            0.5 * grad_omega[:, :1],
            0.5 * (grad_omega[:, :-1] + grad_omega[:, 1:]),
            0.5 * grad_omega[:, -1:],
        ], dim=-1)
        grad_prefix = -torch.flip(torch.cumsum(torch.flip(grad_mu[:, 1:], [-1]), -1), [-1])
        delta_B = state.column.B_interface[1:] - state.column.B_interface[:-1]
        grad_div = grad_prefix - (gp + (grad_prefix * delta_B).sum(-1, keepdim=True))
        pressure_covector = -dp_edge * (horizontal_adjoint + divergence_transpose(grad_div))

        # Fixed-vorticity skew transpose in the actual diagonal KE metric.
        vorticity = v(mesh.curl(u)) + self.rotation_rate * mesh.coriolis_unit_vertex[:, None]
        vertices = mesh.verticesOnEdge
        q_edge = 0.5 * (vorticity[vertices[:, 0]] + vorticity[vertices[:, 1]])
        indices = mesh.safe_rotation_stencil
        coefficients = -0.5 * (q_edge[:, None, :] + q_edge[indices]) * mesh.rotation_weights_masked[:, :, None]
        metric_diagonal = edge_area * dp_edge / self.gravity
        transpose_rotation = torch.zeros_like(u).scatter_add(
            0, indices.reshape(-1, 1).expand(-1, u.shape[1]),
            (coefficients * (metric_diagonal * u)[:, None, :]).reshape(-1, u.shape[1]),
        )
        rotation_covector = 0.5 * (
            metric_diagonal * (coefficients * u[indices]).sum(1) - transpose_rotation
        )

        # Centered vertical exchange and the time-dependent metric correction.
        mu_edge = v(state.cell_to_edge(mu))

        def metric(z, thickness=dp):
            return self.metric_action(state, z, thickness)

        def vertical(z):
            zi = torch.cat([z[:, :1], 0.5 * (z[:, :-1] + z[:, 1:]), z[:, -1:]], dim=-1)
            return (mu_edge[:, :-1] * (zi[:, :-1] - z)
                    - mu_edge[:, 1:] * (zi[:, 1:] - z)) / dp_edge

        weighted_u = metric(u) / dp_edge
        exchange = 0.5 * mu_edge[:, 1:-1] * (weighted_u[:, :-1] + weighted_u[:, 1:])
        zero_end = torch.zeros_like(weighted_u[:, :1])
        vertical_transpose = (torch.cat([exchange, zero_end], dim=-1)
                              - torch.cat([zero_end, exchange], dim=-1))
        vertical_covector = 0.5 * (
            metric(vertical(u)) - vertical_transpose - metric(u, mu[:, :-1] - mu[:, 1:])
        )
        bernoulli_covector = metric_diagonal * v(-state.grad(self.cell_kinetic_energy(state)))
        momentum = {"momentum_diffusion": torch.zeros_like(u), "diffusion": torch.zeros_like(u)}
        if combine:
            momentum["total"] = self.solve_kinetic_metric(
                state, pressure_covector + rotation_covector + vertical_covector + bernoulli_covector,
            )
        else:
            for name, covector in [
                ("pgf", pressure_covector), ("rotation", rotation_covector),
                ("vertical", vertical_covector), ("bernoulli", bernoulli_covector),
            ]:
                momentum[name] = self.solve_kinetic_metric(state, covector)
            momentum["total"] = sum(momentum[name] for name in ["pgf", "rotation", "vertical", "bernoulli"])

        flux = omega * 0.5 * (theta[:, :-1] + theta[:, 1:])
        vertical_scalar = torch.cat([-flux[:, :1], flux[:, :-1] - flux[:, 1:], flux[:, -1:]], dim=-1)
        scalar = {"horizontal": horizontal, "vertical": vertical_scalar,
                  "coord_work": torch.zeros_like(theta),
                  "advective": horizontal + vertical_scalar, "total": horizontal + vertical_scalar}
        return {"u_e": momentum, "mtheta_i": scalar,
                "ps": self.ps_tendency_terms(state, primitives=ctx)}

    def momentum_tendency_terms(self, state, *, primitives=None):
        return self.tendency_terms(state, primitives=primitives)["u_e"]

    def mtheta_tendency_terms(self, state, *, primitives=None):
        return self.tendency_terms(state, primitives=primitives)["mtheta_i"]

    def rotation_term(self, state):
        return self.momentum_tendency_terms(state)["rotation"]

    def pgf_term(self, state):
        return self.momentum_tendency_terms(state)["pgf"]

    def bernoulli_term(self, state):
        self.check_state_compatible(state)
        return -state.grad(self.cell_kinetic_energy(state))

    def rotation_work_terms(self, state):
        rotation = self.rotation_term(state)
        power = self.kinetic_energy_gradient_u(state) * rotation
        scale = power.abs().sum().clamp_min(torch.finfo(power.dtype).tiny)
        return {"rotation": rotation, "rotation_rms": rotation.square().mean().sqrt(),
                "work": power.sum(), "work_abs_scale": scale,
                "work_relative": power.sum().abs() / scale}

    def boundary_energy_rate_closed(self, state, tendency):
        """Explicit directional derivative of E+Btop, in watts.

        Atmospheric energy_rate_closed retains its original E-only meaning.
        This method includes lid pressure work, and uses no runtime autodiff.
        """
        gm, gp = self.thermal_gradients(state)
        db = state.column.B_interface[1:] - state.column.B_interface[:-1]
        gp = gp + (state.mesh.cell_area[:, None] * db *
                   self.cell_kinetic_energy(state) / self.gravity).sum(-1, keepdim=True)
        return ((self.kinetic_energy_gradient_u(state) * v(tendency.u_e)).sum()
                + (gm * v(tendency.mtheta_i)).sum() + (gp * v(tendency.ps)).sum())
