"""Matched dry hydrostatic dynamics with diagonal edge kinetic energy.

The closed semidiscrete invariant is atmospheric energy plus the fixed-pressure
lid reservoir. Time integrators need not preserve it exactly. This retained scheme
uses centered scalar transport, without positivity or monotonicity guarantees.
See docs/CURRENT_PATHS.md for explicit and ARK4 entry paths.
"""
import torch
from ..core.state import interface_mass
from ..core.state import State
from .constants import GRAVITY, OMEGA
from .dry_energy import DryEnergy
from .scalar_transport import ScalarTransportState


def v(x):
    """Unwrap a Field while retaining the computation graph."""
    return x.values if hasattr(x, "location") else x


class DryHydrostaticConservativeDynamics(DryEnergy, torch.nn.Module):
    """Unfitted diagonal KE, matched pressure/rotation/momentum transport.

    Currently unbatched, dry, direct_u_gl2_adjoint columns with a fixed positive
    pressure top. No runtime autodiff or iterative kinetic-metric solve.
    Strategy substitution is intentionally unavailable: the operators form a pair.
    """
    recipe_name = "conservative_diagonal"

    def __init__(self, omega=OMEGA, gravity=GRAVITY, validate_state=True):
        super().__init__()
        self.rotation_rate = float(omega)
        self.gravity = float(gravity)
        self.validate_state = bool(validate_state)

    def check_state_compatible(self, state):
        if state.mesh is None or state.column is None:
            raise ValueError("conservative dynamics requires State mesh and column")
        if state.tracer_m is not None:
            raise ValueError("conservative_diagonal currently supports dry states only")
        if v(state.u_e).ndim != 2:
            raise ValueError("conservative_diagonal requires an unbatched state")
        if state.column.phi_full_mode != "direct_u_gl2_adjoint":
            raise ValueError("conservative_diagonal requires direct_u_gl2_adjoint")
        if float(state.column.B_interface[0]) != 0 or float(state.column.A_interface[0]) <= 0:
            raise ValueError("conservative_diagonal requires a fixed positive pressure top")

    def centered_scalar_transport(
        self,
        state: State,
        u_e: torch.Tensor,
        specific_scalar: torch.Tensor,
        dp: torch.Tensor,
        *,
        dp_e: torch.Tensor | None = None,
        mass_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Centered horizontal CP content flux divergence.

        specific_scalar is K (dry theta) or J/(kg K) (moist entropy), on
        cell/interfaces. mass_flux is edge/full Pa m/s, positive along the
        mesh normal. Half of each layer flux feeds each bounding interface;
        minus divergence returns content per second, with half end weights.
        """
        mesh = state.mesh
        if mass_flux is None:
            dp_e = mesh.cell_to_edge(dp) if dp_e is None else dp_e
            mass_flux = dp_e * u_e

        c0 = mesh.cellsOnEdge[:, 0]
        c1 = mesh.cellsOnEdge[:, 1]
        scalar_upper_e = 0.5 * (specific_scalar[..., c0, :-1] + specific_scalar[..., c1, :-1])
        scalar_lower_e = 0.5 * (specific_scalar[..., c0, 1:] + specific_scalar[..., c1, 1:])

        div_upper = mesh.div(0.5 * mass_flux * scalar_upper_e)
        div_lower = mesh.div(0.5 * mass_flux * scalar_lower_e)

        content_tendency = torch.zeros_like(specific_scalar)
        content_tendency[..., :-1] -= div_upper
        content_tendency[..., 1:] -= div_lower
        return content_tendency

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
        dm = interface_mass(dp)
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
        dmdp = interface_mass((column.B_interface[1:] - column.B_interface[:-1]).expand_as(dp))
        gp = gp + factor * (0.5 * (theta[:, :-1] + theta[:, 1:])
                            * (dpi[:, 1:] - dpi[:, :-1])).sum(-1, keepdim=True)
        gp = gp - (lid_gradient * theta * dmdp).sum(-1, keepdim=True)
        return gm + lid_gradient, gp

    def tendency_terms(self, state, *, primitives=None):
        return self._assemble(state, primitives=primitives, combine=False)

    def tendency(self, state):
        return self.terms_total(self._assemble(state, combine=True), like=state)

    def _assemble(self, state, *, primitives=None, combine=False):
        """Assemble the matched mass, heat-content and momentum exchanges.

        Inputs: u_e (edge,full) m/s; ps (cell,1) Pa; mtheta_i
        (cell,interface) Pa K. Outputs use those locations and units/second.
        gm and gp are integrated thermal-energy covectors, J/(Pa K) and
        J/Pa. Transport transposes turn their work into edge momentum work;
        division by the kinetic metric gives acceleration in m/s².
        The sphere has no lateral boundary; vertical mass flux vanishes at
        the fixed lid and surface. The lid energy reservoir closes the budget.
        """
        self.check_state_compatible(state)
        ctx = self._transport_primitives(state) if primitives is None else primitives
        mesh = state.mesh
        u, theta = v(state.u_e), v(state.theta())
        dp, dp_edge = v(ctx["dp"]), v(ctx["dp_e"])
        mu, omega = v(ctx["mu"]), v(ctx["omega"])
        horizontal = v(self.centered_scalar_transport(
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


    def _validate_mass_geometry(
        self,
        state: State,
        dp: torch.Tensor,
        dm: torch.Tensor | None = None,
    ) -> None:
        """Raise when the hybrid column geometry becomes nonphysical."""
        if not self.validate_state:
            return
        ps = state.ps
        p_top = float(state.column.A_interface[0].item())

        if not torch.isfinite(ps).all().item():
            raise FloatingPointError("surface pressure is not finite")
        if ps.min().item() <= p_top:
            raise FloatingPointError(
                f"surface pressure fell below model top pressure: "
                f"ps_min={ps.min().item():.3e}, p_top={p_top:.3e}"
            )

        if not torch.isfinite(dp).all().item():
            raise FloatingPointError("layer thickness is not finite")
        if dp.min().item() <= 0.0:
            raise FloatingPointError(f"nonpositive layer thickness: dp_min={dp.min().item():.3e}")

        if dm is not None:
            if not torch.isfinite(dm).all().item():
                raise FloatingPointError("interface mass is not finite")
            if dm.min().item() <= 0.0:
                raise FloatingPointError(f"nonpositive interface mass: dm_min={dm.min().item():.3e}")

    def forward(self, state: State) -> State:
        """Alias for tendency() — makes dynamics usable as an nn.Module."""
        return self.tendency(state)

    def terms_total(self, terms: dict[str, dict[str, torch.Tensor]], *, like: State | None = None) -> State:
        """Collapse `tendency_terms` output to a `State`-shaped RHS.

        Pass ``like=state`` to preserve mesh, column, and attrs on the returned
        RHS. Omitting ``like`` keeps the historical contextless return value.
        """
        tracer_total = None
        if "tracer_m" in terms:
            tracer_total = terms["tracer_m"]["total"]
        mesh = None if like is None else like.mesh
        column = None if like is None else like.column
        attrs = {} if like is None else dict(like.attrs or {})
        return State(
            u_e=terms["u_e"]["total"],
            mtheta_i=terms["mtheta_i"]["total"],
            ps=terms["ps"]["total"],
            tracer_m=tracer_total,
            mesh=mesh,
            column=column,
            attrs=attrs,
        )


    def _layer_divergence(
        self,
        state: State,
        u_e: torch.Tensor,
        dp:  torch.Tensor,
        *,
        dp_e: torch.Tensor | None = None,
        mass_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Horizontal mass-flux divergence in each full level.

            div_k = div( u_e_k * cell_to_edge(dp_k) )

        This quantity is needed both for ∂ps/∂t and for the vertical
        mass flux recurrence, so it is computed once and shared.

        Input:  u_e [..., nEdges, N]
                dp  [..., nCells, N]
        Output:     [..., nCells, N]
        """
        if mass_flux is None:
            dp_e = state.mesh.cell_to_edge(dp) if dp_e is None else dp_e
            mass_flux = u_e * dp_e
        return state.mesh.div(mass_flux)               # [..., nCells, N]

    def _ps_tendency(self, layer_div: torch.Tensor) -> torch.Tensor:
        """
        Surface pressure tendency: vertical integral of layer divergence.

        ∂ps/∂t = -Σ_k div_k

        Input:  layer_div [..., nCells, N]
        Output:           [..., nCells, 1]
        """
        return -layer_div.sum(dim=-1, keepdim=True)

    def _vertical_mass_flux(
        self,
        state: State,
        layer_div: torch.Tensor,
        dps_dt:    torch.Tensor,
    ) -> torch.Tensor:
        """
        Downward mass flux at each CP interface (Pa/s).

        Derived from the layer continuity equation by a downward prefix
        cumulative sum starting from the top boundary condition μ_0 = 0:

            μ_k = -Σ_{j=0}^{k-1} [ δB_j * ∂ps/∂t + div_j ]

        where δB_j = B_{j+1} - B_j accounts for the time tendency of
        the hybrid coordinate.

        Boundary conditions:
            μ_0 = 0  (no mass through model top)
            μ_N ≈ 0  (self-consistent if ps obeys its budget)

        Positive μ means downward (toward higher pressure).

        Input:  layer_div [..., nCells, N]   from _layer_divergence()
                dps_dt    [..., nCells, 1]   from _ps_tendency()
        Output: mu        [..., nCells, N+1]
        """
        # residual_k = δB_k * dps_dt + div_k
        # dB: [N], dps_dt: [..., nCells, 1] → broadcasts to [..., nCells, N]
        dB = state.column.B_interface[1:] - state.column.B_interface[:-1]
        dB = dB.to(dtype=layer_div.dtype, device=layer_div.device)
        residual = dB * dps_dt + layer_div           # [..., nCells, N]

        # Exclusive prefix sum: μ_k = -Σ_{j<k} residual_j
        # cumsum[k] = Σ_{j=0}^{k} residual_j
        # μ = cat([0, -cumsum[:-1]], which shifts right by one
        neg_cum = -torch.cumsum(residual, dim=-1)    # [..., nCells, N]
        zero    = torch.zeros_like(neg_cum[..., :1]) # [..., nCells, 1]
        return torch.cat([zero, neg_cum], dim=-1)    # [..., nCells, N+1]

    def _omega_full(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Vertical mass flux at full levels: average of bounding interfaces.

            ω_k = ½ (μ_k + μ_{k+1})

        The same averaging as the hydrostatic layer-mean θ̄_k — required
        for energy consistency of the vertical theta flux.

        Input:  mu    [..., nCells, N+1]
        Output: omega [..., nCells, N]
        """
        return 0.5 * (mu[..., :-1] + mu[..., 1:])

    def ps_tendency_terms(
        self,
        state: State,
        *,
        primitives: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return the surface-pressure continuity terms owned by transport."""

        ctx = self._transport_primitives(state) if primitives is None else primitives
        return {
            "mass_flux": ctx["dps_dt"],
            "total": ctx["dps_dt"],
        }

    def _scalar_transport_primitives(self, state) -> dict[str, torch.Tensor]:
        """Shared mass/transport primitives used by dynamics and diagnostics."""

        dp = state.layer_dp()
        dm = interface_mass(dp)
        self._validate_mass_geometry(state, dp, dm)
        specific_scalar = state.scalar_content / dm
        dp_e = state.mesh.cell_to_edge(dp)
        mass_flux = state.u_e * dp_e
        layer_div = self._layer_divergence(state, state.u_e, dp, dp_e=dp_e, mass_flux=mass_flux)
        dps_dt = self._ps_tendency(layer_div)
        mu = self._vertical_mass_flux(state, layer_div, dps_dt)
        omega = self._omega_full(mu)
        return {
            "dp": dp,
            "dp_e": dp_e,
            "dm": dm,
            "specific_scalar": specific_scalar,
            "mass_flux": mass_flux,
            "layer_div": layer_div,
            "dps_dt": dps_dt,
            "mu": mu,
            "omega": omega,
        }

    def _transport_primitives(self, state):
        self.check_state_compatible(state)
        view = ScalarTransportState(state.u_e, state.ps, state.mtheta_i, state.mesh, state.column)
        # Dry inspection keeps its physical theta label; shared transport is neutral.
        fields = self._scalar_transport_primitives(view)
        fields["theta"] = fields.pop("specific_scalar")
        return fields
