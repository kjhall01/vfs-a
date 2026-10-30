"""Conservative mixture RHS and energy derivatives for retained moist stepping.

Public state uses entropy/species contents. The shared transport input uses
ScalarTransportState.scalar_content as an explicitly named entropy buffer.
Bounded explicit and ARK4 steppers share these centered candidate equations.

Read tendency -> transport_state -> _assemble; the latter calls mixture_gradients
before pairing scalar transport with momentum work. Energy diagnostics use
the same two-point pressure quadrature. C/E/L denote cells/edges/layers,
I=L+1 denotes CP interfaces; vertical indices increase from top to bottom.
"""

from dataclasses import dataclass
import torch
from ..core.state import interface_mass
from .scalar_transport import ScalarTransportState
from .conservative import DryHydrostaticConservativeDynamics, v
from .moist_cp import interface_pressure_mass, reconstruct_moist_cp
from .mixture_thermodynamics import MixtureThermodynamics


@dataclass
class MoistState:
    """Tensor contents on fixed geometry, with no implicit conversion.

    u_e (E,L): edge-normal wind, m/s. ps (C,1): surface pressure, Pa.
    entropy_content (C,I): interface pressure thickness times specific entropy,
    Pa J/(kg K). water_content (C,I,3): thickness times vapor/liquid/ice mass
    fractions, Pa. Divide area-integrated contents by gravity for inventories.
    mesh/column are shared geometry references, not copied by combine.
    """

    u_e: torch.Tensor
    ps: torch.Tensor
    entropy_content: torch.Tensor
    water_content: torch.Tensor  # (cells, interfaces, vapor/liquid/ice)
    mesh: object
    column: object

    def combine(self, other, a=1.0, b=1.0):
        """Allocate weighted tensor contents; preserve geometry and autograd."""
        if self.mesh is not other.mesh or self.column is not other.column:
            raise ValueError("state geometries must agree")
        return MoistState(
            *(
                a * getattr(self, k) + b * getattr(other, k)
                for k in ("u_e", "ps", "entropy_content", "water_content")
            ),
            self.mesh,
            self.column,
        )


class _MoistKernel(DryHydrostaticConservativeDynamics):
    """Reuse geometric transport/kinetic operators with a explicit entropy transport input.

    The shared centered scalar-flux formula accepts entropy explicitly.
    Mixture thermodynamics below replaces dry thermodynamic interpretation.
    """

    def __init__(self, thermo, **kwargs):
        super().__init__(**kwargs)
        self.mixture = thermo

    def check_state_compatible(self, state):
        if state.mesh is None or state.column is None:
            raise ValueError("transport input requires mesh and column")
        if state.u_e.ndim != 2:
            raise ValueError("moist transport requires an unbatched state")
        if state.column.phi_full_mode != "direct_u_gl2_adjoint":
            raise ValueError("moist transport requires direct_u_gl2_adjoint")
        if float(state.column.B_interface[0]) != 0 or float(state.column.A_interface[0]) <= 0:
            raise ValueError("moist transport requires a fixed positive pressure top")

    def _transport_primitives(self, state):
        return self._scalar_transport_primitives(state)

    def mixture_gradients(self, state, water):
        """Analytic derivatives of internal + potential + top-reservoir energy.

        state is the borrowed scalar transport view; water (C,I,3) is pressure-weighted water,
        Pa. p/dp/dm are interface/layer/interface-weight pressures (Pa).
        S is entropy content, Pa J/(kg K). Reconstruction gives T/pg (C,L,2),
        K/Pa, and q (C,L,2,4), dry/vapor/liquid/ice fractions at Gauss points.

        Return entropy_gradient (C,I), J / [Pa J/(kg K)]; pressure_gradient
        (C,1), J/Pa; water_gradient (C,I,3), J/Pa. These are derivatives of
        global energy with respect to stored contents, not specific potentials
        or spatial gradients. Kinetic energy is added by energy_gradients.
        """
        # 1. Reconstruct thermodynamics from CP contents at two points per layer.
        p = v(state.column.interface_pressure(state.ps))
        dp = p.diff()
        dm = interface_pressure_mass(dp)
        S = v(state.scalar_content)
        quadrature = reconstruct_moist_cp(p, S, water, self.mixture)
        T, q, pg = (
            quadrature["temperature"],
            quadrature["fractions"],
            quadrature["pressure"],
        )
        thermo = self.mixture
        # 2. Species enthalpies/chemical potentials: J/kg; entropies: J/(kg K).
        # cp/R/href/pref are species constants in J/(kg K), J/(kg K), J/kg, Pa.
        # safe only selects logarithm arguments; it never clips stored water.
        cp = T.new_tensor([thermo.cp_d, thermo.cp_v, thermo.cp_l, thermo.cp_i])
        R = T.new_tensor([thermo.R_d, thermo.R_v, 0, 0])
        href = T.new_tensor([0, thermo.L_v_ref, 0, -thermo.L_f_ref])
        pref = T.new_tensor([thermo.p_d_ref, thermo.p_v_ref, 1, 1])
        partial = pg[..., None] * q * R / thermo.gas_constant(q)[..., None]
        safe = torch.where(partial > 0, partial, pref)
        partial_enthalpy = (T[..., None] - thermo.T_ref) * cp + href
        partial_entropy = (
            cp * torch.log(T[..., None] / thermo.T_ref)
            + href / thermo.T_ref
            - R * torch.log(safe / pref)
        )
        chemical_potential = partial_enthalpy - T[..., None] * partial_entropy
        water_chemical_difference = (
            chemical_potential[..., 1:] - chemical_potential[..., :1]
        )
        x = T.new_tensor([0.5 - 0.5 / 3**0.5, 0.5 + 0.5 / 3**0.5])
        weight = state.mesh.cell_area[:, None] / self.gravity

        def adj(z):
            """Transpose linear CP interpolation and pressure quadrature.

            z (C,L,2) -> (C,I), with added units kg from dp*area/gravity.
            x is the dimensionless Gauss coordinate; each point has weight 1/2.
            """
            upper = (dp[..., None] * 0.5 * z * (1 - x)).sum(-1)
            lower = (dp[..., None] * 0.5 * z * x).sum(-1)
            return (
                torch.cat([upper, torch.zeros_like(upper[:, :1])], -1)
                + torch.cat([torch.zeros_like(lower[:, :1]), lower], -1)
            ) * weight

        # 3. Pull back T and water-minus-dry chemical potentials to CP contents.
        # Dividing by dm accounts for content -> specific entropy/fraction.
        entropy_gradient = adj(T) / dm
        water_gradient = torch.stack(
            [adj(water_chemical_difference[..., i]) / dm for i in range(3)], -1
        )
        db = state.column.B_interface.diff()
        dm_dps = interface_pressure_mass(db)
        B_gauss = state.column.B_interface[:-1, None] + db[:, None] * x
        # 4. Pressure derivative: layer thickness and Gauss-pressure motion,
        # then subtract normalization changes at fixed entropy/water contents.
        # db, dm_dps and B_gauss are dimensionless derivatives with respect to ps.
        pressure_gradient = weight * (
            db * thermo.enthalpy(T, q).mean(-1)
            + dp * (quadrature["alpha"] * B_gauss).mean(-1)
        ).sum(-1, keepdim=True)
        pressure_gradient -= (entropy_gradient * (S / dm) * dm_dps).sum(
            -1, keepdim=True
        )
        pressure_gradient -= (
            water_gradient * (water / dm[..., None]) * dm_dps[None, :, None]
        ).sum((-1, -2))[:, None]
        surface_geopotential = torch.as_tensor(
            state.column.phi_s, dtype=p.dtype, device=p.device
        ).reshape(-1, 1)
        pressure_gradient += weight * surface_geopotential
        return entropy_gradient, pressure_gradient, water_gradient

    def _assemble(self, state, water, *, primitives=None, combine=False):
        """Assemble centered transport and matched momentum terms in that order.

        Private state: wind (E,L), m/s; entropy buffer (C,I), Pa J/(kg K);
        ps (C,1), Pa; separate water (C,I,3), Pa. Return term dictionaries:
        u_e in m/s², entropy_content in Pa J/(kg K s), water_content
        in Pa/s and ps in Pa/s. Covectors below are edge-layer forces (N);
        solve_kinetic_metric converts them to acceleration using the kg metric.
        No forward tendency is generated by autograd.
        """
        # 1. Continuity and scalar transport reuse geometry-only dry machinery.
        self.check_state_compatible(state)
        transport = (
            self._transport_primitives(state) if primitives is None else primitives
        )
        mesh = state.mesh
        # Both entropy access paths divide by CP mass. Keep their
        # existing evaluations: do not replace one with the other or cache them.
        u, specific_entropy = v(state.u_e), v(state.specific_scalar())
        dp, dp_edge = v(transport["dp"]), v(transport["dp_e"])
        mu, omega = v(transport["mu"]), v(transport["omega"])
        # mu (C,I) and its interface-to-layer average omega (C,L): Pa/s.
        horizontal_entropy = v(
            self.centered_scalar_transport(
                state,
                state.u_e,
                transport["specific_scalar"],
                transport["dp"],
                dp_e=transport["dp_e"],
                mass_flux=transport["mass_flux"],
            )
        )
        entropy_gradient, pressure_gradient, water_gradient = self.mixture_gradients(
            state, water
        )
        edge_area = mesh.dc_edge[:, None] * mesh.dv_edge[:, None]

        def divergence_transpose(covector):
            # Euclidean transpose; cell areas belong inside the gradient.
            return -edge_area * v(mesh.grad(covector / mesh.cell_area[:, None]))

        # 2. Transpose centered horizontal entropy transport against dE/dS.
        c0, c1 = mesh.cellsOnEdge.T
        entropy_edge = 0.5 * (specific_entropy[c0] + specific_entropy[c1])
        horizontal_adjoint = -0.5 * (
            entropy_edge[:, :-1] * divergence_transpose(entropy_gradient[:, :-1])
            + entropy_edge[:, 1:] * divergence_transpose(entropy_gradient[:, 1:])
        )
        # Transpose vertical entropy flux, omega average, mu prefix sum,
        # and ps_dot=-sum(div). These are all Euclidean covectors.
        grad_omega = (
            (entropy_gradient[:, 1:] - entropy_gradient[:, :-1])
            * 0.5
            * (specific_entropy[:, :-1] + specific_entropy[:, 1:])
        )
        grad_mu = torch.cat(
            [
                0.5 * grad_omega[:, :1],
                0.5 * (grad_omega[:, :-1] + grad_omega[:, 1:]),
                0.5 * grad_omega[:, -1:],
            ],
            dim=-1,
        )
        grad_prefix = -torch.flip(
            torch.cumsum(torch.flip(grad_mu[:, 1:], [-1]), -1), [-1]
        )
        delta_B = state.column.B_interface[1:] - state.column.B_interface[:-1]
        grad_div = grad_prefix - (
            pressure_gradient + (grad_prefix * delta_B).sum(-1, keepdim=True)
        )
        pressure_covector = -dp_edge * (
            horizontal_adjoint + divergence_transpose(grad_div)
        )

        # 3. Each water fraction q (C,I), kg/kg, uses the same CP transport.
        # Its transpose adds to pressure force; its divergence gives Pa/s.
        # Dry air is implicit, so the chemical derivative replaces dry by water.
        water_tendencies = []
        dm = interface_mass(dp)
        for a in range(3):
            q = water[..., a] / dm
            species_gradient = water_gradient[..., a]
            fraction_edge = 0.5 * (q[c0] + q[c1])
            species_horizontal_adjoint = -0.5 * (
                fraction_edge[:, :-1] * divergence_transpose(species_gradient[:, :-1])
                + fraction_edge[:, 1:] * divergence_transpose(species_gradient[:, 1:])
            )
            species_grad_omega = (
                (species_gradient[:, 1:] - species_gradient[:, :-1])
                * 0.5
                * (q[:, :-1] + q[:, 1:])
            )
            species_grad_mu = torch.cat(
                [
                    0.5 * species_grad_omega[:, :1],
                    0.5 * (species_grad_omega[:, :-1] + species_grad_omega[:, 1:]),
                    0.5 * species_grad_omega[:, -1:],
                ],
                -1,
            )
            species_grad_prefix = -torch.flip(
                torch.cumsum(torch.flip(species_grad_mu[:, 1:], [-1]), -1), [-1]
            )
            species_grad_div = species_grad_prefix - (
                species_grad_prefix * delta_B
            ).sum(-1, keepdim=True)
            pressure_covector -= dp_edge * (
                species_horizontal_adjoint + divergence_transpose(species_grad_div)
            )
            horizontal_species = v(
                self.centered_scalar_transport(
                    state,
                    state.u_e,
                    q,
                    transport["dp"],
                    dp_e=transport["dp_e"],
                    mass_flux=transport["mass_flux"],
                )
            )
            flux = omega * 0.5 * (q[:, :-1] + q[:, 1:])
            water_tendencies.append(
                horizontal_species
                + torch.cat(
                    [-flux[:, :1], flux[:, :-1] - flux[:, 1:], flux[:, -1:]], -1
                )
            )

        # 4. Fixed-vorticity skew transpose in the diagonal kinetic-energy metric.
        # Vorticity is 1/s on vertex layers, averaged to edge layers; the metric
        # is edge_area*dp_edge/gravity (kg). Skew pairing cancels rotational work.
        vorticity = (
            v(mesh.curl(u)) + self.rotation_rate * mesh.coriolis_unit_vertex[:, None]
        )
        vertices = mesh.verticesOnEdge
        edge_vorticity = 0.5 * (vorticity[vertices[:, 0]] + vorticity[vertices[:, 1]])
        indices = mesh.safe_rotation_stencil
        coefficients = (
            -0.5
            * (edge_vorticity[:, None, :] + edge_vorticity[indices])
            * mesh.rotation_weights_masked[:, :, None]
        )
        metric_diagonal = edge_area * dp_edge / self.gravity
        transpose_rotation = torch.zeros_like(u).scatter_add(
            0,
            indices.reshape(-1, 1).expand(-1, u.shape[1]),
            (coefficients * (metric_diagonal * u)[:, None, :]).reshape(-1, u.shape[1]),
        )
        rotation_covector = 0.5 * (
            metric_diagonal * (coefficients * u[indices]).sum(1) - transpose_rotation
        )

        # 5. Centered vertical velocity exchange and time-dependent mass metric.
        # Keep the metric-rate correction: layer mass changes during transport.
        mu_edge = v(state.mesh.cell_to_edge(mu))

        def metric(z, thickness=dp):
            return self.metric_action(state, z, thickness)

        def vertical(z):
            zi = torch.cat([z[:, :1], 0.5 * (z[:, :-1] + z[:, 1:]), z[:, -1:]], dim=-1)
            return (
                mu_edge[:, :-1] * (zi[:, :-1] - z) - mu_edge[:, 1:] * (zi[:, 1:] - z)
            ) / dp_edge

        weighted_u = metric(u) / dp_edge
        exchange = 0.5 * mu_edge[:, 1:-1] * (weighted_u[:, :-1] + weighted_u[:, 1:])
        zero_end = torch.zeros_like(weighted_u[:, :1])
        vertical_transpose = torch.cat([exchange, zero_end], dim=-1) - torch.cat(
            [zero_end, exchange], dim=-1
        )
        vertical_covector = 0.5 * (
            metric(vertical(u)) - vertical_transpose - metric(u, mu[:, :-1] - mu[:, 1:])
        )
        bernoulli_covector = metric_diagonal * v(
            -state.mesh.grad(self.cell_kinetic_energy(state))
        )
        # 6. Bernoulli work uses cell kinetic energy (J/kg). Preserve the two
        # assembly paths: solve the sum for tendency, or solve each term for
        # diagnostics. Reassociating these operations can change roundoff.
        momentum = {
            "momentum_diffusion": torch.zeros_like(u),
            "diffusion": torch.zeros_like(u),
        }
        if combine:
            momentum["total"] = self.solve_kinetic_metric(
                state,
                pressure_covector
                + rotation_covector
                + vertical_covector
                + bernoulli_covector,
            )
        else:
            for name, covector in [
                ("pgf", pressure_covector),
                ("rotation", rotation_covector),
                ("vertical", vertical_covector),
                ("bernoulli", bernoulli_covector),
            ]:
                momentum[name] = self.solve_kinetic_metric(state, covector)
            momentum["total"] = sum(
                momentum[name] for name in ["pgf", "rotation", "vertical", "bernoulli"]
            )

        # 7. Finish entropy divergence (closed top/bottom fluxes), then continuity.
        # The transported scalar is entropy throughout this assembly.
        flux = omega * 0.5 * (specific_entropy[:, :-1] + specific_entropy[:, 1:])
        vertical_scalar = torch.cat(
            [-flux[:, :1], flux[:, :-1] - flux[:, 1:], flux[:, -1:]], dim=-1
        )
        entropy_terms = {
            "horizontal": horizontal_entropy,
            "vertical": vertical_scalar,
            "coord_work": torch.zeros_like(specific_entropy),
            "advective": horizontal_entropy + vertical_scalar,
            "total": horizontal_entropy + vertical_scalar,
        }
        return {
            "u_e": momentum,
            "entropy_content": entropy_terms,
            "water_content": {"total": torch.stack(water_tendencies, -1)},
            "ps": self.ps_tendency_terms(state, primitives=transport),
        }


class MoistConservativeDynamics(torch.nn.Module):
    """Experimental mixture-energy RHS on the spherical mesh.

    Positive vapor, nonnegative suspended liquid/ice, common velocity and T.
    Fixed pressure top and direct_u_gl2_adjoint column geometry are required.
    No saturation, sedimentation, physics coupling, or positivity limiter.

    By default, condensate roundoff down to 32*eps times each species' largest
    positive fraction is admitted without changing the stored state. Condensate
    thermodynamics have a smooth algebraic extension through zero; vapor does
    not. This numerical tolerance is not a physical negative-water allowance.
    Set allow_condensate_roundoff=False for strictly nonnegative validation.
    """

    def __init__(
        self, *, thermodynamics=None, allow_condensate_roundoff=True, **kwargs
    ):
        super().__init__()
        self.allow_condensate_roundoff = allow_condensate_roundoff
        self.thermodynamics = thermodynamics or MixtureThermodynamics()
        self.kernel = _MoistKernel(self.thermodynamics, **kwargs)

    def _condensate_tolerance(self, q):
        scale = q[..., 1:].clamp_min(0).amax(dim=(0, 1))
        factor = (
            32 * torch.finfo(q.dtype).eps if self.allow_condensate_roundoff else 0.0
        )
        return factor * scale

    def composition_bounds(self, s):
        """Pure tensor diagnostics, also usable on a rejected composition.

        Return water_fraction_min (vapor/liquid/ice, kg/kg), condensate_tolerance
        (liquid/ice, kg/kg), and negative_condensate_mass (liquid/ice, kg).
        The latter is the magnitude of the negative inventory, not a correction.
        Nothing is clipped or written back to the state.
        """
        dm = interface_pressure_mass(v(s.column.interface_pressure(s.ps)).diff())
        q = s.water_content / dm[..., None]
        negative = -s.water_content[..., 1:].clamp_max(0)
        return dict(
            water_fraction_min=q.amin(dim=(0, 1)),
            condensate_tolerance=self._condensate_tolerance(q),
            negative_condensate_mass=(s.mesh.cell_area[:, None, None] * negative).sum(
                dim=(0, 1)
            )
            / self.kernel.gravity,
        )

    def transport_state(self, s):
        """Validate mixture contents and borrow an explicitly named entropy view.

        scalar_content is exactly s.entropy_content: no clone, detach, cast,
        or conversion to potential temperature. No dry thermodynamic diagnostic
        accepts this view. Every call recomputes its masses from supplied ps.
        """
        p = v(s.column.interface_pressure(s.ps))
        dp = p.diff()
        dm = interface_pressure_mass(dp)
        if s.water_content.shape != (*p.shape, 3):
            raise ValueError("water contents must have shape (cells, interfaces, 3)")
        q = s.water_content / dm[..., None]
        if not torch.isfinite(q).all() or not torch.isfinite(s.entropy_content).all():
            raise ValueError("nonfinite moist state")
        tolerance = self._condensate_tolerance(q)
        if (
            (dp <= 0).any()
            or (q[..., 0] <= 0).any()
            or (q[..., 1:] < -tolerance).any()
            or (q.sum(-1) >= 1).any()
        ):
            raise ValueError(
                "require positive layer mass/vapor/dry air and nonnegative condensates (apart from admitted roundoff)"
            )
        # No dry thermodynamic diagnostics are evaluated on this scalar transport view.
        transport_view = ScalarTransportState(s.u_e, s.ps, s.entropy_content, s.mesh, s.column)
        self.kernel.check_state_compatible(transport_view)
        return transport_view

    def tendency_terms(self, s):
        """Public component RHS dictionaries, with entropy named explicitly.

        Units/staggering are those of MoistState per second. Momentum components
        are solved separately before summing, unlike tendency's combined solve.
        """
        transport_view = self.transport_state(s)
        terms = self.kernel._assemble(transport_view, s.water_content)
        return terms

    def tendency(self, s):
        """Centered moist RHS in MoistState units per second; no limiter.

        Validate/wrap -> assemble with one combined momentum solve -> unwrap
        continuity and return entropy/water under their public field names.
        """
        transport_view = self.transport_state(s)
        terms = self.kernel._assemble(transport_view, s.water_content, combine=True)
        return MoistState(
            terms["u_e"]["total"],
            v(terms["ps"]["total"]),
            terms["entropy_content"]["total"],
            terms["water_content"]["total"],
            s.mesh,
            s.column,
        )

    def energy_terms(self, s):
        """Global internal/potential/kinetic/top-reservoir energies, J.

        Also return geopotential_interface (C,I), m²/s². Reconstruct from CP
        entropy/water at two Gauss points per layer; alpha (C,L,2), m³/kg,
        integrates hydrostatic geopotential from fixed surface phi_s upward.
        All outputs retain gradients; no reporting detach occurs here.
        """
        transport_view = self.transport_state(s)
        p = v(s.column.interface_pressure(s.ps))
        quadrature = reconstruct_moist_cp(
            p, s.entropy_content, s.water_content, self.thermodynamics
        )
        dp = quadrature["dp"]
        alpha = quadrature["alpha"]
        phi_s = torch.as_tensor(s.column.phi_s, dtype=p.dtype, device=p.device).reshape(
            -1, 1
        )
        geopotential_increments = dp * alpha.mean(-1)
        phi = phi_s + torch.cat(
            [
                torch.flip(
                    torch.cumsum(torch.flip(geopotential_increments, [-1]), -1), [-1]
                ),
                torch.zeros_like(dp[:, :1]),
            ],
            -1,
        )
        # Discrete integration by parts uses exactly the same pressure quadrature.
        layer_mean_geopotential = phi[:, 1:] + (
            (quadrature["pressure"] - p[:, :-1, None]) * alpha
        ).mean(-1)
        area_over_gravity = s.mesh.cell_area[:, None] / self.kernel.gravity
        internal = (
            area_over_gravity * dp * quadrature["internal_energy"].mean(-1)
        ).sum()
        potential = (area_over_gravity * dp * layer_mean_geopotential).sum()
        boundary = (area_over_gravity[:, 0] * p[:, 0] * phi[:, 0]).sum()
        # The fixed-pressure lid reservoir is part of the semidiscrete invariant.
        kinetic = self.kernel.total_kinetic_energy(transport_view)
        return dict(
            internal=internal,
            potential=potential,
            boundary=boundary,
            kinetic=kinetic,
            total=internal + potential + kinetic + boundary,
            geopotential_interface=phi,
        )

    def boundary_total_energy(self, s):
        """Atmosphere plus fixed-pressure lid reservoir, scalar J."""
        return self.energy_terms(s)["total"]

    def energy_gradients(self, s):
        """Analytic total-energy derivatives in a MoistState-shaped container.

        Start with mixture_gradients (no kinetic energy), add the ps derivative
        of layer kinetic mass, then form the edge-wind kinetic derivative.
        Output units: u_e (E,L), kg m/s; ps (C,1), J/Pa; entropy_content
        (C,I), J / [Pa J/(kg K)]; water_content (C,I,3), J/Pa.
        These slots hold derivatives, not a physical state. Geometry is shared
        and derivatives themselves remain connected for further differentiation.
        """
        transport_view = self.transport_state(s)
        entropy_gradient, pressure_gradient, water_gradient = (
            self.kernel.mixture_gradients(transport_view, s.water_content)
        )
        db = s.column.B_interface.diff()
        pressure_gradient += (
            s.mesh.cell_area[:, None]
            * db
            * self.kernel.cell_kinetic_energy(transport_view)
            / self.kernel.gravity
        ).sum(-1, keepdim=True)
        wind_gradient = self.kernel.kinetic_energy_gradient_u(transport_view)
        return MoistState(
            wind_gradient,
            pressure_gradient,
            entropy_gradient,
            water_gradient,
            s.mesh,
            s.column,
        )

    def boundary_energy_rate(self, s, rhs):
        """Contract analytic energy derivatives with an RHS to obtain watts.

        Sum over all cells/edges, levels/interfaces and species. This is an
        instantaneous work diagnostic, not a finite-step energy assertion.
        """
        gradient = self.energy_gradients(s)
        return sum(
            (getattr(gradient, k) * getattr(rhs, k)).sum()
            for k in ("u_e", "ps", "entropy_content", "water_content")
        )
