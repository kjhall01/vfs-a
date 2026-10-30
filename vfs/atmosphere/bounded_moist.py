"""Bounded Gibbs moist configuration, independent of research directories.

Entry: MoistAtmosphere.step -> step -> step_with_info -> euler_with_info.
Each Euler evaluation builds Gibbs traces, limits the species budget, splits
limiter work into mechanical correction/heat, then assembles scalar divergence.
Carrier transpose, Gibbs correction and bounded updates are explicit methods
of one class; no historical wrapper inheritance is needed.

Shapes below: C cells, E horizontal edges, L layers, I=L+1 CP interfaces;
N=C*I graph nodes and H=E*I+C*L oriented graph faces. Nodes are cell-major.
Four species are dry/vapor/liquid/ice; four trace components are instead
specific entropy/vapor/liquid/ice. Pressure-weighted amounts / gravity give kg.
"""

import torch
from .moist_conservative import MoistState
from .conservative import v
from .moist_cp import interface_pressure_mass
from .bounded_transport import graph, divergence, bounded_transfer, limited_traces
from .gibbs_means import gibbs_face, composition_entropy_derivatives


class BoundedGibbs:
    """Bounded moist explicit operator extracted from checkpoint 54.

    Set release_as_heat=True to reproduce the frozen bounded moist baseline.
    The old no-heat experiment is not a retained stepping configuration.
    Call step/euler, not a generic RHS Euler update: the species amount
    ledger is evaluated in cancellation-safe form. dt is a stage budget.
    Semidiscrete work closure does not imply exact finite-step energy.
    """

    def __init__(
        self, base, vapor_floor=1e-14, dry_floor=1e-14, *, release_as_heat
    ):
        if release_as_heat is not True:
            raise ValueError("retained bounded dynamics requires release_as_heat=True")
        self.base = base
        self.vapor_floor, self.dry_floor = vapor_floor, dry_floor
        # Heat allocation closes the work from the actual limited Gibbs flux.
        self.release_as_heat = release_as_heat

    def evaluate(self, s, dt):
        """Return the dt-budgeted RHS and ledger/limiter diagnostics.

        State: u_e (E,L) m/s; ps (C,1) Pa; entropy_content (C,I)
        Pa J/(kg K); water_content (C,I,3) Pa. RHS units are these per second.
        `info['amounts']` is the authoritative Euler species ledger (N,4),
        Pa m². `info['integrated_flux']` is its face transfer (H,4), Pa m².
        """
        d, mesh = self.base, s.mesh
        area = mesh.cell_area[:, None]
        # 1. Carrier geometry: mass (C,I) in Pa m², F (H,) in Pa m²/s.
        # dm is interface pressure thickness (Pa), not mass in kg.
        mass, i, j, F = graph(d, s)
        dm = mass / area
        specific_entropy = s.entropy_content / dm
        water_fractions = s.water_content / dm[..., None]
        species_fractions = torch.cat(
            [1 - water_fractions.sum(-1, keepdim=True), water_fractions], -1
        )
        n = specific_entropy.shape[1]
        # 2. Gibbs high traces: entropy J/(kg K), fractions kg/kg; (H,4).
        # Only horizontal traces differ from arithmetic means. The matched
        # momentum correction is retained when scalar RHS entries are replaced.
        gibbs_rhs_correction, gibbs_info = self._gibbs_corrections(s)
        node_traces = torch.cat(
            [specific_entropy[..., None], water_fractions], -1
        ).reshape(-1, 4)
        high_traces = 0.5 * (node_traces[i] + node_traces[j]) + gibbs_info["delta_trace"]
        high_fractions = torch.cat(
            [1 - high_traces[:, 1:].sum(-1, keepdim=True), high_traces[:, 1:]], -1
        )
        pressure = v(s.column.interface_pressure(s.ps))
        T = d.thermodynamics.temperature_from_entropy(specific_entropy, pressure, species_fractions)
        composition_entropy = composition_entropy_derivatives(
            d.thermodynamics, T, pressure, species_fractions
        ).reshape(-1, 3)
        base = d.tendency(s)
        # 3. Continuity sets the actual Euler carrier endpoint (N,), Pa m².
        # Existing fraction lower bounds constrain transfer; no state is clipped
        # or given a vapor floor. Positive vapor is required on entry.
        target_ps = s.ps + dt * base.ps
        target_amount = (
            area
            * interface_pressure_mass(v(s.column.interface_pressure(target_ps)).diff())
        ).flatten()
        floors = species_fractions.new_tensor([self.dry_floor, self.vapor_floor, 0.0, 0.0])
        face_entropy, face_fractions, info = limited_traces(
            mass.flatten(),
            species_fractions.reshape(-1, 4),
            high_traces[:, 0],
            high_fractions,
            0.5 * (composition_entropy[i] + composition_entropy[j]),
            i,
            j,
            F,
            dt,
            target=target_amount,
            floors=floors,
        )
        trace_change = torch.cat([info["delta_s"][:, None], info["delta_q"][:, 1:]], -1)
        # 4. Limiting changes Gibbs traces. Area-normalized energy derivatives
        # have units K/g (entropy) and m (water); their jump times trace_change
        # gives work_per_carrier (H,) in m. F * work_per_carrier is power (W).
        energy_gradient = d.energy_gradients(s)
        energy_covector = (
            torch.cat(
                [energy_gradient.entropy_content[..., None], energy_gradient.water_content], -1
            )
            / area[..., None]
        ).reshape(-1, 4)
        work_per_carrier = ((energy_covector[j] - energy_covector[i]) * trace_change).sum(-1)
        work = F * work_per_carrier
        heat = (-work).clamp_min(0) if self.release_as_heat else torch.zeros_like(work)
        mechanical_b = (
            torch.where(work < 0, 0.0, work_per_carrier)
            if self.release_as_heat else work_per_carrier
        )
        # With heat release disabled, all work stays mechanical and heat is zero.
        # Released scalar work becomes nonnegative face heat (H,), W. Split
        # it equally between endpoint energy budgets; divide by each endpoint's
        # dE/d(entropy_content) so entropy_source (N,) has Pa J/(kg K s).
        # This is equal power allocation, not equal entropy allocation.
        entropy_energy_gradient = energy_gradient.entropy_content.flatten()
        if bool((entropy_energy_gradient <= 0).any()):
            raise ValueError("nonpositive entropy-energy derivative")
        entropy_source = (
            torch.zeros_like(entropy_energy_gradient)
            .index_add(0, i, 0.5 * heat / entropy_energy_gradient[i])
            .index_add(0, j, 0.5 * heat / entropy_energy_gradient[j])
        )
        transport = d.kernel._transport_primitives(d.transport_state(s))
        # 5. Carrier transpose returns an edge force (E,L), N. The kinetic
        # velocity metric (E,L), kg, converts it to acceleration (m/s²).
        velocity_metric = (
            mesh.dc_edge[:, None]
            * mesh.dv_edge[:, None]
            * v(transport["dp_e"])
            / d.kernel.gravity
        )
        limiter_force = -self.pullback(s, mechanical_b) / velocity_metric
        rhs = base.combine(gibbs_rhs_correction)
        # 6. Keep corrected momentum/continuity, replace scalar tendencies with
        # the limited fluxes. Antisymmetric divergence conserves global species.
        # Water RHS is diagnostic here: Euler uses amounts directly below.
        rhs = MoistState(
            rhs.u_e + limiter_force,
            rhs.ps,
            divergence(i, j, F * face_entropy, mass.numel()).reshape_as(specific_entropy) / area
            + entropy_source.reshape_as(specific_entropy),
            divergence(
                i, j, info["integrated_flux"][:, 1:] / dt, mass.numel()
            ).reshape_as(water_fractions)
            / area[..., None],
            mesh,
            s.column,
        )
        info.update(
            limiter_force=limiter_force,
            limiter_work=work,
            carrier=F,
            heat=heat,
            entropy_source=entropy_source.reshape_as(specific_entropy),
            face_entropy=face_entropy,
            face_fractions=face_fractions,
            target_mass=target_amount,
        )
        return rhs, info

    def tendency(self, s, dt):
        return self.evaluate(s, dt)[0]

    def corrections(self, s, dt):
        rhs, info = self.evaluate(s, dt)
        return rhs.combine(self.base.tendency(s), b=-1), info

    def euler_with_info(self, s, dt):
        """Euler stage: advance wind/pressure/entropy, install ledger water."""
        rhs, info = self.evaluate(s, dt)
        updated = s.combine(rhs, b=dt)
        # (N,4) Pa m² -> (C,I,3) Pa, excluding the dry-air ledger column.
        # Do not replace this by water_content + dt*rhs.water_content: that
        # subtraction/addition loses the cancellation-safe bounded amounts.
        water = (
            info["amounts"][:, 1:].reshape_as(s.water_content)
            / s.mesh.cell_area[:, None, None]
        )
        updated = MoistState(
            updated.u_e, updated.ps, updated.entropy_content, water, s.mesh, s.column
        )
        self.base.transport_state(updated)
        return updated, info

    def step_with_info(self, s, dt):
        """SSPRK3 with three bounded Euler maps, each using the full dt budget.

        Convex combinations act on the same prognostic contents, not fractions.
        Each Euler map returns its own ledger; info1/2/3 are stage diagnostics,
        not a single ledger for the final convex combination.
        """
        stage1, info1 = self.euler_with_info(s, dt)
        stage1_euler, info2 = self.euler_with_info(stage1, dt)
        # Second RK state: 3/4 initial + 1/4 twice-Euler state.
        stage2 = s.combine(stage1_euler, a=0.75, b=0.25)
        stage2_euler, info3 = self.euler_with_info(stage2, dt)
        # Final RK state: 1/3 initial + 2/3 Euler-from-second-stage state.
        updated = s.combine(stage2_euler, a=1 / 3, b=2 / 3)
        self.base.transport_state(updated)
        return updated, [info1, info2, info3]

    def step(self, s, dt):
        return self.step_with_info(s, dt)[0]

    def euler(self, s, dt):
        return self.euler_with_info(s, dt)[0]

    def pullback(self, s, b):
        """Euclidean transpose of u -> area-integrated CP carrier flux.

        b (H,) is face work per carrier amount, m; return (E,L) force, N.
        Reverse the graph's horizontal interface averaging, vertical omega
        averaging, mu prefix sum and ps continuity constraint in that order.
        This analytic transpose pairs scalar work with momentum; no autograd
        tendency or independent pressure-gradient operator is introduced.
        """
        d = self.base
        m = s.mesh
        n = s.water_content.shape[1]
        a = d.transport_state(s)
        ctx = d.kernel._transport_primitives(a)
        dp_e = v(ctx["dp_e"])
        nh = m.nEdges * n
        # bh (E,I), m: horizontal face work; averaging reverses layer -> CP.
        bh = b[:nh].reshape(m.nEdges, n)
        horizontal = m.dv_edge[:, None] * dp_e * 0.5 * (bh[:, :-1] + bh[:, 1:])
        # go (C,L), m³: vertical work times area. gmu (C,I), m³, reverses
        # omega averaging; gr/gd (C,L), m³, reverse the prefix sum and the
        # surface-pressure constraint. These are transpose weights, not fluxes.
        go = b[nh:].reshape(m.nCells, n - 1) * m.cell_area[:, None]
        gmu = torch.cat(
            [0.5 * go[:, :1], 0.5 * (go[:, :-1] + go[:, 1:]), 0.5 * go[:, -1:]], -1
        )
        gr = -torch.flip(torch.cumsum(torch.flip(gmu[:, 1:], [-1]), -1), [-1])
        db = s.column.B_interface.diff()
        gd = gr - (gr * db).sum(-1, keepdim=True)
        # divT (E,L), m², reverses horizontal divergence; dp_e*divT is force.
        divT = -(m.dc_edge[:, None] * m.dv_edge[:, None]) * v(
            m.grad(gd / m.cell_area[:, None])
        )
        return horizontal + dp_e * divT

    def cfl(self, s, dt):
        """Diagnostic maximum of outgoing carrier fraction and layer shrinkage.

        dt is seconds; both ratios are dimensionless. Does not choose a timestep
        or subcycle SSPRK3; bounded_transfer subcycles only its donor reference.
        """
        mass, i, j, F = graph(self.base, s)
        out = torch.zeros(mass.numel(), dtype=F.dtype, device=F.device)
        out.index_add_(0, i, F.clamp_min(0))
        out.index_add_(0, j, (-F).clamp_min(0))
        a = self.base.transport_state(s)
        ctx = self.base.kernel._transport_primitives(a)
        dp = v(ctx["dp"])
        pd = v(ctx["dps_dt"])
        dpdot = pd * s.column.B_interface.diff()
        return max(
            float((dt * out / mass.flatten()).max()),
            float((dt * (-dpdot).clamp_min(0) / dp).max()),
        )

    def _gibbs_corrections(self, s):
        """Match horizontal Gibbs traces to their momentum work correction.

        Return an RHS increment and face diagnostics. delta_trace (H,4) is
        Gibbs minus arithmetic entropy/water trace; vertical entries are zero.
        Its flux divergence is a scalar increment, paired with the negative
        carrier transpose to cancel instantaneous total-energy work.
        """
        d = self.base
        m = s.mesh
        area = m.cell_area[:, None]
        n = s.water_content.shape[1]
        mass, i, j, F = graph(d, s)
        dm = mass / area
        specific_entropy = s.entropy_content / dm
        water_fractions = s.water_content / dm[..., None]
        species_fractions = torch.cat(
            [1 - water_fractions.sum(-1, keepdim=True), water_fractions], -1
        )
        pressure = v(s.column.interface_pressure(s.ps))
        T = d.thermodynamics.temperature_from_entropy(specific_entropy, pressure, species_fractions)
        c0, c1 = m.cellsOnEdge.T
        face_entropy, face_fractions, alpha = gibbs_face(
            d.thermodynamics, T[c0], pressure[c0], species_fractions[c0], T[c1], pressure[c1], species_fractions[c1]
        )
        trace_change = torch.cat(
            [
                (face_entropy - 0.5 * (specific_entropy[c0] + specific_entropy[c1]))[..., None],
                face_fractions[..., 1:] - 0.5 * (water_fractions[c0] + water_fractions[c1]),
            ],
            -1,
        )
        trace_change = torch.cat(
            [trace_change.reshape(-1, 4), trace_change.new_zeros(m.nCells * (n - 1), 4)], 0
        )
        # F transports entropy at Pa m² J/(kg K s), water at Pa m²/s.
        # Divide node divergence by area to recover pressure-weighted RHS units.
        flux = F[:, None] * trace_change
        scalar_correction = (
            torch.zeros(m.nCells * n, 4, dtype=flux.dtype, device=flux.device)
            .index_add(0, i, -flux)
            .index_add(0, j, flux)
            .reshape(m.nCells, n, 4)
            / area[..., None]
        )
        energy_gradient = d.energy_gradients(s)
        # Same work pairing as the limiter correction in evaluate, here applied
        # to Gibbs-minus-arithmetic traces before any species-budget limiting.
        energy_covector = (
            torch.cat(
                [energy_gradient.entropy_content[..., None], energy_gradient.water_content], -1
            )
            / area[..., None]
        ).reshape(-1, 4)
        work_per_carrier = ((energy_covector[j] - energy_covector[i]) * trace_change).sum(-1)
        transport = d.kernel._transport_primitives(d.transport_state(s))
        velocity_metric = (
            m.dc_edge[:, None] * m.dv_edge[:, None] * v(transport["dp_e"]) / d.kernel.gravity
        )
        rhs_correction = MoistState(
            -self.pullback(s, work_per_carrier) / velocity_metric,
            torch.zeros_like(s.ps),
            scalar_correction[..., 0],
            scalar_correction[..., 1:],
            m,
            s.column,
        )
        return rhs_correction, dict(
            delta_trace=trace_change,
            face_work=F * work_per_carrier,
            carrier_flux=F,
            face_entropy=face_entropy,
            face_fractions=face_fractions,
            face_alpha=alpha,
        )
