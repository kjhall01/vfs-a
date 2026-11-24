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
from .moist_cp import interface_pressure_mass
from .bounded_transport import _carrier_graph, bounded_transfer, divergence, graph, limited_traces
from .gibbs_means import gibbs_face, partial_chemical, composition_entropy_derivatives


def bounded_ssprk3_with_info(euler_with_info, s, time, dt):
    """Apply the existing bounded SSPRK3 sequence using timed Euler maps.

    The callback receives (state, evaluation_time, full_positive_dt) and must
    retain the authoritative Euler species ledger. Evaluation times are
    t, t+dt, t+dt/2: the second convex combination returns to the half-time
    stage, so RHS times are nonmonotonic even though every Euler budget is
    positive. These are evaluation times, not three consecutive time intervals.

    This function combines prognostic tensors exactly as the atmospheric step
    does. A prescribed-velocity caller must replace wind before EVERY Euler
    evaluation and at the final output time; intermediate combined winds are
    not prescribed velocities. Physical validation remains the caller's job.
    """
    stage1, info1 = euler_with_info(s, time, dt)
    stage1_euler, info2 = euler_with_info(stage1, time + dt, dt)
    # Second RK state: 3/4 initial + 1/4 twice-Euler state.
    stage2 = s.combine(stage1_euler, a=0.75, b=0.25)
    stage2_euler, info3 = euler_with_info(stage2, time + dt / 2, dt)
    # Final RK state: 1/3 initial + 2/3 Euler-from-second-stage state.
    updated = s.combine(stage2_euler, a=1 / 3, b=2 / 3)
    return updated, [info1, info2, info3]


class BoundedGibbs:
    """Bounded moist explicit operator extracted from checkpoint 54.

    Heat release is intrinsic to this frozen bounded moist formulation.
    Call step/euler, not a generic RHS Euler update: the species amount
    ledger is evaluated in cancellation-safe form. dt is a stage budget.
    Semidiscrete work closure does not imply exact finite-step energy.
    """

    def __init__(self, base, vapor_floor=1e-14, dry_floor=1e-14):
        self.base = base
        self.vapor_floor, self.dry_floor = vapor_floor, dry_floor
        # Heat allocation closes the work from the actual limited Gibbs flux.

    def evaluate(self, s, dt, *, include_carrier_graph=False):
        """Return the dt-budgeted RHS and ledger/limiter diagnostics.

        State: u_e (E,L) m/s; ps (C,1) Pa; entropy_content (C,I)
        Pa J/(kg K); water_content (C,I,3) Pa. RHS units are these per second.
        `info['amounts']` is the authoritative Euler species ledger (N,4),
        Pa m². `info['integrated_flux']` is its face transfer (H,4), Pa m².
        """
        return self._stage(s, dt, endpoint=False, include_carrier_graph=include_carrier_graph)

    def _stage(self, s, dt, *, endpoint, include_carrier_graph):
        """Evaluate one unchanged state, then form a diagnostic RHS or an Euler endpoint."""
        d, mesh = self.base, s.mesh
        area = mesh.cell_area[:, None]
        # 1. Carrier geometry: mass (C,I) in Pa m², F (H,) in Pa m²/s.
        # dm is interface pressure thickness (Pa), not mass in kg.
        view = d.transport_state(s)
        transport = d.kernel._transport_primitives(view)
        mass, i, j, F = _carrier_graph(s, transport)
        dm = mass / area
        specific_entropy = s.entropy_content / dm
        water_fractions = s.water_content / dm[..., None]
        species_fractions = torch.cat(
            [1 - water_fractions.sum(-1, keepdim=True), water_fractions], -1
        )
        # 2. Gibbs high traces: entropy J/(kg K), fractions kg/kg; (H,4).
        # Only horizontal traces differ from arithmetic means. The matched
        # momentum correction is retained when scalar RHS entries are replaced.
        # Sharing tensors changes reverse-mode summation order. The campaign
        # uses no_grad; differentiable callers retain independent evaluations.
        shared = not torch.is_grad_enabled()
        if shared:
            pressure = s.column.interface_pressure(s.ps)
            T = d.thermodynamics.temperature_from_entropy(specific_entropy, pressure, species_fractions)
            gibbs_change = self._gibbs_trace_change(
                s, specific_entropy, water_fractions, species_fractions, pressure, T, shared=True
            )
        else:
            gibbs_change, gibbs_force = self._independent_gibbs_force(s)
        node_traces = torch.cat(
            [specific_entropy[..., None], water_fractions], -1
        ).reshape(-1, 4)
        high_traces = 0.5 * (node_traces[i] + node_traces[j]) + gibbs_change
        high_fractions = torch.cat(
            [1 - high_traces[:, 1:].sum(-1, keepdim=True), high_traces[:, 1:]], -1
        )
        if not shared:
            pressure = s.column.interface_pressure(s.ps)
            T = d.thermodynamics.temperature_from_entropy(specific_entropy, pressure, species_fractions)
        composition_entropy = composition_entropy_derivatives(
            d.thermodynamics, T, pressure, species_fractions
        ).reshape(-1, 3)
        if shared:
            gradients = d.kernel.mixture_gradients(view, s.water_content)
            centered = d.kernel._assemble(
                view, s.water_content, primitives=transport, gradients=gradients,
                combine=True, scalar_tendencies=False,
            )
            base_u = centered["u_e"]["total"]
            base_ps = centered["ps"]["total"]
        else:
            base = d.tendency(s)
            base_u, base_ps = base.u_e, base.ps
        # 3. Continuity sets the actual Euler carrier endpoint (N,), Pa m².
        # Existing fraction lower bounds constrain transfer; no state is clipped
        # or given a vapor floor. Positive vapor is required on entry.
        target_ps = s.ps + dt * base_ps
        target_amount = (
            area
            * interface_pressure_mass(s.column.interface_pressure(target_ps).diff())
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
        if shared:
            entropy_gradient, _, water_gradient = gradients
        else:
            gradient = d.energy_gradients(s)
            entropy_gradient, water_gradient = gradient.entropy_content, gradient.water_content
        energy_covector = (
            torch.cat([entropy_gradient[..., None], water_gradient], -1) / area[..., None]
        ).reshape(-1, 4)
        covector_jump = energy_covector[j] - energy_covector[i]
        work_per_carrier = (covector_jump * trace_change).sum(-1)
        work = F * work_per_carrier
        heat = (-work).clamp_min(0)
        mechanical_b = torch.where(work < 0, 0.0, work_per_carrier)
        # Released scalar work becomes nonnegative face heat (H,), W. Split
        # it equally between endpoint energy budgets; divide by each endpoint's
        # dE/d(entropy_content) so entropy_source (N,) has Pa J/(kg K s).
        # This is equal power allocation, not equal entropy allocation.
        entropy_energy_gradient = entropy_gradient.flatten()
        if bool((entropy_energy_gradient <= 0).any()):
            raise ValueError("nonpositive entropy-energy derivative")
        entropy_source = (
            torch.zeros_like(entropy_energy_gradient)
            .index_add(0, i, 0.5 * heat / entropy_energy_gradient[i])
            .index_add(0, j, 0.5 * heat / entropy_energy_gradient[j])
        )
        # 5. Carrier transpose returns an edge force (E,L), N. The kinetic
        # velocity metric (E,L), kg, converts it to acceleration (m/s²).
        if shared:
            dp_edge = transport["dp_e"]
            velocity_metric = mesh.dc_edge[:, None] * mesh.dv_edge[:, None] * dp_edge / d.kernel.gravity
            gibbs_work = (covector_jump * gibbs_change).sum(-1)
            gibbs_force = -self._pullback(s, gibbs_work, dp_edge) / velocity_metric
            limiter_force = -self._pullback(s, mechanical_b, dp_edge) / velocity_metric
        else:
            limiter_force = self._independent_force(s, mechanical_b)
        wind_rhs = (base_u + gibbs_force) + limiter_force
        entropy_rhs = (
            divergence(i, j, F * face_entropy, mass.numel()).reshape_as(specific_entropy) / area
            + entropy_source.reshape_as(specific_entropy)
        )
        if endpoint:
            # The bounded species amounts are the authoritative Euler update.
            water = info["amounts"][:, 1:].reshape_as(s.water_content) / area[..., None]
            # The differentiable endpoint has its own pressure path. Sharing
            # the limiter target would reassociate its active-bound adjoint.
            endpoint_ps = target_ps if shared else s.ps + dt * base_ps
            result = MoistState(
                s.u_e + dt * wind_rhs, endpoint_ps, s.entropy_content + dt * entropy_rhs,
                water, mesh, s.column,
            )
            d.transport_state(result)
        else:
            water_rhs = divergence(i, j, info["integrated_flux"][:, 1:] / dt, mass.numel())
            result = MoistState(
                wind_rhs, base_ps, entropy_rhs,
                water_rhs.reshape_as(water_fractions) / area[..., None], mesh, s.column,
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
        # Keep the established public report schema unchanged. The cloud
        # transport integration explicitly opts into these already-computed
        # tensors, avoiding redundant graph/transport primitive construction.
        if include_carrier_graph:
            info["carrier_graph"] = (mass, i, j, F)
        return result, info

    def tendency(self, s, dt):
        return self.evaluate(s, dt)[0]

    def corrections(self, s, dt):
        rhs, info = self.evaluate(s, dt)
        return rhs.combine(self.base.tendency(s), b=-1), info

    def euler_with_info(self, s, dt, *, include_carrier_graph=False):
        """Euler stage: advance wind/pressure/entropy, install ledger water."""
        return self._stage(s, dt, endpoint=True, include_carrier_graph=include_carrier_graph)

    def step_with_info(self, s, dt):
        """SSPRK3 with three bounded Euler maps, each using the full dt budget.

        Convex combinations act on the same prognostic contents, not fractions.
        Each Euler map returns its own ledger; info1/2/3 are stage diagnostics,
        not a single ledger for the final convex combination.
        """
        def euler_at_time(state, time, budget):
            return self.euler_with_info(state, budget)

        updated, info = bounded_ssprk3_with_info(euler_at_time, s, 0.0, dt)
        self.base.transport_state(updated)
        return updated, info

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
        view = self.base.transport_state(s)
        dp_edge = self.base.kernel._transport_primitives(view)["dp_e"]
        return self._pullback(s, b, dp_edge)

    @staticmethod
    def _pullback(s, b, dp_e):
        m = s.mesh
        n = s.water_content.shape[1]
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
        divT = -(m.dc_edge[:, None] * m.dv_edge[:, None]) * m.grad(gd / m.cell_area[:, None])
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
        dp = ctx["dp"]
        pd = ctx["dps_dt"]
        dpdot = pd * s.column.B_interface.diff()
        return max(
            float((dt * out / mass.flatten()).max()),
            float((dt * (-dpdot).clamp_min(0) / dp).max()),
        )

    def _gibbs_trace_change(self, s, specific_entropy, water_fractions, species_fractions, pressure, T, *, shared):
        """Gibbs minus centered horizontal traces, extended by zero on vertical faces."""
        m = s.mesh
        n = s.water_content.shape[1]
        thermo = self.base.thermodynamics
        c0, c1 = m.cellsOnEdge.T
        potentials = None
        if shared:
            mu = partial_chemical(thermo, T, pressure, species_fractions)
            potentials = (mu[c0], mu[c1])
        face_entropy, face_fractions, _ = gibbs_face(
            thermo, T[c0], pressure[c0], species_fractions[c0], T[c1], pressure[c1], species_fractions[c1],
            chemical_potentials=potentials,
        )
        trace_change = torch.cat(
            [
                (face_entropy - 0.5 * (specific_entropy[c0] + specific_entropy[c1]))[..., None],
                face_fractions[..., 1:] - 0.5 * (water_fractions[c0] + water_fractions[c1]),
            ],
            -1,
        )
        return torch.cat(
            [trace_change.reshape(-1, 4), trace_change.new_zeros(m.nCells * (n - 1), 4)], 0
        )

    def _independent_force(self, s, work):
        """Separate metric and transport-transpose paths for the original adjoint."""
        d, mesh = self.base, s.mesh
        dp_edge = d.kernel._transport_primitives(d.transport_state(s))["dp_e"]
        metric = mesh.dc_edge[:, None] * mesh.dv_edge[:, None] * dp_edge / d.kernel.gravity
        return -self.pullback(s, work) / metric

    def _independent_gibbs_force(self, s):
        """Preserve the Gibbs correction's independent thermodynamic derivative path."""
        d, mesh = self.base, s.mesh
        area = mesh.cell_area[:, None]
        mass, i, j, _ = graph(d, s)
        dm = mass / area
        entropy, water = s.entropy_content / dm, s.water_content / dm[..., None]
        fractions = torch.cat([1 - water.sum(-1, keepdim=True), water], -1)
        pressure = s.column.interface_pressure(s.ps)
        T = d.thermodynamics.temperature_from_entropy(entropy, pressure, fractions)
        change = self._gibbs_trace_change(s, entropy, water, fractions, pressure, T, shared=False)
        gradient = d.energy_gradients(s)
        covector = (torch.cat([gradient.entropy_content[..., None], gradient.water_content], -1)
                    / area[..., None]).reshape(-1, 4)
        work = ((covector[j] - covector[i]) * change).sum(-1)
        return change, self._independent_force(s, work)
