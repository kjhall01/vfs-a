"""Bounded Gibbs moist configuration, independent of research directories.

Read evaluate in order: carrier graph, Gibbs traces, species budget,
energy work split, momentum transpose, entropy heating, scalar divergence.
Carrier transpose, Gibbs correction and bounded updates are explicit methods
of one class; no historical wrapper inheritance is needed.
"""

import torch
from .moist_conservative import MoistState
from .conservative import v
from .moist_cp import interface_pressure_mass
from .bounded_transport import graph, divergence, bounded_transfer, limited_traces
from .gibbs_means import gibbs_face, composition_entropy_derivatives


class BoundedGibbs:
    """Bounded moist explicit operator extracted from checkpoint 54.

    Set release_as_heat=True to reproduce the verification campaign.
    The default remains False for fidelity to the original wrapper.
    Call step/euler, not a generic RHS Euler update: the species amount
    ledger is evaluated in cancellation-safe form. dt is a stage budget.
    Semidiscrete work closure does not imply exact finite-step energy.
    """

    def __init__(
        self, base, vapor_floor=1e-14, dry_floor=1e-14, *, release_as_heat=False
    ):
        self.base = base
        self.vapor_floor, self.dry_floor = vapor_floor, dry_floor
        # Existing composition_work_split experiment, now applied to the
        # correction from the actual Gibbs high flux. Research switch only.
        self.release_as_heat = release_as_heat

    def evaluate(self, s, dt):
        d, mesh = self.base, s.mesh
        area = mesh.cell_area[:, None]
        mass, i, j, F = graph(d, s)
        dm = mass / area
        ss = s.entropy_content / dm
        qw = s.water_content / dm[..., None]
        q = torch.cat([1 - qw.sum(-1, keepdim=True), qw], -1)
        n = ss.shape[1]
        high_delta, h = self._gibbs_corrections(s)
        z = torch.cat([ss[..., None], qw], -1).reshape(-1, 4)
        zh = 0.5 * (z[i] + z[j]) + h["delta_trace"]
        qh = torch.cat([1 - zh[:, 1:].sum(-1, keepdim=True), zh[:, 1:]], -1)
        pressure = v(s.column.interface_pressure(s.ps))
        T = d.thermodynamics.temperature_from_entropy(ss, pressure, q)
        eta = composition_entropy_derivatives(d.thermodynamics, T, pressure, q).reshape(
            -1, 3
        )
        base = d.tendency(s)
        target_ps = s.ps + dt * base.ps
        target = (
            area
            * interface_pressure_mass(v(s.column.interface_pressure(target_ps)).diff())
        ).flatten()
        floors = q.new_tensor([self.dry_floor, self.vapor_floor, 0.0, 0.0])
        sf, qf, info = limited_traces(
            mass.flatten(),
            q.reshape(-1, 4),
            zh[:, 0],
            qh,
            0.5 * (eta[i] + eta[j]),
            i,
            j,
            F,
            dt,
            target=target,
            floors=floors,
        )
        dz = torch.cat([info["delta_s"][:, None], info["delta_q"][:, 1:]], -1)
        g = d.energy_gradients(s)
        cov = (
            torch.cat([g.entropy_content[..., None], g.water_content], -1)
            / area[..., None]
        ).reshape(-1, 4)
        b = ((cov[j] - cov[i]) * dz).sum(-1)
        work = F * b
        heat = (-work).clamp_min(0) if self.release_as_heat else torch.zeros_like(work)
        mechanical_b = torch.where(work < 0, 0.0, b) if self.release_as_heat else b
        gs = g.entropy_content.flatten()
        if bool((gs <= 0).any()):
            raise ValueError("nonpositive entropy-energy derivative")
        sigma = (
            torch.zeros_like(gs)
            .index_add(0, i, 0.5 * heat / gs[i])
            .index_add(0, j, 0.5 * heat / gs[j])
        )
        ctx = d.kernel._transport_primitives(d._adapter(s))
        metric = (
            mesh.dc_edge[:, None]
            * mesh.dv_edge[:, None]
            * v(ctx["dp_e"])
            / d.kernel.gravity
        )
        force = -self.pullback(s, mechanical_b) / metric
        rhs = base.combine(high_delta)
        rhs = MoistState(
            rhs.u_e + force,
            rhs.ps,
            divergence(i, j, F * sf, mass.numel()).reshape_as(ss) / area
            + sigma.reshape_as(ss),
            divergence(
                i, j, info["integrated_flux"][:, 1:] / dt, mass.numel()
            ).reshape_as(qw)
            / area[..., None],
            mesh,
            s.column,
        )
        info.update(
            limiter_force=force,
            limiter_work=work,
            carrier=F,
            heat=heat,
            entropy_source=sigma.reshape_as(ss),
            face_entropy=sf,
            face_fractions=qf,
            target_mass=target,
        )
        return rhs, info

    def tendency(self, s, dt):
        return self.evaluate(s, dt)[0]

    def corrections(self, s, dt):
        rhs, info = self.evaluate(s, dt)
        return rhs.combine(self.base.tendency(s), b=-1), info

    def euler_with_info(self, s, dt):
        rhs, info = self.evaluate(s, dt)
        z = s.combine(rhs, b=dt)
        water = (
            info["amounts"][:, 1:].reshape_as(s.water_content)
            / s.mesh.cell_area[:, None, None]
        )
        z = MoistState(z.u_e, z.ps, z.entropy_content, water, s.mesh, s.column)
        self.base._adapter(z)
        return z, info

    def step_with_info(self, s, dt):
        a, ia = self.euler_with_info(s, dt)
        aa, ib = self.euler_with_info(a, dt)
        b = s.combine(aa, a=0.75, b=0.25)
        bb, ic = self.euler_with_info(b, dt)
        z = s.combine(bb, a=1 / 3, b=2 / 3)
        self.base._adapter(z)
        return z, [ia, ib, ic]

    def step(self, s, dt):
        return self.step_with_info(s, dt)[0]

    def euler(self, s, dt):
        return self.euler_with_info(s, dt)[0]

    def pullback(self, s, b):
        """Euclidean transpose of u -> area-integrated CP carrier flux.
        Includes omega averaging, mu prefix sum, and ps continuity constraint.
        """
        d = self.base
        m = s.mesh
        n = s.water_content.shape[1]
        a = d._adapter(s)
        ctx = d.kernel._transport_primitives(a)
        dp_e = v(ctx["dp_e"])
        nh = m.nEdges * n
        bh = b[:nh].reshape(m.nEdges, n)
        horizontal = m.dv_edge[:, None] * dp_e * 0.5 * (bh[:, :-1] + bh[:, 1:])
        go = b[nh:].reshape(m.nCells, n - 1) * m.cell_area[:, None]
        gmu = torch.cat(
            [0.5 * go[:, :1], 0.5 * (go[:, :-1] + go[:, 1:]), 0.5 * go[:, -1:]], -1
        )
        gr = -torch.flip(torch.cumsum(torch.flip(gmu[:, 1:], [-1]), -1), [-1])
        db = s.column.B_interface.diff()
        gd = gr - (gr * db).sum(-1, keepdim=True)
        divT = -(m.dc_edge[:, None] * m.dv_edge[:, None]) * v(
            m.grad(gd / m.cell_area[:, None])
        )
        return horizontal + dp_e * divT

    def cfl(self, s, dt):
        mass, i, j, F = graph(self.base, s)
        out = torch.zeros(mass.numel(), dtype=F.dtype, device=F.device)
        out.index_add_(0, i, F.clamp_min(0))
        out.index_add_(0, j, (-F).clamp_min(0))
        a = self.base._adapter(s)
        ctx = self.base.kernel._transport_primitives(a)
        dp = v(ctx["dp"])
        pd = v(ctx["dps_dt"])
        dpdot = pd * s.column.B_interface.diff()
        return max(
            float((dt * out / mass.flatten()).max()),
            float((dt * (-dpdot).clamp_min(0) / dp).max()),
        )

    def _gibbs_corrections(self, s):
        d = self.base
        m = s.mesh
        area = m.cell_area[:, None]
        n = s.water_content.shape[1]
        mass, i, j, F = graph(d, s)
        dm = mass / area
        ss = s.entropy_content / dm
        qw = s.water_content / dm[..., None]
        q = torch.cat([1 - qw.sum(-1, keepdim=True), qw], -1)
        pressure = v(s.column.interface_pressure(s.ps))
        T = d.thermodynamics.temperature_from_entropy(ss, pressure, q)
        c0, c1 = m.cellsOnEdge.T
        sf, qf, alpha = gibbs_face(
            d.thermodynamics, T[c0], pressure[c0], q[c0], T[c1], pressure[c1], q[c1]
        )
        dz = torch.cat(
            [
                (sf - 0.5 * (ss[c0] + ss[c1]))[..., None],
                qf[..., 1:] - 0.5 * (qw[c0] + qw[c1]),
            ],
            -1,
        )
        dz = torch.cat([dz.reshape(-1, 4), dz.new_zeros(m.nCells * (n - 1), 4)], 0)
        flux = F[:, None] * dz
        D = (
            torch.zeros(m.nCells * n, 4, dtype=flux.dtype, device=flux.device)
            .index_add(0, i, -flux)
            .index_add(0, j, flux)
            .reshape(m.nCells, n, 4)
            / area[..., None]
        )
        g = d.energy_gradients(s)
        cov = (
            torch.cat([g.entropy_content[..., None], g.water_content], -1)
            / area[..., None]
        ).reshape(-1, 4)
        b = ((cov[j] - cov[i]) * dz).sum(-1)
        ctx = d.kernel._transport_primitives(d._adapter(s))
        metric = (
            m.dc_edge[:, None] * m.dv_edge[:, None] * v(ctx["dp_e"]) / d.kernel.gravity
        )
        delta = MoistState(
            -self.pullback(s, b) / metric,
            torch.zeros_like(s.ps),
            D[..., 0],
            D[..., 1:],
            m,
            s.column,
        )
        return delta, dict(
            delta_trace=dz,
            face_work=F * b,
            carrier_flux=F,
            face_entropy=sf,
            face_fractions=qf,
            face_alpha=alpha,
        )
