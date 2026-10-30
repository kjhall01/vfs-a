"""Analytic smooth-mixture pressure/transport wave blocks; no runtime AD.

Differentiate the native GL2 mixture enthalpy energy and its matched transport
transpose at fixed contents (S, C_v, C_l, C_i), including ps-dependent masses.
The off-diagonal block is an exact derivative of those parts of the RHS, not
of the whole nonlinear tendency. Advection, rotation and KE remain explicit.
"""
import torch
from ...atmosphere.moist_cp import interface_pressure_mass, reconstruct_moist_cp


class MoistPressureWave:
    """Freeze thermodynamic coefficients once per outer ARK step."""
    def __init__(self, dynamics, state):
        self.d = dynamics
        self.s = state
        self.transport_view = dynamics.transport_state(state)
        self.p = state.column.interface_pressure(state.ps)
        self.dp = self.p.diff()
        self.dm = interface_pressure_mass(self.dp)
        self.db = state.column.B_interface.diff()
        self.dm1 = interface_pressure_mass(self.db)
        self.x = self.p.new_tensor([.5 - .5 / 3**.5, .5 + .5 / 3**.5])
        self.Bg = state.column.B_interface[:-1, None] + self.db[:, None] * self.x
        self.r = reconstruct_moist_cp(
            self.p, state.entropy_content, state.water_content, dynamics.thermodynamics,
        )
        self.gs, self.gp, self.gc = dynamics.kernel.mixture_gradients(self.transport_view, state.water_content)
        self.scalars = torch.cat([state.entropy_content[..., None], state.water_content], -1) / self.dm[..., None]
        thermo = dynamics.thermodynamics
        T, q, pg = self.r['temperature'], self.r['fractions'], self.r['pressure']
        self.cp = T.new_tensor([thermo.cp_d, thermo.cp_v, thermo.cp_l, thermo.cp_i])
        self.R = T.new_tensor([thermo.R_d, thermo.R_v, 0, 0])
        self.href = T.new_tensor([0, thermo.L_v_ref, 0, -thermo.L_f_ref])
        pref = T.new_tensor([thermo.p_d_ref, thermo.p_v_ref, 1, 1])
        self.Rm = thermo.gas_constant(q)
        self.cpm = thermo.heat_capacity_p(q)
        partial = pg[..., None] * q * self.R / self.Rm[..., None]
        self.sa = (self.cp * torch.log(T[..., None] / thermo.T_ref)
                   + self.href / thermo.T_ref
                   - self.R * torch.log(torch.where(partial > 0, partial, pref) / pref))
        self.ha = (T[..., None] - thermo.T_ref) * self.cp + self.href
        chemical = self.ha - T[..., None] * self.sa
        self.mu = chemical[..., 1:] - chemical[..., :1]
        self.weight = state.mesh.cell_area[:, None] / dynamics.kernel.gravity
        self.edge_area = state.mesh.dc_edge[:, None] * state.mesh.dv_edge[:, None]
        self.dp_e = state.mesh.cell_to_edge(self.dp)

    def interpolate(self, z):
        """Interface scalar to the two native Gauss points in each layer."""
        return z[:, :-1, None] * (1 - self.x) + z[:, 1:, None] * self.x

    def adj(self, z, dp):
        """Transpose interpolation with GL2, pressure and area/g weights."""
        a = (dp[..., None] * .5 * z * (1 - self.x)).sum(-1)
        b = (dp[..., None] * .5 * z * self.x).sum(-1)
        zero = torch.zeros_like(a[:, :1])
        return (torch.cat([a, zero], -1) + torch.cat([zero, b], -1)) * self.weight

    def gradient_tangent(self, S, C, ps):
        """Directional derivatives of H_S, H_ps, H_C and specific scalars.

        All arguments are perturbations. For each species a,
        dT = T/cp * (ds + R dp/p - sum(s_a dq_a)),
        d(h_a-T s_a) = -s_a dT + T R_a (dp/p - dR/R + dq_a/q_a).
        Condensates have R_a=0 and a regular derivative through q_a=0;
        dry air and vapor must be strictly positive.
        """
        T, q, pg = self.r['temperature'], self.r['fractions'], self.r['pressure']
        ddp, ddm = self.db * ps, self.dm1 * ps
        ds = (S - self.scalars[..., 0] * ddm) / self.dm
        dq = (C - self.scalars[..., 1:] * ddm[..., None]) / self.dm[..., None]
        dsg = self.interpolate(ds)
        dqg = torch.stack([self.interpolate(dq[..., i]) for i in range(3)], -1)
        dqg = torch.cat([-dqg.sum(-1, keepdim=True), dqg], -1)
        dpg = self.Bg * ps[..., None]
        dR = (dqg * self.R).sum(-1)
        dT = T / self.cpm * (dsg + self.Rm * dpg / pg - (self.sa * dqg).sum(-1))
        safeq = torch.cat([q[..., :2], torch.ones_like(q[..., 2:])], -1)
        dchemical = (-self.sa * dT[..., None] + T[..., None] * self.R
                     * (dpg[..., None] / pg[..., None] - dR[..., None] / self.Rm[..., None] + dqg / safeq))
        dmu = dchemical[..., 1:] - dchemical[..., :1]
        dgs = (self.adj(dT, self.dp) + self.adj(T, ddp) - self.gs * ddm) / self.dm
        dgc = torch.stack([
            (self.adj(dmu[..., i], self.dp) + self.adj(self.mu[..., i], ddp)
             - self.gc[..., i] * ddm) / self.dm for i in range(3)
        ], -1)
        dh = self.cpm * dT + (self.ha * dqg).sum(-1)
        dalpha = self.r['alpha'] * (dR / self.Rm + dT / T - dpg / pg)
        dgp = self.weight * (
            self.db * dh.mean(-1) + ddp * (self.r['alpha'] * self.Bg).mean(-1)
            + self.dp * (dalpha * self.Bg).mean(-1)
        ).sum(-1, keepdim=True)
        dgp -= ((dgs * self.scalars[..., 0] + self.gs * ds) * self.dm1).sum(-1, keepdim=True)
        dgp -= ((dgc * self.scalars[..., 1:] + self.gc * dq) * self.dm1[None, :, None]).sum((-1, -2))[:, None]
        return dgs, dgp, dgc, ds, dq

    def wind(self, S, C, ps):
        """Exact thermodynamic-to-PGF derivative, including trace variations."""
        dgs, dgp, dgc, ds, dq = self.gradient_tangent(S, C, ps)
        gradients = torch.cat([self.gs[..., None], self.gc], -1)
        dgradients = torch.cat([dgs[..., None], dgc], -1)
        dscalars = torch.cat([ds[..., None], dq], -1)
        mesh = self.s.mesh
        i, j = mesh.cellsOnEdge.T

        def div_transpose(x):
            return -self.edge_area * mesh.grad(x / mesh.cell_area[:, None])

        horizontal = torch.zeros_like(self.s.u_e)
        grad_omega = torch.zeros_like(self.dp)
        for k in range(4):
            z, dz = self.scalars[..., k], dscalars[..., k]
            a, da = gradients[..., k], dgradients[..., k]
            ze, dze = .5 * (z[i] + z[j]), .5 * (dz[i] + dz[j])
            horizontal -= .5 * (
                dze[:, :-1] * div_transpose(a[:, :-1]) + ze[:, :-1] * div_transpose(da[:, :-1])
                + dze[:, 1:] * div_transpose(a[:, 1:]) + ze[:, 1:] * div_transpose(da[:, 1:])
            )
            grad_omega += da.diff() * .5 * (z[:, :-1] + z[:, 1:]) + a.diff() * .5 * (dz[:, :-1] + dz[:, 1:])
        grad_mu = torch.cat([
            .5 * grad_omega[:, :1], .5 * (grad_omega[:, :-1] + grad_omega[:, 1:]),
            .5 * grad_omega[:, -1:],
        ], -1)
        prefix = -torch.flip(torch.cumsum(torch.flip(grad_mu[:, 1:], [-1]), -1), [-1])
        grad_div = prefix - (dgp + (prefix * self.db).sum(-1, keepdim=True))
        # dp_edge cancels exactly between the transport transpose and KE metric.
        return -self.d.kernel.gravity / self.edge_area * (horizontal + div_transpose(grad_div))

    def transport(self, u):
        """Exact wind-to-content tendency at frozen ps, entropy and fractions."""
        flux = self.dp_e * u
        div = self.s.mesh.div(flux)
        ps = -div.sum(-1, keepdim=True)
        mu = self.d.kernel._vertical_mass_flux(self.transport_view, div, ps)
        omega = .5 * (mu[:, :-1] + mu[:, 1:])
        fields = []
        for k in range(4):
            z = self.scalars[..., k]
            horizontal = self.d.kernel.centered_scalar_transport(
                self.transport_view, u, z, self.dp, dp_e=self.dp_e, mass_flux=flux,
            )
            vertical = omega * .5 * (z[:, :-1] + z[:, 1:])
            fields.append(horizontal + torch.cat([
                -vertical[:, :1], vertical[:, :-1] - vertical[:, 1:], vertical[:, -1:],
            ], -1))
        return ps, fields[0], torch.stack(fields[1:], -1)
