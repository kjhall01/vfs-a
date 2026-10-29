"""Analytic pressure-wave block of the dry conservative diagonal operator.

No automatic differentiation or finite differences are used here. The column
thermal energy (including the fixed-pressure lid) is affine in mtheta:
H_th = sum_i a_i(ps) mtheta_i + A/g * p_bottom * phi_s.
Cache a, a', a'' once per step; apply the exact derivative of the matched PGF.
Advection, rotation and kinetic forces remain in the nonlinear ARK residual.
"""
import torch
from ...atmosphere.conservative import v


def thermal_coefficients(state, gravity):
    """Return dH/dmtheta and its first two LOCAL surface-pressure derivatives.

    Explicit product/quotient rules for the existing direct_u_gl2_adjoint
    energy quadrature. Derivatives are at fixed mtheta, not fixed theta.
    """
    col = state.column
    p = v(col.interface_pressure(state.ps))
    b = col.B_interface.expand_as(p)
    dp, db = p.diff(), b.diff()
    pi = v(col.exner_interface(state.ps))
    pi1 = col.kappa * pi * b / p
    pi2 = col.kappa * (col.kappa - 1) * pi * (b / p).square()
    pm, bm = .5 * (p[:, :-1] + p[:, 1:]), .5 * (b[:, :-1] + b[:, 1:])
    u = v(col.exner_full_pmid(state.ps))
    u1 = col.kappa * u * bm / pm
    u2 = col.kappa * (col.kappa - 1) * u * (bm / pm).square()
    delta, d1, d2 = pi.diff(), pi1.diff(), pi2.diff()
    s = (u - pi[:, :-1]) / delta
    s1 = (u1 - pi1[:, :-1] - s * d1) / delta
    s2 = (u2 - pi2[:, :-1] - s * d2 - 2 * s1 * d1) / delta
    cv_cp = 1 - col.R_d / col.c_p
    # Weight from all layers above this layer, plus the lid reservoir.
    prefix = .5 * p[:, :-1] * delta
    prefix1 = .5 * (b[:, :-1] * delta + p[:, :-1] * d1)
    prefix2 = b[:, :-1] * d1 + .5 * p[:, :-1] * d2
    coeff = []
    for lower in (False, True):
        if lower:
            f, f1, f2 = 1 - s.square(), -2*s*s1, -2*(s1.square()+s*s2)
            t, t1, t2 = s, s1, s2
        else:
            f, f1, f2 = (1-s).square(), -2*(1-s)*s1, 2*s1.square()-2*(1-s)*s2
            t, t1, t2 = 1-s, -s1, -s2
        w = .5*delta*f + cv_cp*u*t
        w1 = .5*(d1*f+delta*f1) + cv_cp*(u1*t+u*t1)
        w2 = .5*(d2*f+2*d1*f1+delta*f2) + cv_cp*(u2*t+2*u1*t1+u*t2)
        coeff.append((prefix+dp*w, prefix1+db*w+dp*w1,
                      prefix2+2*db*w1+dp*w2))
    def assemble(top, bottom):
        z = torch.zeros_like(top[:, :1])
        return torch.cat([top,z],-1) + torch.cat([z,bottom],-1)
    c,c1,c2 = (assemble(coeff[0][i],coeff[1][i]) for i in range(3))
    dm, dm1 = .5*assemble(dp,dp), .5*assemble(db,db)
    scale = state.mesh.cell_area[:,None] / gravity
    a = scale*c/dm
    a1 = (scale*c1-a*dm1)/dm
    a2 = (scale*c2-2*a1*dm1)/dm
    return a,a1,a2


class ConservativePressureWave:
    """Frozen exact thermodynamic-to-pressure-force derivative."""
    def __init__(self, dynamics, anchor):
        dynamics.check_state_compatible(anchor)
        self.state = anchor
        self.gravity = dynamics.gravity
        self.a, self.a1, self.a2 = thermal_coefficients(anchor, dynamics.gravity)
        self.theta = v(anchor.theta())
        self.dm = dynamics._interface_mass(v(anchor.layer_dp()))
        self.db = anchor.column.B_interface.diff()
        self.dm1 = dynamics._interface_mass(self.db.expand_as(v(anchor.layer_dp())))

    def wind(self, mtheta, ps):
        s, mesh = self.state, self.state.mesh
        theta = self.theta
        dt = (mtheta-theta*self.dm1*ps)/self.dm
        dg = self.a1*ps
        dgp = (self.a1*mtheta+self.a2*v(s.mtheta_i)*ps).sum(-1,keepdim=True)
        edge_area = mesh.dc_edge[:,None]*mesh.dv_edge[:,None]
        def div_transpose(x):
            return -edge_area*v(mesh.grad(x/mesh.cell_area[:,None]))
        i,j = mesh.cellsOnEdge.T
        te, dte = .5*(theta[i]+theta[j]), .5*(dt[i]+dt[j])
        h = -.5*(dte[:,:-1]*div_transpose(self.a[:,:-1])
                 +te[:,:-1]*div_transpose(dg[:,:-1])
                 +dte[:,1:]*div_transpose(self.a[:,1:])
                 +te[:,1:]*div_transpose(dg[:,1:]))
        domega = dg.diff()*.5*(theta[:,:-1]+theta[:,1:]) + self.a.diff()*.5*(dt[:,:-1]+dt[:,1:])
        dmu = torch.cat([.5*domega[:,:1],.5*(domega[:,:-1]+domega[:,1:]),.5*domega[:,-1:]],-1)
        prefix = -torch.flip(torch.cumsum(torch.flip(dmu[:,1:],[-1]),-1),[-1])
        ddiv = prefix - (dgp+(prefix*self.db).sum(-1,keepdim=True))
        # dp_edge cancels exactly between the transport transpose and KE metric.
        return -self.gravity/edge_area*(h+div_transpose(ddiv))
