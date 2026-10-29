"""Opt-in research moist dynamics. No phase change, limiter or SCM integration.

Public state uses entropy/species contents. The private transport adapter uses
State.mtheta_i ONLY as an internal scalar buffer; it is never exposed as dry theta.
The established dry dynamics source and defaults are unchanged.
"""
from dataclasses import dataclass
import torch
from ..core.state import State
from .conservative import DryHydrostaticConservativeDynamics, v
from .moist_cp import interface_pressure_mass, reconstruct_moist_cp
from .mixture_thermodynamics import MixtureThermodynamics

@dataclass
class MoistState:
    u_e: torch.Tensor
    ps: torch.Tensor
    entropy_content: torch.Tensor
    water_content: torch.Tensor  # (cells, interfaces, vapor/liquid/ice)
    mesh: object
    column: object

    def combine(self, other, a=1., b=1.):
        if self.mesh is not other.mesh or self.column is not other.column:
            raise ValueError("state geometries must agree")
        return MoistState(*(a*getattr(self,k)+b*getattr(other,k) for k in
            ('u_e','ps','entropy_content','water_content')),self.mesh,self.column)

class _MoistKernel(DryHydrostaticConservativeDynamics):
    def __init__(self, thermo, **kwargs):
        super().__init__(**kwargs)
        self.mixture = thermo

    def mixture_gradients(self, state, water):
        p=v(state.column.interface_pressure(state.ps));dp=p.diff();dm=interface_pressure_mass(dp)
        S=v(state.mtheta_i);r=reconstruct_moist_cp(p,S,water,self.mixture)
        T,q,pg=r['temperature'],r['fractions'],r['pressure'];t=self.mixture
        cp=T.new_tensor([t.cp_d,t.cp_v,t.cp_l,t.cp_i]);R=T.new_tensor([t.R_d,t.R_v,0,0])
        href=T.new_tensor([0,t.L_v_ref,0,-t.L_f_ref]);pref=T.new_tensor([t.p_d_ref,t.p_v_ref,1,1])
        partial=pg[...,None]*q*R/t.gas_constant(q)[...,None]
        safe=torch.where(partial>0,partial,pref)
        ha=(T[...,None]-t.T_ref)*cp+href
        sa=cp*torch.log(T[...,None]/t.T_ref)+href/t.T_ref-R*torch.log(safe/pref)
        chem=ha-T[...,None]*sa;mu=chem[...,1:]-chem[...,:1]
        x=T.new_tensor([.5-.5/3**.5,.5+.5/3**.5])
        weight=state.mesh.cell_area[:,None]/self.gravity
        def adj(z):
            a=(dp[...,None]*.5*z*(1-x)).sum(-1);b=(dp[...,None]*.5*z*x).sum(-1)
            return (torch.cat([a,torch.zeros_like(a[:,:1])],-1)+torch.cat([torch.zeros_like(b[:,:1]),b],-1))*weight
        gs=adj(T)/dm;gc=torch.stack([adj(mu[...,i])/dm for i in range(3)],-1)
        db=state.column.B_interface.diff();dmdp=interface_pressure_mass(db)
        Bg=state.column.B_interface[:-1,None]+db[:,None]*x
        gp=weight*(db*t.enthalpy(T,q).mean(-1)+dp*(r['alpha']*Bg).mean(-1)).sum(-1,keepdim=True)
        gp-=(gs*(S/dm)*dmdp).sum(-1,keepdim=True)
        gp-=(gc*(water/dm[...,None])*dmdp[None,:,None]).sum((-1,-2))[:,None]
        phi=torch.as_tensor(state.column.phi_s,dtype=p.dtype,device=p.device).reshape(-1,1)
        gp+=weight*phi
        return gs,gp,gc

    def _assemble(self, state, water, *, primitives=None, combine=False):
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
        gm, gp, gc = self.mixture_gradients(state, water)
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

        # Active species use the same CP flux and its explicit transpose.
        species = []
        dm = self._interface_mass(dp)
        for a in range(3):
            q = water[..., a] / dm
            gradient = gc[..., a]
            qe = 0.5 * (q[c0] + q[c1])
            adj = -0.5 * (qe[:, :-1] * divergence_transpose(gradient[:, :-1])
                          + qe[:, 1:] * divergence_transpose(gradient[:, 1:]))
            go = (gradient[:, 1:] - gradient[:, :-1]) * 0.5 * (q[:, :-1] + q[:, 1:])
            gmu = torch.cat([0.5*go[:, :1], 0.5*(go[:, :-1]+go[:, 1:]), 0.5*go[:, -1:]], -1)
            gr = -torch.flip(torch.cumsum(torch.flip(gmu[:, 1:], [-1]), -1), [-1])
            gd = gr - (gr * delta_B).sum(-1, keepdim=True)
            pressure_covector -= dp_edge * (adj + divergence_transpose(gd))
            h = v(self._mtheta_tendency(state, state.u_e, q, ctx['dp'],
                    dp_e=ctx['dp_e'], mass_flux=ctx['mass_flux']))
            flux = omega * 0.5 * (q[:, :-1] + q[:, 1:])
            species.append(h + torch.cat([-flux[:, :1],flux[:, :-1]-flux[:, 1:],flux[:, -1:]],-1))

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
        return {"u_e": momentum, "mtheta_i": scalar, "water_content": {"total": torch.stack(species,-1)},
                "ps": self.ps_tendency_terms(state, primitives=ctx)}


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
    def __init__(self, *, thermodynamics=None, allow_condensate_roundoff=True, **kwargs):
        super().__init__()
        self.allow_condensate_roundoff = allow_condensate_roundoff
        self.thermodynamics=thermodynamics or MixtureThermodynamics()
        self.kernel=_MoistKernel(self.thermodynamics,**kwargs)

    def _condensate_tolerance(self, q):
        scale = q[..., 1:].clamp_min(0).amax(dim=(0, 1))
        factor = 32 * torch.finfo(q.dtype).eps if self.allow_condensate_roundoff else 0.
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
        return dict(water_fraction_min=q.amin(dim=(0, 1)),
                    condensate_tolerance=self._condensate_tolerance(q),
                    negative_condensate_mass=(
                        s.mesh.cell_area[:, None, None] * negative
                    ).sum(dim=(0, 1)) / self.kernel.gravity)

    def _adapter(self,s):
        p=v(s.column.interface_pressure(s.ps));dp=p.diff();dm=interface_pressure_mass(dp)
        if s.water_content.shape != (*p.shape,3): raise ValueError("water contents must have shape (cells, interfaces, 3)")
        q=s.water_content/dm[...,None]
        if not torch.isfinite(q).all() or not torch.isfinite(s.entropy_content).all(): raise ValueError("nonfinite moist state")
        tolerance = self._condensate_tolerance(q)
        if (dp<=0).any() or (q[...,0]<=0).any() or (q[...,1:] < -tolerance).any() or (q.sum(-1)>=1).any():
            raise ValueError("require positive layer mass/vapor/dry air and nonnegative condensates (apart from admitted roundoff)")
        # No dry thermodynamic diagnostics are evaluated on this private adapter.
        a=State(s.u_e,s.entropy_content,s.ps,mesh=s.mesh,column=s.column)
        self.kernel.check_state_compatible(a)
        return a

    def tendency_terms(self,s):
        a=self._adapter(s);t=self.kernel._assemble(a,s.water_content)
        t['entropy_content']=t.pop('mtheta_i')
        return t

    def tendency(self,s):
        a=self._adapter(s);t=self.kernel._assemble(a,s.water_content,combine=True)
        return MoistState(t['u_e']['total'],v(t['ps']['total']),t['mtheta_i']['total'],
                          t['water_content']['total'],s.mesh,s.column)

    def step(self, state, dt):
        """SSPRK3 with stage/output validation; no clipping or exact-energy claim."""
        if dt <= 0: raise ValueError("dt must be positive")
        a = state.combine(self.tendency(state), b=dt)
        b = state.combine(a.combine(self.tendency(a), b=dt), a=.75, b=.25)
        result = state.combine(b.combine(self.tendency(b), b=dt), a=1/3, b=2/3)
        self._adapter(result)
        return result

    def energy_terms(self,s):
        a=self._adapter(s);p=v(s.column.interface_pressure(s.ps));r=reconstruct_moist_cp(p,s.entropy_content,s.water_content,self.thermodynamics)
        dp=r['dp'];alpha=r['alpha'];phi_s=torch.as_tensor(s.column.phi_s,dtype=p.dtype,device=p.device).reshape(-1,1)
        increments=dp*alpha.mean(-1)
        phi=phi_s+torch.cat([torch.flip(torch.cumsum(torch.flip(increments,[-1]),-1),[-1]),torch.zeros_like(dp[:,:1])],-1)
        # Discrete integration by parts uses exactly the same pressure quadrature.
        mean_phi=phi[:,1:]+((r['pressure']-p[:,:-1,None])*alpha).mean(-1)
        w=s.mesh.cell_area[:,None]/self.kernel.gravity
        internal=(w*dp*r['internal_energy'].mean(-1)).sum()
        potential=(w*dp*mean_phi).sum()
        boundary=(w[:,0]*p[:,0]*phi[:,0]).sum()
        kinetic=self.kernel.total_kinetic_energy(a)
        return dict(internal=internal,potential=potential,boundary=boundary,kinetic=kinetic,
                    total=internal+potential+kinetic+boundary,geopotential_interface=phi)

    def boundary_total_energy(self,s):return self.energy_terms(s)['total']

    def energy_gradients(self,s):
        a=self._adapter(s);gs,gp,gc=self.kernel.mixture_gradients(a,s.water_content)
        db=s.column.B_interface.diff()
        gp+=(s.mesh.cell_area[:,None]*db*self.kernel.cell_kinetic_energy(a)/self.kernel.gravity).sum(-1,keepdim=True)
        gu=self.kernel.kinetic_energy_gradient_u(a)
        return MoistState(gu,gp,gs,gc,s.mesh,s.column)

    def boundary_energy_rate(self,s,rhs):
        grad=self.energy_gradients(s)
        return sum((getattr(grad,k)*getattr(rhs,k)).sum() for k in ('u_e','ps','entropy_content','water_content'))
