"""Research-only simultaneous column mixing/TKE solve, wet columns only.

End TKE sets momentum, scalar and TKE diffusivities and dissipation. Positive
Patankar iteration is checked against the ACTUAL nonlinear-work residual.
No production defaults, clipping, hidden substeps or forward autograd.
"""
from dataclasses import dataclass, replace
import math
import torch
from vfs.ocean.stratified import TKEClosure
from vfs.ocean.stratified.tke import diffuse, positive_sqrt
from vfs.ocean.stratified.heating import thermal_parts, apply_heat


def tridiagonal(diagonal, off, rhs):
    """Batched symmetric tridiagonal solve; no dense matrix or column loop."""
    cp=[];yp=[]
    for j in range(rhs.shape[-1]):
        d=diagonal[:,j] if j==0 else diagonal[:,j]-off[:,j-1]*cp[-1]
        if (d<=0).any() or not torch.isfinite(d).all():raise FloatingPointError('nonpositive column pivot')
        y=rhs[:,j] if j==0 else rhs[:,j]-off[:,j-1]*yp[-1]
        yp.append(y/d)
        if j<rhs.shape[-1]-1:cp.append(off[:,j]/d)
    out=[yp[-1]]
    for j in range(rhs.shape[-1]-2,-1,-1):out.append(yp[j]-cp[j]*out[-1])
    return torch.stack(out[::-1],-1)


@dataclass(frozen=True)
class CoupledReport:
    iterations: int
    fixed_point_error: torch.Tensor
    budget_error: torch.Tensor
    global_energy_residual: torch.Tensor
    column_energy_residual: torch.Tensor
    kinetic_column_exchange: torch.Tensor
    buoyancy_layer_work: torch.Tensor
    buoyancy_energy_flux: torch.Tensor
    tke_energy_flux: torch.Tensor
    dissipated_heat: torch.Tensor
    entropy_change: torch.Tensor
    local_tke_residual: torch.Tensor
    caloric_reconstruction_residual: torch.Tensor
    mixing_limiter: dict  # Compatibility with the offline diagnostic accumulator.


@dataclass(frozen=True)
class CoupledClosure(TKEClosure):
    """End-coefficient backward-Euler candidate; no calibrated closure claim.

    Iterates do not advance time or commit ledgers. Tolerances control the
    nonlinear solve, not positivity floors. Failure preserves caller inputs.
    """
    max_iterations: int = 80
    rtol: float = 1e-6
    atol: float = 1e-12
    relaxation: float = .7
    accelerate: bool = True
    interface_scheme: str = "resistance"
    budget_atol: float = 1e-13
    budget_rtol: float = 1e-9

    def __post_init__(self):
        super().__post_init__()
        if self.interface_scheme not in ("arithmetic","resistance"):raise ValueError("unknown interface scheme")
        if self.max_iterations<1:raise ValueError('positive iteration limit required')
        if not (math.isfinite(self.rtol) and self.rtol>0 and math.isfinite(self.atol) and self.atol>0):raise ValueError('positive finite solve tolerances required')
        if self.budget_atol<=0 or self.budget_rtol<=0 or not math.isfinite(self.budget_atol+self.budget_rtol):raise ValueError('positive finite budget tolerances required')
        if not 0<self.relaxation<=1:raise ValueError('relaxation must lie in (0,1]')

    def apply(self,ocean,state,tke_h,dt):
        if not math.isfinite(dt) or dt<=0:raise ValueError('positive finite dt required')
        physical=ocean.check_state(state)
        h=state.h;mesh=state.mesh;energy=ocean.energy
        if tke_h.shape!=h.shape or tke_h.dtype!=h.dtype or tke_h.device!=h.device:raise ValueError('TKE must match thickness')
        if not torch.isfinite(tke_h).all() or (tke_h<0).any():raise ValueError('TKE must be finite and nonnegative')
        area=energy.mass_area(state);mass=area*h
        if (area<=0).any():raise ValueError('coupled research solver supports fully wet columns only')
        k0=tke_h/h;guess=k0
        zero=h.new_zeros(h.shape[0],1)
        div=lambda f:torch.cat((f,zero),-1)-torch.cat((zero,f),-1)
        add=lambda f:torch.cat((f,zero),-1)+torch.cat((zero,f),-1)
        he=mesh.cell_to_edge(h);mechanical=energy.base.mechanical
        metric=mechanical.metric(physical)
        allocate=lambda e:h*mechanical.interpolation_transpose(mesh,e/he)
        psi0,ref0=thermal_parts(ocean,physical)
        s0=physical.salinity();theta0=physical.temperature();H0=ref0['h0']
        node,weight=energy.base._quadrature(h)
        z=energy.base.height(physical)[...,None]+h[...,None]*(node-.5)
        alpha=h[:,:-1]/(h[:,:-1]+h[:,1:])
        theta_round=energy.reference.temperature_from_enthalpy(s0,H0)
        eta_round=energy.reference.evaluate(s0,theta_round,entropy_derivatives=True)['entropy']

        def resistance(diffusivity, thickness):
            upper,lower=diffusivity[:,:-1],diffusivity[:,1:]
            denominator=thickness[:,:-1]*lower+thickness[:,1:]*upper
            # Two half-layer diffusive resistances in series; exact zero is
            # handled without evaluating a singular reciprocal or adding a floor.
            return 2*upper*lower/torch.where(denominator>0,denominator,torch.ones_like(denominator))

        def trial(k):
            km=self.c_mu*self.mixing_length*positive_sqrt(k)
            g=.5*(km[:,:-1]+km[:,1:])/(.5*(h[:,:-1]+h[:,1:]))
            if self.interface_scheme=="resistance":g=resistance(km,h)
            ek=mesh.cell_to_edge(km)
            eg=ocean.column.rho0*(mesh.dc_edge*mesh.dv_edge)[:,None]*.5*(ek[:,:-1]+ek[:,1:])/(.5*(he[:,:-1]+he[:,1:]))
            if self.interface_scheme=="resistance":
                eg=ocean.column.rho0*(mesh.dc_edge*mesh.dv_edge)[:,None]*resistance(ek,he)
            if ocean.domain is not None:eg=eg*ocean.domain.active_edge[:,None]
            u=diffuse(state.u_e,metric,eg,dt)
            es=dt*eg*(u[:,1:]-u[:,:-1]).square();ez=torch.zeros_like(u[:,:1])
            shear=allocate(.5*(torch.cat((es,ez),-1)+torch.cat((ez,es),-1)))
            production=shear+allocate(.5*metric*(u-state.u_e).square())
            gh=g/self.prandtl if self.mix_scalars else torch.zeros_like(g)
            salt=diffuse(s0,h,gh,dt);enthalpy=diffuse(H0,h,gh,dt)
            fs=dt*gh*(salt[:,1:]-salt[:,:-1]);fh=dt*gh*(enthalpy[:,1:]-enthalpy[:,:-1])
            salt_h=state.salt_h+div(fs);salt=salt_h/h
            enthalpy=H0+div(fh)/h
            theta=energy.reference.temperature_from_enthalpy(salt,enthalpy)
            eta=energy.reference.evaluate(salt,theta,entropy_derivatives=True)['entropy']
            tracer=state.tracer_h
            if tracer is not None:
                fields=[]
                for j in range(tracer.shape[-1]):
                    q=diffuse(tracer[...,j]/h,h,gh,dt)
                    fields.append(tracer[...,j]+div(dt*gh*(q[:,1:]-q[:,:-1])))
                tracer=torch.stack(fields,-1) if fields else tracer
            mixed=replace(state,u_e=u,salt_h=salt_h,entropy_h=state.entropy_h+h*(eta-eta_round),tracer_h=tracer)
            after=ocean.check_state(mixed);psi1,ref1=thermal_parts(ocean,after)
            change=mass*((psi1-psi0)+(ref1['h0']-H0))
            # Centred conjugate energy flux defines the local work allocation;
            # the actual finite H difference retains its nonlinear remainder.
            tm=.5*(theta0+after.temperature());sm=.5*(s0+salt)
            ref=energy.reference.evaluate(sm,tm)
            _,pt,ps,_=energy.base.parcel_potential(tm[...,None],sm[...,None],z)
            a=(pt*weight).sum(-1)/ref['h_theta'];b=(ps*weight).sum(-1)-a*ref['h_s']
            face=lambda q:(1-alpha)*q[:,:-1]+alpha*q[:,1:]
            flux=area*(fh+face(a)*fh+face(b)*fs)
            # The h0 transfer is conservative by construction. Cancel it
            # algebraically, rather than making TKE pay inverse-roundtrip noise.
            # Full-H acceptance below independently checks that noise.
            work=mass*(psi1-psi0)-div(area*(face(a)*fh+face(b)*fs))
            reconstruction=change-div(flux)-work
            if self.interface_scheme=="resistance":
                # For TKE itself, sqrt(k) grad(k) = (2/3) grad(k**1.5).
                # The divided difference stays finite when either side is zero.
                root=positive_sqrt(k);den=root[:,:-1]+root[:,1:]
                mean=(2/3)*(k[:,:-1]+root[:,:-1]*root[:,1:]+k[:,1:])/torch.where(den>0,den,torch.ones_like(den))
                gk=self.c_mu*self.mixing_length*mean/(.5*(h[:,:-1]+h[:,1:]))
            else:gk=g
            conductance=area*gk/self.sigma_k
            diss=dt*mass*self.c_epsilon*positive_sqrt(k)/self.mixing_length
            diag=mass+add(dt*conductance)+diss
            return mixed,production,work,flux,conductance,diss,diag,-dt*conductance,reconstruction

        previous=None;accelerations=0
        budget_error=h.new_tensor(float("inf"));correction=budget_error
        for iteration in range(1,self.max_iterations+1):
            mixed,production,work,flux,g,diss,diagonal,off,reconstruction=trial(guess)
            # Positive source/sink decomposition is algebra, not state clipping.
            positive=work.clamp_min(0);source=(-work).clamp_min(0)
            denom=torch.where(guess>0,guess,torch.ones_like(guess))
            if ((guess==0)&(positive>0)).any():
                # A zero layer may receive TKE through represented transport.
                # This is an uncommitted warm start, not a physical energy floor.
                warm=tridiagonal(diagonal,off,mass*k0+production+source)
                if ((warm<=0)&(positive>0)).any():raise FloatingPointError('no positive transported TKE warm start')
                guess=warm;previous=None
                continue
            predicted=tridiagonal(diagonal+positive/denom,off,mass*k0+production+source)
            if not torch.isfinite(predicted).all() or (predicted<0).any():raise FloatingPointError('positive coupled iteration failed')
            scale=self.atol+self.rtol*torch.maximum(k0,torch.maximum(guess,predicted))
            error=((predicted-guess).abs()/scale).max()
            accepted=False
            if error<=1:
                # Evaluate the ACTUAL energy-equation residual. A signed direct
                # solve minimizes it, but may return tiny negative k at exhausted
                # layers. Never clip that solution: the positive iterate is an
                # alternative only when it independently meets the strict budget
                # residual test. A residual tolerance is not a TKE floor.
                direct=tridiagonal(diagonal,off,mass*k0+production-work)
                budget_scale=mass*self.budget_atol+self.budget_rtol*(mass*k0+production+work.abs())
                for final in (direct,predicted):
                    if not torch.isfinite(final).all() or (final<0).any():continue
                    correction=((final-guess).abs()/(self.atol+self.rtol*torch.maximum(k0,torch.maximum(guess,final)))).max()
                    tflux=dt*g*(final[:,1:]-final[:,:-1])
                    actual_residual=mass*(final-k0)-production+work+diss*final-div(tflux)
                    budget_error=(actual_residual.abs()/budget_scale).max()
                    if correction<=1 and budget_error<=1:
                        accepted=True
                        break
                if accepted:break
            next_guess=(1-self.relaxation)*guess+self.relaxation*predicted
            f=predicted-guess
            if self.accelerate and previous is not None:
                old_f,old_predicted=previous
                difference=(f-old_f)/scale
                denominator=difference.square().sum(-1,keepdim=True)
                beta=((f/scale)*difference).sum(-1,keepdim=True)/torch.where(denominator>0,denominator,torch.ones_like(denominator))
                extrapolated=predicted-beta*(predicted-old_predicted)
                # Reject unsafe ITERATES, never clip physical TKE or its budget.
                valid=(denominator>0)&torch.isfinite(beta)&(beta.abs()<=2)
                valid=valid&torch.isfinite(extrapolated).all(-1,keepdim=True)
                valid=valid&(extrapolated>=0).all(-1,keepdim=True)
                valid=valid&((extrapolated>0)|(predicted==0)).all(-1,keepdim=True)
                valid=valid&(extrapolated<=2*torch.maximum(guess,predicted)).all(-1,keepdim=True)
                next_guess=torch.where(valid,extrapolated,next_guess)
                accelerations+=int(valid.sum())
            previous=(f,predicted)
            guess=next_guess
        else:raise FloatingPointError(f'coupled column failed to converge in {self.max_iterations} iterations; iterate residual {float(error):.6g}, reconciled residual {float(correction):.6g}, budget residual {float(budget_error):.6g}')
        heat=diss*final
        end,thermal_residual=apply_heat(ocean,mixed,heat/mass)
        final_physical=ocean.check_state(end);psi2,ref2=thermal_parts(ocean,final_physical)
        local_ke=allocate(.5*metric*(end.u_e-state.u_e)*(end.u_e+state.u_e))
        exchange=local_ke+production
        column=(mass*((psi2-psi0)+(ref2['h0']-H0))+local_ke+mass*(final-k0)).sum(-1)-exchange.sum(-1)
        entropy=(area*(end.entropy_h-state.entropy_h)).sum(-1)
        tol=1024*torch.finfo(h.dtype).eps*(mass*(psi0.abs()+H0.abs())+area*tke_h+local_ke.abs()).sum(-1).clamp_min(1)
        if (column.abs()>tol).any():raise FloatingPointError('coupled full-H closure failed')
        stol=512*torch.finfo(h.dtype).eps*(area*state.entropy_h.abs()).sum(-1).clamp_min(1)
        if (entropy < -stol).any():raise ValueError('coupled update decreases column entropy')
        tkeflux=dt*g*(final[:,1:]-final[:,:-1])
        residual=mass*(final-k0)-production+work+heat-div(tkeflux)
        report=CoupledReport(iteration,correction,budget_error,column.sum(),column,exchange.sum(-1),work,flux,tkeflux,heat,entropy,residual,reconstruction,
            {'limited_pairs':h.new_zeros(()),'coupled_iterations':h.new_tensor(iteration),'accelerated_columns':h.new_tensor(accelerations),'scaled_residual_sum':correction,'budget_scaled_residual_sum':budget_error})
        return end,h*final,report
