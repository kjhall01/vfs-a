"""Research semi-implicit mixing: predicted coefficients, matched energy transfers.

Momentum is solved once. Scalar right-hand sides share one factorization per
proposal. TKE diffusion and linearized dissipation use fixed predicted k; actual mixing
work is funded by a signed linear TKE solve, never by clipping the state.
Unfunded scalar proposals are reduced and re-solved with positive implicit transport.
After bounded retries, remaining unfunded columns defer scalar mixing explicitly.
"""
from dataclasses import dataclass,replace
import math
import torch
from path_work import PathWorkClosure
from coupled import CoupledReport
from vfs.ocean.stratified.tke import diffuse,positive_sqrt
from vfs.ocean.stratified.heating import thermal_parts,apply_heat

class Factor:
    """Batched tridiagonal factors, reused for arbitrary trailing RHS fields."""
    def __init__(self,diagonal,off):
        self.off=off;self.pivots=[];self.upper=[]
        for j in range(diagonal.shape[-1]):
            d=diagonal[:,j] if j==0 else diagonal[:,j]-off[:,j-1]*self.upper[-1]
            if (d<=0).any() or not torch.isfinite(d).all():raise FloatingPointError('invalid pivot')
            self.pivots.append(d)
            if j<off.shape[-1]:self.upper.append(off[:,j]/d)
    def solve(self,rhs):
        scalar=rhs.ndim==2
        if scalar:rhs=rhs[...,None]
        y=[]
        for j,d in enumerate(self.pivots):
            q=rhs[:,j] if j==0 else rhs[:,j]-self.off[:,j-1,None]*y[-1]
            y.append(q/d[:,None])
        x=[y[-1]]
        for j in range(len(y)-2,-1,-1):x.append(y[j]-self.upper[j][:,None]*x[-1])
        out=torch.stack(x[::-1],1)
        return out[...,0] if scalar else out

def diffuse_many(values,mass,g,dt):
    zero=torch.zeros_like(mass[:,:1]);off=-dt*g
    diagonal=mass+torch.cat((dt*g,zero),-1)+torch.cat((zero,dt*g),-1)
    flux=dt*g[...,None]*(values[:,1:]-values[:,:-1]);z=torch.zeros_like(values[:,:1])
    return values+Factor(diagonal,off).solve(torch.cat((flux,z),1)-torch.cat((z,flux),1))

@dataclass(frozen=True)
class LaggedClosure(PathWorkClosure):
    fast_thermodynamics: bool = False
    max_reductions: int = 8
    reduction: float = .9
    predictor_steps: int = 1
    funding: str = "split"
    def __post_init__(self):
        super().__post_init__()
        if self.max_reductions<0 or self.predictor_steps<0 or self.funding not in ('column','local','split') or not 0<self.reduction<1:raise ValueError('invalid scalar retry controls')

    def apply(self,ocean,state,tke_h,dt):
        return self._apply(ocean,state,tke_h,dt)[:3]

    def _workspace(self,ocean,physical):
        if not self.fast_thermodynamics:return None
        from fast_thermo import MixingThermodynamics
        return MixingThermodynamics(ocean,physical)

    def _enthalpy_temperature(self,energy,workbuf,salt,enthalpy,guess):
        if workbuf:
            from fast_thermo import Caloric
            return Caloric(energy.reference,salt).inverse(enthalpy,guess,'h0')
        return energy.reference.temperature_from_enthalpy(salt,enthalpy)

    def _baseline_entropy(self,energy,workbuf,salt,theta,enthalpy):
        if workbuf:
            from fast_thermo import Caloric
            recovered=Caloric(energy.reference,salt).inverse(enthalpy,theta,'h0')
        else:recovered=energy.reference.temperature_from_enthalpy(salt,enthalpy)
        return energy.reference.evaluate(salt,recovered,entropy_derivatives=True)['entropy']

    def _mixed_state(self,ocean,state,u,salt_h,tracer,theta,baseline,workbuf):
        eta=ocean.reference.evaluate(salt_h/state.h,theta,entropy_derivatives=True)['entropy']
        mixed=replace(state,u_e=u,salt_h=salt_h,entropy_h=state.entropy_h+state.h*(eta-baseline),tracer_h=tracer)
        after=workbuf.physical(mixed,theta) if workbuf else ocean.check_state(mixed)
        return mixed,after

    def _entropy_content(self,ocean,state):
        return state.entropy_h

    def _apply(self,ocean,state,tke_h,dt,physical=None):
        if not math.isfinite(dt) or dt<=0:raise ValueError('positive finite dt required')
        if physical is None:physical=ocean.check_state(state)
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
        workbuf=self._workspace(ocean,physical)
        parts=workbuf.parts if workbuf else lambda p:thermal_parts(ocean,p)
        psi0,ref0=parts(physical)
        s0=physical.salinity();theta0=physical.temperature();H0=ref0['h0']
        node,weight=energy.base._quadrature(h)
        z=energy.base.height(physical)[...,None]+h[...,None]*(node-.5)
        alpha=h[:,:-1]/(h[:,:-1]+h[:,1:])
        if workbuf:
            from fast_thermo import Caloric
        eta_round=self._baseline_entropy(energy,workbuf,s0,theta0,H0)

        def resistance(diffusivity, thickness):
            upper,lower=diffusivity[:,:-1],diffusivity[:,1:]
            denominator=thickness[:,:-1]*lower+thickness[:,1:]*upper
            # Two half-layer diffusive resistances in series; exact zero is
            # handled without evaluating a singular reciprocal or adding a floor.
            return 2*upper*lower/torch.where(denominator>0,denominator,torch.ones_like(denominator))

        momentum_cache=[]
        trial_evaluations=0
        def trial(k, scalar_scale):
            nonlocal trial_evaluations
            trial_evaluations+=1
            km=self.c_mu*self.mixing_length*positive_sqrt(k)
            g=.5*(km[:,:-1]+km[:,1:])/(.5*(h[:,:-1]+h[:,1:]))
            if self.interface_scheme=="resistance":g=resistance(km,h)
            ek=mesh.cell_to_edge(km)
            eg=ocean.column.rho0*(mesh.dc_edge*mesh.dv_edge)[:,None]*.5*(ek[:,:-1]+ek[:,1:])/(.5*(he[:,:-1]+he[:,1:]))
            if self.interface_scheme=="resistance":
                eg=ocean.column.rho0*(mesh.dc_edge*mesh.dv_edge)[:,None]*resistance(ek,he)
            if ocean.domain is not None:eg=eg*ocean.domain.active_edge[:,None]
            if not momentum_cache:
                u=diffuse(state.u_e,metric,eg,dt)
                es=dt*eg*(u[:,1:]-u[:,:-1]).square();ez=torch.zeros_like(u[:,:1])
                shear=allocate(.5*(torch.cat((es,ez),-1)+torch.cat((ez,es),-1)))
                production=shear+allocate(.5*metric*(u-state.u_e).square())
                momentum_cache.append((u,production))
            u,production=momentum_cache[0]
            gh=g/self.prandtl*scalar_scale if self.mix_scalars else torch.zeros_like(g)
            values=[s0,H0]
            if state.tracer_h is not None:
                values.extend((state.tracer_h/h[...,None]).unbind(-1))
            solutions=diffuse_many(torch.stack(values,-1),h,gh,dt)
            salt,enthalpy=solutions[...,0],solutions[...,1]
            fs=dt*gh*(salt[:,1:]-salt[:,:-1]);fh=dt*gh*(enthalpy[:,1:]-enthalpy[:,:-1])
            salt_h=state.salt_h+div(fs);salt=salt_h/h
            enthalpy=H0+div(fh)/h
            theta=self._enthalpy_temperature(energy,workbuf,salt,enthalpy,theta0)
            tracer=state.tracer_h
            if tracer is not None:
                q=solutions[...,2:]
                f=dt*gh[...,None]*(q[:,1:]-q[:,:-1])
                tz=torch.zeros_like(tracer[:,:1])
                tracer=tracer+torch.cat((f,tz),1)-torch.cat((tz,f),1)
            mixed,after=self._mixed_state(ocean,state,u,salt_h,tracer,theta,eta_round,workbuf)
            psi1,ref1=parts(after)
            change=mass*((psi1-psi0)+(ref1['h0']-H0))
            # Centred conjugate energy flux defines the local work allocation;
            # the actual finite H difference retains its nonlinear remainder.
            tm=.5*(theta0+after.temperature());sm=.5*(s0+salt)
            if workbuf:a,b=workbuf.conjugates(tm,sm)
            else:
                ref=energy.reference.evaluate(sm,tm)
                _,pt,ps,_=energy.base.parcel_potential(tm[...,None],sm[...,None],z)
                a=(pt*weight).sum(-1)/ref['h_theta'];b=(ps*weight).sum(-1)-a*ref['h_s']
            face=lambda q:(1-alpha)*q[:,:-1]+alpha*q[:,1:]
            flux=area*(fh+face(a)*fh+face(b)*fs)
            # The h0 transfer is conservative by construction. Cancel it
            # algebraically, rather than making TKE pay inverse-roundtrip noise.
            # Full-H acceptance below independently checks that noise.
            if workbuf:work=workbuf.path_work(self,physical,H0,fh,fs,(a,b),after=after)
            else:
                work=mass*(psi1-psi0)-div(area*(face(a)*fh+face(b)*fs))
                work=self._buoyancy_work(ocean,physical,after,fh,fs,work)
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
            return mixed,production,work,flux,conductance,diss,diag,-dt*conductance,reconstruction,after

        scalar_scale=torch.ones_like(h[:,:-1]);limited=torch.zeros_like(scalar_scale,dtype=torch.bool)
        deferred=torch.zeros_like(h[:,:1],dtype=torch.bool)
        fixed_k=k0
        for _ in range(self.predictor_steps):
            momentum_cache.clear()
            _,p,b,_,_,_,a,off,_,_=trial(fixed_k,scalar_scale)
            denom=torch.where(fixed_k>0,fixed_k,torch.ones_like(fixed_k))
            fixed_k=Factor(a+b.clamp_min(0)/denom,off).solve(mass*k0+p+(-b).clamp_min(0))
            if not torch.isfinite(fixed_k).all() or (fixed_k<0).any():raise FloatingPointError('invalid coefficient predictor')
        momentum_cache.clear()
        factor=None
        for attempt in range(self.max_reductions+2):
            mixed,production,work,flux,g,diss,diagonal,off,reconstruction,after=trial(fixed_k,scalar_scale)
            if factor is None:
                factor=Factor(mass+add(dt*g) if self.funding=='split' else diagonal,off)
                transported=factor.solve(mass*k0+production)
            if self.funding=='split':
                # Ordered split: transport produced TKE first, then pay the
                # actual local buoyancy work and dissipate the remainder.
                final=(mass*transported-work)/(mass+diss)
            else:final=factor.solve(mass*k0+production-work)
            if not torch.isfinite(final).all():raise FloatingPointError('nonfinite lagged TKE')
            bad=final<0
            if not bad.any():break
            if attempt<self.max_reductions:
                if self.funding=='split':
                    ratio=torch.where(bad,(mass*transported/torch.where(work>0,work,torch.ones_like(work))).clamp(0,1)*self.reduction,torch.ones_like(work))
                    face_ratio=torch.minimum(ratio[:,:-1],ratio[:,1:])
                    interfaces=face_ratio<1
                    scalar_scale=scalar_scale*face_ratio
                elif self.funding=='local':
                    interfaces=bad[:,:-1]|bad[:,1:]
                    scalar_scale=torch.where(interfaces,scalar_scale*self.reduction,scalar_scale)
                else:
                    # Signed TKE response includes transport across the column.
                    free=factor.solve(mass*k0+production)
                    debit=factor.solve(work)
                    ratio=torch.where(debit>0,free/torch.where(debit>0,debit,torch.ones_like(debit)),torch.ones_like(debit))
                    ratio=ratio.amin(-1,keepdim=True).clamp(0,1)*self.reduction
                    interfaces=bad.any(-1,keepdim=True).expand_as(scalar_scale)
                    scalar_scale=torch.where(interfaces,scalar_scale*ratio,scalar_scale)
                limited=limited|interfaces
            elif attempt==self.max_reductions:
                deferred=bad.any(-1,keepdim=True)
                scalar_scale=torch.where(deferred,torch.zeros_like(scalar_scale),scalar_scale)
            else:raise FloatingPointError('lagged TKE remains unfunded with scalar mixing deferred')
        transport_k=transported if self.funding=='split' else final
        tflux=dt*g*(transport_k[:,1:]-transport_k[:,:-1])
        actual_residual=mass*(final-k0)-production+work+diss*final-div(tflux)
        budget_scale=mass*self.budget_atol+self.budget_rtol*(mass*k0+production+work.abs())
        budget_error=(actual_residual.abs()/budget_scale).max()
        if not torch.isfinite(budget_error) or budget_error>1:raise FloatingPointError('lagged local budget failed')
        heat=diss*final
        if workbuf:end,thermal_residual,final_physical=workbuf.heat(mixed,after,heat/mass)
        else:
            end,thermal_residual=apply_heat(ocean,mixed,heat/mass)
            final_physical=ocean.check_state(end)
        psi2,ref2=parts(final_physical)
        local_ke=allocate(.5*metric*(end.u_e-state.u_e)*(end.u_e+state.u_e))
        exchange=local_ke+production
        column=(mass*((psi2-psi0)+(ref2['h0']-H0))+local_ke+mass*(final-k0)).sum(-1)-exchange.sum(-1)
        entropy=(area*(self._entropy_content(ocean,end)-self._entropy_content(ocean,state))).sum(-1)
        tol=1024*torch.finfo(h.dtype).eps*(mass*(psi0.abs()+H0.abs())+area*tke_h+local_ke.abs()).sum(-1).clamp_min(1)
        if (column.abs()>tol).any():raise FloatingPointError('coupled full-H closure failed')
        stol=512*torch.finfo(h.dtype).eps*(area*self._entropy_content(ocean,state).abs()).sum(-1).clamp_min(1)
        if (entropy < -stol).any():raise ValueError('coupled update decreases column entropy')
        tkeflux=dt*g*(transport_k[:,1:]-transport_k[:,:-1])
        residual=mass*(final-k0)-production+work+heat-div(tkeflux)
        report=CoupledReport(trial_evaluations,h.new_zeros(()),budget_error,column.sum(),column,exchange.sum(-1),work,flux,tkeflux,heat,entropy,residual,reconstruction,
            {'limited_pairs':limited.sum().to(h.dtype),'semi_trials':h.new_tensor(trial_evaluations),
             'limited_interfaces':limited.sum().to(h.dtype),'deferred_columns':deferred.sum().to(h.dtype),
             'minimum_fraction':scalar_scale.min() if scalar_scale.numel() else h.new_ones(())})
        return end,h*final,report,final_physical

@dataclass(frozen=True)
class SubcycledLaggedClosure(LaggedClosure):
    """Explicit, fixed mixing substeps; outer forcing/dynamics are unchanged."""
    substeps: int = 4
    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.substeps,int) or self.substeps<1:raise ValueError('positive integer substeps required')
    def apply(self,ocean,state,tke_h,dt):
        reports=[];physical=None
        for _ in range(self.substeps):
            state,tke_h,r,physical=self._apply(ocean,state,tke_h,dt/self.substeps,
                physical=physical if self.fast_thermodynamics else None)
            reports.append(r)
        summed={name:sum(getattr(r,name) for r in reports) for name in ('global_energy_residual','column_energy_residual','kinetic_column_exchange','buoyancy_layer_work','buoyancy_energy_flux','tke_energy_flux','dissipated_heat','entropy_change','local_tke_residual','caloric_reconstruction_residual')}
        metrics={key:(torch.stack([r.mixing_limiter[key] for r in reports]).min() if key=='minimum_fraction' else sum(r.mixing_limiter[key] for r in reports)) for key in reports[0].mixing_limiter}
        metrics['mixing_substeps']=tke_h.new_tensor(self.substeps)
        report=replace(reports[-1],iterations=sum(r.iterations for r in reports),budget_error=torch.stack([r.budget_error for r in reports]).max(),mixing_limiter=metrics,**summed)
        return state,tke_h,report
