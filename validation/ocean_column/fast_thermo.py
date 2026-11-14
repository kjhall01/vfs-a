"""Call-local thermodynamic work for the unchanged JMD95 mixing Hamiltonian.

No persistent model cache, state clipping, approximate work surrogate or forward
AD. Warm inversions retain the original residual tests and bracket fallback.
"""
from dataclasses import replace
import math
import torch
from vfs.ocean.stratified.thermodynamics import polynomial, SurfaceGibbs
from vfs.ocean.stratified.equations import validate
from vfs.ocean.state import OceanState
from vfs.ocean.jmd95_energy import jmd95_partials, _polynomial, JMD95Energy


def temperature_partial(s, t, pressure_dbar):
    """JMD95 temperature partial only; checked against the full kernel."""
    r0, r0t = _polynomial(t, (999.842594, 6.793952e-2, -9.095290e-3,
                             1.001685e-4, -1.120083e-6, 6.536332e-9))
    r1, r1t = _polynomial(t, (.824493, -4.0899e-3, 7.6438e-5,
                             -8.2467e-7, 5.3875e-9))
    r15, r15t = _polynomial(t, (-5.72466e-3, 1.0227e-4, -1.6546e-6))
    s15 = s.pow(1.5)
    rho_surface = r0 + s*r1 + s15*r15 + 4.8314e-4*s.square()
    rho_t = r0t + s*r1t + s15*r15t

    k0, k0t = _polynomial(t, (1.965933e4, 1.444304e2, -1.706103,
                             9.648704e-3, -4.190253e-5))
    k1, k1t = _polynomial(t, (5.284855e1, -.3101089, 6.283263e-3, -5.084188e-5))
    k15, k15t = _polynomial(t, (.388664, 9.085835e-3, -4.619924e-4))
    a0, a0t = _polynomial(t, (3.186519, 2.212276e-2, -2.984642e-4, 1.956415e-6))
    a1, a1t = _polynomial(t, (6.704388e-3, -1.847318e-4, 2.059331e-7))
    b0, b0t = _polynomial(t, (2.102898e-4, -1.202016e-5, 1.394680e-7))
    b1, b1t = _polynomial(t, (-2.040237e-6, 6.128773e-8, 6.207323e-10))
    p = .1 * pressure_dbar  # polynomial pressure is bar
    a = a0 + s*a1 + 1.480266e-4*s15
    b = b0 + s*b1
    bulk = k0 + s*k1 + s15*k15 + p*a + p.square()*b
    bulk_t = k0t + s*k1t + s15*k15t + p*(a0t+s*a1t) + p.square()*(b0t+s*b1t)
    denominator = 1 - p/bulk
    # d(1-p/K)/dx = p K_x/K², except p itself contributes -0.1/K.
    common = rho_surface/denominator.square()
    dt = rho_t/denominator - common*p*bulk_t/bulk.square()
    return dt


class Caloric:
    """Fixed-salinity factors; preserve polynomial summation order."""
    def __init__(self, reference, salt):
        self.reference=reference;self.salt=salt
        d=reference.data;sa=d['ups']*salt
        self.factors=[sa.pow(1+.5*i)/d['sa_scale']**(.5*i) for i in range(len(d['salt_power_coefficients']))]
        self.log_entropy=d['entropy_log_coefficient']*torch.special.xlogy(sa,sa/35.16504)
    def evaluate(self,theta):
        kernel=getattr(self.reference,'_compiled_caloric',None)
        return self._values(theta) if kernel is None else kernel(self,theta)
    def _values(self,theta):
        r=self.reference;d=r.data;t=r.to_90*theta
        y=(t-d['temperature_midpoint'])/d['temperature_halfspan']
        g,gt,gtt=polynomial(y,d['pure_power_coefficients'])
        gt,gtt=gt/d['temperature_halfspan'],gtt/d['temperature_halfspan']**2
        for factor,row in zip(self.factors,d['salt_power_coefficients']):
            c,ct,ctt=polynomial(y,row)
            gt=gt+factor*(ct/d['temperature_halfspan'])
            gtt=gtt+factor*(ctt/d['temperature_halfspan']**2)
            g=g+factor*c
        kelvin=t+273.15
        return dict(h0=g-kelvin*gt,h_theta=-kelvin*gtt*r.to_90,
                    entropy=-gt-self.log_entropy,entropy_theta=-gtt*r.to_90)
    def inverse(self,target,guess,kind):
        r=self.reference;d=r.data
        if not torch.isfinite(target).all():raise ValueError('nonfinite caloric target')
        # Original evaluator enforces salt/dtype/domain checks independently.
        lo=torch.full_like(target,d['temperature_ITS90_range_C'][0]/r.to_90)
        hi=torch.full_like(target,d['temperature_ITS90_range_C'][1]/r.to_90)
        domain_lo,domain_hi=lo,hi
        low=r.evaluate(self.salt,lo)[kind];high=r.evaluate(self.salt,hi)[kind]
        if ((target<low)|(target>high)).any():raise ValueError('caloric target outside reference domain')
        theta=torch.where(torch.isfinite(guess)&(guess>=lo)&(guess<=hi),guess,lo+(hi-lo)*(target-low)/(high-low))
        offset=4000. if kind=='h0' else 20.
        tolerance=128*torch.finfo(target.dtype).eps*(target.abs()+offset)
        for _ in range(13):
            values=self.evaluate(theta);error=values[kind]-target
            if (error.abs()<=tolerance).all():
                # Never accept solely on the cheaper kernel's residual.
                # One Newton polishing step also retains the implicit first
                # derivative when the warm guess already matches the target.
                polished=theta-error/values['h_theta' if kind=='h0' else 'entropy_theta']
                if ((polished>=domain_lo)&(polished<=domain_hi)).all():
                    exact=r.evaluate(self.salt,polished)[kind]-target
                    if (exact.abs()<=tolerance).all():return polished
                error=r.evaluate(self.salt,theta)[kind]-target
            lo=torch.where(error<0,theta,lo);hi=torch.where(error>0,theta,hi)
            proposal=theta-error/values['h_theta' if kind=='h0' else 'entropy_theta']
            theta=torch.where((proposal>=lo)&(proposal<=hi),proposal,.5*(lo+hi))
        raise FloatingPointError('warm caloric inversion failed original residual check')


class MixingThermodynamics:
    """Explicit substep workspace; valid only for this fixed layer geometry."""
    def __init__(self,ocean,physical):
        if type(ocean.reference) is not SurfaceGibbs or type(ocean.energy.base) is not JMD95Energy:
            raise TypeError('fast mixing requires the unmodified SurfaceGibbs/JMD95 kernels')
        self.ocean=ocean;self.reference=ocean.reference;self.base=ocean.energy.base
        options=getattr(self.reference,'_compile_options',None)
        if options is not None and not hasattr(self.reference,'_compiled_caloric'):
            self.reference._compiled_caloric=torch.compile(Caloric._values,**options)
            self.base._compiled_temperature_partial=torch.compile(temperature_partial,**options)
        self.temperature_kernel=getattr(self.base,'_compiled_temperature_partial',temperature_partial)
        self.node,self.weight=self.base._quadrature(physical.h)
        self.z=self.base.height(physical)[...,None]+physical.h[...,None]*(self.node-.5)
        self.pressure=-self.base.rho0*self.base.gravity*(self.z[...,None]*self.node)/1e4
        t,s=physical.temperature()[...,None,None],physical.salinity()[...,None,None]
        self.rho_ref,_,_,self.rp_ref=self.base._partials(torch.full_like(s,self.base.S0),torch.full_like(t,self.base.T0),self.pressure)
    def parcel(self,theta,salt):
        b=self.base;z=self.z;x=z[...,None]*self.node
        rho,rt,rs,rp=b._partials(salt[...,None,None],theta[...,None,None],self.pressure)
        f=b.gravity/b.rho0*(rho-self.rho_ref);fz=-b.gravity**2/1e4*(rp-self.rp_ref)
        return (z*(f*self.weight).sum(-1),z*(b.gravity/b.rho0*rt*self.weight).sum(-1),
                z*(b.gravity/b.rho0*rs*self.weight).sum(-1),((f+x*fz)*self.weight).sum(-1))
    def temperature_derivative(self,theta,salt):
        # Reference density has zero temperature derivative; never evaluate it.
        rt=self.temperature_kernel(salt[...,None,None],theta[...,None,None],self.pressure)
        return (self.z*(self.base.gravity/self.base.rho0*rt*self.weight).sum(-1)*self.weight).sum(-1)
    def parts(self,physical):
        t,s=physical.temperature(),physical.salinity()
        return (self.parcel(t,s)[0]*self.weight).sum(-1),self.reference.evaluate(s,t)
    def conjugates(self,t,s):
        ref=self.reference.evaluate(s,t);_,pt,ps,_=self.parcel(t,s)
        a=(pt*self.weight).sum(-1)/ref['h_theta']
        return a,(ps*self.weight).sum(-1)-a*ref['h_s']
    def physical(self,state,guess):
        salt=state.salt_h/state.h
        theta=Caloric(self.reference,salt).inverse(state.entropy_h/state.h,guess,'entropy')
        physical=OceanState(state.u_e,state.h,state.h*theta,state.salt_h,state.tracer_h,mesh=state.mesh,column=state.column)
        return validate(self.ocean.base,state,self.ocean.energy,physical=physical)
    def path_work(self,closure,physical,H0,fh,fs,mid,*,after=None):
        h=physical.h;area=self.ocean.energy.mass_area(physical)
        zero=torch.zeros_like(h[:,:1]);div=lambda f:torch.cat((f,zero),-1)-torch.cat((zero,f),-1)
        ds,dh=div(fs)/h,div(fh)/h;s0,t0=physical.salinity(),physical.temperature()
        a_bar,b_bar=torch.zeros_like(h),torch.zeros_like(h)
        for fraction,w in zip(closure._path_nodes,closure._path_weights):
            salt=s0+fraction*ds
            theta=Caloric(self.reference,salt).inverse(H0+fraction*dh,t0,'h0')
            a,b=self.conjugates(theta,salt);a_bar=a_bar+w*a;b_bar=b_bar+w*b
        a_mid,b_mid=mid;alpha=h[:,:-1]/(h[:,:-1]+h[:,1:])
        face=lambda q:(1-alpha)*q[:,:-1]+alpha*q[:,1:]
        interface=area*((a_bar[:,:-1]-a_bar[:,1:])*fh+(b_bar[:,:-1]-b_bar[:,1:])*fs)
        paired=torch.cat((alpha*interface,zero),-1)+torch.cat((zero,(1-alpha)*interface),-1)
        return paired+div(area*(face(a_bar-a_mid)*fh+face(b_bar-b_mid)*fs))
    def heat(self,state,physical,heat):
        if not torch.any(heat!=0):return state,torch.zeros_like(heat),physical
        theta,residual=self.heat_temperature(physical,heat)
        eta=self.reference.evaluate(physical.salinity(),theta,entropy_derivatives=True)['entropy']
        end=replace(state,entropy_h=torch.where(heat!=0,state.h*eta,state.entropy_h))
        return end,residual,self.physical(end,theta)

    def heat_temperature(self,physical,heat,*,allow_cooling=False):
        """Solve only the finite thermal increment, without changing coordinates."""
        t0,s=physical.temperature(),physical.salinity();cal=Caloric(self.reference,s)
        def derivative(t):return cal.evaluate(t)['h_theta']+self.temperature_derivative(t,s)
        nodes=(.5-math.sqrt(15)/10,.5,.5+math.sqrt(15)/10);weights=(5/18,4/9,5/18)
        delta=heat/derivative(t0)
        tolerance=128*torch.finfo(heat.dtype).eps*heat.abs().clamp_min(torch.finfo(heat.dtype).tiny)
        for iteration in range(7):
            gained=delta*sum(w*derivative(t0+n*delta) for n,w in zip(nodes,weights))
            residual=gained-heat
            if (residual.abs()<=tolerance).all():break
            if iteration==6:raise FloatingPointError('fast heat increment residual failed')
            delta=delta-residual/derivative(t0+delta)
        if not torch.isfinite(delta).all() or (not allow_cooling and (delta<0).any()):raise FloatingPointError('invalid fast heating increment')
        return t0+delta,residual
