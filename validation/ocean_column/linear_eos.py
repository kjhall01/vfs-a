"""Column-only linear-density and constant-cp thermodynamic controls.

No supported dycore defaults change. The same native energy metric and coupled
solver are used. Linear coefficients match JMD95 partials at T=12.5 IPTS-68 C,
SP=34.75, p=0 dbar; this does not match the complete nonlinear density profile.
"""
import torch
from vfs.ocean.jmd95_energy import JMD95Energy,jmd95_partials
from vfs.ocean.stratified import StratifiedOcean
from vfs.ocean.stratified.equations import EntropyEnergy


class LinearDensityEnergy(JMD95Energy):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        _,rt,rs,_=jmd95_partials(torch.tensor(34.75,dtype=torch.float64),torch.tensor(12.5,dtype=torch.float64),torch.tensor(0.,dtype=torch.float64))
        self.alpha=float(-rt/self.rho0);self.beta=float(rs/self.rho0)

    def density(self,state):
        return self.rho0*(1-self.alpha*(state.temperature()-self.T0)+self.beta*(state.salinity()-self.S0))

    def parcel_potential(self,temperature,salinity,z):
        anomaly=-self.alpha*(temperature-self.T0)+self.beta*(salinity-self.S0)
        return self.gravity*z*anomaly,-self.gravity*self.alpha*z, self.gravity*self.beta*z,self.gravity*anomaly


class ConstantHeatCapacity:
    """h0=cp*T, eta=cp*log((T+273.15)/273.15); salinity is conservative.

    Temperature uses the existing harness coordinate directly (IPTS-68 degrees).
    This idealized caloric control is not a TEOS/GSW replacement.
    """
    temperature_scale='IPTS-68'
    to_90=1.
    data={'temperature_ITS90_range_C':[-2.5,40.],'practical_salinity_range':[0.,42.]}
    cp=3990.

    def evaluate(self,sp,theta,*,entropy_derivatives=False):
        if not torch.isfinite(sp).all() or not torch.isfinite(theta).all():raise ValueError('finite thermodynamic inputs required')
        if ((sp<0)|(sp>42)|(theta<-2.5)|(theta>40)).any():raise ValueError('outside control reference domain')
        kelvin=theta+273.15;eta=self.cp*torch.log(kelvin/273.15)
        zero=torch.zeros_like(theta);cp=zero+self.cp
        result=dict(h0=self.cp*theta,entropy=eta,cp=cp,h_theta=cp,h_s=zero,entropy_theta=cp/kelvin,kelvin=kelvin,gibbs=self.cp*theta-kelvin*eta)
        if entropy_derivatives:result['entropy_s']=zero
        return result

    def temperature_from_enthalpy(self,sp,enthalpy):
        theta=enthalpy/self.cp
        self.evaluate(sp,theta)
        return theta


class LinearColumnOcean(StratifiedOcean):
    """Research adapter for boundary/mixing tests; horizontal stepping disabled."""
    def __init__(self,mesh,column,*,constant_cp=True):
        super().__init__(mesh,column)
        base=LinearDensityEnergy(gravity=column.gravity,rho0=column.rho0)
        self.base.dynamics.energy=base
        if constant_cp:self.reference=ConstantHeatCapacity()
        self.energy=EntropyEnergy(base,self.reference)

    def step_with_report(self,*args,**kwargs):
        raise NotImplementedError('linear control is qualified only for offline column physics')


# Local research import; this module is never installed into the VFS package.
from coupled import CoupledClosure

class ExactLinearWorkClosure(CoupledClosure):
    """Analytic interface work eliminates subtraction/inversion noise from B.

    Only valid for the affine-density, constant-cp control at fixed geometry.
    The full actual-H check remains independent and unchanged.
    """
    def _buoyancy_work(self,ocean,before,after,enthalpy_flux,salt_flux,default_work):
        e=ocean.energy.base;ref=ocean.reference
        if not isinstance(e,LinearDensityEnergy) or not isinstance(ref,ConstantHeatCapacity):
            raise ValueError('exact linear work requires linear density and constant cp')
        h=before.h;z=e.height(before)
        w=ocean.energy.mass_area(before)*e.gravity*(z[:,:-1]-z[:,1:])*(-e.alpha*enthalpy_flux/ref.cp+e.beta*salt_flux)
        fraction=h[:,:-1]/(h[:,:-1]+h[:,1:]);zero=torch.zeros_like(h[:,:1])
        return torch.cat((fraction*w,zero),-1)+torch.cat((zero,(1-fraction)*w),-1)
