"""Opt-in Roquet Freezing density with analytic Hamiltonian and mixing work.

State temperature means Conservative Temperature [deg C], salinity means
Absolute Salinity [g/kg]. Calorics are explicitly idealized, not TEOS entropy.
Coefficients: Roquet et al. (2015), Table 3, Freezing; JPO-D-15-0080.1.
"""
from dataclasses import dataclass
import math
import torch
from vfs.ocean.energy import ConstantDensityEnergy
from vfs.ocean.jmd95_energy import JMD95Energy
from vfs.ocean.stratified import StratifiedOcean,TKEState,OceanSurfaceHeating
from vfs.ocean.stratified.equations import EntropyEnergy
from linear import LinearCaloric,LinearMixing,LinearRemapper
from model import TemperatureOcean
from system import TemperatureSystem
from surface import FastSurfaceExchange,TemperatureBoundary


@dataclass(frozen=True)
class RoquetFreezingEOS:
    """Published density coefficients; density offsets cancel in buoyancy.

    P(CT,SA,z)=s*SA+t*CT+tt*CT**2-tz*CT*z, with z positive UP.
    s: kg/m3/(g/kg), t: kg/m3/K, tt: kg/m3/K2, tz: kg/m4/K.
    CT0/SA0 choose the fixed reference profile P(CT0,SA0,z).
    """
    s: float = .7718
    t: float = -.0491
    tt: float = -.005027
    tz: float = -2.5681e-5
    CT0: float = 10.
    SA0: float = 35.
    cp: float = 3991.86795711963  # TEOS potential-enthalpy scale; not full calorics

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):raise ValueError('finite EOS coefficients required')
        if self.cp<=0 or not -2.5<self.CT0<40 or not 0<self.SA0<42:raise ValueError('invalid caloric/reference parameters')

    def polynomial(self,ct,sa,z):return self.s*sa+self.t*ct+self.tt*ct.square()-self.tz*ct*z

    def anomaly(self,ct,sa,z):
        # Difference form avoids subtracting the O(30 kg/m3) salinity datum.
        return self.s*(sa-self.SA0)+(ct-self.CT0)*(self.t+self.tt*(ct+self.CT0)-self.tz*z)


class RoquetCaloric(LinearCaloric):
    """h0=cp*CT; idealized eta=cp*log((CT+273.15)/273.15).

    eta is a model entropy coordinate, not physical TEOS-10 entropy. Likewise
    the 'kelvin' conjugate is CT+273.15, not in-situ thermodynamic temperature.
    Inherited analytic means/inverses are exact for this declared reference.
    """
    temperature_scale='Conservative Temperature'
    to_90=1.  # Compatibility scale in shared guards; CT is NOT potential T90.
    data={'temperature_ITS90_range_C':[-2.5,40.],'practical_salinity_range':[0.,42.]}


class RoquetEnergy(JMD95Energy):
    """Exact layer integral of the fixed-datum polynomial dynamic enthalpy."""
    def __init__(self,config,*,gravity,rho0,domain=None):
        self.config=config;self.T0=config.CT0;self.S0=config.SA0
        self.gravity=float(gravity);self.rho0=float(rho0);self.domain=domain
        self.mechanical=ConstantDensityEnergy(gravity=gravity,rho0=rho0,domain=domain)

    def _quadrature(self,like):
        # Shared ALE diagnostics use this exact two-node rule for degree <=2.
        d=1/math.sqrt(12)
        return like.new_tensor([.5-d,.5+d]),like.new_tensor([.5,.5])

    def density(self,state):
        return self.rho0+self.config.anomaly(state.temperature(),state.salinity(),self.height(state))

    def parcel_potential(self,temperature,salinity,z):
        c=self.config;dt=temperature-c.CT0
        f=c.s*(salinity-c.SA0)+dt*(c.t+c.tt*(temperature+c.CT0))
        k=self.gravity/self.rho0
        psi=k*(z*f-.5*c.tz*dt*z.square())
        pt=k*(z*(c.t+2*c.tt*temperature)-.5*c.tz*z.square())
        ps=k*c.s*z;pz=k*(f-c.tz*dt*z)
        return psi,pt,ps,pz

    def coefficients(self,state):
        """Layer-mean Psi=A*CT²+B*CT+C*SA+D, at fixed h."""
        c=self.config;z=self.height(state);z2=z.square()+state.h.square()/12;k=self.gravity/self.rho0
        a=k*c.tt*z;b=k*(c.t*z-.5*c.tz*z2);cc=k*c.s*z
        return a,b,cc,-a*c.CT0**2-b*c.CT0-cc*c.SA0

    def stratification(self,state):
        h=state.h;t=state.temperature();s=state.salinity();z=self.height(state)
        c=self.config;k=self.gravity/self.rho0;a,b,cc,d=self.coefficients(state)
        # Reference differences preserve exact zero at the reference water mass.
        psi=a*(t-c.CT0)*(t+c.CT0)+b*(t-c.CT0)+cc*(s-c.SA0)
        qt=2*a*t+b;qs=cc
        pz=k*c.anomaly(t,s,z);lift=h*pz;above=lift.cumsum(-1)-lift
        # Moving layer endpoints and overlying water, at fixed h*CT and h*SA.
        qh=psi-t*qt-s*qs+.5*h*pz-k*c.tz*(t-c.CT0)*h.square()/12+above
        mass=self.rho0*state.mesh.cell_area[:,None]
        mask=(lambda x:x) if self.domain is None else self.domain.apply_cell
        return (mass*mask(h*psi)).sum(),dict(h=mass*mask(qh),temp_h=mass*mask(qt),salt_h=mass*mask(qs))


class RoquetEntropyEnergy(EntropyEnergy):
    """Same coordinate-chain-rule assembly with an explicit CT/SA type guard."""
    def __init__(self,base,reference):
        if not isinstance(base,RoquetEnergy) or not isinstance(reference,RoquetCaloric):
            raise TypeError('Roquet CT/SA energy and caloric reference must match')
        self.base,self.reference=base,reference


class RoquetWorkspace:
    def __init__(self,ocean,physical):
        if not isinstance(ocean.energy.base,RoquetEnergy) or not isinstance(ocean.reference,RoquetCaloric):raise TypeError('matched Roquet energy/calorics required')
        self.ocean=ocean;self.base=ocean.energy.base;self.reference=ocean.reference
        self.a,self.b,self.c,self.d=self.base.coefficients(physical)

    def parts(self,physical):
        t=physical.temperature();s=physical.salinity();cfg=self.base.config
        psi=self.a*(t-cfg.CT0)*(t+cfg.CT0)+self.b*(t-cfg.CT0)+self.c*(s-cfg.SA0)
        return psi,self.reference.evaluate(s,t)

    def conjugates(self,t,s):return (2*self.a*t+self.b)/self.reference.cp_theta,self.c

    def path_work(self,closure,physical,H0,fh,fs,mid,*,after=None):
        # Exact discrete gradient of a quadratic: evaluate at the midpoint of
        # the TOTAL increment. This includes cross terms from both interfaces.
        h=physical.h;zero=torch.zeros_like(h[:,:1])
        div=lambda x:torch.cat((x,zero),-1)-torch.cat((zero,x),-1)
        dt=div(fh)/(h*self.reference.cp_theta)
        a,b=self.conjugates(physical.temperature()+.5*dt,None)
        work=self.ocean.energy.mass_area(physical)*((a[:,:-1]-a[:,1:])*fh+(b[:,:-1]-b[:,1:])*fs)
        fraction=h[:,:-1]/(h[:,:-1]+h[:,1:])
        return torch.cat((fraction*work,zero),-1)+torch.cat((zero,(1-fraction)*work),-1)

    def heat(self,state,physical,heat,*,allow_cooling=False):
        if heat.shape!=state.h.shape or heat.dtype!=state.h.dtype or heat.device!=state.h.device:raise ValueError('heat must match layers')
        if not torch.isfinite(heat).all() or (not allow_cooling and (heat<0).any()):raise ValueError('invalid heating')
        t=physical.temperature();capacity=self.reference.cp_theta+2*self.a*t+self.b
        discriminant=capacity.square()+4*self.a*heat
        if (capacity<=0).any() or (discriminant<=0).any():raise ValueError('heating has no admissible increasing-energy root')
        delta=2*heat/(capacity+torch.sqrt(discriminant))
        if (capacity+2*self.a*delta<=0).any():raise ValueError('nonpositive final heat capacity')
        end=state.with_tensors(state.u_e,state.h,state.temp_h+state.h*delta,state.salt_h,state.tracer_h)
        TemperatureOcean(self.ocean).fields(end)
        return end,(capacity+self.a*delta)*delta-heat,end


@dataclass(frozen=True)
class RoquetMixing(LinearMixing):
    def _workspace(self,ocean,physical):return RoquetWorkspace(ocean,physical)


def roquet_heat(ocean,state,specific,physical):
    return RoquetWorkspace(getattr(ocean,'original',ocean),physical).heat(state,physical,specific,allow_cooling=True)


class RoquetHeating(OceanSurfaceHeating):
    _thermal_update=staticmethod(roquet_heat)


class RoquetExchange(FastSurfaceExchange):
    _thermal_update=staticmethod(roquet_heat)


class RoquetOcean(TemperatureSystem):
    def __init__(self,mesh,column,*,config=None,domain=None,closure=None,boundary=None,heating=None,exchange=None):
        config=RoquetFreezingEOS() if config is None else config
        if not isinstance(config,RoquetFreezingEOS):raise TypeError('RoquetFreezingEOS required')
        for value,kind in ((closure,RoquetMixing),(heating,RoquetHeating),(exchange,RoquetExchange)):
            if value is not None and not isinstance(value,kind):raise TypeError('use matched Roquet physics adapters')
        ocean=StratifiedOcean(mesh,column,domain=domain,warm_start=True)
        base=RoquetEnergy(config,gravity=column.gravity,rho0=column.rho0,domain=domain)
        ocean.base.dynamics.energy=base;ocean.base.dynamics.eos='roquet_freezing';ocean.base.dynamics.energy_label='roquet_hamiltonian'
        ocean.reference=RoquetCaloric(config);ocean.energy=RoquetEntropyEnergy(base,ocean.reference)
        super().__init__(ocean,closure=closure,boundary=boundary,heating=heating,exchange=exchange)
        self.config=config;self.remapper=LinearRemapper()

    def compile_thermodynamics(self, *, backend='inductor'):
        """Compile selected pure kernels; validation and timestep control stay eager."""
        previous=getattr(self,'_compile_backend',None)
        if previous is not None:
            if previous!=backend:raise ValueError('rebuild the model to change compile backend')
            return self
        options=dict(backend=backend,fullgraph=True,dynamic=True)
        base=self.ocean.energy.base
        base.stratification=torch.compile(base.stratification,**options)
        base.parcel_potential=torch.compile(base.parcel_potential,**options)
        base.coefficients=torch.compile(base.coefficients,**options)
        self.ocean.reference._values=torch.compile(self.ocean.reference._values,**options)
        self._compile_backend=backend
        return self

    def state(self,*,conservative_temperature,absolute_salinity,tke=0.,**fields):
        """No implicit conversion: CT [deg C], SA [g/kg]."""
        if 'temperature' in fields or 'salinity' in fields:raise TypeError('use explicit CT/SA keyword names')
        s=self.ocean.base.state(temperature=conservative_temperature,salinity=absolute_salinity,**fields)
        out=TKEState(s,s.h*torch.as_tensor(tke,dtype=s.h.dtype,device=s.h.device));self.check(out);return out


def convert_jmd95_initialization(*,potential_temperature_ipts68,practical_salinity,pressure_dbar,longitude,latitude):
    """Explicit offline GSW conversion; optional dependency, no runtime use.

    Pressure [dbar], lon/lat [degrees] must be supplied, not inferred. This NumPy
    initialization boundary does not support autograd; native CT/SA needs no GSW.
    """
    import gsw
    values=(potential_temperature_ipts68,practical_salinity,pressure_dbar,longitude,latitude)
    if any(not isinstance(v,torch.Tensor) for v in values):raise TypeError('conversion inputs must be tensors')
    if any(v.requires_grad for v in values):raise ValueError('offline GSW conversion is not differentiable')
    if any(not torch.isfinite(v).all() for v in values):raise ValueError('finite conversion inputs required')
    t,s,p,lon,lat=[v.detach().cpu().numpy() for v in values]
    sa=gsw.SA_from_SP(s,p,lon,lat);ct=gsw.CT_from_pt(sa,t/1.00024)
    like=potential_temperature_ipts68
    result=dict(conservative_temperature=torch.as_tensor(ct,dtype=like.dtype,device=like.device),absolute_salinity=torch.as_tensor(sa,dtype=like.dtype,device=like.device))
    if any(not torch.isfinite(v).all() for v in result.values()):raise ValueError('GSW conversion returned nonfinite values')
    return result
