"""Separate affine-density, constant-cp research ocean; no JMD95 evaluations.

The shared finite-volume transport/rotation and TKE funding algorithm are retained.
The Hamiltonian, its derivatives, caloric means and all physics use this same EOS.
"""
from dataclasses import dataclass
import math
import torch
from vfs.ocean.energy import ConstantDensityEnergy
from vfs.ocean.jmd95_energy import JMD95Energy
from vfs.ocean.stratified import StratifiedOcean,OceanSurfaceHeating,TKEState
from vfs.ocean.stratified.equations import EntropyEnergy
from vfs.ocean.temperature.physics import TemperatureMixing
from vfs.ocean.temperature.model import TemperatureOcean
from vfs.ocean.temperature.system import TemperatureSystem
from vfs.ocean.temperature.surface import FastSurfaceExchange,TemperatureBoundary
from vfs.ocean.temperature.ale import TemperatureEnthalpyRemapper,TemperatureRemapper


@dataclass(frozen=True)
class LinearEOS:
    # Frozen JMD95 local slopes at T68=12.5 C, SP=34.75, p=0 dbar,
    # divided by rho0=1025. They are parameters, not recomputed from JMD95.
    alpha: float = .0001908821582383638
    beta: float = .0007568878444130999
    T0: float = 10.
    S0: float = 35.
    cp: float = 3990.  # J/kg/K, for T90; T90=T68/1.00024

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.alpha,self.beta,self.T0,self.S0,self.cp)):
            raise ValueError('finite linear EOS parameters required')
        if self.alpha<0 or self.beta<0 or self.cp<=0 or not 0<self.S0<42 or not -2.5<self.T0/1.00024<40:
            raise ValueError('invalid linear EOS coefficients or reference state')


class LinearEnergy(JMD95Energy):
    """Reuse energy assembly/height only; anomaly integrals are analytic.

    H_b=sum rho0*A*h*g*z*(-alpha*(T-T0)+beta*(S-S0)).
    At fixed h*T and h*S, a thickness change also lifts the layers above it.
    """
    def __init__(self,config,*,gravity,rho0,domain=None):
        self.config=config;self.alpha=config.alpha;self.beta=config.beta
        self.T0,self.S0=config.T0,config.S0
        self.gravity,self.rho0=float(gravity),float(rho0);self.domain=domain
        self.mechanical=ConstantDensityEnergy(gravity=gravity,rho0=rho0,domain=domain)

    def _quadrature(self,like):
        # Exact midpoint integration for this affine-in-z potential, including
        # callers that decompose the energy into column heating/remap budgets.
        return like.new_tensor([.5]),like.new_tensor([1.])

    def density(self,state):
        return self.rho0*(1-self.alpha*(state.temperature()-self.T0)+self.beta*(state.salinity()-self.S0))

    def parcel_potential(self,temperature,salinity,z):
        anomaly=-self.alpha*(temperature-self.T0)+self.beta*(salinity-self.S0)
        return self.gravity*z*anomaly,-self.gravity*self.alpha*z,self.gravity*self.beta*z,self.gravity*anomaly

    def stratification(self,state):
        h=state.h;z=self.height(state)
        anomaly=-self.alpha*(state.temperature()-self.T0)+self.beta*(state.salinity()-self.S0)
        lift=h*anomaly;above=lift.cumsum(-1)-lift
        datum=self.alpha*self.T0-self.beta*self.S0
        qh=self.gravity*(datum*z+.5*lift+above)
        mass=self.rho0*state.mesh.cell_area[:,None]
        mask=(lambda v:v) if self.domain is None else self.domain.apply_cell
        total=(mass*mask(self.gravity*z*lift)).sum()
        return total,dict(h=mass*mask(qh),temp_h=mass*mask(-self.gravity*self.alpha*z),salt_h=mass*mask(self.gravity*self.beta*z))


from .caloric import ConstantHeatCapacity as LinearCaloric
from .ale import ConstantHeatCapacityRemapper as LinearRemapper


class LinearWorkspace:
    """Fixed-geometry exact work and algebraic heat increments."""
    def __init__(self,ocean,physical):
        if not isinstance(ocean.energy.base,LinearEnergy) or not isinstance(ocean.reference,LinearCaloric):
            raise TypeError('linear workspace requires the matched linear energy and calorics')
        self.ocean=ocean;self.base=ocean.energy.base;self.reference=ocean.reference
        self.z=self.base.height(physical)
        self.capacity=self.reference.cp_theta-self.base.gravity*self.base.alpha*self.z
        if not torch.isfinite(self.capacity).all() or (self.capacity<=0).any():raise ValueError('nonpositive full-H thermal derivative')

    def parts(self,physical):
        t,s=physical.temperature(),physical.salinity()
        return self.base.parcel_potential(t,s,self.z)[0],self.reference.evaluate(s,t)

    def conjugates(self,t,s):
        return -self.base.gravity*self.base.alpha*self.z/self.reference.cp_theta,self.base.gravity*self.base.beta*self.z

    def path_work(self,closure,physical,H0,fh,fs,mid,*,after=None):
        h=physical.h;area=self.ocean.energy.mass_area(physical)
        a,b=self.conjugates(None,None)
        work=area*((a[:,:-1]-a[:,1:])*fh+(b[:,:-1]-b[:,1:])*fs)
        fraction=h[:,:-1]/(h[:,:-1]+h[:,1:]);zero=torch.zeros_like(h[:,:1])
        return torch.cat((fraction*work,zero),-1)+torch.cat((zero,(1-fraction)*work),-1)

    def heat(self,state,physical,heat,*,allow_cooling=False):
        if heat.shape!=state.h.shape or heat.dtype!=state.h.dtype or heat.device!=state.h.device:
            raise ValueError('heat must match layer shape, dtype and device')
        if not torch.isfinite(heat).all() or (not allow_cooling and (heat<0).any()):raise ValueError('invalid linear heating input')
        delta=heat/self.capacity
        end=state.with_tensors(state.u_e,state.h,state.temp_h+state.h*delta,state.salt_h,state.tracer_h)
        TemperatureOcean(self.ocean).fields(end)
        return end,self.capacity*delta-heat,end


@dataclass(frozen=True)
class LinearMixing(TemperatureMixing):
    def _workspace(self,ocean,physical):return LinearWorkspace(ocean,physical)
    def _enthalpy_temperature(self,energy,workbuf,salt,enthalpy,guess):
        return energy.reference.temperature_from_enthalpy(salt,enthalpy)


def linear_heat(ocean,state,specific,physical):
    original=getattr(ocean,'original',ocean)
    return LinearWorkspace(original,physical).heat(state,physical,specific,allow_cooling=True)


class LinearHeating(OceanSurfaceHeating):
    _thermal_update=staticmethod(linear_heat)


class LinearExchange(FastSurfaceExchange):
    _thermal_update=staticmethod(linear_heat)


class LinearOcean(TemperatureSystem):
    """Explicit full research configuration, using h*T68 prognostic storage.

    Reuses checked transport, rotation, splitting and local TKE funding. It does
    not reuse JMD95 density/caloric evaluations or nonlinear mixing-path work.
    """
    def __init__(self,mesh,column,*,config=None,domain=None,closure=None,boundary=None,heating=None,exchange=None):
        config=LinearEOS() if config is None else config
        if not isinstance(config,LinearEOS):raise TypeError('config must be LinearEOS')
        for value,kind in ((closure,LinearMixing),(heating,LinearHeating),(exchange,LinearExchange)):
            if value is not None and not isinstance(value,kind):raise TypeError('use matched linear physics adapters')
        ocean=StratifiedOcean(mesh,column,domain=domain,warm_start=True)
        base=LinearEnergy(config,gravity=column.gravity,rho0=column.rho0,domain=domain)
        ocean.base.dynamics.energy=base
        ocean.base.dynamics.eos='linear_fixed_datum';ocean.base.dynamics.energy_label='linear_hamiltonian'
        ocean.reference=LinearCaloric(config);ocean.energy=EntropyEnergy(base,ocean.reference)
        super().__init__(ocean,closure=closure,boundary=boundary,heating=heating,exchange=exchange)
        self.config=config;self.remapper=LinearRemapper()

    def state(self,*,tke=0.,**fields):
        physical=self.ocean.base.state(**fields)
        result=TKEState(physical,physical.h*torch.as_tensor(tke,dtype=physical.h.dtype,device=physical.h.device))
        self.check(result);return result
