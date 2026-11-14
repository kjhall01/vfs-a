"""Conserved-flux work from a straight T/S path; no intermediate inversions."""
from dataclasses import dataclass
import torch
from physics import TemperatureMixing,TemperatureThermodynamics


class TemperaturePath(TemperatureThermodynamics):
    def path_work(self,closure,physical,H0,fh,fs,mid,*,after=None):
        if after is None:raise ValueError('temperature path needs the actual proposal endpoint')
        h=physical.h;area=self.ocean.energy.mass_area(physical)
        t0,s0=physical.temperature(),physical.salinity()
        dt,ds=after.temperature()-t0,after.salinity()-s0
        ptbar,psbar,htbar,hsbar=[torch.zeros_like(h) for _ in range(4)]
        for fraction,w in zip(closure._path_nodes,closure._path_weights):
            t,s=t0+fraction*dt,s0+fraction*ds
            ref=self.reference.evaluate(s,t)
            _,pt,ps,_=self.parcel(t,s)
            ptbar=ptbar+w*(pt*self.weight).sum(-1)
            psbar=psbar+w*(ps*self.weight).sum(-1)
            htbar=htbar+w*ref['h_theta'];hsbar=hsbar+w*ref['h_s']
        if not torch.isfinite(htbar).all() or (htbar<=0).any():raise FloatingPointError('invalid path heat capacity')
        # Eliminate dT between dPsi=PT*dT+PS*dS and dh0=HT*dT+HS*dS.
        a_bar=ptbar/htbar;b_bar=psbar-a_bar*hsbar
        a_mid,b_mid=mid;alpha=h[:,:-1]/(h[:,:-1]+h[:,1:])
        face=lambda q:(1-alpha)*q[:,:-1]+alpha*q[:,1:]
        zero=torch.zeros_like(h[:,:1])
        div=lambda f:torch.cat((f,zero),-1)-torch.cat((zero,f),-1)
        interface=area*((a_bar[:,:-1]-a_bar[:,1:])*fh+(b_bar[:,:-1]-b_bar[:,1:])*fs)
        paired=torch.cat((alpha*interface,zero),-1)+torch.cat((zero,(1-alpha)*interface),-1)
        # Same midpoint energy-flux allocation as the enthalpy-path method.
        return paired+div(area*(face(a_bar-a_mid)*fh+face(b_bar-b_mid)*fs))


@dataclass(frozen=True)
class TemperaturePathMixing(TemperatureMixing):
    def _workspace(self,ocean,physical):
        return TemperaturePath(ocean,physical)


@dataclass(frozen=True)
class GuardedTemperaturePathMixing(TemperaturePathMixing):
    """Experimental midpoint work, retrying order 3 on budget rejection.

    This is a guarded cheap proposal, not a quadrature error estimator. A small
    accepted global residual does not bound each layer's allocation error.
    Fixed order 3 remains the comparison/recommended research configuration.
    """
    def _apply(self,ocean,state,tke_h,dt,physical=None):
        from dataclasses import replace
        refinements=0
        try:
            result=TemperatureMixing._apply(replace(self,quadrature_order=1),ocean,state,tke_h,dt,physical=physical)
            order=1
        except (ValueError,FloatingPointError) as error:
            if str(error) not in ('coupled full-H closure failed','lagged local budget failed','coupled update decreases column entropy'):
                raise
            result=TemperatureMixing._apply(replace(self,quadrature_order=3),ocean,state,tke_h,dt,physical=physical)
            order=3;refinements=1
        end,k,report,final=result
        metrics=dict(report.mixing_limiter,path_refinements=state.h.new_tensor(refinements),
                     accepted_path_points=state.h.new_tensor(order))
        return end,k,replace(report,mixing_limiter=metrics),final
