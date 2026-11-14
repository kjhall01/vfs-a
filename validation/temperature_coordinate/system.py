"""Complete research temperature-state composition with explicit ALE schedule."""
from dataclasses import dataclass
import torch
from vfs.ocean.stratified.tke_model import TKEState,TKEStepReport
from model import TemperatureOcean
from surface import TemperatureFacade,FastSurfaceExchange,FastSurfaceHeating,TemperatureBoundary
from ale import TemperatureEnergy,TemperatureRemapper
from path import TemperaturePathMixing


class TemperatureSystem:
    def __init__(self,ocean,*,closure=None,boundary=None,heating=None,exchange=None):
        self.ocean=ocean;self.dynamics=TemperatureOcean(ocean)
        self.facade=TemperatureFacade(ocean);self.energy=TemperatureEnergy(ocean)
        self.closure=closure;self.boundary=boundary;self.heating=heating;self.exchange=exchange
        self.remapper=TemperatureRemapper()

    def check(self,state):
        self.dynamics.fields(state.ocean)
        k=state.tke_h;h=state.ocean.h
        if k.shape!=h.shape or k.dtype!=h.dtype or k.device!=h.device or not torch.isfinite(k).all() or (k<0).any():
            raise ValueError('TKE must match h and be finite/nonnegative')
        wet=(self.energy.mass_area(state.ocean)>0).expand_as(h)
        if (k[~wet]!=0).any():raise ValueError('dry TKE must be zero')

    @staticmethod
    def pack(state):
        s=state.ocean;q=s.tracer_h
        fields=state.tke_h[...,None] if q is None else torch.cat((q,state.tke_h[...,None]),-1)
        return s.with_tensors(s.u_e,s.h,s.temp_h,s.salt_h,fields)

    @staticmethod
    def unpack(packed,original):
        q=None if original.ocean.tracer_h is None else packed.tracer_h[...,:-1]
        return TKEState(packed.with_tensors(packed.u_e,packed.h,packed.temp_h,packed.salt_h,q),packed.tracer_h[...,-1])

    def step_with_report(self,state,dt,*,surface_stress=None,shortwave=None,surface_heat=None,surface_fluxes=None):
        self.check(state)
        packed=self.dynamics.step(self.pack(state),dt)
        end=self.unpack(packed,state);transported=end
        exchange=boundary=heating=mixing=None
        if surface_fluxes is not None:
            if self.exchange is None:raise ValueError('surface fluxes require exchange')
            s,exchange=self.exchange.apply(self.facade,end.ocean,dt,surface_fluxes,_physical=end.ocean)
            end=TKEState(s,end.tke_h)
        if self.boundary is not None:
            s,k,boundary=self.boundary.apply(self.facade,end.ocean,end.tke_h,dt,surface_stress=surface_stress,_physical=end.ocean)
            end=TKEState(s,k)
        elif surface_stress is not None:raise ValueError('stress requires boundary')
        if self.heating is not None:
            s,heating=self.heating.apply(self.facade,end.ocean,dt,shortwave=shortwave,surface_heat=surface_heat,_physical=end.ocean)
            end=TKEState(s,end.tke_h)
        elif shortwave is not None or surface_heat is not None:raise ValueError('heat flux requires heating')
        if self.closure is not None:
            s,k,mixing=self.closure.apply(self.ocean,end.ocean,end.tke_h,dt)
            end=TKEState(s,k)
        self.check(end)
        residual=(self.energy.mass_area(state.ocean)*(transported.tke_h-state.tke_h)).sum()
        return end,TKEStepReport(None,mixing,residual,boundary,heating,exchange)

    def remap_with_report(self,state,target_h):
        self.check(state)
        mapped,report=self.remapper.remap_with_report(self.ocean.base,self.pack(state),target_h,self.energy)
        end=self.unpack(mapped,state);self.check(end)
        return end,report

    def total_energy(self,state):
        return self.energy.terms(state.ocean)['total']+(self.energy.mass_area(state.ocean)*state.tke_h).sum()
