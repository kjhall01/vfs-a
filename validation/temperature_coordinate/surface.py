"""Opt-in surface solvers: shared forcing laws and unchanged report checks."""
from dataclasses import replace
import torch
from vfs.ocean.stratified import OceanSurfaceHeating,OceanSurfaceExchange,OceanBoundaryStress
from vfs.ocean.stratified.equations import EntropyState
from fast_thermo import MixingThermodynamics
from model import TemperatureOcean


class TemperatureFacade:
    """Physical-state interface for existing forcing laws; no inverse calls."""
    def __init__(self,ocean):
        self.original=ocean
        for name in ('base','mesh','column','domain','energy','reference'):
            setattr(self,name,getattr(ocean,name))
    def check_state(self,state,*,_physical=None):
        TemperatureOcean(self.original).fields(state)
        return state


def fast_thermal_update(ocean,state,specific,physical):
    original=getattr(ocean,'original',ocean)
    w=MixingThermodynamics(original,physical)
    if not torch.any(specific!=0):return state,torch.zeros_like(specific),physical
    theta,residual=w.heat_temperature(physical,specific,allow_cooling=True)
    if isinstance(state,EntropyState):
        eta0=original.reference.evaluate(physical.salinity(),physical.temperature(),entropy_derivatives=True)['entropy']
        eta1=original.reference.evaluate(physical.salinity(),theta,entropy_derivatives=True)['entropy']
        end=replace(state,entropy_h=state.entropy_h+state.h*(eta1-eta0))
        final=w.physical(end,theta)
    else:
        end=state.with_tensors(state.u_e,state.h,torch.where(specific!=0,state.h*theta,state.temp_h),state.salt_h,state.tracer_h)
        final=ocean.check_state(end)
    return end,residual,final


class FastSurfaceHeating(OceanSurfaceHeating):
    _thermal_update=staticmethod(fast_thermal_update)


class FastSurfaceExchange(OceanSurfaceExchange):
    _thermal_update=staticmethod(fast_thermal_update)
    def _trial_state(self,ocean,state,physical,new_h,salt,velocity):
        if isinstance(state,EntropyState):
            return super()._trial_state(ocean,state,physical,new_h,salt,velocity)
        return state.with_tensors(velocity,new_h,new_h*physical.temperature(),state.salt_h,state.tracer_h)


class TemperatureBoundary(OceanBoundaryStress):
    def _velocity_state(self,state,velocity):
        return state.with_tensors(velocity,state.h,state.temp_h,state.salt_h,state.tracer_h)
