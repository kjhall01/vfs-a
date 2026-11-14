"""Research temperature-native heating and the existing lagged TKE algorithm.

Reuse its tridiagonal solves, scalar funding retries, local/full-H and entropy
acceptance tests. Only caloric storage/reconstruction hooks differ. No boundary
stress, freshwater exchange, radiative profile or ALE adapter is provided here;
those optional composition adapters live in surface.py, ale.py and system.py.
"""
from dataclasses import dataclass
import torch
from fast_thermo import MixingThermodynamics
from lagged import SubcycledLaggedClosure
from model import TemperatureOcean


class TemperatureThermodynamics(MixingThermodynamics):
    def heat(self,state,physical,heat):
        if not torch.any(heat!=0):return state,torch.zeros_like(heat),physical
        theta,residual=self.heat_temperature(physical,heat)
        end=state.with_tensors(state.u_e,state.h,
            torch.where(heat!=0,state.h*theta,state.temp_h),state.salt_h,state.tracer_h)
        TemperatureOcean(self.ocean).fields(end)
        return end,residual,end


@dataclass(frozen=True)
class TemperatureMixing(SubcycledLaggedClosure):
    fast_thermodynamics: bool = True

    def __post_init__(self):
        super().__post_init__()
        if not self.fast_thermodynamics:
            raise ValueError('temperature mixing requires fast thermodynamic workspace')

    def _workspace(self,ocean,physical):
        return TemperatureThermodynamics(ocean,physical)

    def _baseline_entropy(self,energy,workbuf,salt,theta,enthalpy):
        # No entropy increment is stored, so no inverse-roundtrip baseline.
        return None

    def _mixed_state(self,ocean,state,u,salt_h,tracer,theta,baseline,workbuf):
        # Store the enthalpy solve's result directly; no entropy baseline or
        # entropy inverse. Do not mask equal-valued solutions: their derivatives
        # can still depend on the supplied enthalpy and salinity.
        end=state.with_tensors(u,state.h,state.h*theta,salt_h,tracer)
        TemperatureOcean(ocean).fields(end)
        return end,end

    def _entropy_content(self,ocean,state):
        return state.h*ocean.reference.evaluate(state.salinity(),state.temperature())['entropy']

    def _apply(self,ocean,state,tke_h,dt,physical=None):
        TemperatureOcean(ocean).fields(state)
        return super()._apply(ocean,state,tke_h,dt,physical=state)


def heat(ocean,state,heat_specific,*,allow_cooling=False):
    """Prescribed layer energy increment [J/kg], fixed h/S/u; full-H checked."""
    candidate=TemperatureOcean(ocean);candidate.fields(state)
    if heat_specific.shape!=state.h.shape or heat_specific.dtype!=state.h.dtype or heat_specific.device!=state.h.device:
        raise ValueError('heat must match layer shape, dtype and device')
    if not torch.isfinite(heat_specific).all():raise ValueError('finite heat required')
    if not allow_cooling and (heat_specific<0).any():raise ValueError('cooling must be explicit')
    if not torch.any(heat_specific!=0):return state,torch.zeros_like(heat_specific)
    work=TemperatureThermodynamics(ocean,state)
    theta,residual=work.heat_temperature(state,heat_specific,allow_cooling=allow_cooling)
    end=state.with_tensors(state.u_e,state.h,
        torch.where(heat_specific!=0,state.h*theta,state.temp_h),state.salt_h,state.tracer_h)
    candidate.fields(end)
    p0,r0=work.parts(state);p1,r1=work.parts(end)
    actual=p1-p0+r1['h0']-r0['h0']
    tolerance=1024*torch.finfo(state.h.dtype).eps*(p0.abs()+r0['h0'].abs()+heat_specific.abs()).clamp_min(1)
    if ((actual-heat_specific).abs()>tolerance).any():raise FloatingPointError('temperature heating full-H residual failed')
    return end,residual
