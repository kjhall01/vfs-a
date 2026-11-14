"""Prescribed surface heat and two-band penetrating shortwave radiation."""
from dataclasses import dataclass
import math
import torch
from .heating import apply_heat, thermal_parts


@dataclass(frozen=True)
class SurfaceHeatingReport:
    """All energies/residuals in J; layer arrays [cell,layer], others [cell]."""
    surface_energy: torch.Tensor
    shortwave_energy: torch.Tensor
    bottom_transmitted_energy: torch.Tensor
    bottom_exported_energy: torch.Tensor
    bottom_absorbed_energy: torch.Tensor
    layer_heat: torch.Tensor
    external_heat: torch.Tensor  # global scalar
    flux_residual: torch.Tensor
    inversion_residual: torch.Tensor
    layer_energy_residual: torch.Tensor
    energy_residual: torch.Tensor  # global scalar


@dataclass(frozen=True)
class OceanSurfaceHeating:
    """Optical lengths [m] and first-band fraction are explicit, not calibrated.

    Inputs [cell] are W/m², positive into the ocean. shortwave is nonnegative
    net sunlight AFTER reflection. surface_heat is signed non-solar heat
    (e.g. net longwave), deposited at the top. No atmospheric flux calculation.
    bottom='export' sends transmitted light out of the water energy budget;
    'absorb' deposits it in the lowest layer, an explicit instantaneous-return
    approximation with no separate seabed reservoir.
    """
    first_band_fraction: float
    attenuation_lengths: tuple
    bottom: str = 'export'

    def __post_init__(self):
        if not math.isfinite(self.first_band_fraction) or not 0 <= self.first_band_fraction <= 1:
            raise ValueError('first_band_fraction must be finite in [0,1]')
        if len(self.attenuation_lengths) != 2 or any(not math.isfinite(x) or x <= 0 for x in self.attenuation_lengths):
            raise ValueError('two finite positive attenuation lengths are required')
        if self.bottom not in ('export','absorb'):
            raise ValueError("bottom must be 'export' or 'absorb'")

    def _thermal_update(self,ocean,state,specific,physical):
        end,residual=apply_heat(ocean,state,specific,allow_cooling=True)
        return end,residual,ocean.check_state(end)

    def apply(self, ocean, state, dt, *, shortwave=None, surface_heat=None, _physical=None, _return_physical=False):
        """Heat an EntropyState directly; does not require a TKE state/closure."""
        if not math.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError('dt must be finite and positive')
        physical=(ocean.check_state(state) if _physical is None else ocean.check_state(state, _physical=_physical))
        h=state.h; mass_area=ocean.energy.mass_area(state)
        wet=(mass_area[:,0] > 0)
        def field(value,name):
            if value is None:return torch.zeros_like(h[:,0])
            if not isinstance(value,torch.Tensor):raise TypeError(name+' must be a tensor [cell] in W/m²')
            if value.shape != (h.shape[0],) or value.dtype != h.dtype or value.device != h.device or not torch.isfinite(value).all():
                raise ValueError(name+' must be finite [cell], matching state dtype/device')
            return torch.where(wet,value,torch.zeros_like(value))
        sw=field(shortwave,'shortwave');surface=field(surface_heat,'surface_heat')
        if shortwave is not None and (shortwave < 0).any():
            raise ValueError('net incoming shortwave must be nonnegative')
        interfaces=torch.cat((torch.zeros_like(h[:,:1]),torch.cumsum(h,-1)),-1)
        a=self.first_band_fraction
        absorption=torch.zeros_like(h);transmitted=torch.zeros_like(sw)
        for fraction,length in zip((a,1-a),self.attenuation_lengths):
            # Stable even for layers much thinner than an attenuation length.
            absorption=absorption+fraction*sw[:,None]*torch.exp(-interfaces[:,:-1]/length)*(-torch.expm1(-h/length))
            transmitted=transmitted+fraction*sw*torch.exp(-interfaces[:,-1]/length)
        bottom_heat=transmitted if self.bottom=='absorb' else torch.zeros_like(sw)
        exported=transmitted-bottom_heat
        top=torch.cat((surface[:,None],torch.zeros_like(h[:,1:])), -1)
        bed=torch.cat((torch.zeros_like(h[:,:-1]),bottom_heat[:,None]), -1)
        area=mass_area/ocean.column.rho0
        layer_heat=dt*area*(absorption+top+bed)
        surface_energy=dt*area[:,0]*surface
        shortwave_energy=dt*area[:,0]*sw
        transmitted_energy=dt*area[:,0]*transmitted
        exported_energy=dt*area[:,0]*exported
        bottom_energy=dt*area[:,0]*bottom_heat
        incoming=surface_energy+shortwave_energy-exported_energy
        flux_residual=layer_heat.sum(-1)-incoming
        mass=mass_area*h
        safe_mass=torch.where(mass > 0,mass,torch.ones_like(mass))
        end,inversion,final=self._thermal_update(ocean,state,layer_heat/safe_mass,physical)
        old_psi,old_ref=thermal_parts(ocean,physical)
        new_psi,new_ref=thermal_parts(ocean,final)
        measured=mass*((new_psi-old_psi)+(new_ref['h0']-old_ref['h0']))
        residual=measured-layer_heat
        scale=(mass*(old_psi.abs()+old_ref['h0'].abs())+layer_heat.abs()).clamp_min(1)
        tolerance=1024*torch.finfo(h.dtype).eps*scale
        if not torch.isfinite(residual).all() or (residual.abs()>tolerance).any():
            raise FloatingPointError('surface heating full-H layer energy closure failed')
        flux_scale=(surface_energy.abs()+shortwave_energy+exported_energy).clamp_min(1)
        if (flux_residual.abs()>128*torch.finfo(h.dtype).eps*flux_scale).any():
            raise FloatingPointError('shortwave flux ledger failed')
        result = (end,SurfaceHeatingReport(surface_energy,shortwave_energy,transmitted_energy,
            exported_energy,bottom_energy,layer_heat,incoming.sum(),flux_residual,
            mass*inversion,residual,residual.sum()))
        return (*result,final) if _return_physical else result
