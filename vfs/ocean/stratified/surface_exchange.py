"""Conservative ocean-side exchange of prescribed water, momentum and energy."""
from dataclasses import dataclass,replace
import math
import torch
from .heating import apply_heat
from .energy_columns import column_terms


@dataclass(frozen=True)
class OceanSurfaceFluxes:
    """Interval-mean prescribed fluxes; None is zero except required water carriers.

    precipitation/evaporation: nonnegative [cell], kg/m²/s, salt/tracer free.
    sensible_heat/latent_heat: signed [cell], W/m², positive into the ocean.
    water_energy_flux: signed [cell], W/m², energy transported with water in
    the ocean Hamiltonian reference, INCLUDING mechanical energy, EXCLUDING
    heat already in sensible/latent/radiative terms. Required when either water
    rate is nonzero, including balanced precipitation and evaporation.
    water_momentum_flux: signed native edge-normal [edge], Pa, advected momentum
    per horizontal area/time, not wind stress. Required with nonzero water rates.
    Explicit zeros are permitted: they mean zero energy/momentum transfer.
    """
    precipitation: object = None
    evaporation: object = None
    sensible_heat: object = None
    latent_heat: object = None
    water_energy_flux: object = None
    water_momentum_flux: object = None


@dataclass(frozen=True)
class SurfaceExchangeReport:
    """Cell mass budgets [kg], edge impulses [N s], cell energy budgets [J]."""
    precipitation_mass: torch.Tensor
    evaporation_mass: torch.Tensor
    mass_residual: torch.Tensor
    water_impulse: torch.Tensor
    momentum_residual: torch.Tensor
    sensible_energy: torch.Tensor
    latent_energy: torch.Tensor
    water_energy: torch.Tensor
    external_energy: torch.Tensor  # global scalar J
    provisional_energy_change: dict  # named column Hamiltonian terms [J]
    thermal_adjustment: torch.Tensor  # top-layer correction [cell], J
    inversion_residual: torch.Tensor  # [cell,layer], J
    column_energy_residual: torch.Tensor
    energy_residual: torch.Tensor  # global scalar J


class OceanSurfaceExchange:
    """Top-layer freshwater and heat exchange; no bulk-flux or atmosphere model.

    Salt and all existing passives are nonvolatile and have zero boundary flux.
    The supplied native momentum impulse sets the new edge momentum. A local
    top-layer thermal solve closes the prescribed full-H energy change,
    including changing salinity, mass, free-surface and kinetic energy.
    No automatic freshwater enthalpy, latent heat or TKE source is inferred.
    """
    def apply(self,ocean,state,dt,fluxes):
        if not isinstance(fluxes,OceanSurfaceFluxes):
            raise TypeError('fluxes must be OceanSurfaceFluxes')
        if not math.isfinite(float(dt)) or float(dt)<=0:
            raise ValueError('dt must be finite and positive')
        physical=ocean.check_state(state)
        h=state.h;mesh=state.mesh;energy=ocean.energy
        mass_area=energy.mass_area(state);area=mass_area[:,0]/ocean.column.rho0
        wet=area>0
        active=torch.ones(mesh.nEdges,dtype=torch.bool,device=h.device) if ocean.domain is None else ocean.domain.active_edge
        def field(value,name,size,mask):
            if value is None:return torch.zeros(size,dtype=h.dtype,device=h.device)
            if not isinstance(value,torch.Tensor):raise TypeError(name+' must be a tensor')
            if value.shape!=(size,) or value.dtype!=h.dtype or value.device!=h.device or not torch.isfinite(value).all():
                raise ValueError(name+' must be finite with the declared shape and state dtype/device')
            if name in ('precipitation','evaporation') and (value<0).any():
                raise ValueError(name+' must be nonnegative')
            return torch.where(mask,value,torch.zeros_like(value))
        p=field(fluxes.precipitation,'precipitation',mesh.nCells,wet)
        e=field(fluxes.evaporation,'evaporation',mesh.nCells,wet)
        sensible=field(fluxes.sensible_heat,'sensible_heat',mesh.nCells,wet)
        latent=field(fluxes.latent_heat,'latent_heat',mesh.nCells,wet)
        water=field(fluxes.water_energy_flux,'water_energy_flux',mesh.nCells,wet)
        momentum=field(fluxes.water_momentum_flux,'water_momentum_flux',mesh.nEdges,active)
        if ((p!=0)|(e!=0)).any() and (fluxes.water_energy_flux is None or fluxes.water_momentum_flux is None):
            raise ValueError('water exchange requires explicit water_energy_flux and water_momentum_flux (zeros are allowed)')
        exchanging=(p!=0)|(e!=0)
        edge_exchange=mesh.cell_to_edge((p+e)[:,None])[:,0]>0
        if ((water!=0)&~exchanging).any() or ((momentum!=0)&~edge_exchange).any():
            raise ValueError('water-carried fluxes require local precipitation or evaporation')
        dh=dt*(p-e)/ocean.column.rho0
        new_h=torch.cat(((h[:,0]+dh)[:,None],h[:,1:]),-1)
        if not torch.isfinite(new_h).all() or (new_h<=ocean.base.dynamics.min_thickness).any():
            raise ValueError('water exchange exhausts a layer; reduce dt, no clipping or wetting/drying')
        salt=state.salt_h/new_h
        if ((salt<=0)|(salt>=42)).any():
            raise ValueError('water exchange leaves salinity domain; reduce dt, no salt clipping')
        mechanical=energy.base.mechanical
        provisional=physical.with_tensors(physical.u_e,new_h,new_h*physical.temperature(),state.salt_h,state.tracer_h)
        old_metric=mechanical.metric(physical);new_metric=mechanical.metric(provisional)
        edge_area=mesh.dc_edge*mesh.dv_edge
        delta_metric=ocean.column.rho0*edge_area*mesh.cell_to_edge(torch.cat((dh[:,None],torch.zeros_like(h[:,1:])), -1))[:,0]
        impulse=dt*edge_area*momentum
        u_top=state.u_e[:,0]+(impulse-delta_metric*state.u_e[:,0])/new_metric[:,0]
        velocity=torch.cat((u_top[:,None],state.u_e[:,1:]),-1)
        # Only the top geometry/composition changes. Keep lower/dry entropy exact.
        theta=physical.temperature()
        old_eta=ocean.reference.evaluate(physical.salinity(),theta,entropy_derivatives=True)['entropy']
        new_eta=ocean.reference.evaluate(salt,theta,entropy_derivatives=True)['entropy']
        entropy=state.entropy_h+(new_h-h)*new_eta+h*(new_eta-old_eta)
        trial=replace(state,h=new_h,u_e=velocity,entropy_h=entropy)
        provisional=ocean.check_state(trial)
        before,_=column_terms(energy,physical)
        after,_=column_terms(energy,provisional)
        changes={key:after[key]-before[key] for key in before}
        sensible_energy=dt*area*sensible;latent_energy=dt*area*latent;water_energy=dt*area*water
        external=sensible_energy+latent_energy+water_energy
        correction=external-sum(changes.values())
        top_mass=mass_area[:,0]*new_h[:,0]
        safe_mass=torch.where(wet,top_mass,torch.ones_like(top_mass))
        specific=torch.cat(((correction/safe_mass)[:,None],torch.zeros_like(h[:,1:])), -1)
        end,inversion=apply_heat(ocean,trial,specific,allow_cooling=True)
        final=ocean.check_state(end)
        final_terms,_=column_terms(energy,final)
        residual=sum(final_terms[key]-before[key] for key in before)-external
        scale=(sum(x.abs() for x in before.values())+external.abs()).clamp_min(1)
        if not torch.isfinite(residual).all() or (residual.abs()>2048*torch.finfo(h.dtype).eps*scale).any():
            raise FloatingPointError('surface exchange full-H column energy closure failed')
        mass_residual=mass_area[:,0]*(new_h[:,0]-h[:,0])-dt*area*(p-e)
        momentum_residual=new_metric[:,0]*velocity[:,0]-old_metric[:,0]*state.u_e[:,0]-impulse
        return end,SurfaceExchangeReport(dt*area*p,dt*area*e,mass_residual,impulse,
            momentum_residual,sensible_energy,latent_energy,water_energy,external.sum(),
            changes,correction,mass_area*new_h*inversion,residual,residual.sum())
