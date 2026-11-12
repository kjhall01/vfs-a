"""Ocean driven by prescribed freshwater, heat, radiation and stress fluxes."""
import json
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import (StratifiedOcean,TKEOcean,TKEClosure,
    OceanSurfaceExchange,OceanSurfaceFluxes,OceanSurfaceHeating,OceanBoundaryStress)


def run():
    mesh=VoronoiMesh(n_cells=32,lloyd_iters=2,rotation_scheme='compatible')
    depth=40.+10*mesh.cell_xyz[:,2:3]
    ocean=StratifiedOcean(mesh,OceanColumn(8,total_depth=depth,rest_thickness=torch.full((8,),5.)))
    model=TKEOcean(ocean,closure=TKEClosure(2.),exchange=OceanSurfaceExchange(),
        heating=OceanSurfaceHeating(.6,(.5,20.)),boundary=OceanBoundaryStress(.0025))
    h=(depth/8).expand(mesh.nCells,8).clone()
    state=ocean.state(u_e=torch.zeros(mesh.nEdges,8,dtype=h.dtype),h=h,
        temperature=torch.full_like(h,12.),salinity=torch.full_like(h,35.),tracer=torch.ones(*h.shape,1,dtype=h.dtype))
    state=model.state(state,tke=torch.full_like(h,.001))
    cell=lambda x:torch.full_like(h[:,0],x)
    # Synthetic prescribed fluxes, NOT a bulk law or a precipitation-temperature
    # conversion. The energy convention is explicitly the ocean Hamiltonian's.
    fluxes=OceanSurfaceFluxes(precipitation=cell(.001),evaporation=cell(.000005),
        sensible_heat=cell(-10.),latent_heat=cell(-12.5),
        water_energy_flux=cell(49.75),water_momentum_flux=torch.zeros_like(state.ocean.u_e[:,0]))
    before=model.budgets(state);external=0.;water_mass=0.;residual=0.
    area=ocean.energy.mass_area(state.ocean)[:,0]/1025
    mass=float((ocean.energy.mass_area(state.ocean)*h).sum())
    for _ in range(3):
        state,r=model.step_with_report(state,60.,surface_fluxes=fluxes,
            shortwave=cell(200.),surface_heat=cell(-50.),
            surface_stress=torch.full_like(state.ocean.u_e[:,0],.05))
        external+=float(r.exchange.external_energy+r.heating.external_heat+r.boundary.surface_work)
        water_mass+=float((r.exchange.precipitation_mass-r.exchange.evaporation_mass).sum())
        residual+=float(r.exchange.energy_residual)
    after=model.budgets(state)
    return dict(cells=32,layers=8,steps=3,dt_s=60.,net_water_kg_m2=water_mass/float(area.sum()),
        mass_residual_over_initial_mass=float((1025*(after['volume']-before['volume'])-water_mass)/mass),
        salt_relative_change=float((after['salt']-before['salt'])/before['salt']),
        external_energy_J=external,exchange_energy_residual_J_kg=residual/mass,
        combined_energy_minus_inputs_J_kg=float((after['total']-before['total']-external)/mass),
        min_tke=float(state.specific_tke().min()))

if __name__=='__main__':print(json.dumps(run(),indent=2))
