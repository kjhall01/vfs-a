"""Prescribed ocean radiation with explicit optics, bottom export and TKE."""
import json
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean,TKEOcean,TKEClosure,OceanSurfaceHeating


def run():
    mesh=VoronoiMesh(n_cells=32,lloyd_iters=2,rotation_scheme='compatible')
    depth=40.+10.*mesh.cell_xyz[:,2:3]
    column=OceanColumn(16,total_depth=depth,rest_thickness=torch.full((16,),2.5))
    ocean=StratifiedOcean(mesh,column)
    # Illustrative optics; incoming shortwave is already net of reflection.
    heating=OceanSurfaceHeating(first_band_fraction=.6,attenuation_lengths=(.5,20.),bottom='export')
    model=TKEOcean(ocean,closure=TKEClosure(2.),heating=heating)
    h=(depth/16).expand(mesh.nCells,16).clone()
    s=ocean.state(u_e=torch.zeros(mesh.nEdges,16,dtype=h.dtype),h=h,
        temperature=torch.full_like(h,12.),salinity=torch.full_like(h,35.))
    state=model.state(s,tke=torch.full_like(h,.001))
    shortwave=torch.full_like(depth[:,0],200.)
    # Net longwave cooling supplied externally; no atmospheric radiation solver.
    longwave=torch.full_like(shortwave,-50.)
    before=model.budgets(state);incoming=0.;exported=0.;heat_residual=0.
    for _ in range(5):
        state,report=model.step_with_report(state,60.,shortwave=shortwave,surface_heat=longwave)
        incoming+=float(report.heating.external_heat)
        exported+=float(report.heating.bottom_exported_energy.sum())
        heat_residual+=float(report.heating.energy_residual)
    after=model.budgets(state)
    mass=float((ocean.energy.mass_area(s)*h).sum())
    return dict(cells=32,layers=16,dt_s=60.,steps=5,
        net_water_heating_J=incoming,bottom_export_J=exported,
        heating_residual_J_kg=heat_residual/mass,
        full_step_energy_minus_heat_J_kg=float((after['total']-before['total']-incoming)/mass),
        min_tke=float(state.specific_tke().min()),
        min_temperature_C=float(state.ocean.physical_state(ocean.reference).temperature().min()),
        max_temperature_C=float(state.ocean.physical_state(ocean.reference).temperature().max()))

if __name__=='__main__':print(json.dumps(run(),indent=2))
