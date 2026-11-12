"""Constant prescribed fluxes: closed inventory targets and subdivision audit."""
import json
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean,OceanSurfaceExchange,OceanSurfaceFluxes


def run():
    mesh=VoronoiMesh(n_cells=32,lloyd_iters=2,rotation_scheme='compatible')
    depth=40.+10*mesh.cell_xyz[:,2:3]
    ocean=StratifiedOcean(mesh,OceanColumn(4,total_depth=depth,rest_thickness=torch.full((4,),10.)))
    h=(depth/4).expand(mesh.nCells,4).clone()
    state=ocean.state(u_e=torch.full((mesh.nEdges,4),.2,dtype=h.dtype),h=h,
        temperature=torch.full_like(h,12.),salinity=torch.full_like(h,35.),tracer=torch.ones(*h.shape,1,dtype=h.dtype))
    cell=lambda v:torch.full_like(h[:,0],v)
    fluxes=OceanSurfaceFluxes(precipitation=cell(.001),evaporation=cell(.000005),
        sensible_heat=cell(-10.),latent_heat=cell(-12.5),water_energy_flux=cell(49.75),
        water_momentum_flux=torch.zeros_like(state.u_e[:,0]))
    operator=OceanSurfaceExchange();duration=3600.
    expected_h=h[:,0]+duration*(.001-.000005)/1025
    expected_salt=state.salt_h[:,0]/expected_h
    before=ocean.budgets(state);mass=(ocean.energy.mass_area(state)*h).sum()
    area=ocean.energy.mass_area(state)[:,0]/1025
    expected_energy=duration*area.sum()*(49.75-10.-12.5)
    rows=[];reference=None
    for steps in (1,2,4,8):
        out=state
        for _ in range(steps):out,_=operator.apply(ocean,out,duration/steps,fluxes)
        after=ocean.budgets(out)
        if reference is None:reference=ocean.check_state(out)
        physical=ocean.check_state(out)
        rows.append(dict(substeps=steps,
            maximum_thickness_error_m=float((out.h[:,0]-expected_h).abs().max()),
            maximum_salinity_error=float((physical.salinity()[:,0]-expected_salt).abs().max()),
            salt_content_exact=bool(torch.equal(out.salt_h,state.salt_h)),
            passive_content_exact=bool(torch.equal(out.tracer_h,state.tracer_h)),
            energy_residual_J_kg=float((after['total']-before['total']-expected_energy)/mass),
            temperature_difference_from_single_step_C=float((physical.temperature()-reference.temperature()).abs().max())))
    return dict(scope='Isolated constant prescribed boundary fluxes, 3600 s; no dynamics or turbulence. Subdivision consistency, not a coupled temporal-convergence claim.',results=rows)

if __name__=='__main__':print(json.dumps(run(),indent=2))
