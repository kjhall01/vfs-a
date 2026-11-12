"""Short spherical wind/drag/TKE example; run from an installed VFS wheel."""
import json
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean,TKEOcean,TKEClosure,OceanBoundaryStress


def run(*, bathymetry=False):
    mesh=VoronoiMesh(n_cells=32,lloyd_iters=2,rotation_scheme='compatible')
    depth=100.+40.*mesh.cell_xyz[:,2:3] if bathymetry else torch.full((mesh.nCells,1),100.,dtype=torch.float64)
    column=OceanColumn(16,total_depth=depth,rest_thickness=torch.full((16,),100./16))
    ocean=StratifiedOcean(mesh,column)
    model=TKEOcean(ocean,closure=TKEClosure(2.),
        boundary=OceanBoundaryStress(bottom_drag_coefficient=.0025))
    h=(depth/16).expand(mesh.nCells,16).clone()
    axis=mesh.edge_xyz.new_tensor([0.,0.,1.]).expand_as(mesh.edge_xyz)
    # Smooth eastward forcing, vanishing at the poles; native edge-normal Pa.
    east=torch.linalg.cross(axis,mesh.edge_xyz)
    projection=(east*mesh.edge_normal).sum(-1)
    resolved=ocean.state(u_e=.1*projection[:,None].expand(mesh.nEdges,16),h=h,
        temperature=torch.full_like(h,12.),salinity=torch.full_like(h,35.))
    # Explicit seed: no implicit background TKE or surface turbulence flux.
    state=model.state(resolved,tke=torch.full_like(h,1e-3))
    old=model.budgets(state);work=0.;drag=0.;residual=0.
    for _ in range(5):
        state,report=model.step_with_report(state,10.,surface_stress=.1*projection)
        work+=float(report.boundary.surface_work)
        drag+=float(report.boundary.bottom_ke_loss.sum())
        residual+=float(report.boundary.energy_residual)
    new=model.budgets(state)
    mass=float((ocean.energy.mass_area(state.ocean)*h).sum())
    return dict(cells=32,layers=16,steps=5,dt_s=10.,bathymetry=bathymetry,
        min_depth_m=float(depth.min()),max_depth_m=float(depth.max()),surface_work_J=work,
        drag_to_tke_J=drag,boundary_energy_residual_J=residual,
        full_step_energy_minus_wind_J_kg=float((new['total']-old['total']-work)/mass),
        min_tke_m2_s2=float(state.specific_tke().min()))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bathymetry',action='store_true',help='Use a smooth 60–140 m depth profile')
    print(json.dumps(run(bathymetry=parser.parse_args().bathymetry),indent=2))
