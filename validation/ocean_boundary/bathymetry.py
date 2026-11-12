"""Horizontal drag over varying depth: reconstruction and rest-pressure audit."""
import json
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean,TKEOcean,OceanBoundaryStress


def run():
    rows=[]
    for cells in (32,128,512,2048):
        mesh=VoronoiMesh(n_cells=cells,lloyd_iters=2,rotation_scheme='compatible')
        depth=100.+40.*mesh.cell_xyz[:,2:3]
        column=OceanColumn(4,total_depth=depth,rest_thickness=torch.tensor([10.,20.,30.,40.]))
        ocean=StratifiedOcean(mesh,column)
        model=TKEOcean(ocean,boundary=OceanBoundaryStress(.0025))
        h=depth*depth.new_tensor([.1,.2,.3,.4])
        axis=mesh.edge_xyz.new_tensor([0.,0.,1.]).expand_as(mesh.edge_xyz)
        vectors={'along':torch.linalg.cross(axis,mesh.edge_xyz),
                 'across':axis-mesh.edge_xyz[:,2:3]*mesh.edge_xyz}
        for name,vector in vectors.items():
            vector=.3*vector
            un=(vector*mesh.edge_normal).sum(-1)
            s=ocean.state(u_e=un[:,None].expand(mesh.nEdges,4),h=h,
                temperature=torch.full_like(h,10.),salinity=torch.full_like(h,35.))
            state=model.state(s,tke=torch.zeros_like(h))
            out,report=model.boundary_with_report(state,100.)
            # Analytic vector speed; same native thickness interpolation, so
            # this isolates speed reconstruction rather than thickness quadrature.
            he=mesh.cell_to_edge(h)[:,-1]
            expected=un/(1+100*.0025*vector.norm(dim=-1)/he)
            actual_increment=out.ocean.u_e[:,-1]-un
            reference_increment=expected-un
            area=mesh.dc_edge*mesh.dv_edge
            error=torch.sqrt((area*(actual_increment-reference_increment).square()).sum()/(area*reference_increment.square()).sum())
            rows.append(dict(cells=cells,direction=name,relative_drag_increment_error=float(error),
                energy_residual_over_drag=float(report.energy_residual/report.bottom_ke_loss.sum()),
                min_tke=float(out.specific_tke().min())))
        # A reference-composition rest and a uniform nonreference composition
        # expose existing pressure errors separately from the zero drag at rest.
        for temp in (10.,12.):
            s=ocean.state(u_e=torch.zeros(mesh.nEdges,4,dtype=h.dtype),h=h,
                temperature=torch.full_like(h,temp),salinity=torch.full_like(h,35.))
            state=model.state(s,tke=torch.zeros_like(h))
            out,report=model.boundary_with_report(state,100.)
            pressure=ocean.base.dynamics.tendency_terms(ocean.check_state(s))['u_e']['hamiltonian_pressure']
            rows.append(dict(cells=cells,rest_temperature=temp,
                pressure_rms_m_s2=float(pressure.square().mean().sqrt()),
                boundary_max_velocity_change=float((out.ocean.u_e-s.u_e).abs().max()),
                boundary_tke_gain_J=float(report.bottom_tke_gain.sum())))
    return dict(scope='Horizontal stress per horizontal area; analytic vector-speed comparison at shared discrete thickness. Rest-pressure errors belong to unchanged dynamics. No slope-normal traction or form drag.',results=rows)

if __name__=='__main__':print(json.dumps(run(),indent=2))
