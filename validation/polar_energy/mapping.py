"""Measure energy and tracer mapping defects with the existing NONconservative KNN.

Uses the actual constant-density ocean mechanical Hamiltonian on the source
Voronoi mesh and an analogous face-metric Hamiltonian on the diagnostic grid.
This is a diagnostic experiment, not a new remapper or active-EOS qualification.
"""
import argparse
import json
from pathlib import Path
import torch
from vfs import VoronoiMesh
from vfs.geometry import EquiangularGrid
from vfs.geometry.remap import MeshToLatLonKNN
from vfs.ocean.column import OceanColumn
from vfs.ocean.state import OceanState
from vfs.ocean.energy import ConstantDensityEnergy


def face_kinetic(grid,h,east,north,rho0=1025.):
    """Analogue of rho0/2 sum_e length_e distance_e I(h)_e u_e²."""
    # Polar scalar rows are one cap; east faces there have zero metric weight.
    h=grid._polar_mean(h)
    he=.5*(h+h.roll(-1,-1))
    hn=torch.cat((h[:1],.5*(h[:-1]+h[1:]),h[-1:]))
    return .5*rho0*grid.face_inner(east,north,he*east,hn*north)


def measure(ncells,nlat,poles=False):
    mesh=VoronoiMesh(n_cells=ncells,lloyd_iters=3,rotation_scheme='compatible')
    grid=EquiangularGrid.from_coordinates(torch.linspace(90,-90,nlat+1,dtype=torch.float64),torch.arange(2*nlat,dtype=torch.float64)*180/nlat) if poles else EquiangularGrid(nlat,2*nlat)
    remap=MeshToLatLonKNN.from_mesh(mesh,grid,k=8)
    xyz=mesh.cell_xyz;exyz=mesh.edge_xyz
    h=(100.+2*xyz[:,2]+torch.sin(2*torch.pi*xyz[:,0]))[:,None]
    q=(.2+.8*(xyz[:,0]>.2).double())[:,None]
    salt=(35.+xyz[:,2])[:,None]
    axis=xyz.new_tensor([.3,.4,.5]);divaxis=xyz.new_tensor([.2,-.1,.1])
    velocity=torch.linalg.cross(axis.expand_as(exyz),exyz)+divaxis-(exyz*divaxis).sum(-1,keepdim=True)*exyz
    u=(velocity*mesh.edge_normal).sum(-1,keepdim=True)
    state=OceanState(u,h,10.*h,h*salt,mesh=mesh,column=OceanColumn(1,total_depth=100.))
    energy=ConstantDensityEnergy();source=energy.terms(state)
    # Exact decomposition of the SOURCE diagonal edge Hamiltonian onto cells.
    source_k_density=(h*energy.kinetic_thickness_gradient(state)).sum(-1)/mesh.cell_area
    source_pe_density=.5*energy.rho0*energy.gravity*(h[:,0]-100.).square()
    rec=mesh.cell_wind_vector(u)[:,0,:]
    hg=remap(h)[:,:,0];qg=remap(h*q)[:,:,0]/hg;sg=remap(h*salt)[:,:,0]/hg
    # Map mass-weighted Cartesian components, then project at target locations.
    velocity_g=remap(h*rec)/hg[:,:,None]
    lat,lon=map(torch.deg2rad,grid.meshgrid())
    east_basis=torch.stack((-lon.sin(),lon.cos(),torch.zeros_like(lon)),-1)
    north_basis=torch.stack((-lat.sin()*lon.cos(),-lat.sin()*lon.sin(),lat.cos()),-1)
    ug=(velocity_g*east_basis).sum(-1);vg=(velocity_g*north_basis).sum(-1)
    east,north=grid.wind_to_faces(ug,vg)
    target_k=face_kinetic(grid,hg,east,north,energy.rho0)
    target_pe=.5*energy.rho0*energy.gravity*grid.integral((hg-100.).square())
    density_k=grid.integral(remap(source_k_density))
    density_pe=grid.integral(remap(source_pe_density))
    mass0=energy.rho0*(mesh.cell_area[:,None]*h).sum();mass1=energy.rho0*grid.integral(hg)
    tracer0=energy.rho0*(mesh.cell_area[:,None]*h*q).sum();tracer1=energy.rho0*grid.integral(hg*qg)
    salt0=energy.rho0*(mesh.cell_area[:,None]*h*salt).sum();salt1=energy.rho0*grid.integral(hg*sg)
    fixed=remap(h*.37)[:,:,0]/hg
    result={'mesh_cells':ncells,'target_shape':grid.shape,'layout':'poles' if poles else 'centers',
      'remapper':'existing MeshToLatLonKNN(k=8); NOT conservative',
      'mass_relative_change':float((mass1-mass0)/mass0),
      'tracer_inventory_relative_change':float((tracer1-tracer0)/tracer0),
      'salt_inventory_relative_change':float((salt1-salt0)/salt0),
      'tracer_range':[float(qg.min()),float(qg.max())],
      'constant_tracer_max_error':float((fixed-.37).abs().max()),
      'source_kinetic_J':float(source['kinetic']),
      'source_reconstructed_cell_kinetic_J':float((.5*energy.rho0*mesh.cell_area*h[:,0]*rec.square().sum(-1)).sum()),
      'mapped_kinetic_density_integral_J':float(density_k),
      'target_collocated_kinetic_J':float(.5*energy.rho0*grid.integral(hg*(ug.square()+vg.square()))),
      'target_face_kinetic_J':float(target_k),
      'source_potential_J':float(source['free_surface']),
      'mapped_potential_density_integral_J':float(density_pe),'target_potential_J':float(target_pe),
      'energy_density_transfer_defect_J':float(density_k+density_pe-source['total']),
      'energy_reconstruction_defect_J':float(target_k+target_pe-density_k-density_pe),
      'total_energy_change_J':float(target_k+target_pe-source['total'])}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[measure(n,nlat,poles) for n in (64,256,1024) for nlat in (16,32) for poles in (False,True)]
    a.output.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
