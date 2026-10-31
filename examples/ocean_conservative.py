"""Explicit selection of the constant-density energy-derived ocean."""
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import Ocean, OceanColumn
from vfs.ocean.conservative import ConstantDensityDynamics

torch.set_default_dtype(torch.float64)
mesh=VoronoiMesh(n_cells=32,lloyd_iters=1,verbose=False)
column=OceanColumn(3,total_depth=3000.)
model=Ocean(mesh,column,dynamics=ConstantDensityDynamics())
h=torch.ones(mesh.nCells,3)*1000+.1*mesh.cell_xyz[:,:1]/3
state=model.state(u_e=torch.zeros(mesh.nEdges,3),h=h,
    temperature=torch.full_like(h,10.),salinity=torch.full_like(h,35.))
before=model.budgets(state)
for _ in range(10):
    state=model.step(state,30.)
after=model.budgets(state)
energy='constant_density_mechanical_energy'
print('Relative volume change:',float((after['volume']-before['volume'])/before['volume']))
print('Relative energy change (includes RK time error):',float((after[energy]-before[energy])/before[energy]))
print('T/S are passive; this example does not include stratification energy.')
