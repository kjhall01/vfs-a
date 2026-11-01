"""Ten explicit steps of the recovered ocean reference; not a climate benchmark."""
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import Ocean, OceanColumn

torch.set_default_dtype(torch.float64)
mesh=VoronoiMesh(n_cells=32,lloyd_iters=1,verbose=False)
column=OceanColumn(3,total_depth=3000.)
ocean=Ocean(mesh,column)
h=torch.ones(mesh.nCells,3)*1000 + .1*mesh.cell_xyz[:,:1]/3
state=ocean.state(u_e=torch.zeros(mesh.nEdges,3),h=h,
    temperature=torch.full_like(h,10.),salinity=torch.full_like(h,35.))
initial=ocean.budgets(state)
for _ in range(10):
    state=ocean.step(state,30.)
final=ocean.budgets(state)
print('Relative volume change:',float((final['volume']-initial['volume'])/initial['volume']))
print('Minimum layer thickness (m):',float(state.h.min()))
print('The reference energy is incomplete; no Hamiltonian closure is claimed.')
