"""Seeded moist state and perturbation for atmosphere ownership tests."""
import torch
from test_moist_conservative import case
from vfs.atmosphere import MoistConservativeDynamics
from vfs.geometry.mesh import VoronoiMesh
from vfs.atmosphere.moist_conservative import MoistState

def setup(levels=3):
 torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
 mesh=VoronoiMesh(n_cells=16,lloyd_iters=0,verbose=False)
 d=MoistConservativeDynamics();s=case(mesh,levels)
 g=torch.Generator().manual_seed(42)
 z=MoistState(torch.randn(s.u_e.shape,generator=g),50*torch.randn(s.ps.shape,generator=g),100*torch.randn(s.entropy_content.shape,generator=g),.01*torch.randn(s.water_content.shape,generator=g),mesh,s.column)
 return d,s,z
