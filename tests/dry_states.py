"""Seeded dry state and perturbation for shared atmosphere tests."""
import torch
from vfs.atmosphere import DryHydrostaticConservativeDynamics
from vfs.atmosphere.state import State
from vfs.geometry.mesh import VoronoiMesh
from vfs.atmosphere.column import CharneyPhillipsColumn, make_hybrid_coefficients


def setup(levels=3):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    mesh = VoronoiMesh(n_cells=16, lloyd_iters=0, verbose=False)
    a,b = make_hybrid_coefficients(levels,p_top=5000.,sigma_power=1.4)
    col = CharneyPhillipsColumn(levels,a,b,phi_s=500*mesh.cell_xyz[:,0],phi_full_mode='direct_u_gl2_adjoint')
    gen = torch.Generator().manual_seed(18)
    rand = lambda *shape: torch.randn(*shape,generator=gen)
    s = State.from_theta(10*rand(mesh.nEdges,levels),300+3*rand(mesh.nCells,levels+1),1e5+1000*rand(mesh.nCells,1),mesh=mesh,column=col)
    d = DryHydrostaticConservativeDynamics()
    z = State(rand(*s.u_e.shape),100*rand(*s.mtheta_i.shape),100*rand(*s.ps.shape),mesh=mesh,column=col)
    return d,s,z
