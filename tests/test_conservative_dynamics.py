import pytest
import torch
from unittest.mock import patch
from vfs.atmosphere import DryHydrostaticConservativeDynamics
from vfs.atmosphere.conservative import v
from vfs.geometry.mesh import VoronoiMesh
from vfs.geometry.column import CharneyPhillipsColumn, make_hybrid_coefficients
from vfs.core.state import State
@pytest.fixture(scope='module')
def mesh():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    return VoronoiMesh(n_cells=64,lloyd_iters=2,verbose=False)

def case(mesh,seed,levels=6,top=5000.):
    gen=torch.Generator().manual_seed(seed)
    A,B=make_hybrid_coefficients(levels,p_top=top,sigma_power=1.)
    col=CharneyPhillipsColumn(levels,A,B,phi_s=0.,phi_full_mode='direct_u_gl2_adjoint')
    theta=300+100*(1-col.B_interface[None,:])+2*torch.randn(mesh.nCells,levels+1,generator=gen)
    ps=1e5+2000*torch.randn(mesh.nCells,1,generator=gen)
    u=10*torch.randn(mesh.nEdges,levels,generator=gen)
    return State.from_theta(u,theta,ps,mesh=mesh,column=col)

@pytest.mark.parametrize('seed,levels,top',[(7,6,5000.),(17,1,5000.),(29,6,100.)])
def test_energy_mass_theta(mesh,seed,levels,top):
 s=case(mesh,seed,levels,top);d=DryHydrostaticConservativeDynamics()
 torch.testing.assert_close(d.total_kinetic_energy(s),(s.mesh.cell_area[:,None]*v(s.layer_dp())*d.cell_kinetic_energy(s)).sum()/d.gravity,rtol=1e-12,atol=1.)
 leaves=[v(getattr(s,k)).clone().requires_grad_() for k in ['u_e','mtheta_i','ps']]
 q=State(*leaves,mesh=mesh,column=s.column)
 ge=torch.autograd.grad(v(d.total_energy(q)),leaves,retain_graph=True)
 closed=d.scalar_energy_gradients(s)
 torch.testing.assert_close(v(closed[0]),ge[1],rtol=2e-11,atol=1e-6)
 torch.testing.assert_close(v(closed[1]),ge[2],rtol=2e-11,atol=1e-3)
 torch.testing.assert_close(v(d.kinetic_energy_gradient_u(s)),ge[0],rtol=2e-11,atol=1e-3)
 g=torch.autograd.grad(v(d.boundary_total_energy(q)),leaves)
 with torch.no_grad(),patch('torch.autograd.grad',side_effect=AssertionError('AD in tendency')):rhs=d.tendency(s)
 work=[(gg*v(getattr(rhs,k))).sum() for gg,k in zip(g,['u_e','mtheta_i','ps'])]
 assert float(abs(sum(work))/sum(abs(w) for w in work).clamp_min(1))<2e-10
 for k in ['ps','mtheta_i']:
  z=mesh.cell_area[:,None]*v(getattr(rhs,k));assert float(abs(z.sum())/z.abs().sum().clamp_min(1))<1e-12


def test_conservative_terms_sum(mesh):
    d = DryHydrostaticConservativeDynamics()
    assert isinstance(d, DryHydrostaticConservativeDynamics)
    s = case(mesh, 71)
    rhs = d.tendency(s)
    decomposed = d.terms_total(d.tendency_terms(s), like=s)
    for key in ("u_e", "mtheta_i", "ps"):
        torch.testing.assert_close(v(getattr(rhs,key)), v(getattr(decomposed,key)), rtol=1e-12, atol=1e-12)


def test_reject_unsupported_column_and_tracers(mesh):
    d = DryHydrostaticConservativeDynamics()
    s = case(mesh, 4)
    s.column.phi_full_mode = "midpoint"
    with pytest.raises(ValueError, match="direct_u_gl2_adjoint"):
        d.tendency(s)
    s = case(mesh, 4)
    s.tracer_m = torch.zeros(mesh.nCells, 6, 1)
    with pytest.raises(ValueError, match="dry states only"):
        d.tendency(s)


def test_boundary_rate_against_independent_derivative(mesh):
    d = DryHydrostaticConservativeDynamics()
    s = case(mesh, 91)
    leaves = [v(getattr(s,k)).clone().requires_grad_() for k in ['u_e','mtheta_i','ps']]
    s = State(*leaves, mesh=mesh, column=s.column)
    gradients = torch.autograd.grad(d.boundary_total_energy(s), leaves)
    direction = State(*(torch.randn_like(x) for x in leaves), mesh=mesh, column=s.column)
    expected = sum((g*v(getattr(direction,k))).sum() for g,k in zip(gradients,['u_e','mtheta_i','ps']))
    with patch('torch.autograd.grad',side_effect=AssertionError('AD in diagnostic')):
        actual = d.boundary_energy_rate_closed(s,direction)
    torch.testing.assert_close(actual,expected,rtol=2e-11,atol=1.)
