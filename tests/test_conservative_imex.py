"""Independent derivative oracles and integration checks for the matched split."""
import pytest
import torch
from vfs.atmosphere import DryHydrostaticConservativeDynamics
from vfs.core.state import State
from vfs.core.integrators import ssprk3
from vfs.geometry.mesh import VoronoiMesh
from vfs.geometry.column import CharneyPhillipsColumn, make_hybrid_coefficients
from vfs.imex import DryPELinearOperator, DryPEIMEXARK2, IMEXSolverConfig
from vfs.integrators.imex.conservative_wave import thermal_coefficients


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


@pytest.mark.parametrize('levels',[1,3,12])
def test_coefficients_match_existing_energy_and_gradient(levels):
    d,s,z = setup(levels)
    a,a1,a2 = thermal_coefficients(s,d.gravity)
    gm,gp = d.thermal_gradients(s)
    torch.testing.assert_close(a,gm,rtol=2e-13,atol=1e-4)
    surface = s.mesh.cell_area[:,None]/d.gravity*s.column.phi_s[:,None]
    torch.testing.assert_close((a1*s.mtheta_i).sum(-1,keepdim=True)+surface*s.column.B_interface[-1],gp,rtol=2e-12,atol=1e-3)
    energy = (a*s.mtheta_i).sum()+(surface*s.column.interface_pressure(s.ps)[:,-1:]).sum()
    torch.testing.assert_close(energy,(d.boundary_total_energy(s)-d.total_kinetic_energy(s)).squeeze(),rtol=1e-14,atol=0)


@pytest.mark.parametrize('levels',[1,3,12])
def test_pressure_tangent_against_autodiff_oracle(levels):
    d,s,z=setup(levels)
    op=DryPELinearOperator(d,s,implicit_operator='gravity_wave')
    def pgf(m,p):
        return d.pgf_term(State(s.u_e,m,p,mesh=s.mesh,column=s.column))
    _,expected=torch.autograd.functional.jvp(pgf,(s.mtheta_i,s.ps),(z.mtheta_i,z.ps))
    torch.testing.assert_close(op.gravity_wave_wind(z.mtheta_i,z.ps),expected,rtol=3e-10,atol=2e-16)
    def scalar(u):
        r=d.tendency(State(u,s.mtheta_i,s.ps,mesh=s.mesh,column=s.column))
        return r.mtheta_i,r.ps
    _,expected=torch.autograd.functional.jvp(scalar,(s.u_e,),(z.u_e,))
    got=op.gravity_wave_thermodynamic(z.u_e)
    for x,y in zip(got,expected):torch.testing.assert_close(x,y,rtol=2e-12,atol=1e-12)


def test_analytic_step_uses_no_autodiff_and_conserves_mass(monkeypatch):
    d,s,z=setup()
    def forbidden(*a,**kw):raise AssertionError('runtime autodiff forbidden')
    monkeypatch.setattr(torch.func,'jvp',forbidden)
    monkeypatch.setattr(torch.autograd.functional,'jvp',forbidden)
    monkeypatch.setattr(torch.autograd,'grad',forbidden)
    stepper=DryPEIMEXARK2(d,IMEXSolverConfig(implicit_operator='gravity_wave',preconditioner='hydrostatic_schur',rtol=1e-11,max_iterations=80))
    out,info=stepper.step_with_info(s,1800.)
    assert info.converged
    assert info.linear_operator=='frozen_analytic_conservative_gravity_wave'
    torch.testing.assert_close(d.total_mass(out),d.total_mass(s),rtol=2e-14,atol=0)


def test_schur_and_full_solve_agree():
    d,s,z=setup()
    results=[]
    for pre in ['none','hydrostatic_schur']:
        stepper=DryPEIMEXARK2(d,IMEXSolverConfig(implicit_operator='gravity_wave',preconditioner=pre,rtol=1e-11,max_iterations=80))
        results.append(stepper.step(s,600.))
    for name in ['u_e','ps','mtheta_i']:
        torch.testing.assert_close(getattr(results[0],name),getattr(results[1],name),rtol=1e-10,atol=1e-8)


def test_time_refinement_recovers_explicit_solution():
    d,s,z=setup()
    with torch.no_grad():
        ref=s
        for _ in range(64):ref=ssprk3(d,ref,60/64)
        errors=[]
        for dt in [30.,15.,7.5]:
            stepper=DryPEIMEXARK2(d,IMEXSolverConfig(implicit_operator='gravity_wave',preconditioner='hydrostatic_schur',rtol=1e-12,max_iterations=80))
            out=s
            for _ in range(round(60/dt)):out=stepper.step(out,dt)
            errors.append(float((out.u_e-ref.u_e).square().mean().sqrt()))
        assert errors[0]/errors[1]>3.5
        assert errors[1]/errors[2]>3.5


def test_multilevel_matches_direct_schur():
    d,s,z=setup()
    outputs=[]
    for pre in ['hydrostatic_schur','multilevel_hydrostatic_schur']:
        stepper=DryPEIMEXARK2(d,IMEXSolverConfig(implicit_operator='gravity_wave',preconditioner=pre,rtol=1e-10,max_iterations=80))
        outputs.append(stepper.step(s,1800.))
    for name in ['u_e','ps','mtheta_i']:
        torch.testing.assert_close(getattr(outputs[0],name),getattr(outputs[1],name),rtol=1e-9,atol=1e-7)
