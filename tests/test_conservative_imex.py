"""Independent derivative oracles and integration checks for the matched split."""
import pytest
import torch
from vfs.atmosphere import DryHydrostaticConservativeDynamics
from vfs.core.state import State
from vfs.core.integrators import ssprk3
from vfs.geometry.mesh import VoronoiMesh
from vfs.geometry.column import CharneyPhillipsColumn, make_hybrid_coefficients
from vfs.imex import DryPELinearOperator, DryPEIMEXARK, IMEXSolverConfig
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
    stepper=DryPEIMEXARK(d,IMEXSolverConfig(implicit_operator='gravity_wave',preconditioner='hydrostatic_schur',rtol=1e-11,max_iterations=80))
    out,info=stepper.step_with_info(s,1800.)
    assert info.converged
    assert info.linear_operator=='frozen_analytic_conservative_gravity_wave'
    torch.testing.assert_close(d.total_mass(out),d.total_mass(s),rtol=2e-14,atol=0)


def test_schur_and_full_solve_agree():
    d,s,z=setup()
    results=[]
    for pre in ['none','hydrostatic_schur']:
        stepper=DryPEIMEXARK(d,IMEXSolverConfig(implicit_operator='gravity_wave',preconditioner=pre,rtol=1e-11,max_iterations=80))
        results.append(stepper.step(s,600.))
    for name in ['u_e','ps','mtheta_i']:
        torch.testing.assert_close(getattr(results[0],name),getattr(results[1],name),rtol=1e-10,atol=1e-8)


def test_ark4_time_refinement_against_independent_dop853():
    # ARK2's old 60-second interval puts ARK4 at roundoff; retain a meaningful
    # independent time-refinement check at the existing moist test's scales.
    import numpy as np
    from scipy.integrate import solve_ivp

    d, s, _ = setup()
    keys = ('u_e', 'mtheta_i', 'ps')
    shapes = [getattr(s, k).shape for k in keys]
    sizes = [int(np.prod(shape)) for shape in shapes]
    def pack(state):
        return np.concatenate([getattr(state, k).numpy().ravel() for k in keys])
    def unpack(values):
        parts = np.split(values, np.cumsum(sizes)[:-1])
        return State(*(torch.from_numpy(x.reshape(shape).copy()) for x, shape in zip(parts, shapes)),
                     mesh=s.mesh, column=s.column)
    with torch.no_grad():
        reference = solve_ivp(lambda t, y: pack(d.tendency(unpack(y))), [0, 3600], pack(s),
                              method='DOP853', rtol=2e-12, atol=1e-11)
        assert reference.success
        reference = unpack(reference.y[:, -1])
        errors = []
        for dt in (600, 300, 150):
            stepper = DryPEIMEXARK(d, IMEXSolverConfig(
                implicit_operator='gravity_wave', preconditioner='hydrostatic_schur',
                rtol=1e-12, max_iterations=100,
            ))
            out = s
            for _ in range(3600 // dt):
                out = stepper.step(out, dt)
            errors.append(float((out.u_e-reference.u_e).square().mean().sqrt()))
        print('dry ARK4 wind RMS errors', errors)
        assert min(errors[0]/errors[1], errors[1]/errors[2]) > 2**(4-.4)


def test_multilevel_matches_direct_schur():
    d,s,z=setup()
    outputs=[]
    for pre in ['hydrostatic_schur','multilevel_hydrostatic_schur']:
        stepper=DryPEIMEXARK(d,IMEXSolverConfig(implicit_operator='gravity_wave',preconditioner=pre,rtol=1e-10,max_iterations=80))
        outputs.append(stepper.step(s,1800.))
    for name in ['u_e','ps','mtheta_i']:
        torch.testing.assert_close(getattr(outputs[0],name),getattr(outputs[1],name),rtol=1e-9,atol=1e-7)


@pytest.mark.parametrize('batch_shape', [(), (2,), (2, 3)])
def test_coarse_system_restricts_tensor_tracers(batch_shape):
    """Exercise real coarse transfers, independently of the unbatched modal fit."""
    from vfs.geometry.mesh.multilevel import VoronoiMeshHierarchy, restrict_state


    _, state, _ = setup()
    hierarchy = VoronoiMeshHierarchy.build(state.mesh)
    assert hierarchy.transfers  # A no-op hierarchy would miss the regression.
    shape = (*batch_shape, state.mesh.nCells, state.column.n_levels, 2)
    tracer = torch.arange(torch.tensor(shape).prod().item(), dtype=state.ps.dtype).reshape(shape) + 1
    anchor = State(
        state.u_e.expand(*batch_shape, *state.u_e.shape),
        state.mtheta_i.expand(*batch_shape, *state.mtheta_i.shape),
        state.ps.expand(*batch_shape, *state.ps.shape),
        tracer, mesh=state.mesh, column=state.column,
    )
    saved = tracer.clone()
    # Tracer transport in the removed canonical dycore is not retained; test
    # the extracted geometry operation directly, including supported batches.
    coarse = restrict_state(anchor, hierarchy)

    # Independent weighted donor sum, retaining every batch/level/species index.
    expected = tracer
    for transfer in hierarchy.transfers:
        restricted = expected.new_zeros((*batch_shape, transfer.n_coarse_cells, shape[-2], shape[-1]))
        for fine in range(transfer.n_fine_cells):
            for slot in range(transfer.coarse_indices.shape[1]):
                donor = int(transfer.coarse_indices[fine, slot])
                restricted[..., donor, :, :] += (
                    expected[..., fine, :, :] * transfer.restriction_weights[fine, slot]
                )
        expected = restricted
    assert coarse.tracer_m.shape == (*batch_shape, hierarchy.meshes[-1].nCells, shape[-2], shape[-1])
    torch.testing.assert_close(coarse.tracer_m, expected, rtol=0, atol=0)
    assert coarse.mesh is hierarchy.meshes[-1]
    assert anchor.tracer_m is tracer
    torch.testing.assert_close(tracer, saved, rtol=0, atol=0)
