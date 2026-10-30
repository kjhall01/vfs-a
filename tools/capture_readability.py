"""Same-environment refactor capture; never writes or regenerates fixtures.

Run the baseline installed wheel with --save /tmp/vfs-readability-before.pt,
then the changed wheel with --compare that file. --original selects the
pre-refactor ARK4 constructor. Capture detaches only copies for serialization;
all diagnostic gradients are evaluated on the original connected graph.
"""
import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path
import torch
from vfs import State, MoistState, VoronoiMesh, CharneyPhillipsColumn, make_hybrid_coefficients
from vfs.atmosphere import DryHydrostaticConservativeDynamics, MoistConservativeDynamics, BoundedGibbs
from vfs.core.integrators import ssprk3
from vfs.imex import DryPEIMEXARK, MoistPEIMEXARK, IMEXSolverConfig

DRY = ('u_e', 'mtheta_i', 'ps')
MOIST = ('u_e', 'ps', 'entropy_content', 'water_content')


def clean(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, (State, MoistState)):
        return {k: clean(getattr(x, k)) for k in (MOIST if isinstance(x, MoistState) else DRY)}
    if is_dataclass(x):
        return clean(asdict(x))
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [clean(v) for v in x]
    return x


def cases():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    mesh = VoronoiMesh(n_cells=16, lloyd_iters=0, verbose=False)
    A, B = make_hybrid_coefficients(3, p_top=5000., sigma_power=1.4)
    col = CharneyPhillipsColumn(3, A, B, phi_s=500*mesh.cell_xyz[:, 0], phi_full_mode='direct_u_gl2_adjoint')
    gen = torch.Generator().manual_seed(18)
    rand = lambda *shape: torch.randn(*shape, generator=gen)
    dry = State.from_theta(10*rand(mesh.nEdges, 3), 300+3*rand(mesh.nCells, 4),
                           1e5+1000*rand(mesh.nCells, 1), mesh=mesh, column=col)
    # The existing test_moist_imex.setup / test_moist_conservative.case state.
    torch.manual_seed(44)
    A, B = make_hybrid_coefficients(3, p_top=5000., sigma_power=1.)
    col = CharneyPhillipsColumn(3, A, B, phi_s=1000*torch.rand(mesh.nCells), phi_full_mode='direct_u_gl2_adjoint')
    ps = torch.ones(mesh.nCells, 1)*1e5
    ps += 2000*torch.randn_like(ps)
    p = col.interface_pressure(ps)
    from vfs.atmosphere.moist_cp import interface_pressure_mass
    dm = interface_pressure_mass(p.diff())
    entropy = 200+80*torch.log(1e5/p)
    q = torch.ones(mesh.nCells, 4, 3)*torch.tensor([.01, .001, .0001])
    u = torch.zeros(mesh.nEdges, 3)
    entropy += torch.randn_like(entropy)*10
    q *= .5+torch.rand_like(q)
    u += torch.randn_like(u)*10
    return dry, MoistState(u, ps, dm*entropy, dm[..., None]*q, mesh, col)


def capture(original=False):
    dry, moist = cases()
    result = {}
    for name, state, dynamics, keys in (
        ('dry', dry, DryHydrostaticConservativeDynamics(), DRY),
        ('moist', moist, MoistConservativeDynamics(), MOIST),
    ):
        values = {'initial': clean(state), 'rhs': clean(dynamics.tendency(state)),
                  'terms': clean(dynamics.tendency_terms(state)),
                  'energy': clean(dynamics.energy_terms(state))}
        leaves = [getattr(state, k).clone().requires_grad_() for k in keys]
        s = (MoistState(*leaves, state.mesh, state.column) if name == 'moist'
             else State(*leaves, mesh=state.mesh, column=state.column))
        values['energy_gradient'] = clean(torch.autograd.grad(dynamics.boundary_total_energy(s), leaves))
        rhs = dynamics.tendency(s)
        values['rhs_vjp'] = clean(torch.autograd.grad(sum(getattr(rhs, k).sum() for k in keys), leaves))
        if name == 'dry':
            values['explicit'] = clean(ssprk3(dynamics, state, 1.))
        else:
            bounded = BoundedGibbs(dynamics, release_as_heat=True)
            values['bounded_rhs'], values['ledger'] = clean(bounded.evaluate(state, 1.))
            values['euler'], values['euler_ledger'] = clean(bounded.euler_with_info(state, 1.))
            values['explicit'], values['stage_ledgers'] = clean(bounded.step_with_info(state, 1.))
        for pre in (('none', 'hydrostatic_schur', 'multilevel_hydrostatic_schur') if name == 'dry'
                    else ('none', 'hydrostatic_schur')):
            cfg = IMEXSolverConfig(implicit_operator='gravity_wave', preconditioner=pre,
                                   rtol=1e-11, max_iterations=100)
            factory = DryPEIMEXARK if name == 'dry' else MoistPEIMEXARK
            kwargs = {'method': 'ark4'} if original else {}
            solver = factory(dynamics, cfg, **kwargs)
            stages = []
            real = dynamics.tendency
            def record(s):
                stages.append(clean(s))
                return real(s)
            # Dry captures its callable on construction; moist reads dynamics.
            if name == 'dry':
                solver._tendency = record
            else:
                dynamics.tendency = record
            try:
                out, info = solver.step_with_info(state, 600.)
            finally:
                if name == 'moist':
                    del dynamics.tendency
            assert info.converged
            values['ark4_'+pre] = dict(state=clean(out), stages=stages, info=clean(info))
            print(name, pre, info.total_iterations, [i.relative_residual for i in info.stages])
        result[name] = values
    return result


def compare(a, b, path='capture'):
    if isinstance(b, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True, msg=lambda s: f'{path}: {s}')
        return 1
    if isinstance(b, dict):
        assert a.keys() == b.keys(), path
        return sum(compare(a[k], b[k], path+'/'+k) for k in b)
    if isinstance(b, list):
        assert len(a) == len(b), path
        return sum(compare(x, y, path+f'/{i}') for i, (x, y) in enumerate(zip(a, b)))
    assert a == b, (path, a, b)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--save', type=Path)
    group.add_argument('--compare', type=Path)
    p.add_argument('--original', action='store_true')
    args = p.parse_args()
    result = capture(args.original)
    if args.save:
        if args.save.exists():
            raise FileExistsError(args.save)
        torch.save(result, args.save)
    else:
        print('Exactly equal tensors:', compare(result, torch.load(args.compare, weights_only=True)))
