"""Research composition of bounded contents and existing centered JMD95 forces.

NOT an energy-closing production stepper. No mechanical work correction or
invented heat reservoir is installed. Each full Euler stage retains its ledger.
The optional all-mechanical pairing is diagnosed only, to expose its zero-speed
discontinuity. Run from an installed wheel; outputs contain measurements only.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import vfs
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import Ocean, OceanColumn, OceanDomain, OceanState
from vfs.ocean.jmd95 import JMD95Dynamics
from vfs.ocean.transport import HorizontalTransport, divergence


def transport_for(model, scheme='fct'):
    if not isinstance(model.dynamics, JMD95Dynamics) or model.sources:
        raise ValueError('research composition requires unforced JMD95 dynamics')
    return HorizontalTransport(model.mesh, domain=model.domain, scheme=scheme,
                               min_thickness=model.dynamics.min_thickness)


def ledger_rhs(state, ledger, dt, velocity_rhs):
    area = state.mesh.cell_area[:, None]
    fields = divergence(ledger.i, ledger.j, ledger.integrated_flux/dt,
                        state.h.numel()).reshape(*state.h.shape, -1)/area[..., None]
    hdot = divergence(ledger.i, ledger.j, ledger.carrier, state.h.numel()).reshape_as(state.h)/area
    return state.with_tensors(velocity_rhs, hdot, fields[..., 0], fields[..., 1],
                             None if state.tracer_h is None else fields[..., 2:])


def work_diagnostics(model, state, ledger, dt, rhs):
    """Independent contraction of analytic H derivatives and flux work."""
    gradients = model.dynamics.energy.gradients(state)
    local = [gradients[k]*getattr(rhs, k) for k in ('u_e', 'h', 'temp_h', 'salt_h')]
    area = state.mesh.cell_area[:, None]
    covectors = torch.stack([gradients[k]/area for k in ('temp_h', 'salt_h')], -1).reshape(-1, 2)
    difference = (ledger.integrated_flux-ledger.high_integrated_flux)[:, :2]/dt
    face_work = ((covectors[ledger.j]-covectors[ledger.i])*difference).sum()
    return dict(energy_rate_W=float(sum(x.sum() for x in local).detach()),
                absolute_local_work_W=float(sum(x.abs().sum() for x in local).detach()),
                limiter_work_W=float(face_work.detach()))


def paired_force(model, state, ledger):
    """Counterfactual all-mechanical force: closes work but can jump at u=0.

    Diagnostic only. No division by velocity and no force installed in a step.
    """
    gradients = model.dynamics.energy.gradients(state)
    norm = model.dynamics.rho0*state.mesh.cell_area[:, None]
    cov_grad = torch.stack([state.mesh.grad(gradients[k]/norm)
                           for k in ('temp_h', 'salt_h')], -1)
    if model.domain is None:
        return -(ledger.trace_change[:, :2].reshape(*state.u_e.shape, 2)*cov_grad).sum(-1)
    correction = torch.zeros_like(state.u_e)
    active = model.domain.active_edge
    correction[active] = -(ledger.trace_change[:, :2].reshape(-1, state.h.shape[-1], 2)*cov_grad[active]).sum(-1)
    return correction


def euler(model, state, dt, scheme='fct'):
    model.check_state(state)
    centered = model.tendency(state)
    ledger = transport_for(model, scheme).advance(state, dt)
    result = ledger.install(state, velocity=state.u_e+dt*centered.u_e)
    model.check_state(result)
    model.dynamics.check_state(result)
    return result, ledger, ledger_rhs(state, ledger, dt, centered.u_e)


def step(model, state, dt, scheme='fct'):
    """Three FULL-dt Euler budgets; convex combinations of stored contents."""
    first, l1, _ = euler(model, state, dt, scheme)
    second, l2, _ = euler(model, first, dt, scheme)
    middle = .75*state+.25*second
    third, l3, _ = euler(model, middle, dt, scheme)
    result = (1/3)*state+(2/3)*third
    model.check_state(result)
    model.dynamics.check_state(result)
    return result, (l1, l2, l3)


def active_case(coast=False, zero_salt=False):
    mesh = VoronoiMesh(n_cells=32, lloyd_iters=2, verbose=False)
    domain = OceanDomain(mesh, mesh.cell_xyz[:, 2] > -.4) if coast else None
    column = OceanColumn(3, total_depth=3000.)
    model = Ocean(mesh, column, domain=domain, dynamics=JMD95Dynamics(domain=domain))
    x, y = mesh.cell_xyz[:, :1], mesh.cell_xyz[:, 1:2]
    h = torch.tensor([500., 1000., 1500.])[None, :]*(1+.001*x)
    t = torch.tensor([17., 10., 4.])[None, :]+torch.where(y > 0, 2., -2.)
    s = torch.full_like(h, 35.)+torch.where(x > 0, 1., -1.)
    if zero_salt:
        s = torch.where(x > 0, s, torch.zeros_like(s))
    u = .03*torch.sin(torch.arange(mesh.nEdges)[:, None]*.3+torch.arange(3)[None, :])
    state = model.state(u_e=u, h=h, temperature=t, salinity=s,
                        tracer=(x > 0).to(h.dtype).expand_as(h)[..., None])
    if domain is not None:
        state = domain.project_state(state)
    return model, state


class RingMesh:
    """Unit periodic finite-volume ring; transport-only analytic reference."""
    def __init__(self, n):
        self.nCells = self.nEdges = n
        self.cell_area = torch.full((n,), 1/n)
        self.dv_edge = torch.ones(n)
        k = torch.arange(n)
        self.cellsOnEdge = torch.stack((k, (k+1) % n), -1)

    def cell_to_edge(self, q):
        i, j = self.cellsOnEdge.unbind(-1)
        return .5*(q[i]+q[j])


def ring_state(n=32, sharp=False):
    mesh, column = RingMesh(n), OceanColumn(1, total_depth=1.)
    x = (torch.arange(n)+.5)/n
    # Exact cell averages, not point samples, of the smooth reference.
    q = (x < .5).to(x.dtype) if sharp else .5+.4*torch.sinc(torch.tensor(1/n))*torch.sin(2*math.pi*x)
    h = torch.ones(n, 1)
    return OceanState.from_temperature(torch.ones(n, 1), h, (q-1)[:, None],
                                      q[:, None], q[:, None, None], mesh=mesh, column=column)


def transport_step(state, dt, scheme):
    transport = HorizontalTransport(state.mesh, scheme=scheme)
    def euler_map(s):
        return transport.advance(s, dt).install(s)
    a = euler_map(state)
    b = .75*state+.25*euler_map(a)
    return (1/3)*state+(2/3)*euler_map(b)


def ring_probe(sizes):
    rows = []
    for sharp in (False, True):
        for scheme in ('donor', 'fct'):
            for n in sizes:
                state = ring_state(n, sharp)
                initial = state.salinity().clone()
                steps = math.ceil(.25*n/.2)
                dt = .25/steps
                for _ in range(steps):
                    state = transport_step(state, dt, scheme)
                x = (torch.arange(n)+.5)/n-.25
                reference = torch.roll(initial, n//4, 0) if sharp else (.5+.4*torch.sinc(torch.tensor(1/n))*torch.sin(2*math.pi*x))[:, None]
                q = state.salinity()
                rows.append(dict(sharp=sharp, scheme=scheme, cells=n,
                    L1=float((q-reference).abs().mean()), L2=float((q-reference).square().mean().sqrt()),
                    minimum=float(q.min()), maximum=float(q.max()),
                    inventory_error=float(q.mean()-initial.mean())))
    return rows


def active_probe():
    rows = []
    for coast in (False, True):
        for zero in (False, True):
            model, state = active_case(coast, zero)
            for scheme in ('donor', 'fct'):
                dt = 10.
                end, ledger, rhs = euler(model, state, dt, scheme)
                work = work_diagnostics(model, state, ledger, dt, rhs)
                centered = model.tendency(state)
                centered_end = state+dt*centered
                wet_area = state.mesh.cell_area.sum() if model.domain is None else state.mesh.cell_area[model.domain.wet_cell].sum()
                work['limiter_work_W_m2'] = work['limiter_work_W']/float(wet_area)
                work['centered_energy_rate_W'] = work_diagnostics(model, state, ledger, dt, centered)['energy_rate_W']
                # Counterfactual work pairing is never installed in end.
                corrected = rhs.with_tensors(rhs.u_e+paired_force(model, state, ledger),
                                              rhs.h, rhs.temp_h, rhs.salt_h, rhs.tracer_h)
                work['counterfactual_paired_rate_W'] = work_diagnostics(model, state, ledger, dt, corrected)['energy_rate_W']
                final, stages = step(model, state, dt, scheme)
                budgets0, budgets1 = model.budgets(state), model.budgets(final)
                relative = {key:float(((budgets1[key]-budgets0[key])/budgets0[key].abs()).abs().max())
                            for key in ('volume', 'temperature_content', 'salinity_content', 'tracer_content')}
                rows.append(dict(coast=coast, zero_salt=zero, scheme=scheme, dt_seconds=dt,
                    **work, euler_energy_change_J=float(model.dynamics.total_energy(end)-model.dynamics.total_energy(state)),
                    ssprk3_energy_change_J=float(model.dynamics.total_energy(final)-model.dynamics.total_energy(state)),
                    centered_euler_minimum_salinity=float(centered_end.salinity().min()),
                    relative_inventory_errors=relative, minimum_stage_salinity=min(float((l.amounts[..., 1]/l.volume).min()) for l in stages),
                    maximum_stage_outflow_fraction=max(float(l.outflow_fraction.max()) for l in stages),
                    limited_face_field_fraction=float((ledger.beta < 1).to(state.h.dtype).mean())))
    return rows


def zero_velocity_probe():
    model, initial = active_case(zero_salt=True)
    rows = []
    for scheme in ('donor', 'fct'):
        for eps in (1e-3, 1e-6, 1e-9, 1e-12):
            forces, paired, scalar_rates = [], [], []
            for sign in (-1, 1):
                state = initial.with_tensors(torch.full_like(initial.u_e, sign*eps), initial.h,
                                            initial.temp_h, initial.salt_h, initial.tracer_h)
                _, ledger, rhs = euler(model, state, 10., scheme)
                forces.append(rhs.u_e)
                paired.append(rhs.u_e+paired_force(model, state, ledger))
                scalar_rates.append(rhs.salt_h)
            rows.append(dict(scheme=scheme, epsilon_m_s=eps,
                centered_force_jump_max_m_s2=float((forces[1]-forces[0]).abs().max()),
                counterfactual_paired_jump_max_m_s2=float((paired[1]-paired[0]).abs().max()),
                salt_rate_jump_max=float((scalar_rates[1]-scalar_rates[0]).abs().max())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    root = Path(vfs.__file__).parent
    hashes = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob('*.py'))}
    with torch.no_grad():
        result = dict(scope='CPU float64, prescribed ring and one active SSPRK3 step; no accuracy gates',
            torch_version=torch.__version__, runtime_hashes=hashes,
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            ring=ring_probe((32, 64) if args.quick else (32, 64, 128, 256)),
            active=active_probe(), zero_velocity=zero_velocity_probe())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(f'Measurements written to {args.output}')


if __name__ == '__main__':
    main()
