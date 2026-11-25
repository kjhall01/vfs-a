"""Williamson (1992) case 2: steady rotating shallow water on a sphere.

Measures the frozen constant-density ocean with production Ocean.step/SSPRK3.
No balancing correction, damping, clipping or numerical acceptance thresholds.
References: doi:10.1016/S0021-9991(05)80016-6; Polaris geostrophic task.
Only the axis-aligned (alpha=0) case is implemented. Constants are explicit and
shared by the analytic initial state and dynamics; temperature/salinity are passive.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import Ocean, OceanColumn, OceanState
from vfs.ocean.conservative import ConstantDensityDynamics


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_tool('run_verification')
capture = load_tool('capture_ocean')
RADIUS = 6371220.0                   # m
GRAVITY = 9.80616                   # m s^-2
OMEGA = 7.292e-5                    # s^-1
U0 = 2 * math.pi * RADIUS / (12 * 86400)  # m s^-1, equatorial speed
H0 = 29400.0 / GRAVITY              # m, equatorial water-column thickness
PARAMETERS = dict(radius_m=RADIUS, gravity_m_s2=GRAVITY, omega_s_inverse=OMEGA,
                  equatorial_speed_m_s=U0, equatorial_thickness_m=H0, alpha=0.)


def analytic_fields(xyz):
    """At unit-sphere positions r, v=U0 (k cross r), h=H0-C (k dot r)^2.

    k is the north-pole unit vector and C=(a Omega U0+U0^2/2)/g.
    These are the steady continuum solution, sampled at cells/edge midpoints;
    they are not adjusted to be an exact discrete equilibrium or cell averages.
    """
    pole = xyz.new_tensor([0., 0., 1.]).expand_as(xyz)
    velocity = U0 * torch.cross(pole, xyz, dim=-1)
    thickness = H0 - (RADIUS * OMEGA * U0 + .5 * U0**2) / GRAVITY * xyz[:, 2:3]**2
    return velocity, thickness


def make_model(mesh):
    column = OceanColumn(1, total_depth=H0, gravity=GRAVITY)
    dynamics = ConstantDensityDynamics(gravity=GRAVITY, omega=OMEGA)
    return Ocean(mesh, column, dynamics=dynamics)


def initial_state(model):
    mesh = model.mesh
    velocity, _ = analytic_fields(mesh.edge_xyz)
    _, h = analytic_fields(mesh.cell_xyz)
    u = (velocity * mesh.edge_normal).sum(-1, keepdim=True)
    return model.state(u_e=u, h=h, temperature=torch.full_like(h, 10.),
                       salinity=torch.full_like(h, 35.), tracer=torch.ones_like(h)[..., None])


def norms(error, weights):
    """Weighted L1, RMS (L2), and unweighted maximum, in the field's units."""
    return dict(l1=float((weights * error.abs()).sum() / weights.sum()),
                l2=float(((weights * error.square()).sum() / weights.sum()).sqrt()),
                linf=float(error.abs().max()))


def initial_balance(model, state):
    """Compare actual RHS terms with independent continuum edge accelerations."""
    mesh = model.mesh
    edge_weight = (mesh.dc_edge * mesh.dv_edge)[:, None]
    terms = model.dynamics.tendency_terms(state)
    pressure = -GRAVITY * mesh.grad(state.eta())
    kinetic = terms['u_e']['bernoulli'] - pressure
    rotation = terms['u_e']['rotation']
    r = mesh.edge_xyz
    z = r[:, 2:3]
    pole = r.new_tensor([0., 0., 1.])
    northward = pole - z * r  # tangent gradient of sin(latitude), times radius
    projection = (northward * mesh.edge_normal).sum(-1, keepdim=True)
    # -g grad(h), -grad(|v|^2/2), and -(f+zeta) r cross v.
    analytic = {
        'pressure': (2 * OMEGA * U0 + U0**2 / RADIUS) * z * projection,
        'kinetic': U0**2 / RADIUS * z * projection,
        'rotation': -2 * (OMEGA + U0 / RADIUS) * U0 * z * projection,
    }
    actual = dict(pressure=pressure, kinetic=kinetic, rotation=rotation)
    rhs_u = rotation + terms['u_e']['bernoulli']
    rhs_h = terms['h']['horizontal']
    gradient = model.dynamics.energy.gradients(state)
    local_u, local_h = gradient['u_e'] * rhs_u, gradient['h'] * rhs_h
    scale = float(local_u.abs().sum() + local_h.abs().sum())
    rate = float(local_u.sum() + local_h.sum())
    report = dict(
        momentum_residual_m_s2=norms(rhs_u, edge_weight),
        thickness_residual_m_s=norms(rhs_h, mesh.cell_area[:, None]),
        term_magnitudes_m_s2={key: norms(value, edge_weight) for key, value in actual.items()},
        term_errors_m_s2={key: norms(actual[key] - analytic[key], edge_weight) for key in actual},
        energy_rate_W=rate,
        energy_work_scale_W=scale,
        energy_rate_normalized=common.normalized_residual(rate, scale),
    )
    arrays = {**{key: value.numpy() for key, value in actual.items()},
              **{'analytic_' + key: value.numpy() for key, value in analytic.items()},
              'rhs_h': rhs_h.numpy(), 'rhs_u': rhs_u.numpy()}
    return report, arrays


def diagnostics(model, state, reference, initial_budgets):
    mesh = model.mesh
    budgets = model.budgets(state)
    drift = {key: float((value - initial_budgets[key]) / initial_budgets[key])
             for key, value in budgets.items() if value.numel() == 1}
    return dict(h_error_m=norms(state.h - reference.h, mesh.cell_area[:, None]),
                u_error_m_s=norms(state.u_e - reference.u_e, (mesh.dc_edge * mesh.dv_edge)[:, None]),
                relative_budget_change=drift, minimum_thickness_m=float(state.h.min()),
                maximum_temperature_error_degC=float((state.temperature() - 10.).abs().max()),
                maximum_salinity_error=float((state.salinity() - 35.).abs().max()),
                maximum_uniform_tracer_error=float((state.tracers() - 1.).abs().max()))


def integrate(model, initial, dt, end_time, sample_interval=86400):
    """Run unchanged production SSPRK3, saving daily metrics and the endpoint."""
    if dt <= 0 or end_time <= 0 or end_time % dt or sample_interval % dt:
        raise ValueError('Positive timestep must divide duration and sample interval')
    state = initial
    budgets = model.budgets(initial)
    series = [dict(time_s=0, **diagnostics(model, state, initial, budgets))]
    with torch.no_grad():
        for step in range(1, int(end_time / dt) + 1):
            state = model.step(state, dt)
            if step % int(sample_interval / dt) == 0 or step * dt == end_time:
                series.append(dict(time_s=step * dt, **diagnostics(model, state, initial, budgets)))
    return state, series


def save_inputs(path, model, state, lloyd_iters):
    """Freeze every buffer, including derived geometry, for exact replay."""
    mesh = model.mesh
    arrays = {'mesh.' + name: value.numpy() for name, value in mesh.named_buffers()}
    arrays.update({key: getattr(state, key).numpy() for key in capture.FIELDS})
    arrays['mesh_metadata'] = np.array(json.dumps({key: getattr(mesh, key) for key in
                                                  ('nCells', 'nEdges', 'nVertices', 'maxEdges', 'radius')}))
    arrays['case_metadata'] = np.array(json.dumps(dict(parameters=PARAMETERS, lloyd_iters=lloyd_iters)))
    np.savez_compressed(path, **arrays)


def case_inputs(cells, lloyd_iters, directory=None):
    if directory is None:
        mesh = VoronoiMesh(n_cells=cells, lloyd_iters=lloyd_iters, radius=RADIUS, verbose=False)
        model = make_model(mesh)
        return model, initial_state(model)
    path = directory / f'inputs_{cells}.npz'
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['case_metadata']))
    if metadata != dict(parameters=PARAMETERS, lloyd_iters=lloyd_iters):
        raise ValueError('Archived benchmark parameters differ')
    mesh, fields = capture.load_inputs(path)
    if mesh.nCells != cells or mesh.radius != RADIUS:
        raise ValueError('Archived mesh size or radius differs')
    model = make_model(mesh)
    state = OceanState(**fields, mesh=mesh, column=model.column)
    reference = initial_state(model)
    for key in capture.FIELDS:
        if not torch.equal(getattr(state, key), getattr(reference, key)):
            raise ValueError(f'Archived analytic initial field differs: {key}')
    return model, state


def refinement_summary(cases):
    """Report observed orders; never turn a slope into a scientific pass/fail."""
    rows = []
    for a, b in zip(cases[:-1], cases[1:]):
        for dt in sorted(set(a['runs']) & set(b['runs']), key=int):
            for field in ('h_error_m', 'u_error_m_s'):
                ea, eb = (case['runs'][dt]['series'][-1][field]['l2'] for case in (a, b))
                order = math.log(ea / eb) / math.log(a['mesh_spacing_m'] / b['mesh_spacing_m']) if ea and eb else None
                rows.append(dict(coarse_cells=a['cells'], fine_cells=b['cells'], dt_s=int(dt), field=field,
                                 observed_order=order))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cells', nargs='+', type=int, default=[64, 256, 1024])
    parser.add_argument('--dt', nargs='+', type=int, default=[600, 300, 150])
    parser.add_argument('--days', type=int, default=5)
    parser.add_argument('--lloyd-iters', type=int, default=100)
    parser.add_argument('--inputs', type=Path, help='replay frozen inputs from a previous run')
    args = parser.parse_args()
    if (args.days <= 0 or args.lloyd_iters < 0 or min(args.cells) < 16 or
        len(set(args.cells)) != len(args.cells) or len(set(args.dt)) != len(args.dt) or
        any(dt <= 0 or 86400 % dt for dt in args.dt)):
        parser.error('Use distinct cells >=16, distinct positive dt dividing one day, and positive days')
    config = dict(parameters=PARAMETERS, cells=sorted(args.cells), dt_s=sorted(args.dt, reverse=True),
                  days=args.days, lloyd_iters=args.lloyd_iters, stepper='production SSPRK3',
                  input_directory=str(args.inputs) if args.inputs else None,
                  sampling='cell-center h and edge-midpoint normal velocity; alpha=0')
    with common.recorded_run(args.repo, args.output, 'OCEAN-WILLIAMSON-2', config, __file__,
            numerical_revision=common.NUMERICAL_COMMIT, structural_revision=common.STRUCTURAL_COMMIT,
            ocean_revision=common.OCEAN_COMMIT):
        # Archive the helper that restores all frozen mesh buffers verbatim.
        (args.output / 'capture_ocean.py').write_bytes(Path(capture.__file__).read_bytes())
        cases = []
        for cells in sorted(args.cells):
            model, initial = case_inputs(cells, args.lloyd_iters, args.inputs)
            save_inputs(args.output / f'inputs_{cells}.npz', model, initial, args.lloyd_iters)
            balance, arrays = initial_balance(model, initial)
            np.savez_compressed(args.output / f'initial_terms_{cells}.npz', **arrays)
            mesh = model.mesh
            case = dict(cells=cells, edges=mesh.nEdges, mesh_spacing_m=math.sqrt(4 * math.pi * RADIUS**2 / cells),
                        initial_balance=balance, runs={}, timestep_comparisons=[])
            previous = None
            for dt in sorted(args.dt, reverse=True):
                endpoint, series = integrate(model, initial, dt, args.days * 86400)
                case['runs'][str(dt)] = dict(steps=args.days * 86400 // dt, series=series)
                np.savez_compressed(args.output / f'final_{cells}_dt{dt}.npz',
                    **{key: getattr(endpoint, key).numpy() for key in capture.FIELDS})
                if previous is not None:
                    before_dt, before = previous
                    case['timestep_comparisons'].append(dict(coarse_dt_s=before_dt, fine_dt_s=dt,
                        h_difference_m=norms(before.h - endpoint.h, mesh.cell_area[:, None]),
                        u_difference_m_s=norms(before.u_e - endpoint.u_e, (mesh.dc_edge * mesh.dv_edge)[:, None])))
                previous = dt, endpoint
                print(json.dumps(dict(cells=cells, dt_s=dt, final=series[-1]), allow_nan=False), flush=True)
                # Retain completed runs even if a later configuration fails.
                common.write_json(args.output / f'case_{cells}.json', case)
            cases.append(case)
        common.write_json(args.output / 'results.json', dict(configuration=config, cases=cases,
            spatial_orders=refinement_summary(cases), limitations=[
                'Point-sampled analytic initial data, not exact finite-volume cell/edge averages',
                'Axis-aligned global one-layer constant-density flow only; no coast or stratification',
                'Finite-time errors mix mesh representation, spatial dynamics and temporal error',
                'No scientific acceptance thresholds or general positivity/long-run qualification']))


if __name__ == '__main__':
    main()
