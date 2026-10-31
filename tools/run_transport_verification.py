"""C5 prescribed horizontal transport through production bounded Euler/SSPRK3.

A run records measurements, not scientific acceptance thresholds. It never
resets carrier, replaces reconstructed fluxes, or uses computed momentum to
advance the prescribed velocity. See verification/INCREMENT_1C.md.
"""
import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import time as clock

import numpy as np
import torch
from vfs import MoistAtmosphere, VoronoiMesh, CharneyPhillipsColumn, make_hybrid_coefficients
from vfs.atmosphere.bounded_moist import bounded_ssprk3_with_info
from vfs.atmosphere.bounded_transport import graph
from vfs.atmosphere.moist_budgets import inventories


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = sibling('transport_references')
energy = sibling('run_energy_verification')
common = energy.common


def normal_velocity(mesh, time, flow):
    cartesian = reference.velocity(mesh.edge_xyz.numpy(), time, flow, mesh.radius)
    return torch.from_numpy((cartesian * mesh.edge_normal.numpy()).sum(-1)).reshape(-1, 1)


def make_case(cells, flow, kind, order):
    mesh = VoronoiMesh(n_cells=cells, lloyd_iters=0, radius=6371220., verbose=False)
    a, b = make_hybrid_coefficients(1, p_top=5000.)
    model = MoistAtmosphere(mesh, CharneyPhillipsColumn(1, a, b, phi_full_mode='direct_u_gl2_adjoint'))
    q, quadrature = reference.cell_average(mesh, flow, 0., kind, order)
    q = torch.from_numpy(q)
    state = model.state_from_fields(
        temperature=280., surface_pressure=100000., normal_wind=normal_velocity(mesh, 0., flow),
        vapor=q[:, :1], liquid=q[:, 1:2], ice=q[:, 2:3],
    )
    return model, state, quadrature


def face_sum(i, j, values, nodes):
    """Independent NumPy histogram audit; no VFS divergence/index_add reuse.

    Checked against the original node-by-node all-face oracle on small meshes
    and in tests. The expensive oracle is not run per node on larger meshes.
    """
    shape = values.shape
    array = values.numpy().reshape(shape[0], -1)
    result = np.stack([np.bincount(j.numpy(), array[:, k], minlength=nodes)
                       - np.bincount(i.numpy(), array[:, k], minlength=nodes)
                       for k in range(array.shape[1])], -1)
    return torch.from_numpy(result).reshape(nodes, *shape[1:])


def audit_stage(model, initial, updated, info, dt, oracle=False):
    gravity = model.dynamics.kernel.gravity
    m0, q0 = energy.independent_amounts(initial, gravity)
    m1, q1 = energy.independent_amounts(updated, gravity)
    _, i, j, flux = graph(model.dynamics, initial)
    carrier_flux = dt*flux/gravity
    species_flux = info['integrated_flux']/gravity
    expected = info['amounts'][:, 1:].reshape_as(updated.water_content)/initial.mesh.cell_area[:, None, None]
    if not torch.equal(updated.water_content, expected):
        raise ValueError('Euler water is not the exact authoritative ledger')
    if not all(bool(torch.isfinite(x).all()) for x in (m1, q1, species_flux, info['beta'])):
        raise ValueError('Nonfinite stage budget')
    if bool((m1 <= 0).any()) or bool((q1[:, :2] <= 0).any()) or bool((q1[:, 2:] < 0).any()):
        raise ValueError('Nonpositive carrier/dry/vapor or negative condensate')
    if bool(((info['beta'] < 0) | (info['beta'] > 1)).any()) or info['limited_faces'] != int((info['beta'] < 1).sum()):
        raise ValueError('Inconsistent discrete limiter diagnostics')
    net_carrier = face_sum(i, j, carrier_flux, m0.numel())
    net_species = face_sum(i, j, species_flux, m0.numel())
    oracle_difference = None
    if oracle:
        oracle_difference = dict(
            carrier_kg=float((net_carrier-energy.incident_sum(i,j,carrier_flux,m0.numel())).abs().max()),
            species_kg=(net_species-energy.incident_sum(i,j,species_flux,m0.numel())).abs().amax(0).tolist())
    # Production may subcycle only the donor remap; report its actual graph
    # outflow divided by the smaller carrier endpoint, not a velocity CFL proxy.
    out = np.bincount(i.numpy(), np.maximum(carrier_flux.numpy(), 0), minlength=m0.numel())
    out += np.bincount(j.numpy(), np.maximum(-carrier_flux.numpy(), 0), minlength=m0.numel())
    q = q1/m1[:, None]
    production = inventories(model.dynamics, updated)
    return dict(
        exact_water_ledger=True, minimum_carrier_kg=float(m1.min()),
        minimum_fraction=q.amin(0).tolist(), maximum_fraction=q.amax(0).tolist(),
        carrier_inventory_kg=float(m1.sum()), species_inventory_kg=q1.sum(0).tolist(),
        carrier_change_kg=float(m1.sum()-m0.sum()), species_change_kg=(q1.sum(0)-q0.sum(0)).tolist(),
        carrier_incidence_max_error_kg=float((m1-m0-net_carrier).abs().max()),
        species_incidence_max_error_kg=(info['amounts']/gravity-q0-net_species).abs().amax(0).tolist(),
        species_flux_sum_minus_carrier_max_kg=float((species_flux.sum(-1)-carrier_flux).abs().max()),
        ledger_sum_minus_carrier_max_kg=float((info['amounts'].sum(-1)/gravity-m1).abs().max()),
        production_mass_minus_independent_kg=float(production['mass']-m1.sum()),
        production_water_minus_independent_kg=(production['species']-q1[:, 1:].sum(0)).tolist(),
        constant_composition_error=float((q[:, 1:]-q.new_tensor([.01,.001,.0001])).abs().max()),
        limited_faces=info['limited_faces'], donor_substeps=info['substeps'], beta_min=float(info['beta'].min()),
        max_carrier_outflow_fraction=float((torch.from_numpy(out)/torch.minimum(m0,m1)).max()),
        max_discrete_velocity_divergence_per_s=float(initial.mesh.div(initial.u_e).abs().max()),
        max_vertical_carrier_flux_kg_per_s=float((flux[initial.mesh.nEdges*2:]/gravity).abs().max()),
        small_oracle_difference=oracle_difference,
    )


def advance(model, state, time, dt, flow, *, raw=False, stage_log=None, step_index=None):
    """Impose wind at RHS times; preserve all production scalar Euler updates.

    RK combinations still compute unused wind tensors. They never enter the
    next transport evaluation; returned wind is the analytical endpoint value.
    This is a CPU float64 verification driver, not a model velocity callback API.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Positive finite timestep required')
    rows, stages = [], []
    def euler(stage, evaluation_time, budget):
        imposed = replace(stage, u_e=normal_velocity(stage.mesh, evaluation_time, flow))
        updated, info = model.transport.euler_with_info(imposed, budget)
        row = audit_stage(model, imposed, updated, info, budget, oracle=state.mesh.nCells <= 16)
        row.update(time_s=evaluation_time, dt_s=budget, stage=len(rows)+1)
        rows.append(row)
        if stage_log is not None:
            row['step'] = step_index
            stage_log.write(json.dumps(row, allow_nan=False)+'\n')
            stage_log.flush()
        if raw:
            stages.append(energy.archive(dict(initial=energy.payload(imposed), output=energy.payload(updated), info=info)))
        return updated, info
    updated, _ = bounded_ssprk3_with_info(euler, state, time, dt)
    updated = replace(updated, u_e=normal_velocity(state.mesh, time+dt, flow))
    model.dynamics.transport_state(updated)
    return updated, rows, stages


def norms(error, weights):
    weights = weights / weights.sum()
    return dict(L1=(weights[:,None]*error.abs()).sum(0).tolist(),
                L2=torch.sqrt((weights[:,None]*error**2).sum(0)).tolist(),
                Linf=error.abs().amax(0).tolist())


def snapshot(model, state, reference_q):
    mass, species = energy.independent_amounts(state, model.dynamics.kernel.gravity)
    mass, species = mass.reshape(-1,2), species.reshape(-1,2,4)
    q = species[...,1:]/mass[...,None]
    reference_q = torch.from_numpy(reference_q)
    # The analytic fraction is vertically uniform, but production Gibbs means
    # need not preserve that property. Include BOTH CP interfaces in errors.
    error = (q-reference_q[:,None,:]).reshape(-1,3)
    area, carrier = state.mesh.cell_area, mass.sum(-1)
    area_nodes = area.repeat_interleave(2)/2
    weights = mass.flatten()/mass.sum()
    flat_q = q.reshape(-1,3)
    mean = (weights[:,None]*flat_q).sum(0)
    variance = (weights[:,None]*(flat_q-mean)**2).sum(0)
    excess = (species[...,1].sum(-1)/carrier)-.01
    centroid = (area[:,None]*excess[:,None]*state.mesh.cell_xyz).sum(0)
    exact_centroid = (area[:,None]*(reference_q[:,:1]-.01)*state.mesh.cell_xyz).sum(0)
    phase = None
    if torch.linalg.norm(centroid)>0 and torch.linalg.norm(exact_centroid)>0:
        phase = float(torch.atan2(torch.linalg.norm(torch.linalg.cross(centroid,exact_centroid)), centroid@exact_centroid))
    return dict(area_error_kg_per_kg=norms(error,area_nodes), carrier_error_kg_per_kg=norms(error,mass.flatten()),
        per_interface_area_error=[norms(q[:,k]-reference_q,area) for k in range(2)],
        carrier_inventory_kg=float(carrier.sum()), species_inventory_kg=species.sum((0,1)).tolist(),
        min_fraction=(species/mass[...,None]).amin((0,1)).tolist(),
        max_fraction=(species/mass[...,None]).amax((0,1)).tolist(),
        max_interface_fraction_difference=float((q[:,0]-q[:,1]).abs().max()),
        pressure_max_error_from_uniform_Pa=float((state.ps-100000).abs().max()),
        peak_loss_kg_per_kg=(reference_q.amax(0)-flat_q.amax(0)).tolist(),
        carrier_weighted_variance=variance.tolist(),
        carrier_weighted_q_one_minus_q=(weights[:,None]*flat_q*(1-flat_q)).sum(0).tolist(),
        excess_vapor_centroid_displacement_rad=phase,
        constant_composition_error=float((q-q.new_tensor([.01,.001,.0001])).abs().max()))


def run_case(repo, output, cells, flow, kind, steps, order=12, numerical_revision=None, structural_revision=None):
    dt = reference.PERIOD/steps
    configuration = dict(cells=cells, layers=1, flow=flow, profile=kind, steps=steps, dt_s=dt,
        period_s=reference.PERIOD, radius_m=6371220., p_top_Pa=5000., temperature_K=280.,
        surface_pressure_Pa=100000., lloyd_iters=0, dtype='float64', device='cpu', threads=1,
        quadrature_orders=[order-4,order], velocity='direct edge-midpoint sample; no projection',
        cadence='metrics every Euler stage; raw stages at first, half, final step; fields at 0,T/2,T',
        guarantees='transport only; not coupled dynamical energy closure')
    started = clock.perf_counter()
    with common.recorded_run(repo, output, 'C5-H-001', configuration, __file__, numerical_revision, structural_revision=structural_revision):
        for source in ('transport_references.py', 'run_energy_verification.py'):
            shutil.copyfile(Path(__file__).with_name(source), output/source)
        model, state, initial_quad = make_case(cells, flow, kind, order)
        torch.save(dict(mesh=model.mesh.state_dict(), column=model.column.state_dict(), initial=energy.payload(state)), output/'inputs.pt')
        references, quad = {}, {}
        for label, t in [('initial',0.),('maximum_deformation',reference.PERIOD/2),('final',reference.PERIOD)]:
            high, detail = reference.cell_average(model.mesh, flow, t, kind, order)
            low, _ = reference.cell_average(model.mesh, flow, t, kind, order-4)
            detail['high_minus_low_error_kg_per_kg'] = norms(torch.from_numpy(high-low),model.mesh.cell_area)
            if flow == 'NL2010_case4' and t == reference.PERIOD/2:
                refined, _ = reference.cell_average(model.mesh, flow, t, kind, order, rtol=2e-12)
                detail['trajectory_tolerance_refinement_max'] = float(np.max(abs(high-refined)))
            references[label], quad[label] = high, detail
        torch.save(references,output/'reference-fields.pt')
        common.write_json(output/'reference-error.json',quad)
        metrics = {'initial':snapshot(model,state,references['initial'])}
        torch.save(energy.payload(state),output/'initial.pt')
        # Flush each stage so a failed run retains all completed stage evidence.
        with (output/'stages.jsonl').open('w') as log:
            for step in range(steps):
                raw = step in (0,steps//2-1,steps-1)
                torch.save(energy.payload(state),output/'last-step-input.pt')
                updated, rows, stages = advance(model,state,step*dt,dt,flow,raw=raw,
                                                 stage_log=log,step_index=step)
                if raw:
                    torch.save(stages,output/f'euler-stages-{step}.pt')
                state = updated
                if step+1 in (steps//2,steps):
                    label = 'final' if step+1 == steps else 'maximum_deformation'
                    metrics[label] = snapshot(model,state,references[label])
                    torch.save(energy.payload(state),output/(label+'.pt'))
                    common.write_json(output/'metrics.json',metrics)
        metrics['runtime_seconds'] = clock.perf_counter()-started
        metrics['artifact_bytes_before_manifest'] = sum(p.stat().st_size for p in output.iterdir())
        common.write_json(output/'metrics.json',metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-revision", choices=[common.STRUCTURAL_COMMIT])
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--numerical-revision', choices=[common.NUMERICAL_COMMIT])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cells', type=int, nargs='+', default=[16,64,256])
    parser.add_argument('--steps', type=int, nargs='+', default=[120,240])
    parser.add_argument('--flows', nargs='+', choices=reference.FLOWS, default=list(reference.FLOWS))
    parser.add_argument('--profile', choices=['smooth','constant','empty_condensate'], default='smooth')
    parser.add_argument('--quadrature-order', type=int, default=12)
    args = parser.parse_args()
    if any(n<4 for n in args.cells) or any(n<2 or n%2 for n in args.steps) or args.quadrature_order<6:
        parser.error('At least four cells, even positive steps, quadrature order >=6 required')
    args.output.mkdir(parents=True,exist_ok=False)
    failures = []
    for cells in args.cells:
        for flow in args.flows:
            for steps in args.steps:
                name = f'{flow}-{args.profile}-{cells}-{steps}'
                try:
                    run_case(args.repo,args.output/name,cells,flow,args.profile,steps,args.quadrature_order,args.numerical_revision, structural_revision=args.structural_revision)
                    print(name,'completed',flush=True)
                except Exception as error:
                    failures.append(dict(case=name,error=f'{type(error).__name__}: {error}'))
                    common.write_json(args.output/'failures.json',failures)
                    print(name,failures[-1]['error'],flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} failed cases; see failures.json and individual manifests')


if __name__ == '__main__':
    main()
