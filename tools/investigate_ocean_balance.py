"""Separate sampling and spatial errors in the frozen ocean Williamson case.

This driver never advances a modified equation. Algebraic ablations below are
measurements of stored stencils, not alternative selectable production solvers.
All accelerations are edge-normal m/s²; kinetic Bernoulli values are cell m²/s².
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import importlib.util
_spec = importlib.util.spec_from_file_location('williamson', Path(__file__).with_name('verify_ocean_williamson.py'))
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)
common = w.common
from vfs.geometry.mesh import VoronoiMesh



def parallel_transport(vector, start, end):
    """Shortest-geodesic transport of a tangent vector between unit positions."""
    return vector - ((vector * end).sum(-1, keepdim=True) /
                     (1 + (start * end).sum(-1, keepdim=True))) * (start + end)


def rotation_parts(mesh, u, q_edge, metric):
    """Audit R_q u and its metric adjoint with fixed q, using saved coefficients.

    q is absolute vorticity [edge, layer], s^-1, not potential vorticity.
    The sparse scatter explicitly transposes the same coefficients. This isolates
    the layer-metric adjustment without changing production vorticity or state.
    """
    stencil = mesh.safe_rotation_stencil
    coefficients = -.5 * (q_edge[:, None] + q_edge[stencil]) * mesh.rotation_weights_masked[..., None]
    forward = (coefficients * u[stencil]).sum(1)
    adjoint = torch.zeros_like(u).index_add(
        0, stencil.flatten(), (coefficients * (metric*u)[:, None]).reshape(-1, u.shape[-1]))
    return forward, .5 * (metric*forward-adjoint)/metric


def sampled_circulation(mesh, axis):
    """Exact solid-body line-average velocity along each dual triangle side.

    For v=U0 axis×r and a great-circle arc c0→c1, v·t is constant,
    U0 axis·normalize(c0×c1). It differs from a Voronoi-face midpoint sample.
    This is a diagnostic sampling alternative, not a replacement initial state.
    """
    c0, c1 = mesh.cell_xyz[mesh.cellsOnEdge].unbind(1)
    normal = torch.linalg.cross(c0, c1)
    normal = normal / normal.norm(dim=-1, keepdim=True)
    return w.U0 * (normal * axis).sum(-1, keepdim=True)


def quadrature_thickness(mesh, order):
    """Independent cell averages; reuse only the geometric Gauss-Duffy rule."""
    quadrature = w.load_tool('transport_references').quadrature
    points, weights, owner, area = quadrature(mesh, order)
    h = w.H0 - (w.RADIUS*w.OMEGA*w.U0 + .5*w.U0**2)/w.GRAVITY * points[:, 2]**2
    average = np.bincount(owner, weights*h, minlength=mesh.nCells)/area
    return torch.from_numpy(average[:, None])


def measure(model, state, *, quadrature=False):
    mesh, dynamics = model.mesh, model.dynamics
    u = state.u_e
    area = mesh.cell_area[:, None]
    edge_area = (mesh.dc_edge * mesh.dv_edge)[:, None]
    edge_rms = lambda a: w.norms(a, edge_area)['l2']
    z = mesh.edge_xyz[:, 2:3]
    pole = z.new_tensor([0., 0., 1.])
    projection = ((pole-z*mesh.edge_xyz)*mesh.edge_normal).sum(-1, keepdim=True)
    exact_rotation = -2*(w.OMEGA+w.U0/w.RADIUS)*w.U0*z*projection
    exact_kinetic_force = w.U0**2/w.RADIUS*z*projection
    q_vertex = mesh.curl(u)+w.OMEGA*mesh.coriolis_unit_vertex[:, None]
    q_edge = q_vertex[mesh.verticesOnEdge].mean(1)
    q_exact = 2*(w.OMEGA+w.U0/w.RADIUS)*z
    forward, paired = rotation_parts(mesh, u, q_edge, dynamics.energy.metric(state))
    exact_q_forward, exact_q_paired = rotation_parts(mesh, u, q_exact, dynamics.energy.metric(state))
    raw_weights = -mesh.tan_weights_masked/mesh.thuburn_edge_sign[:, None]
    raw_stencil = mesh.safe_tan_stencil
    raw_rotation = (-.5*(q_exact[:, None]+q_exact[raw_stencil]) *
                    raw_weights[..., None]*u[raw_stencil]).sum(1)

    # Zeroth spatial moment: rotate a vector parallel-transported from each
    # target to its source edges. The target response should be n×r.
    stencil = mesh.safe_rotation_stencil
    r = mesh.edge_xyz[:, None]
    source_normals = parallel_transport(mesh.edge_normal[stencil], mesh.edge_xyz[stencil], r)
    moment = (mesh.rotation_weights_masked[..., None]*source_normals).sum(1)
    expected = torch.linalg.cross(mesh.edge_normal, mesh.edge_xyz)
    moment_error = (moment-expected).norm(dim=-1, keepdim=True)

    kinetic = dynamics.energy.kinetic_thickness_gradient(state)/(dynamics.rho0*area)
    velocity, _ = w.analytic_fields(mesh.cell_xyz)
    analytic_kinetic = .5*velocity.square().sum(-1, keepdim=True)
    numerical_force = -mesh.grad(kinetic)
    reference_stencil_force = -mesh.grad(analytic_kinetic)

    # Cell KE second moment, mapped to the generator tangent plane. Q=P is
    # necessary to reproduce |v|²/2 for locally constant tangent velocity.
    edges = mesh.safe_edges_on_cell
    normals = parallel_transport(mesh.edge_normal[edges], mesh.edge_xyz[edges], mesh.cell_xyz[:, None])
    weights = mesh.edge_side_area/mesh.edge_side_area.sum(-1, keepdim=True)
    side = (mesh.cellsOnEdge[edges, 1] == torch.arange(mesh.nCells)[:, None]).long()
    local_weight = weights[edges, side] * edge_area[edges, 0] * mesh.valid_edges_on_cell
    Q = torch.einsum('ce,cei,cej->cij',local_weight,normals,normals)/area[..., None]
    P = torch.eye(3)-mesh.cell_xyz[..., None]*mesh.cell_xyz[:, None]

    c0,c1 = mesh.cell_xyz[mesh.cellsOnEdge].unbind(1)
    crossing = (c0+c1)/(c0+c1).norm(dim=-1,keepdim=True)
    displacement = torch.atan2(torch.linalg.cross(crossing,mesh.edge_xyz).norm(dim=-1),
                               (crossing*mesh.edge_xyz).sum(-1))*mesh.radius
    report, _ = w.initial_balance(model,state)
    report.update(
        rotation_ablation_rms_m_s2={
            'raw_with_exact_q':edge_rms(raw_rotation-exact_rotation),
            'geometry_skew_with_exact_q':edge_rms(exact_q_forward-exact_rotation),
            'layer_skew_with_exact_q':edge_rms(exact_q_paired-exact_rotation),
            'layer_skew_increment':edge_rms(paired-forward),
            'raw_geometry_projection_increment':edge_rms(raw_rotation-exact_q_forward),
            'audit_minus_production':edge_rms(paired-dynamics.rotation_acceleration(state))},
        kinetic={'value_error_rms_m2_s2':w.norms(kinetic-analytic_kinetic,area)['l2'],
            'force_error_rms_m_s2':edge_rms(numerical_force-exact_kinetic_force),
            'analytic_value_stencil_error_rms_m_s2':edge_rms(reference_stencil_force-exact_kinetic_force),
            'reconstruction_force_error_rms_m_s2':edge_rms(numerical_force-reference_stencil_force),
            'second_moment_defect_rms':w.norms((Q-P).square().sum((-1,-2)).sqrt()[:,None],area)['l2']},
        geometry={'rotation_constant_vector_defect_rms':edge_rms(moment_error),
            'face_dual_midpoint_offset_over_dc_rms':edge_rms((displacement/mesh.dc_edge)[:,None]),
            'normal_radial_max':float((mesh.edge_normal*mesh.edge_xyz).sum(-1).abs().max()),
            'area_relative_sphere_error':float(area.sum()/(4*np.pi*mesh.radius**2)-1),
            'cell_area_coefficient_variation':float(area.std()/area.mean())},
        manufactured_rotation_unit_speed={})
    for label,axis in [('z',pole),('x',pole.new_tensor([1.,0.,0.])),('tilted',pole.new_tensor([1.,2.,3.])/14**.5)]:
        v=torch.linalg.cross(axis.expand_as(mesh.edge_xyz),mesh.edge_xyz)
        normal=(v*mesh.edge_normal).sum(-1,keepdim=True)
        expected=(torch.linalg.cross(mesh.edge_xyz,v)*mesh.edge_normal).sum(-1,keepdim=True)
        actual=(mesh.rotation_weights_masked[...,None]*normal[stencil]).sum(1)
        report['manufactured_rotation_unit_speed'][label]=edge_rms(actual-expected)
    exact_circulation=sampled_circulation(mesh,pole)
    vertex_reference=2*w.U0/w.RADIUS*mesh.vertex_xyz[:,2:3]
    report['curl']={'face_midpoint_error_rms_s_inverse':w.norms(mesh.curl(u)-vertex_reference,mesh.dual_area[:,None])['l2'],
        'exact_dual_circulation_error_rms_s_inverse':w.norms(mesh.curl(exact_circulation)-vertex_reference,mesh.dual_area[:,None])['l2'],
        'sampling_difference_rms_m_s':edge_rms(exact_circulation-u),
        'reference':'vertex point vorticity; exact dual circulation instead gives its triangle average'}
    # The exact face-average solid-body flux is midpoint flux times sinc(dv/2a).
    face_average=u*torch.sinc(mesh.dv_edge[:,None]/(2*np.pi*mesh.radius))
    report['face_average_minus_point_velocity_rms_m_s']=edge_rms(face_average-u)
    if quadrature:
        coarse,fine=(quadrature_thickness(mesh,k) for k in (6,10))
        report['cell_average']={'orders':[6,10],
            'quadrature_difference_rms_m':w.norms(coarse-fine,area)['l2'],
            'average_minus_point_rms_m':w.norms(fine-state.h,area)['l2'],
            'pressure_force_change_rms_m_s2':edge_rms(-w.GRAVITY*mesh.grad(fine-state.h))}
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cells',type=int,nargs='+',default=[64,256,1024])
    parser.add_argument('--lloyd-iters',type=int,default=100)
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--quadrature',action='store_true')
    args=parser.parse_args()
    if min(args.cells)<16 or len(set(args.cells))!=len(args.cells) or args.lloyd_iters<0:
        parser.error('Use distinct cells >=16 and nonnegative Lloyd count')
    configuration=dict(cells=args.cells,lloyd_iters=args.lloyd_iters,quadrature=args.quadrature,
                       parameters=w.PARAMETERS,scope='stationary spatial diagnostics; production unchanged')
    with common.recorded_run(args.repo,args.output,'OCEAN-ROTATION-DIAGNOSIS',configuration,__file__,
        numerical_revision=common.NUMERICAL_COMMIT,structural_revision=common.STRUCTURAL_COMMIT,
        ocean_revision=common.OCEAN_COMMIT):
        for name in ('verify_ocean_williamson.py','capture_ocean.py','transport_references.py'):
            (args.output/name).write_bytes(Path(__file__).with_name(name).read_bytes())
        rows=[]
        for cells in args.cells:
            started=time.perf_counter()
            model,state=w.case_inputs(cells,args.lloyd_iters,args.inputs)
            w.save_inputs(args.output/f'inputs_{cells}.npz',model,state,args.lloyd_iters)
            row=dict(cells=cells,**measure(model,state,quadrature=args.quadrature))
            row['runtime_seconds']=time.perf_counter()-started
            rows.append(row)
            common.write_json(args.output/'results.json',rows)
            print(cells,row['term_errors_m_s2'],flush=True)


if __name__=='__main__':
    main()
