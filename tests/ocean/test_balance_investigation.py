"""Independent identities and corruption sensitivity for diagnostic probes."""
import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('balance',Path(__file__).resolve().parents[2]/'tools/investigate_ocean_balance.py')
b=importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def test_parallel_transport_preserves_tangency_norm_and_inverse():
    torch.set_default_dtype(torch.float64)
    start=torch.tensor([1.,0.,0.]);end=torch.tensor([0.,1.,0.]);v=torch.tensor([0.,2.,3.])
    transported=b.parallel_transport(v,start,end)
    torch.testing.assert_close(transported,torch.tensor([-2.,0.,3.]),rtol=0,atol=0)
    torch.testing.assert_close(transported.norm(),v.norm())
    torch.testing.assert_close(b.parallel_transport(transported,end,start),v)


def test_exact_dual_circulation_against_independent_arc_quadrature():
    torch.set_default_dtype(torch.float64)
    model,_=b.w.case_inputs(16,0)
    mesh=model.mesh;axis=torch.tensor([1.,2.,3.])/14**.5
    c0,c1=mesh.cell_xyz[mesh.cellsOnEdge].unbind(1)
    angle=torch.acos((c0*c1).sum(-1))
    r0=c0;t0=(c1-c0*torch.cos(angle)[:,None])/torch.sin(angle)[:,None]
    samples=[]
    for fraction in (.13,.5,.89):
        x=angle[:,None]*fraction
        r=r0*torch.cos(x)+t0*torch.sin(x)
        tangent=-r0*torch.sin(x)+t0*torch.cos(x)
        velocity=b.w.U0*torch.linalg.cross(axis.expand_as(r),r)
        samples.append((velocity*tangent).sum(-1,keepdim=True))
    exact=b.sampled_circulation(mesh,axis)
    for value in samples:
        torch.testing.assert_close(value,exact,rtol=2e-13,atol=2e-13)
    assert (samples[0]+exact).abs().max()>1  # reversed orientation is detectable


def test_rotation_audit_is_production_and_fixed_q_transpose():
    torch.set_default_dtype(torch.float64)
    model,state=b.w.case_inputs(24,1)
    mesh=model.mesh;dyn=model.dynamics
    q=(mesh.curl(state.u_e)+b.w.OMEGA*mesh.coriolis_unit_vertex[:,None])[mesh.verticesOnEdge].mean(1)
    metric=dyn.energy.metric(state)
    forward,paired=b.rotation_parts(mesh,state.u_e,q,metric)
    assert torch.equal(paired,dyn.rotation_acceleration(state))
    # A separately assembled dense tiny matrix verifies the transpose.
    matrix=torch.zeros(mesh.nEdges,mesh.nEdges)
    for e in range(mesh.nEdges):
        for slot in range(mesh.rotation_weights.shape[1]):
            j=int(mesh.safe_rotation_stencil[e,slot])
            matrix[e,j]+=-.5*(q[e,0]+q[j,0])*mesh.rotation_weights_masked[e,slot]
    expected=.5*(matrix@state.u_e-(matrix.T@(metric*state.u_e))/metric)
    torch.testing.assert_close(paired,expected,rtol=1e-13,atol=1e-17)
    wrong=.5*(matrix@state.u_e+(matrix.T@(metric*state.u_e))/metric)
    assert (wrong-paired).abs().max()>1e-6


def test_diagnostics_separate_components_and_detect_corrupted_rotation():
    torch.set_default_dtype(torch.float64)
    model,state=b.w.case_inputs(16,0)
    report=b.measure(model,state,quadrature=True)
    assert report['rotation_ablation_rms_m_s2']['audit_minus_production']==0
    # This deliberate test-local corruption is not a benchmark modification.
    before=report['geometry']['rotation_constant_vector_defect_rms']
    model.mesh.rotation_weights_masked.zero_()
    after=b.measure(model,state)['geometry']['rotation_constant_vector_defect_rms']
    torch.testing.assert_close(torch.tensor(after),torch.tensor(1.),rtol=2e-14,atol=0)
    assert abs(before-after)>.1
