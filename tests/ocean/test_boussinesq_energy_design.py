"""Tests of a proposed energy derivation, not a selectable ocean implementation."""
import importlib.util
from pathlib import Path
import pytest
import torch

spec=importlib.util.spec_from_file_location('design',Path(__file__).resolve().parents[2]/'tools/probe_ocean_boussinesq_energy.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def case():
    torch.set_default_dtype(torch.float64)
    mesh,fields=p.w.capture.load_inputs(Path(__file__).resolve().parents[1]/'fixtures/ocean/inputs.npz')
    return p.active_case(mesh,fields)


def test_column_integral_and_fixed_content_derivatives_independently():
    model,state=case()
    leaves=[getattr(state,k).clone().requires_grad_() for k in ('h','temp_h','salt_h')]
    state=state.with_tensors(state.u_e,*leaves,state.tracer_h)
    energy,grad=p.anomaly_energy(state)
    rho0,g=state.column.rho0,state.column.gravity
    density=rho0*(-p.OCEAN_ALPHA_T*(state.temperature()-p.OCEAN_T0)+p.OCEAN_BETA_S*(state.salinity()-p.OCEAN_S0))
    z_top=state.eta();independent=0.
    for k in range(3):
        z_bottom=z_top-state.h[:,k:k+1]
        independent=independent+(.5*g*state.mesh.cell_area[:,None]*density[:,k:k+1]*(z_top.square()-z_bottom.square())).sum()
        z_top=z_bottom
    torch.testing.assert_close(energy,independent,rtol=128*torch.finfo(state.h.dtype).eps,atol=0)
    ad=torch.autograd.grad(independent,leaves)
    for name,value in zip(('h','temp_h','salt_h'),ad):
        torch.testing.assert_close(grad[name],value,rtol=256*torch.finfo(state.h.dtype).eps,atol=0)
    # At fixed T,S the h derivative is different: do not silently use it.
    fixed_TS=grad['h']+state.temperature()*grad['temp_h']+state.salinity()*grad['salt_h']
    assert (fixed_TS-grad['h']).abs().max()>1e15


@pytest.mark.parametrize('field',['h','temp_h','salt_h'])
def test_directional_derivative_roundoff_bound(field):
    _,state=case();base=getattr(state,field)
    direction=torch.cos(torch.arange(base.numel()).reshape_as(base)*.31)
    energy,grad=p.anomaly_energy(state);predicted=(grad[field]*direction).sum()
    delta=.1;values=[]
    for sign in (-1,1):
        fields={k:getattr(state,k) for k in p.w.capture.FIELDS}
        fields[field]=base+sign*delta*direction
        values.append(p.anomaly_energy(state.with_tensors(*(fields[k] for k in p.w.capture.FIELDS)))[0])
    fd=(values[1]-values[0])/(2*delta)
    # The energy is quadratic/affine in these variables; central differences
    # have no truncation term. Bound only floating-point subtraction error.
    scale=max(abs(values[0]),abs(values[1]))
    assert abs(fd-predicted)<=128*torch.finfo(base.dtype).eps*scale/delta


def test_content_force_pairing_and_deliberately_wrong_trace():
    model,state=case();_,grad=p.anomaly_energy(state)
    rhs=model.tendency(state);Hu=model.dynamics.energy.gradients(state)['u_e']
    force=p.paired_force(state,grad)
    local=[Hu*force]+[grad[k]*getattr(rhs,k) for k in grad]
    residual=sum(x.sum() for x in local);scale=sum(x.abs().sum() for x in local)
    assert abs(residual)<=128*torch.finfo(state.h.dtype).eps*scale
    # Dropping the active temperature pairing must be detectable.
    c0,c1=state.mesh.cellsOnEdge.unbind(1)
    T=torch.where(state.u_e>=0,state.temperature()[c0],state.temperature()[c1])
    omitted=T*state.mesh.grad(grad['temp_h']/(state.column.rho0*state.mesh.cell_area[:,None]))
    wrong=residual+(Hu*omitted).sum()
    assert abs(wrong)>1e-6*scale


def test_constant_density_limit_is_exact_and_rest_cases_are_measured():
    model,state=case();zero,grad=p.anomaly_energy(state,alpha=0.,beta=0.)
    assert zero==0
    assert torch.count_nonzero(p.paired_force(state,grad))==0
    rhs=model.tendency(state)
    assert torch.equal(rhs.u_e+p.paired_force(state,grad),rhs.u_e)
    for topography in (False,True):
        row=p.rest_probe(state.mesh,topography=topography)
        assert row['maximum_eta_m']==0
        # No scientific rest-error threshold: retain finite measured residuals.
        assert torch.isfinite(torch.tensor(row['force_m_s2']['l2']))
