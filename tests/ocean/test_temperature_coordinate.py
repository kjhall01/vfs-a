"""Independent tests for the research coordinate change, not full-ocean support."""
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from test_jmd95_hamiltonian import mesh,case
from vfs.ocean.stratified import StratifiedOcean
from vfs.ocean.stratified import transport
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'validation/temperature_coordinate'))
from model import TemperatureOcean

@pytest.mark.parametrize('coast',[False,True])
def test_derivatives_work_and_chain_rule(mesh,coast):
    old,s=case(mesh,coast)
    ocean=StratifiedOcean(mesh,old.column,domain=old.domain)
    candidate=TemperatureOcean(ocean)
    keys=('u_e','h','temp_h','salt_h')
    leaves=[getattr(s,k).clone().requires_grad_() for k in keys]
    s=s.with_tensors(*leaves,s.tracer_h)
    expected=torch.autograd.grad(candidate.energy(s),leaves)
    analytic=candidate.gradients(s)
    for key,g in zip(keys,expected):
        torch.testing.assert_close(analytic[key],g,rtol=2e-12,atol=.1)
    rhs,_=candidate.tendency_with_report(s,1.)
    terms=[g*getattr(rhs,k) for k,g in zip(keys,expected)]
    assert abs(sum(v.sum() for v in terms)) < 2048*torch.finfo(s.h.dtype).eps*sum(v.abs().sum() for v in terms)
    eta,ref=candidate.fields(s)
    erhs,_=transport.evaluate(ocean.base,eta,1.,ocean.energy,physical=s)
    reconstructed=ref['entropy_theta']*rhs.temp_h+ref['entropy_s']*rhs.salt_h+(ref['entropy']-s.temperature()*ref['entropy_theta']-s.salinity()*ref['entropy_s'])*rhs.h
    torch.testing.assert_close(reconstructed,erhs.entropy_h,rtol=2e-12,atol=1e-15)
    torch.testing.assert_close(rhs.u_e,erhs.u_e,rtol=0,atol=0)
    # Independent forward directional derivative through the coordinate map.
    _,jvp=torch.autograd.functional.jvp(lambda h,t,salt:h*ocean.reference.evaluate(salt/h,t/h)['entropy'],
        (s.h,s.temp_h,s.salt_h),(rhs.h,rhs.temp_h,rhs.salt_h))
    torch.testing.assert_close(jvp,erhs.entropy_h,rtol=2e-12,atol=1e-15)


def test_step_has_no_inverse_and_keeps_inventories(mesh):
    old,s=case(mesh)
    candidate=TemperatureOcean(StratifiedOcean(mesh,old.column))
    with patch('vfs.ocean.stratified.equations.temperature_from_entropy',side_effect=AssertionError('inverse called')):
        end=candidate.step(s,1.)
    area=mesh.cell_area[:,None]
    for key in ('h','salt_h','tracer_h'):
        a=area[...,None] if key=='tracer_h' else area
        torch.testing.assert_close((a*getattr(end,key)).sum(),(a*getattr(s,key)).sum(),rtol=3e-15,atol=0)
    assert end.salt_h.min()>0 and end.tracer_h.min()>=0
    with pytest.raises(ValueError):candidate.step(s,0.)


def test_front_step_backprop_and_stage_rejection(mesh):
    old,s=case(mesh)
    candidate=TemperatureOcean(StratifiedOcean(mesh,old.column))
    front=(mesh.cell_xyz[:,:1]>0)
    t=torch.where(front,torch.full_like(s.h,18.),torch.full_like(s.h,4.))
    salt=torch.where(front,torch.full_like(s.h,34.),torch.full_like(s.h,36.))
    theta=(s.h*t).requires_grad_()
    s=s.with_tensors(s.u_e,s.h,theta,s.h*salt,s.tracer_h)
    end=candidate.step(s,1.)
    derivative=torch.autograd.grad(end.temp_h.sum(),theta)[0]
    assert torch.isfinite(derivative).all()
    assert end.salt_h.min()>0 and end.tracer_h.min()>=0
    bad=s.with_tensors(s.u_e,s.h,s.h*50,s.salt_h,s.tracer_h)
    with pytest.raises(ValueError):candidate.step(bad,1.)
