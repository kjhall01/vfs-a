"""Stage work is transient, differentiable, and retains physical rejection."""
from dataclasses import replace
import pytest
import torch
from test_coupled_column import ocean, state
from vfs.ocean.stratified.equations import _StageEvaluation


def test_stage_gradients_match_independent_public_evaluations(ocean):
    _,s=state(ocean,None,unequal=True)
    s=s.ocean
    work=_StageEvaluation(ocean.base,s,1.,ocean.energy)
    for dynamic,actual in [(False,work.gradient),(True,work.dynamic_gradient)]:
        expected=ocean.energy.gradients(s,dynamic_only=dynamic)
        for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)


def test_reconstruction_changes_with_new_stage(ocean):
    _,s=state(ocean,None,unequal=True)
    s=s.ocean;old=_StageEvaluation(ocean.base,s,1.,ocean.energy)
    changed=replace(s,entropy_h=s.entropy_h+s.h*.1)
    new=_StageEvaluation(ocean.base,changed,1.,ocean.energy)
    assert not torch.equal(old.physical.temp_h,new.physical.temp_h)
    torch.testing.assert_close(new.physical.temp_h,ocean.check_state(changed).temp_h,rtol=0,atol=0)
    assert not hasattr(ocean,'_stage')


def test_shared_gradient_retains_backward(ocean):
    _,s=state(ocean,None,unequal=True)
    leaf=s.ocean.entropy_h.detach().requires_grad_()
    s=replace(s.ocean,entropy_h=leaf)
    shared=_StageEvaluation(ocean.base,s,1.,ocean.energy)
    reference=ocean.energy.gradients(s)
    # Nonuniform weighting probes spatial derivatives without cancellation.
    weight=torch.linspace(.5,1.5,leaf.numel(),dtype=leaf.dtype).reshape_as(leaf)
    actual=torch.autograd.grad((shared.gradient['entropy_h']*weight).sum(),leaf)[0]
    expected=torch.autograd.grad((reference['entropy_h']*weight).sum(),leaf)[0]
    torch.testing.assert_close(actual,expected,rtol=2e-14,atol=0)


@pytest.mark.parametrize('field',['salt_h','tracer_h'])
def test_stage_retains_invalid_content_rejection(ocean,field):
    _,s=state(ocean,None)
    bad=replace(s.ocean,**{field:-torch.ones_like(getattr(s.ocean,field))})
    with pytest.raises(ValueError):_StageEvaluation(ocean.base,bad,1.,ocean.energy)
