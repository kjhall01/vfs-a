"""Algebraic transport checks and explicit counterexamples, not climate cutoffs."""
import importlib.util
from pathlib import Path
import pytest
import torch
from vfs.integrators.explicit import ssprk3

spec=importlib.util.spec_from_file_location('admissibility',Path(__file__).resolve().parents[2]/'tools/investigate_ocean_admissibility.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def case(throughflow=True,stepper=None):
    torch.set_default_dtype(torch.float64)
    mesh,_=a.w.capture.load_inputs(Path(__file__).resolve().parents[1]/'fixtures/ocean/inputs.npz')
    return a.make_case(mesh,throughflow=throughflow,**({} if stepper is None else {'stepper':stepper}))


@pytest.mark.parametrize('fraction',[.1,.5,.9])
def test_actual_outflow_condition_preserves_signed_temperature_and_tracer_bounds(fraction):
    model,state=case()
    incoming,outgoing,bound=a.carrier_rates(state)
    rhs=model.tendency(state);dt=fraction*float(bound.min())
    scale=(incoming+outgoing).max()
    assert (state.mesh.cell_area[:,None]*rhs.h-incoming+outgoing).abs().max()<=128*torch.finfo(state.h.dtype).eps*scale
    output=model.step(state,dt)
    eps=128*torch.finfo(state.h.dtype).eps
    assert output.h.min()>model.dynamics.min_thickness
    assert output.temperature().min()>=-2.-eps and output.temperature().max()<=10.+eps
    assert output.salinity().min()>=-eps and output.salinity().max()<=35.+eps
    assert output.tracers().min()>=-eps and output.tracers().max()<=1.+eps


def test_positive_thickness_does_not_imply_salinity_or_tracer_admissibility():
    model,state=case()
    bound=a.carrier_rates(state)[2].min()
    output=model.step(state,1.1*float(bound))
    assert output.h.min()>model.dynamics.min_thickness
    torch.testing.assert_close(output.salinity()[0],torch.tensor([-3.5]),rtol=1e-12,atol=0)
    torch.testing.assert_close(output.tracers()[0],torch.tensor([[-.1]]),rtol=1e-12,atol=0)
    assert output.temperature().max()>10.
    # Conservation remains true while bounds fail.
    for key in ('volume','temperature_content','salinity_content','tracer_content'):
        torch.testing.assert_close(model.budgets(output)[key],model.budgets(state)[key],rtol=128*torch.finfo(state.h.dtype).eps,atol=0)


def test_outflow_only_rejects_negative_thickness_without_repair_or_input_mutation():
    model,state=case(throughflow=False)
    saved={k:getattr(state,k).clone() for k in a.w.capture.FIELDS}
    dt=1.1*float(a.carrier_rates(state)[2].min())
    with pytest.raises(FloatingPointError,match='thickness below minimum'):
        model.step(state,dt)
    for name,value in saved.items():assert torch.equal(getattr(state,name),value)


def test_stage_observer_delegates_exact_ssprk3_and_preserves_gradients():
    observer=a.RecordedSSPRK3();model,state=case(stepper=observer)
    leaves=[getattr(state,k).clone().requires_grad_() for k in a.w.capture.FIELDS]
    state=state.with_tensors(*leaves)
    dt=.1*float(a.carrier_rates(state)[2].min())
    observed=model.step(state,dt);direct=ssprk3(model,state,dt)
    assert len(observer.stages)==3
    for key in a.w.capture.FIELDS:assert torch.equal(getattr(observed,key),getattr(direct,key))
    objective=lambda s:sum((i+1)*getattr(s,k).square().mean() for i,k in enumerate(a.w.capture.FIELDS))
    one=torch.autograd.grad(objective(observed),leaves)
    two=torch.autograd.grad(objective(direct),leaves)
    for x,y in zip(one,two):assert torch.equal(x,y)


def test_final_ssprk3_bounds_do_not_hide_inadmissible_euler_stage():
    observer=a.RecordedSSPRK3();model,state=case(stepper=observer)
    result=model.step(state,1.1*float(a.carrier_rates(state)[2].min()))
    assert result.salinity().min()>=0
    assert observer.stages[0]['euler_concentrations']['salinity']['minimum']<0
    assert observer.stages[0]['maximum_carrier_outflow_fraction']>1


def test_zero_velocity_has_no_advective_bound_but_can_have_wave_acceleration():
    model,state=case()
    state=state.with_tensors(torch.zeros_like(state.u_e),state.h,state.temp_h,state.salt_h,state.tracer_h)
    assert torch.isinf(a.carrier_rates(state)[2]).all()
    assert model.tendency(state).u_e.abs().max()>0
