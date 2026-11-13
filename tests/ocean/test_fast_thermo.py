"""Independent kernels, residuals, unchanged guards and native fast mixing."""
import sys
from dataclasses import replace
import torch
import pytest
from test_linear_column import load
from test_lagged_column import lag
from test_coupled_column import ocean,state
sys.modules.setdefault('lagged',lag)
# Loading by file keeps research helpers outside the installed public package.
fast=load('fast_thermo');sys.modules.setdefault('fast_thermo',fast)
from vfs.ocean.jmd95_energy import jmd95_partials


def test_temperature_partial_against_autograd():
    s=torch.tensor([.1,15.,35.,41.9],dtype=torch.float64)
    t=torch.tensor([-2.,5.,20.,39.],dtype=s.dtype,requires_grad=True)
    p=torch.tensor([0.,100.,2000.,6000.],dtype=s.dtype)
    rho=jmd95_partials(s,t,p)[0]
    actual=torch.autograd.grad(rho.sum(),t)[0]
    torch.testing.assert_close(fast.temperature_partial(s,t,p),actual,rtol=1e-12,atol=1e-13)


def test_caloric_and_warm_inverse_against_full_reference(ocean):
    r=ocean.reference;s=torch.linspace(.1,41.9,21,dtype=torch.float64)[None]
    t=torch.linspace(-2.49,39.99,21,dtype=s.dtype)[None]/r.to_90
    c=fast.Caloric(r,s);full=r.evaluate(s,t);cheap=c.evaluate(t)
    for key in cheap:torch.testing.assert_close(cheap[key],full[key],rtol=0,atol=0)
    for key in ('h0','entropy'):
        for guess in (t,t*0+15,t*0+100):
            solved=c.inverse(full[key],guess,key)
            residual=(r.evaluate(s,solved)[key]-full[key]).abs()
            tol=128*torch.finfo(t.dtype).eps*(full[key].abs()+(4000 if key=='h0' else 20))
            assert bool((residual<=tol).all())
            torch.testing.assert_close(solved,t,rtol=0,atol=1e-10)
        with pytest.raises(ValueError):c.inverse(full[key]*0+1e20,t,key)


def test_cached_parcel_and_derivative(ocean):
    _,s=state(ocean,lag.LaggedClosure(2.),unequal=True)
    p=ocean.check_state(s.ocean);w=fast.MixingThermodynamics(ocean,p)
    expected=ocean.energy.base.parcel_potential(p.temperature()[...,None],p.salinity()[...,None],w.z)
    got=w.parcel(p.temperature(),p.salinity())
    for x,y in zip(got,expected):torch.testing.assert_close(x,y,rtol=0,atol=0)
    torch.testing.assert_close(w.temperature_derivative(p.temperature(),p.salinity()),(expected[1]*w.weight).sum(-1),rtol=0,atol=0)


@pytest.mark.parametrize('unstable',[False,True])
@pytest.mark.parametrize('seed',[0.,1e-12,1e-4])
def test_fast_native_budget_and_agreement(ocean,unstable,seed):
    baseline,s=state(ocean,lag.SubcycledLaggedClosure(2.),unstable=unstable,unequal=True,seed=seed)
    candidate=type(baseline)(ocean,closure=lag.SubcycledLaggedClosure(2.,fast_thermodynamics=True))
    expected,old=baseline.mix_with_report(s,30.)
    actual,r=candidate.mix_with_report(s,30.)
    assert bool((actual.tke_h>=0).all()) and r.budget_error<=1
    for field in ('u_e','h','salt_h','entropy_h','tracer_h'):
        torch.testing.assert_close(getattr(actual.ocean,field),getattr(expected.ocean,field),rtol=2e-12,atol=2e-11)
    torch.testing.assert_close(actual.tke_h,expected.tke_h,rtol=2e-9,atol=1e-16)
    for field in ('salt_h','tracer_h'):
        dim=-1 if field=='salt_h' else -2
        torch.testing.assert_close(getattr(actual.ocean,field).sum(dim),getattr(s.ocean,field).sum(dim),rtol=2e-15,atol=0)
    mass=ocean.energy.mass_area(s.ocean)*s.ocean.h
    assert abs(float((candidate.budgets(actual)['total']-candidate.budgets(s)['total'])/mass.sum()))<1e-8


def test_fast_reuses_all_physical_guards(ocean):
    _,s=state(ocean,lag.LaggedClosure(2.));p=ocean.check_state(s.ocean);w=fast.MixingThermodynamics(ocean,p)
    with pytest.raises(ValueError,match='passive tracers'):
        w.physical(replace(s.ocean,tracer_h=-torch.ones_like(s.ocean.tracer_h)),p.temperature())
    with pytest.raises(ValueError):
        w.physical(replace(s.ocean,salt_h=s.ocean.h*43),p.temperature())

@pytest.mark.parametrize('increment',[0.,1e-9,100.,10000.])
def test_heat_increment_independent_energy(ocean,increment):
    from vfs.ocean.stratified.heating import apply_heat
    _,s=state(ocean,lag.LaggedClosure(2.),unequal=True)
    physical=ocean.check_state(s.ocean);w=fast.MixingThermodynamics(ocean,physical)
    q=torch.full_like(s.ocean.h,increment)
    actual,residual,end=w.heat(s.ocean,physical,q)
    expected,_=apply_heat(ocean,s.ocean,q)
    torch.testing.assert_close(actual.entropy_h,expected.entropy_h,rtol=2e-13,atol=2e-11)
    mass=ocean.energy.mass_area(s.ocean)*s.ocean.h
    before=ocean.energy.terms(s.ocean)['total'];after=ocean.energy.terms(actual)['total']
    assert abs(float((after-before-(mass*q).sum())/mass.sum()))<1e-8
    if increment==0:assert actual is s.ocean

@pytest.mark.parametrize('kind',['h0','entropy'])
def test_exact_warm_guess_retains_inverse_derivative(ocean,kind):
    r=ocean.reference;s=torch.full((1,3),35.,dtype=torch.float64)
    t=torch.tensor([[5.,15.,25.]],dtype=s.dtype)
    target=r.evaluate(s,t)[kind].detach().requires_grad_()
    inverse=fast.Caloric(r,s).inverse(target,t,kind)
    derivative=torch.autograd.grad(inverse.sum(),target)[0]
    expected=1/r.evaluate(s,t)['h_theta' if kind=='h0' else 'entropy_theta']
    torch.testing.assert_close(derivative,expected,rtol=2e-14,atol=0)

@pytest.mark.parametrize('kind',['h0','entropy'])
def test_mixed_convergence_batch_keeps_physical_bracket(ocean,kind):
    r=ocean.reference;x=torch.linspace(0,1,4096,dtype=torch.float64)[None]
    salt=35-.5*x;guess=10+5*x
    target_temperature=guess+torch.sin(100*x)*1e-3
    target_temperature[:,::3]=guess[:,::3]
    target=r.evaluate(salt,target_temperature)[kind]
    got=fast.Caloric(r,salt).inverse(target,guess,kind)
    residual=(r.evaluate(salt,got)[kind]-target).abs()
    tolerance=128*torch.finfo(target.dtype).eps*(target.abs()+(4000 if kind=='h0' else 20))
    assert bool((residual<=tolerance).all())


def test_fast_rejects_alternative_energy_kernel(ocean):
    from test_linear_column import linear
    other=linear.LinearColumnOcean(ocean.mesh,ocean.column,constant_cp=False)
    _,s=state(other,lag.LaggedClosure(2.))
    with pytest.raises(TypeError,match='unmodified'):
        fast.MixingThermodynamics(other,other.check_state(s.ocean))

@pytest.mark.parametrize('kind',['h0','entropy'])
def test_warm_inverse_salinity_derivative(ocean,kind):
    r=ocean.reference;s=torch.full((1,3),35.,dtype=torch.float64,requires_grad=True)
    t=torch.tensor([[5.,15.,25.]],dtype=s.dtype)
    ref=r.evaluate(s,t,entropy_derivatives=True)
    target=ref[kind].detach()
    result=fast.Caloric(r,s).inverse(target,t,kind)
    actual=torch.autograd.grad(result.sum(),s)[0]
    expected=-ref['h_s' if kind=='h0' else 'entropy_s']/ref['h_theta' if kind=='h0' else 'entropy_theta']
    torch.testing.assert_close(actual,expected,rtol=2e-12,atol=1e-13)


def test_fast_composes_with_horizontal_tke_step(ocean):
    baseline,s=state(ocean,lag.SubcycledLaggedClosure(2.),unequal=True)
    candidate=type(baseline)(ocean,closure=lag.SubcycledLaggedClosure(2.,fast_thermodynamics=True))
    expected,_=baseline.step_with_report(s,1.)
    actual,report=candidate.step_with_report(s,1.)
    assert report.turbulence.budget_error<=1 and bool((actual.tke_h>=0).all())
    for field in ('u_e','h','salt_h','entropy_h','tracer_h'):
        torch.testing.assert_close(getattr(actual.ocean,field),getattr(expected.ocean,field),rtol=2e-12,atol=2e-11)
    torch.testing.assert_close(actual.tke_h,expected.tke_h,rtol=2e-9,atol=1e-16)
