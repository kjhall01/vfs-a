"""Temperature-native thermal updates keep the existing acceptance budgets."""
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'validation/ocean_column'))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'validation/temperature_coordinate'))
from resolution_mesh import setup
from model import TemperatureOcean
from physics import TemperatureMixing,heat
from lagged import SubcycledLaggedClosure
from fast_thermo import Caloric

@pytest.fixture
def column():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    o,e,k=setup(16,4)
    return o,e,o.check_state(e),k

@pytest.mark.parametrize('amount',[0.,1.,-1.,100.])
def test_prescribed_heat_no_entropy_inverse(column,amount):
    o,e,s,k=column;c=TemperatureOcean(o)
    with patch('vfs.ocean.stratified.equations.temperature_from_entropy',side_effect=AssertionError('entropy inversion')):
        end,res=heat(o,s,torch.full_like(s.h,amount),allow_cooling=amount<0)
    mass=o.energy.mass_area(s)*s.h
    torch.testing.assert_close((c.energy(end)-c.energy(s))/mass.sum(),s.h.new_tensor(amount),rtol=0,atol=1e-9)
    assert torch.equal(end.h,s.h) and torch.equal(end.salt_h,s.salt_h)
    if amount==0:assert end is s

@pytest.mark.parametrize('factor',[1.,1e-6])
def test_mixing_budgets_inventory_and_no_entropy_inverse(column,factor):
    o,e,s,k=column;c=TemperatureOcean(o);initial=s.h*k*factor
    kinds=[];original=Caloric.inverse
    def counted(self,target,guess,kind):
        kinds.append(kind);return original(self,target,guess,kind)
    with patch('vfs.ocean.stratified.equations.temperature_from_entropy',side_effect=AssertionError('entropy inversion')),patch.object(Caloric,'inverse',counted):
        end,kh,r=TemperatureMixing(2.,substeps=2).apply(o,s,initial,10.)
    assert kinds and set(kinds)=={'h0'}
    assert (kh>=0).all() and (end.tracer_h>=0).all()
    assert r.budget_error<=1
    m=o.energy.mass_area(s)
    drift=(c.energy(end)-c.energy(s)+(m*(kh-initial)).sum())/(m*s.h).sum()
    assert abs(drift)<1e-8
    torch.testing.assert_close((m*end.salt_h).sum(),(m*s.salt_h).sum(),rtol=3e-15,atol=0)
    torch.testing.assert_close((m[...,None]*end.tracer_h).sum(),(m[...,None]*s.tracer_h).sum(),rtol=3e-15,atol=0)


def test_mixing_compare_and_heating_backward(column):
    o,e,s,k=column;initial=s.h*k
    a,ka,ra=SubcycledLaggedClosure(2.,substeps=2,fast_thermodynamics=True).apply(o,e,initial,10.)
    b,kb,rb=TemperatureMixing(2.,substeps=2).apply(o,s,initial,10.)
    ap=o.check_state(a)
    torch.testing.assert_close(b.temperature(),ap.temperature(),rtol=0,atol=2e-10)
    torch.testing.assert_close(kb,ka,rtol=1e-6,atol=1e-12)
    h=torch.full_like(s.h,1.,requires_grad=True)
    end,_=heat(o,s,h)
    derivative=torch.autograd.grad(end.temp_h.sum(),h)[0]
    assert torch.isfinite(derivative).all() and (derivative>0).all()
    with pytest.raises(ValueError):heat(o,s,-torch.ones_like(s.h))
    with pytest.raises(ValueError):heat(o,s,torch.full_like(s.h,float('nan')))

@pytest.mark.parametrize('unstable',[False,True])
@pytest.mark.parametrize('seed',[0.,1e-12,1e-4])
def test_native_stable_unstable_tke(column,unstable,seed):
    o,e,s,k=column
    t=s.h.new_tensor([15.,14.,13.,12.])
    if unstable:t=t.flip(0)
    s=s.with_tensors(s.u_e,s.h,s.h*t,s.salt_h,s.tracer_h)
    initial=s.h*seed
    end,kh,r=TemperatureMixing(2.,substeps=2).apply(o,s,initial,30.)
    assert (kh>=0).all() and r.budget_error<=1
    mass=o.energy.mass_area(s)*s.h
    assert (r.column_energy_residual.abs()/mass.sum(-1)).max()<1e-8
    if seed==0:
        assert torch.equal(kh,initial)
        torch.testing.assert_close(end.temp_h,s.temp_h,rtol=0,atol=2e-10)


def test_explicit_deferral(column):
    o,e,s,k=column
    s=s.with_tensors(torch.zeros_like(s.u_e),s.h,s.temp_h,s.salt_h,s.tracer_h)
    end,kh,r=TemperatureMixing(2.,substeps=1,predictor_steps=0,max_reductions=0).apply(o,s,s.h*1e-12,300.)
    assert r.mixing_limiter['deferred_columns']>0
    assert torch.equal(end.salt_h,s.salt_h)
    assert (kh>=0).all() and r.budget_error<=1
