"""Each cheaper quadrature defines its own energy and matched derivatives."""
import pytest
import torch
from test_jmd95_hamiltonian import mesh,case
from vfs.ocean.jmd95 import JMD95Dynamics
from vfs.ocean.stratified import StratifiedOcean


@pytest.mark.parametrize('order',[2,3,4])
@pytest.mark.parametrize('coast',[False,True])
def test_quadrature_energy_derivative_and_work(mesh,order,coast):
    model,original=case(mesh,coast)
    model.dynamics=JMD95Dynamics(domain=model.domain,quadrature_order=order)
    fields=('u_e','h','temp_h','salt_h')
    leaves=[getattr(original,k).clone().requires_grad_() for k in fields]
    state=original.with_tensors(*leaves,original.tracer_h)
    energy=model.dynamics.energy
    independent=torch.autograd.grad(energy.terms(state)['total'],leaves)
    analytic=energy.gradients(state)
    for key,ref in zip(fields,independent):
        torch.testing.assert_close(analytic[key],ref,rtol=2e-12,atol=.1)
    rhs=model.tendency(state)
    works=[g*getattr(rhs,k) for k,g in zip(fields,independent)]
    assert abs(sum(w.sum() for w in works)) < 512*torch.finfo(state.h.dtype).eps*sum(w.abs().sum() for w in works)


def test_stratified_default_unchanged_and_order_explicit(mesh):
    model,physical=case(mesh)
    a=StratifiedOcean(mesh,model.column)
    b=StratifiedOcean(mesh,model.column,quadrature_order=4)
    c=StratifiedOcean(mesh,model.column,quadrature_order=2)
    state=a.from_temperature(physical)
    assert a.energy.base.quadrature_order==4
    assert c.energy.base.quadrature_order==2
    assert torch.equal(a.energy.terms(state)['total'],b.energy.terms(state)['total'])
    c.check_state(c.step(state,1.))
    with pytest.raises(ValueError):StratifiedOcean(mesh,model.column,quadrature_order=1)


@pytest.mark.parametrize('order',[2,3])
def test_reduced_energy_heating_and_remap_close_own_energy(mesh,order):
    from vfs.ocean.stratified import AdaptiveThermalRemapper,OceanSurfaceHeating
    model,physical=case(mesh)
    ocean=StratifiedOcean(mesh,model.column,quadrature_order=order,remapper=AdaptiveThermalRemapper())
    state=ocean.from_temperature(physical)
    heating=OceanSurfaceHeating(.6,(.5,20.))
    heated,report=heating.apply(ocean,state,1.,surface_heat=torch.full((mesh.nCells,),20.))
    ocean.check_state(heated)
    target=heated.h.sum(-1,keepdim=True).expand_as(heated.h)/heated.h.shape[-1]
    remapped,remap=ocean.remap_with_report(heated,target)
    ocean.check_state(remapped)
    # Operators independently enforce their full-H energy and inversion guards.
    area=ocean.energy.mass_area(state)
    torch.testing.assert_close((area*remapped.salt_h).sum(),(area*state.salt_h).sum(),rtol=2e-15,atol=0)


def test_compilation_is_per_model_and_keeps_guards(mesh):
    model,physical=case(mesh)
    plain=StratifiedOcean(mesh,model.column)
    compiled=StratifiedOcean(mesh,model.column).compile_thermodynamics(backend='eager')
    assert compiled.compile_thermodynamics(backend='eager') is compiled
    assert not hasattr(plain.reference,'_compile_options')
    state=plain.from_temperature(physical)
    x,y=plain.step(state,1.),compiled.step(state,1.)
    for key in ('u_e','h','entropy_h','salt_h','tracer_h'):
        torch.testing.assert_close(getattr(x,key),getattr(y,key),rtol=0,atol=0)
    with pytest.raises(ValueError,match='salinity outside'):
        compiled.reference.evaluate(torch.full_like(physical.h,-1.),physical.temperature())
    with pytest.raises(ValueError,match='another backend'):
        compiled.compile_thermodynamics(backend='inductor')
