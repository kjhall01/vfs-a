"""Independent derivative/work checks for the constant-density numerical change."""
import torch
import pytest
from vfs.ocean import Ocean, OceanColumn, OceanDomain, OceanState
from vfs.ocean.conservative import ConstantDensityDynamics
from vfs.geometry.mesh import VoronoiMesh


def case(layers=3,coast=False):
    torch.set_default_dtype(torch.float64)
    gen=torch.Generator().manual_seed(371)
    mesh=VoronoiMesh(n_cells=24,lloyd_iters=1,verbose=False)
    column=OceanColumn(layers,total_depth=3000.)
    domain=OceanDomain(mesh,mesh.cell_xyz[:,2]>-.4) if coast else None
    h=3000./layers*(1+.1*torch.randn(24,layers,generator=gen))
    u=.1*torch.randn(mesh.nEdges,layers,generator=gen)
    state=OceanState.from_temperature(u,h,torch.full_like(h,10.),torch.full_like(h,35.),
        torch.ones(24,layers,2),mesh=mesh,column=column)
    if domain:state=domain.project_state(state)
    dyn=ConstantDensityDynamics(domain=domain)
    return Ocean(mesh,column,domain=domain,dynamics=dyn),state


@pytest.mark.parametrize('layers',[1,3])
@pytest.mark.parametrize('coast',[False,True])
def test_independent_autograd_energy_and_work(layers,coast):
    model,s=case(layers,coast)
    u=s.u_e.detach().requires_grad_();h=s.h.detach().requires_grad_()
    state=s.with_tensors(u,h,s.temp_h,s.salt_h,s.tracer_h)
    dyn=model.dynamics
    gu,gh=torch.autograd.grad(dyn.total_energy(state),(u,h))
    analytic=dyn.energy.gradients(state)
    eps=torch.finfo(h.dtype).eps
    for key,reference in [('u_e',gu),('h',gh)]:
        assert (analytic[key]-reference).abs().max() <= 32*eps*reference.abs().max()
    rhs=dyn.tendency(state)
    local_u,local_h=gu*rhs.u_e,gh*rhs.h
    scale=local_u.abs().sum()+local_h.abs().sum()
    assert (local_u.sum()+local_h.sum()).abs() <= 128*eps*scale
    rotation=gu*dyn.rotation_acceleration(state)
    assert rotation.sum().abs() <= 128*eps*rotation.abs().sum()


def test_nonzero_free_surface_exchange_and_passive_scalars():
    model,s=case()
    dyn=model.dynamics
    h=s.h.detach().requires_grad_();u=s.u_e.detach().requires_grad_()
    s=s.with_tensors(u,h,s.temp_h,s.salt_h,s.tracer_h)
    terms=dyn.energy.terms(s)
    ku,=torch.autograd.grad(terms['kinetic'],u)
    vh,=torch.autograd.grad(terms['free_surface'],h)
    work=(ku*(-dyn.gravity*s.mesh.grad(s.eta()))).sum()
    surface_rate=(vh*dyn.tendency(s).h).sum()
    assert work != 0
    assert abs(work+surface_rate) <= 128*torch.finfo(h.dtype).eps*max(abs(work),abs(surface_rate))
    changed=s.with_tensors(u,h,s.temp_h*2,s.salt_h*.8,s.tracer_h)
    assert torch.equal(dyn.tendency(changed).u_e,dyn.tendency(s).u_e)
    assert torch.equal(model.physical_fields(changed)['density'],torch.full_like(h,dyn.rho0))


@pytest.mark.parametrize('coast',[False,True])
def test_inventories_and_constant_tracers(coast):
    model,state=case(coast=coast)
    rhs=model.tendency(state)
    assert torch.equal(rhs.tracer_h, rhs.h[...,None].expand_as(rhs.tracer_h))
    area=state.mesh.cell_area[:,None]
    for name in ('h','temp_h','salt_h','tracer_h'):
        value=getattr(rhs,name)
        local=(area if value.ndim==2 else area[...,None])*value
        assert local.sum().abs() <= 128*torch.finfo(value.dtype).eps*local.abs().sum()
    before=model.budgets(state)
    after=model.budgets(model.step(state,30.))
    for key in ('volume','temperature_content','salinity_content','tracer_content'):
        torch.testing.assert_close(after[key],before[key],rtol=128*torch.finfo(state.h.dtype).eps,atol=0)
    assert 'constant_density_mechanical_energy' in after


def test_flat_and_variable_bottom_rest():
    model,state=case()
    for bottom in (torch.full((24,1),3000.),3000*(1+.2*state.mesh.cell_xyz[:,:1])):
        column=OceanColumn(3,total_depth=bottom,rest_thickness=torch.full((3,),1000.))
        h=(bottom/3).expand(24,3).clone()
        # Construct an exactly representable eta=0 by assigning the residual
        # to the last layer; avoids mistaking initial roundoff for PG error.
        h[:,-1]=bottom[:,0]-h[:,:-1].sum(-1)
        resting=OceanState.from_temperature(torch.zeros_like(state.u_e),h,
            torch.full_like(h,10.),torch.full_like(h,35.),mesh=state.mesh,column=column)
        rhs=model.dynamics.tendency(resting)
        assert torch.count_nonzero(rhs.u_e)==0
        assert torch.count_nonzero(rhs.h)==0


def test_rhs_and_one_step_directional_derivatives():
    model,state=case()
    u=state.u_e.detach().requires_grad_()
    direction=torch.cos(torch.arange(u.numel(),dtype=u.dtype)).reshape_as(u)
    def objective(v):
        s=state.with_tensors(v,state.h,state.temp_h,state.salt_h,state.tracer_h)
        return model.step(s,10.).u_e.square().sum()
    grad,=torch.autograd.grad(objective(u),u)
    delta=1e-5
    fd=(objective(u+delta*direction)-objective(u-delta*direction))/(2*delta)
    torch.testing.assert_close((grad*direction).sum(),fd,rtol=1e-7,atol=1e-8)
