"""Energy transfer, invariants and integration of optional ocean TKE."""
from dataclasses import replace
import pytest
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn, OceanDomain
from vfs.ocean.stratified import (StratifiedOcean, AdaptiveThermalRemapper,
    TKEOcean, TKEClosure, TKEState)
from vfs.ocean.stratified.tke import diffuse


@pytest.fixture(scope='module')
def mesh():
    return VoronoiMesh(n_cells=32, lloyd_iters=2, rotation_scheme='compatible')


def case(mesh, *, closure=None, layers=4, thermal=None, k=.01, domain=None):
    o=StratifiedOcean(mesh, OceanColumn(layers,total_depth=100.), domain=domain,
        remapper=AdaptiveThermalRemapper())
    h=torch.full((mesh.nCells,layers),100./layers,dtype=torch.float64)
    temperature=torch.linspace(15.,12.,layers,dtype=h.dtype).expand_as(h) if thermal is None else torch.full_like(h,thermal)
    u=torch.linspace(.3,-.3,layers,dtype=h.dtype).expand(mesh.nEdges,layers).clone()
    if domain is not None:u=domain.apply_edge(u)
    s=o.state(u_e=u,h=h,temperature=temperature,salinity=torch.full_like(h,35.),tracer=torch.ones(*h.shape,1,dtype=h.dtype))
    m=TKEOcean(o,closure=closure)
    kk=torch.full_like(h,k)
    if domain is not None:kk=domain.apply_cell(kk)
    return m,m.state(s,tke=kk)


def mean_mass(m,s):return (m.ocean.energy.mass_area(s.ocean)*s.ocean.h).sum()


def test_diffusion_against_dense_solve_and_weighted_invariants():
    mass=torch.tensor([[2.,1.,3.]],dtype=torch.float64)
    g=torch.tensor([[.3,.4]],dtype=mass.dtype)
    x=torch.tensor([[1.,4.,2.]],dtype=mass.dtype)
    D=torch.tensor([[-1.,1.,0.],[0.,-1.,1.]],dtype=mass.dtype)
    matrix=torch.diag(mass[0])+2*D.T@torch.diag(g[0])@D
    exact=torch.linalg.solve(matrix,(mass*x)[0])
    actual=diffuse(x,mass,g,2.)
    torch.testing.assert_close(actual[0],exact,rtol=2e-15,atol=1e-15)
    torch.testing.assert_close((mass*actual).sum(),(mass*x).sum(),rtol=2e-15,atol=0)
    assert actual.min()>=x.min() and actual.max()<=x.max()
    torch.testing.assert_close(diffuse(x*0+3,mass,g,2.),x*0+3,rtol=0,atol=0)


def test_shear_energy_and_native_momentum(mesh):
    m,s=case(mesh,closure=TKEClosure(2.,c_epsilon=0.,mix_scalars=False))
    out,r=m.mix_with_report(s,100.)
    old,new=m.budgets(s),m.budgets(out)
    scale=mean_mass(m,s)
    assert r.shear_production.sum()>0 and r.implicit_momentum_loss.sum()>0
    assert (old['kinetic']-new['kinetic'])>0
    assert abs(float((new['tke']-old['tke']+new['kinetic']-old['kinetic'])/scale))<1e-16
    metric=m.ocean.energy.base.mechanical.metric(s.ocean.physical_state(m.ocean.reference))
    torch.testing.assert_close((metric*out.ocean.u_e).sum(-1),(metric*s.ocean.u_e).sum(-1),rtol=2e-14,atol=100.)
    assert float(r.kinetic_residual.abs()/scale)<2e-16
    assert torch.equal(out.ocean.entropy_h,s.ocean.entropy_h)


def test_exact_decay_and_heat(mesh):
    c=TKEClosure(2.,c_mu=0.,c_epsilon=.7)
    m,s=case(mesh,closure=c)
    out,r=m.mix_with_report(s,100.)
    exact=.01/(1+.5*100*.7*.1/2)**2
    torch.testing.assert_close(out.specific_tke(),torch.full_like(s.tke_h,exact),rtol=2e-15,atol=0)
    assert (out.ocean.entropy_h>s.ocean.entropy_h).all()
    assert torch.equal(out.ocean.salt_h,s.ocean.salt_h)
    residual=(m.budgets(out)['total']-m.budgets(s)['total'])/mean_mass(m,s)
    assert abs(float(residual))<1e-8
    assert float(r.column_energy_residual.abs().sum()/mean_mass(m,s))<1e-8


@pytest.mark.parametrize('unstable',[False,True])
def test_buoyancy_energy_sign_and_content_conservation(mesh,unstable):
    m,s=case(mesh,closure=TKEClosure(2.,c_epsilon=0.))
    if unstable:
        p=s.ocean.physical_state(m.ocean.reference)
        p=p.with_tensors(p.u_e,p.h,p.temp_h.flip(-1),p.salt_h,p.tracer_h)
        s=replace(s,ocean=m.ocean.from_temperature(p))
    out,r=m.mix_with_report(s,10.)
    assert (r.buoyancy_work.sum()<0) if unstable else (r.buoyancy_work.sum()>0)
    for name in ['salt_h','tracer_h']:
        a=getattr(s.ocean,name);b=getattr(out.ocean,name)
        axis=-2 if name=='tracer_h' else -1
        torch.testing.assert_close(a.sum(axis),b.sum(axis),rtol=2e-14,atol=1e-12)
    assert float(r.column_energy_residual.abs().sum()/mean_mass(m,s))<1e-8
    assert (out.tke_h>=0).all()


def test_zero_tke_has_no_floor_or_mixing(mesh):
    m,s=case(mesh,closure=TKEClosure(2.),k=0.)
    out,r=m.mix_with_report(s,100.)
    assert torch.equal(out.tke_h,s.tke_h)
    for name in ['u_e','h','entropy_h','salt_h','tracer_h']:
        assert torch.equal(getattr(out.ocean,name),getattr(s.ocean,name))
    assert r.dissipated_heat.sum()==0


def test_disabled_closure_reproduces_dynamics_and_advects_tke(mesh):
    m,s=case(mesh,closure=None)
    expected=m.ocean.step(s.ocean,1.)
    out,r=m.step_with_report(s,1.)
    for name in ['u_e','h','entropy_h','salt_h','tracer_h']:
        assert torch.equal(getattr(out.ocean,name),getattr(expected,name))
    torch.testing.assert_close(m.budgets(out)['tke'],m.budgets(s)['tke'],rtol=3e-15,atol=0)
    torch.testing.assert_close(out.specific_tke(),s.specific_tke(),rtol=3e-15,atol=0)
    assert r.turbulence is None


def test_ale_conserves_tke_and_keeps_passive_layout(mesh):
    m,s=case(mesh,closure=TKEClosure(2.))
    target=s.ocean.h.sum(-1,keepdim=True)*s.ocean.h.new_tensor([.2,.25,.25,.3])
    out,r=m.remap_with_report(s,target)
    torch.testing.assert_close(out.tke_h.sum(-1),s.tke_h.sum(-1),rtol=3e-15,atol=0)
    assert out.ocean.tracer_h.shape==s.ocean.tracer_h.shape
    assert (out.tke_h>=0).all()
    assert abs(float((m.budgets(out)['total']-m.budgets(s)['total'])/mean_mass(m,s)))<1e-8


def test_closed_coasts_and_dry_placeholders(mesh):
    wet=mesh.cell_xyz[:,0]>0
    domain=OceanDomain(mesh,wet)
    m,s=case(mesh,closure=TKEClosure(2.),domain=domain)
    out,r=m.mix_with_report(s,10.)
    assert torch.equal(out.tke_h[~wet],s.tke_h[~wet])
    assert torch.equal(out.ocean.entropy_h[~wet],s.ocean.entropy_h[~wet])
    assert torch.equal(out.ocean.u_e[~domain.active_edge],s.ocean.u_e[~domain.active_edge])
    assert float(r.column_energy_residual.abs().sum()/mean_mass(m,s))<1e-8


def test_rejects_invalid_states_and_parameters(mesh):
    m,s=case(mesh,closure=TKEClosure(2.))
    with pytest.raises(ValueError):m.mix_with_report(replace(s,tke_h=-s.tke_h),1.)
    with pytest.raises(ValueError):m.mix_with_report(s,-1.)
    with pytest.raises(ValueError):TKEClosure(0.)
    with pytest.raises(ValueError):TKEClosure(1.,prandtl=float('nan'))


def test_tke_exhaustion_rejects_without_mutating_input(mesh):
    m,s=case(mesh,closure=TKEClosure(100.,c_epsilon=0.),k=1e-8)
    s=replace(s,ocean=replace(s.ocean,u_e=torch.zeros_like(s.ocean.u_e)))
    saved=s.tke_h.clone()
    with pytest.raises(ValueError,match='available TKE'):m.mix_with_report(s,1000.)
    assert torch.equal(saved,s.tke_h)


def test_one_layer_decay(mesh):
    m,s=case(mesh,closure=TKEClosure(2.),layers=1)
    out,r=m.mix_with_report(s,10.)
    assert r.shear_production.sum()==0 and r.buoyancy_work.sum()==0
    assert (out.tke_h<s.tke_h).all()


def test_no_passive_layout_and_combined_step(mesh):
    m,s=case(mesh,closure=TKEClosure(2.))
    s=replace(s,ocean=replace(s.ocean,tracer_h=None))
    out,r=m.step_with_report(s,1.)
    assert out.ocean.tracer_h is None and len(r.ocean.stages)==3
    assert r.turbulence is not None


def test_smooth_tke_directional_derivative(mesh):
    m,s=case(mesh,closure=TKEClosure(2.,mix_scalars=False))
    def loss(amplitude):
        x=replace(s,tke_h=s.tke_h*amplitude)
        y,_=m.mix_with_report(x,2.)
        return y.tke_h.mean()+y.ocean.u_e.square().mean()
    a=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
    grad=torch.autograd.grad(loss(a),a)[0]
    eps=1e-4
    fd=(loss(a.detach()+eps)-loss(a.detach()-eps))/(2*eps)
    torch.testing.assert_close(grad,fd,rtol=2e-7,atol=1e-10)


def test_nonuniform_tke_transport_and_remap(mesh):
    m,s=case(mesh,closure=None)
    mask=(mesh.cell_xyz[:,0]>0).to(s.tke_h.dtype)[:,None]
    s=replace(s,tke_h=s.tke_h*mask*s.tke_h.new_tensor([0.,.5,1.,2.]))
    total=m.budgets(s)['tke']
    for _ in range(3):s=m.step(s,1.)
    out,_=m.remap_with_report(s,s.ocean.h.sum(-1,keepdim=True)*s.tke_h.new_tensor([.2,.25,.25,.3]))
    assert (out.tke_h>=0).all()
    torch.testing.assert_close(m.budgets(out)['tke'],total,rtol=5e-15,atol=0)


def test_heat_and_buoyancy_path_gradients(mesh):
    m,s=case(mesh,closure=TKEClosure(2.))
    def loss(amplitude):
        x=replace(s,tke_h=s.tke_h*amplitude)
        y,_=m.mix_with_report(x,10.)
        temp=y.ocean.physical_state(m.ocean.reference).temperature()
        return y.tke_h.mean()+temp.mean()/1000
    a=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
    grad=torch.autograd.grad(loss(a),a)[0]
    eps=1e-3
    fd=(loss(a.detach()+eps)-loss(a.detach()-eps))/(2*eps)
    torch.testing.assert_close(grad,fd,rtol=2e-6,atol=1e-9)


def test_unequal_neighbor_thickness_matches_column_hamiltonian(mesh):
    from vfs.ocean.stratified.energy_columns import column_terms
    m,s=case(mesh,closure=TKEClosure(2.,mix_scalars=False))
    h=s.ocean.h+mesh.cell_xyz[:,:1]*s.ocean.h.new_tensor([3.,-2.,1.,-2.])
    old=s.ocean.physical_state(m.ocean.reference)
    physical=old.with_tensors(old.u_e,h,h*old.temperature(),h*old.salinity(),h[...,None]*(old.tracer_h/old.h[...,None]))
    s=m.state(m.ocean.from_temperature(physical),tke=torch.full_like(h,.01))
    out,r=m.mix_with_report(s,10.)
    a,_=column_terms(m.ocean.energy,s.ocean.physical_state(m.ocean.reference))
    b,_=column_terms(m.ocean.energy,out.ocean.physical_state(m.ocean.reference))
    mass=m.ocean.energy.mass_area(s.ocean)
    residual=sum(b[k]-a[k] for k in a)+(mass*(out.tke_h-s.tke_h)).sum(-1)
    assert float(((residual-r.kinetic_column_exchange).abs()/(mass*h).sum(-1)).max())<1e-8
    assert float(r.kinetic_column_exchange.abs().sum())>0
    assert float(r.kinetic_column_exchange.sum().abs()/mean_mass(m,s))<1e-16
