"""Native boundary work and drag-to-TKE transfer."""
from dataclasses import replace
import pytest
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn, OceanDomain
from vfs.ocean.stratified import StratifiedOcean, TKEOcean, TKEClosure, OceanBoundaryStress
from vfs.ocean.stratified.boundary import bottom_speed, drag_velocity

@pytest.fixture(scope='module')
def mesh():
    return VoronoiMesh(n_cells=32,lloyd_iters=2,rotation_scheme='compatible')

def case(mesh,layers=4,cd=.0025,domain=None,depth=100.):
    ocean=StratifiedOcean(mesh,OceanColumn(layers,total_depth=depth,rest_thickness=torch.full((layers,),100./layers)),domain=domain)
    h=torch.full((mesh.nCells,layers),100./layers,dtype=torch.float64)
    u=torch.linspace(.2,-.3,layers,dtype=h.dtype).expand(mesh.nEdges,layers).clone()
    if domain is not None:u=domain.apply_edge(u)
    s=ocean.state(u_e=u,h=h,temperature=torch.full_like(h,12.),salinity=torch.full_like(h,35.))
    m=TKEOcean(ocean,boundary=OceanBoundaryStress(cd))
    return m,m.state(s,tke=torch.zeros_like(h))

def ledger(m,s,out,r):
    metric=m.ocean.energy.base.mechanical.metric(s.ocean.physical_state(m.ocean.reference))
    dk=(.5*metric*(out.ocean.u_e-s.ocean.u_e)*(out.ocean.u_e+s.ocean.u_e)).sum()
    de=(m.ocean.energy.mass_area(s.ocean)*(out.tke_h-s.tke_h)).sum()
    scale=(.5*metric*s.ocean.u_e.square()).sum()+r.surface_work.abs()+1
    assert abs(float((dk+de-r.surface_work)/scale))<2e-15
    assert float(r.energy_residual.abs()/scale)<2e-15
    pscale=((metric*s.ocean.u_e.abs()).sum(-1)+r.surface_impulse.abs()+r.bottom_impulse.abs()).clamp_min(1)
    assert float((r.momentum_residual.abs()/pscale).max())<2e-15
    for field in ('h','entropy_h','salt_h'):
        assert torch.equal(getattr(out.ocean,field),getattr(s.ocean,field))

@pytest.mark.parametrize('tau',[.1,-.1])
def test_wind_work(mesh,tau):
    m,s=case(mesh,cd=0.)
    stress=torch.full((mesh.nEdges,),tau,dtype=s.tke_h.dtype)
    out,r=m.boundary_with_report(s,10.,surface_stress=stress)
    torch.testing.assert_close(out.ocean.u_e[:,0],s.ocean.u_e[:,0]+10*stress/(m.ocean.column.rho0*25),rtol=0,atol=0)
    assert torch.equal(out.ocean.u_e[:,1:],s.ocean.u_e[:,1:])
    assert torch.equal(out.tke_h,s.tke_h)
    assert (r.surface_work>0)==(tau>0)
    ledger(m,s,out,r)

@pytest.mark.parametrize('dt',[10.,1e8])
def test_drag(mesh,dt):
    m,s=case(mesh)
    out,r=m.boundary_with_report(s,dt)
    assert (out.ocean.u_e[:,-1].abs()<s.ocean.u_e[:,-1].abs()).all()
    assert torch.equal(out.ocean.u_e[:,-1].sign(),s.ocean.u_e[:,-1].sign())
    assert torch.equal(out.ocean.u_e[:,:-1],s.ocean.u_e[:,:-1])
    assert out.tke_h[:,:-1].count_nonzero()==0 and (out.tke_h[:,-1]>0).all()
    torch.testing.assert_close(r.bottom_tke_gain.sum(),r.bottom_ke_loss.sum(),rtol=3e-15,atol=0)
    ledger(m,s,out,r)

def test_single_layer_and_rest(mesh):
    m,s=case(mesh,layers=1)
    s=replace(s,ocean=replace(s.ocean,u_e=s.ocean.u_e*0))
    out,r=m.boundary_with_report(s,100.,surface_stress=torch.full((mesh.nEdges,),.1,dtype=s.tke_h.dtype))
    assert r.bottom_ke_loss.sum()>0 and r.surface_work>0
    ledger(m,s,out,r)
    zero,rz=m.boundary_with_report(s,100.)
    assert torch.equal(zero.ocean.u_e,s.ocean.u_e) and rz.bottom_ke_loss.sum()==0

def test_unequal_neighbors(mesh):
    m,s=case(mesh)
    h=s.ocean.h+mesh.cell_xyz[:,:1]*s.tke_h.new_tensor([3.,-2.,1.,-2.])
    p=s.ocean.physical_state(m.ocean.reference)
    p=p.with_tensors(p.u_e,h,h*p.temperature(),h*p.salinity(),None)
    s=m.state(m.ocean.from_temperature(p),tke=torch.zeros_like(h))
    out,r=m.boundary_with_report(s,30.)
    from vfs.ocean.stratified.energy_columns import column_terms
    a,_=column_terms(m.ocean.energy,p)
    b,_=column_terms(m.ocean.energy,out.ocean.physical_state(m.ocean.reference))
    residual=b['kinetic']-a['kinetic']+(m.ocean.energy.mass_area(s.ocean)*out.tke_h).sum(-1)
    assert float((residual.abs()/a['kinetic'].abs().clamp_min(1)).max())<3e-15
    ledger(m,s,out,r)

def test_coasts(mesh):
    wet=mesh.cell_xyz[:,0]>0;domain=OceanDomain(mesh,wet)
    m,s=case(mesh,domain=domain)
    out,r=m.boundary_with_report(s,10.,surface_stress=torch.ones(mesh.nEdges,dtype=s.tke_h.dtype))
    assert out.tke_h[~wet].count_nonzero()==0
    assert out.ocean.u_e[~domain.active_edge].count_nonzero()==0
    assert r.surface_impulse[~domain.active_edge].count_nonzero()==0
    ledger(m,s,out,r)

def test_defaults(mesh):
    m,s=case(mesh,cd=0.)
    out,r=m.boundary_with_report(s,10.)
    assert torch.equal(out.ocean.u_e,s.ocean.u_e) and torch.equal(out.tke_h,s.tke_h)
    plain=TKEOcean(m.ocean)
    out,r=plain.boundary_with_report(s,10.)
    assert out is s and r is None
    with pytest.raises(ValueError):plain.step(s,1.,surface_stress=torch.zeros(mesh.nEdges))

@pytest.mark.parametrize('cd',[-1.,float('nan'),float('inf')])
def test_invalid_drag(cd):
    with pytest.raises(ValueError):OceanBoundaryStress(cd)

def test_invalid_inputs(mesh):
    m,s=case(mesh)
    for stress in [torch.zeros(mesh.nEdges,1,dtype=s.tke_h.dtype),torch.zeros(mesh.nEdges,dtype=torch.float32),torch.full((mesh.nEdges,),float('nan'),dtype=s.tke_h.dtype)]:
        with pytest.raises(ValueError):m.boundary_with_report(s,1.,surface_stress=stress)
    for dt in [-1.,0.,float('nan')]:
        with pytest.raises(ValueError):m.boundary_with_report(s,dt)
    m,s=case(mesh,depth=100.+mesh.cell_xyz[:,:1])
    m.boundary_with_report(s,1.)
    m.boundary=OceanBoundaryStress()
    m.boundary_with_report(s,1.,surface_stress=torch.zeros(mesh.nEdges,dtype=s.tke_h.dtype))

def test_reconstruction_and_derivative(mesh):
    m,s=case(mesh)
    speed=bottom_speed(mesh,s.ocean.u_e[:,-1])
    assert (speed>=s.ocean.u_e[:,-1].abs()).all()
    assert (speed>s.ocean.u_e[:,-1].abs()+1e-6).any()
    def loss(a):
        x=replace(s,ocean=replace(s.ocean,u_e=s.ocean.u_e*a))
        y,_=m.boundary_with_report(x,100.,surface_stress=torch.ones(mesh.nEdges,dtype=a.dtype)*a*.1)
        return y.ocean.u_e.square().mean()+y.tke_h.mean()
    a=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
    derivative=torch.autograd.grad(loss(a),a)[0]
    eps=1e-5
    fd=(loss(a.detach()+eps)-loss(a.detach()-eps))/(2*eps)
    torch.testing.assert_close(derivative,fd,rtol=2e-8,atol=1e-10)

def test_combined_step(mesh):
    m,s=case(mesh)
    m.closure=TKEClosure(2.,mix_scalars=False)
    out,r=m.step_with_report(s,1.,surface_stress=torch.full((mesh.nEdges,),.1,dtype=s.tke_h.dtype))
    assert r.boundary is not None and r.turbulence is not None
    assert r.boundary.surface_work>0 and r.turbulence.dissipated_heat.sum()>0
    assert (out.tke_h>=0).all()

def test_quadratic_drag_exact():
    u=torch.tensor([.3,.4],dtype=torch.float64);initial=u.clone()
    for _ in range(100):u=drag_velocity(u,u.norm(),20.,.0025,10.)
    exact=initial/(1+1000*.0025*initial.norm()/20.)
    torch.testing.assert_close(u,exact,rtol=5e-15,atol=0)


def bathymetric_case(mesh, *, direction='along', coast=False, seamount=False):
    depth=100.-30.*torch.exp((mesh.cell_xyz[:,:1]-1.)/.1) if seamount else 100.+40.*mesh.cell_xyz[:,2:3]
    wet=mesh.cell_xyz[:,0]>0 if coast else torch.ones(mesh.nCells,dtype=torch.bool)
    domain=OceanDomain(mesh,wet,depth=depth)
    m,_=case(mesh,depth=depth,domain=domain)
    h=depth*hfractions(depth)
    axis=mesh.edge_xyz.new_tensor([0.,0.,1.]).expand_as(mesh.edge_xyz)
    vector=torch.linalg.cross(axis,mesh.edge_xyz) if direction=='along' else axis-mesh.edge_xyz[:,2:3]*mesh.edge_xyz
    u=domain.apply_edge(.3*(vector*mesh.edge_normal).sum(-1)[:,None].expand(mesh.nEdges,4))
    resolved=m.ocean.state(u_e=u,h=h,temperature=torch.full_like(h,10.),salinity=torch.full_like(h,35.),tracer=torch.ones(*h.shape,1,dtype=h.dtype))
    return m,m.state(resolved,tke=torch.zeros_like(h))


def hfractions(tensor):
    return tensor.new_tensor([.1,.2,.3,.4])


@pytest.mark.parametrize('direction',['along','across'])
@pytest.mark.parametrize('coast',[False,True])
def test_bathymetric_drag_energy_and_locality(mesh,direction,coast):
    m,s=bathymetric_case(mesh,direction=direction,coast=coast)
    out,r=m.boundary_with_report(s,100.)
    ledger(m,s,out,r)
    assert r.bottom_ke_loss.sum()>0
    assert torch.equal(out.ocean.tracer_h,s.ocean.tracer_h)
    assert torch.equal(out.tke_h[:,:-1],s.tke_h[:,:-1])
    assert out.tke_h[~m.ocean.domain.wet_cell].count_nonzero()==0
    from vfs.ocean.stratified.energy_columns import column_terms
    a,_=column_terms(m.ocean.energy,s.ocean.physical_state(m.ocean.reference))
    b,_=column_terms(m.ocean.energy,out.ocean.physical_state(m.ocean.reference))
    residual=b['kinetic']-a['kinetic']+(m.ocean.energy.mass_area(s.ocean)*out.tke_h).sum(-1)
    assert float((residual.abs()/a['kinetic'].abs().clamp_min(1)).max())<3e-15


@pytest.mark.parametrize('seamount',[False,True])
def test_rest_over_bathymetry(mesh,seamount):
    m,s=bathymetric_case(mesh,seamount=seamount)
    s=replace(s,ocean=replace(s.ocean,u_e=s.ocean.u_e*0))
    out,r=m.boundary_with_report(s,100.)
    for field in ('u_e','h','entropy_h','salt_h','tracer_h'):
        assert torch.equal(getattr(out.ocean,field),getattr(s.ocean,field))
    assert torch.equal(out.tke_h,s.tke_h)
    assert r.bottom_ke_loss.count_nonzero()==0


def test_conflicting_depths_reject_without_mutation(mesh):
    depth=100.+40.*mesh.cell_xyz[:,2:3]
    domain=OceanDomain(mesh,torch.ones(mesh.nCells,dtype=torch.bool),depth=depth+1.)
    m,s=case(mesh,depth=depth,domain=domain)
    saved=s.ocean.u_e.clone()
    with pytest.raises(ValueError,match='consistent'):m.boundary_with_report(s,1.)
    assert torch.equal(s.ocean.u_e,saved) and s.tke_h.count_nonzero()==0


def test_flat_formula_unchanged(mesh):
    m,s=case(mesh)
    out,r=m.boundary_with_report(s,10.)
    expected=s.ocean.u_e[:,-1]/(1+10*.0025*bottom_speed(mesh,s.ocean.u_e[:,-1])/25.)
    assert torch.equal(out.ocean.u_e[:,-1],expected)


def test_bathymetric_combined_step(mesh):
    m,s=bathymetric_case(mesh)
    m.closure=TKEClosure(2.,mix_scalars=False)
    old=m.budgets(s)
    out,r=m.step_with_report(s,1.)
    new=m.budgets(out)
    assert r.boundary.bottom_ke_loss.sum()>0
    assert (out.tke_h>=0).all()
    for name in ('volume','salt','tracers'):
        torch.testing.assert_close(new[name],old[name],rtol=3e-15,atol=0)
    mass=(m.ocean.energy.mass_area(s.ocean)*s.ocean.h).sum()
    assert float((new['total']-old['total']).abs()/mass)<1e-8
