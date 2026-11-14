"""Residual-controlled inversions and immutable transport geometry."""
from dataclasses import fields
import pytest
import torch
from test_coupled_column import ocean,state
from vfs.ocean.stratified.faces import temperature_from_entropy
from vfs.ocean.transport import HorizontalTransport


@pytest.mark.parametrize('offset',[0.,.01,20.,float('nan')])
def test_warm_entropy_inverse_keeps_original_residual(ocean,offset):
    r=ocean.reference
    salt=torch.linspace(.1,41.9,129,dtype=torch.float64)[None]
    t=torch.linspace(-2.4,39.8,129,dtype=salt.dtype)[None]/r.to_90
    target=r.evaluate(salt,t)['entropy']
    solved=temperature_from_entropy(r,salt,target,guess=t+offset)
    residual=(r.evaluate(salt,solved)['entropy']-target).abs()
    assert bool((residual<=128*torch.finfo(t.dtype).eps*(target.abs()+20)).all())
    torch.testing.assert_close(solved,t,rtol=0,atol=1e-10)


def test_exact_guess_inverse_target_and_salt_derivatives(ocean):
    r=ocean.reference;s=torch.full((1,3),35.,dtype=torch.float64,requires_grad=True)
    t=torch.tensor([[5.,15.,25.]],dtype=s.dtype,requires_grad=True)
    ref=r.evaluate(s,t,entropy_derivatives=True)
    target=ref['entropy'].detach().requires_grad_()
    out=temperature_from_entropy(r,s,target,guess=t)
    ds,de,dguess=torch.autograd.grad(out.sum(),(s,target,t))
    torch.testing.assert_close(dguess,torch.zeros_like(t),rtol=0,atol=2e-14)
    torch.testing.assert_close(de,1/ref['entropy_theta'],rtol=2e-13,atol=0)
    torch.testing.assert_close(ds,-ref['entropy_s']/ref['entropy_theta'],rtol=2e-13,atol=0)


def test_warm_inverse_rejects_invalid_target_and_guess_layout(ocean):
    s=torch.full((1,3),35.,dtype=torch.float64);t=s*0+10
    with pytest.raises(ValueError):temperature_from_entropy(ocean.reference,s,t*0+1e9,guess=t)
    with pytest.raises(ValueError):temperature_from_entropy(ocean.reference,s,ocean.reference.evaluate(s,t)['entropy'],guess=t[:,0])


def test_prepared_and_unprepared_ledgers_are_exact(ocean):
    _,s=state(ocean,None,unequal=True);p=ocean.check_state(s.ocean)
    ordinary=HorizontalTransport(ocean.mesh,scheme='donor')
    prepared=HorizontalTransport(ocean.mesh,scheme='donor');prepared.prepare(4)
    a,b=ordinary.advance(p,1.),prepared.advance(p,1.)
    for f in fields(a):torch.testing.assert_close(getattr(a,f.name),getattr(b,f.name),rtol=0,atol=0)


def test_prepared_rejects_in_place_connectivity_edit():
    from vfs.geometry.mesh import VoronoiMesh
    mesh=VoronoiMesh(n_cells=12,lloyd_iters=0)
    transport=HorizontalTransport(mesh);transport.prepare(4)
    mesh.cellsOnEdge.add_(0)
    with pytest.raises(ValueError,match='geometry changed'):transport._indices(4,mesh.cellsOnEdge.device)


def test_warm_step_close_to_cold_with_positive_contents(ocean):
    from vfs.ocean.stratified import StratifiedOcean
    _,s=state(ocean,None,unequal=True)
    warm=StratifiedOcean(ocean.mesh,ocean.column,warm_start=True)
    cold=ocean.step(s.ocean,1.);out=warm.step(s.ocean,1.)
    for field in ('u_e','h','entropy_h','salt_h','tracer_h'):
        torch.testing.assert_close(getattr(out,field),getattr(cold,field),rtol=2e-12,atol=2e-11)
    warm.check_state(out)
    assert bool((out.tracer_h>=0).all())


def test_custom_ocean_keeps_public_step_and_check_protocol(ocean):
    from vfs.ocean.stratified import StratifiedOcean, TKEOcean
    class CustomOcean(StratifiedOcean):
        def check_state(self, state):
            return super().check_state(state)
        def step_with_report(self, state, dt, *, target_h=None):
            self.called = True
            return state, None
    custom = CustomOcean(ocean.mesh,ocean.column)
    model = TKEOcean(custom)
    _,s = state(ocean,None,unequal=True)
    out,_ = model.step_with_report(s,1.)
    assert custom.called
    torch.testing.assert_close(out.ocean.entropy_h,s.ocean.entropy_h,rtol=0,atol=0)


def test_public_step_revalidates_changed_contents(ocean):
    from dataclasses import replace
    _,s = state(ocean,None,unequal=True)
    ocean.step(s.ocean,1.)
    bad = replace(s.ocean,entropy_h=torch.full_like(s.ocean.entropy_h,float('nan')))
    with pytest.raises(ValueError,match='entropy outside'):
        ocean.step(bad,1.)


def test_custom_tke_composition_keeps_operator_wrappers(ocean):
    from vfs.ocean.stratified import TKEOcean
    class CustomTKE(TKEOcean):
        def heat_with_report(self, state, dt, **kwargs):
            self.called = True
            return super().heat_with_report(state,dt,**kwargs)
    model = CustomTKE(ocean)
    _,s = state(ocean,None,unequal=True)
    model.step_with_report(s,1.)
    assert model.called


def test_inference_geometry_uses_uncached_indices():
    from vfs.geometry.mesh import VoronoiMesh
    mesh = VoronoiMesh(n_cells=12,lloyd_iters=0)
    transport = HorizontalTransport(mesh)
    transport.prepare(4)
    with torch.inference_mode():
        mesh.cellsOnEdge = mesh.cellsOnEdge.clone()
        transport.prepare(4)
        assert not hasattr(transport,'_prepared')
        i,j = transport._indices(4,mesh.cellsOnEdge.device)
        k = torch.arange(4,device=i.device)
        assert torch.equal(i,(mesh.cellsOnEdge[:,0,None]*4+k).flatten())
        assert torch.equal(j,(mesh.cellsOnEdge[:,1,None]*4+k).flatten())
