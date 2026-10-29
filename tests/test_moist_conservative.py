import pytest,torch
from unittest.mock import patch
from vfs.atmosphere.moist_conservative import MoistState,MoistConservativeDynamics
from vfs.atmosphere.moist_cp import interface_pressure_mass
from vfs.atmosphere.conservative import v
from vfs.geometry.mesh import VoronoiMesh
from vfs.geometry.column import CharneyPhillipsColumn,make_hybrid_coefficients

KEYS=('u_e','ps','entropy_content','water_content')
@pytest.fixture(scope='module')
def mesh():
 torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
 return VoronoiMesh(n_cells=64,lloyd_iters=2,verbose=False)
def case(mesh,n,uniform=False):
 torch.manual_seed(n+41);A,B=make_hybrid_coefficients(n,p_top=5000.,sigma_power=1.)
 col=CharneyPhillipsColumn(n,A,B,phi_s=0. if uniform else 1000*torch.rand(mesh.nCells),phi_full_mode='direct_u_gl2_adjoint')
 ps=torch.ones(mesh.nCells,1)*1e5
 if not uniform:ps+=2000*torch.randn_like(ps)
 p=v(col.interface_pressure(ps));dm=interface_pressure_mass(p.diff())
 entropy=200+80*torch.log(1e5/p)
 q=torch.ones(mesh.nCells,n+1,3)*torch.tensor([.01,.001,.0001])
 u=torch.zeros(mesh.nEdges,n)
 if not uniform:
  entropy+=torch.randn_like(entropy)*10;q*=.5+torch.rand_like(q);u+=torch.randn_like(u)*10
 return MoistState(u,ps,dm*entropy,dm[...,None]*q,mesh,col)

@pytest.mark.parametrize('n',[1,4,8])
def test_full_gradient_energy_and_mass(mesh,n):
 s=case(mesh,n);d=MoistConservativeDynamics()
 leaves=[getattr(s,k).clone().requires_grad_() for k in KEYS]
 a=MoistState(*leaves,mesh,s.column)
 independent=torch.autograd.grad(d.boundary_total_energy(a),leaves)
 explicit=d.energy_gradients(s)
 for k,g in zip(KEYS,independent):torch.testing.assert_close(getattr(explicit,k),g,rtol=2e-10,atol=1e-3)
 with patch('torch.autograd.grad',side_effect=AssertionError('runtime AD')):rhs=d.tendency(s)
 work=[(g*getattr(rhs,k)).sum() for k,g in zip(KEYS,independent)]
 assert float((sum(work).abs()/sum(w.abs() for w in work)).detach())<1e-10
 area=mesh.cell_area[:,None]
 for value in [rhs.ps,rhs.entropy_content,*rhs.water_content.unbind(-1)]:
  z=area*value;assert float(z.sum().abs()/z.abs().sum().clamp_min(1))<1e-12
 terms=d.tendency_terms(s)
 for k in KEYS:torch.testing.assert_close(v(terms[k]['total']),getattr(rhs,k),rtol=1e-12,atol=1e-12)
 if n>1:assert terms['u_e']['vertical'].abs().max()>0
 # The hydrostatic energy decomposition equals the independently reduced H.
 from vfs.atmosphere.moist_cp import reconstruct_moist_cp
 p=v(s.column.interface_pressure(s.ps));r=reconstruct_moist_cp(p,s.entropy_content,s.water_content,d.thermodynamics)
 reduced=(area/d.kernel.gravity*(r['dp']*d.thermodynamics.enthalpy(r['temperature'],r['fractions']).mean(-1)).sum(-1,keepdim=True)).sum()
 reduced+=(area/d.kernel.gravity*s.ps*torch.as_tensor(s.column.phi_s).reshape(-1,1)).sum()
 reduced+=d.energy_terms(s)['kinetic']
 torch.testing.assert_close(reduced,d.boundary_total_energy(s),rtol=1e-12,atol=1.)

def test_uniform_rest_and_short_integration(mesh):
 d=MoistConservativeDynamics();s=case(mesh,4,True)
 assert d.tendency(s).u_e.abs().max()<1e-12
 s=case(mesh,4);e0=d.boundary_total_energy(s);initial=s
 for _ in range(10):
  s=d.step(s,1.)
 assert abs(float((d.boundary_total_energy(s)-e0)/e0))<1e-9
 assert all(torch.isfinite(getattr(s,k)).all() for k in KEYS)


def test_constant_species_and_dry_mass(mesh):
 d=MoistConservativeDynamics();s=case(mesh,4)
 p=v(s.column.interface_pressure(s.ps));dm=interface_pressure_mass(p.diff())
 q=torch.tensor([.02,.002,.0003]);s.water_content=dm[...,None]*q
 rhs=d.tendency(s);dmdt=rhs.ps*interface_pressure_mass(s.column.B_interface.diff())
 torch.testing.assert_close(rhs.water_content,dmdt[...,None]*q,rtol=1e-11,atol=1e-12)
 drydot=dmdt-rhs.water_content.sum(-1)
 torch.testing.assert_close(drydot,dmdt*(1-q.sum()),rtol=1e-11,atol=1e-12)


def test_vanishing_vapor_limit_and_invalid_state(mesh):
 d=MoistConservativeDynamics();s=case(mesh,4)
 dm=interface_pressure_mass(v(s.column.interface_pressure(s.ps)).diff())
 tendencies=[]
 for vapor in [1e-6,1e-8,1e-10]:
  s.water_content=torch.zeros_like(s.water_content);s.water_content[...,0]=dm*vapor
  tendencies.append(d.tendency(s).u_e)
 assert (tendencies[2]-tendencies[1]).norm() < .05*(tendencies[1]-tendencies[0]).norm()
 s.water_content[...,0]=0
 with pytest.raises(ValueError,match='positive layer mass/vapor'):
  d.tendency(s)


@pytest.mark.parametrize('species',[1,2])
def test_condensate_roundoff_retains_state_and_energy_closure(mesh,species):
 # Exercise a negative Gauss-point fraction as well as a negative interface.
 # Condensate thermodynamics are smooth here; no state/energy clipping is used.
 s=case(mesh,4);d=MoistConservativeDynamics()
 dm=interface_pressure_mass(v(s.column.interface_pressure(s.ps)).diff())
 q=s.water_content/dm[...,None]
 scale=q[...,species].max()
 s.water_content[0,:2,species]=-8*torch.finfo(q.dtype).eps*scale*dm[0,:2]
 original=[getattr(s,k).clone() for k in KEYS]
 bounds=d.composition_bounds(s)
 assert bounds['water_fraction_min'][species]<0
 assert bounds['negative_condensate_mass'][species-1]>0
 leaves=[getattr(s,k).clone().requires_grad_() for k in KEYS]
 a=MoistState(*leaves,mesh,s.column)
 oracle=torch.autograd.grad(d.boundary_total_energy(a),leaves)
 explicit=d.energy_gradients(s)
 for k,g in zip(KEYS,oracle):
  torch.testing.assert_close(getattr(explicit,k),g,rtol=2e-10,atol=1e-3)
 with patch('torch.autograd.grad',side_effect=AssertionError('runtime AD')):
  rhs=d.tendency(s)
 work=[(g*getattr(rhs,k)).sum() for k,g in zip(KEYS,oracle)]
 assert abs(sum(work))/sum(w.abs() for w in work)<1e-10
 for value in [rhs.ps,rhs.entropy_content,*rhs.water_content.unbind(-1)]:
  z=mesh.cell_area[:,None]*value
  assert z.sum().abs()/z.abs().sum().clamp_min(1)<1e-12
 terms=d.tendency_terms(s)
 for k,before in zip(KEYS,original):
  assert torch.equal(getattr(s,k),before)
  torch.testing.assert_close(v(terms[k]['total']),getattr(rhs,k),rtol=1e-12,atol=1e-12)
 strict=MoistConservativeDynamics(allow_condensate_roundoff=False)
 for method in [strict.tendency,strict.energy_terms,strict.energy_gradients]:
  with pytest.raises(ValueError,match='nonnegative condensates'):method(s)


@pytest.mark.parametrize('bad',['above_roundoff','absent_species','other_species_scale'])
def test_roundoff_policy_does_not_hide_negative_condensate(mesh,bad):
 s=case(mesh,4);d=MoistConservativeDynamics()
 dm=interface_pressure_mass(v(s.column.interface_pressure(s.ps)).diff())
 if bad=='above_roundoff':
  s.water_content[0,0,1]=-1e-10*dm[0,0]
 elif bad=='absent_species':
  s.water_content[...,1]=0
  s.water_content[0,0,1]=-1e-30*dm[0,0]
 else:
  # Liquid abundance must not provide a tolerance for an almost absent ice phase.
  s.water_content[...,2]=1e-20*dm
  s.water_content[0,0,2]=-1e-25*dm[0,0]
 with pytest.raises(ValueError,match='nonnegative condensates'):d.tendency(s)


def test_roundoff_policy_leaves_positive_composition_bitwise_unchanged(mesh):
 s=case(mesh,4)
 default=MoistConservativeDynamics()
 strict=MoistConservativeDynamics(allow_condensate_roundoff=False)
 for method in ['tendency','energy_gradients']:
  a=getattr(default,method)(s);b=getattr(strict,method)(s)
  for k in KEYS:assert torch.equal(getattr(a,k),getattr(b,k))
 assert torch.equal(default.boundary_total_energy(s),strict.boundary_total_energy(s))
 a=default.step(s,1.);b=strict.step(s,1.)
 for k in KEYS:assert torch.equal(getattr(a,k),getattr(b,k))


def test_step_validates_returned_stage(mesh):
 # Isolate a violation that first appears in the last RHS update. Earlier
 # stage validation cannot detect it, and step must not return it as valid.
 s=case(mesh,4);d=MoistConservativeDynamics()
 zero=MoistState(*(torch.zeros_like(getattr(s,k)) for k in KEYS),mesh,s.column)
 last=MoistState(*(getattr(zero,k).clone() for k in KEYS),mesh,s.column)
 last.water_content[...,2]=-100.
 original=s.water_content.clone()
 with patch.object(d,'tendency',side_effect=[zero,zero,last]):
  with pytest.raises(ValueError,match='nonnegative condensates'):d.step(s,1.)
 assert torch.equal(s.water_content,original)
