import numpy as np
from scipy.integrate import solve_ivp
import pytest,torch
from test_moist_conservative import case
from vfs.atmosphere import MoistConservativeDynamics
from vfs.geometry.mesh import VoronoiMesh
from vfs.atmosphere.moist_conservative import MoistState
from vfs.integrators.imex.moist_wave import MoistPressureWave
from vfs.integrators.imex.moist_ark import MoistPEIMEXARK,_MoistInner,_Vector
from vfs.imex import IMEXSolverConfig

def setup(levels=3):
 torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
 mesh=VoronoiMesh(n_cells=16,lloyd_iters=0,verbose=False)
 d=MoistConservativeDynamics();s=case(mesh,levels)
 g=torch.Generator().manual_seed(42)
 z=MoistState(torch.randn(s.u_e.shape,generator=g),50*torch.randn(s.ps.shape,generator=g),100*torch.randn(s.entropy_content.shape,generator=g),.01*torch.randn(s.water_content.shape,generator=g),mesh,s.column)
 return d,s,z

@pytest.mark.parametrize('levels',[1,3,12])
def test_analytic_pressure_and_transport(levels):
 d,s,z=setup(levels);w=MoistPressureWave(d,s)
 def pressure(S,C,p):return d.tendency_terms(MoistState(s.u_e,p,S,C,s.mesh,s.column))['u_e']['pgf']
 _,expected=torch.autograd.functional.jvp(pressure,(s.entropy_content,s.water_content,s.ps),(z.entropy_content,z.water_content,z.ps))
 torch.testing.assert_close(w.wind(z.entropy_content,z.water_content,z.ps),expected,rtol=2e-9,atol=1e-15)
 def transport(u):
  r=d.tendency(MoistState(u,s.ps,s.entropy_content,s.water_content,s.mesh,s.column))
  return r.ps,r.entropy_content,r.water_content
 _,expected=torch.autograd.functional.jvp(transport,(s.u_e,),(z.u_e,))
 for a,b in zip(w.transport(z.u_e),expected):torch.testing.assert_close(a,b,rtol=2e-12,atol=1e-12)

def test_step_no_autograd_and_conservation(monkeypatch):
 d,s,z=setup()
 def forbidden(*args,**kwargs):raise AssertionError('runtime autograd')
 monkeypatch.setattr(torch.autograd.functional,'jvp',forbidden);monkeypatch.setattr(torch.func,'jvp',forbidden);monkeypatch.setattr(torch.autograd,'grad',forbidden)
 with torch.no_grad():r,info=MoistPEIMEXARK(d).step_with_info(s,1800.)
 assert info.converged
 A=s.mesh.cell_area[:,None]
 torch.testing.assert_close((A*r.ps).sum(),(A*s.ps).sum(),rtol=2e-14,atol=0)
 torch.testing.assert_close((A[...,None]*r.water_content).sum((0,1)),(A[...,None]*s.water_content).sum((0,1)),rtol=2e-13,atol=0)
 assert (r.water_content[...,0]>0).all()


def test_schur_full_agree():
 d,s,z=setup();out=[]
 with torch.no_grad():
  for pre in ['none','hydrostatic_schur']:
   out.append(MoistPEIMEXARK(d,IMEXSolverConfig(preconditioner=pre,rtol=1e-11,max_iterations=100)).step(s,600))
 for k in ['u_e','ps','entropy_content','water_content']:
  torch.testing.assert_close(getattr(out[0],k),getattr(out[1],k),rtol=2e-9,atol=1e-7)


def test_entropy_reference_invariant_norm():
 from dataclasses import replace
 from vfs.atmosphere import MoistConservativeDynamics
 d,s,z=setup();w=MoistPressureWave(d,s);cd,cv=80.,-40.
 t=replace(d.thermodynamics,p_d_ref=d.thermodynamics.p_d_ref*__import__('math').exp(cd/d.thermodynamics.R_d),p_v_ref=d.thermodynamics.p_v_ref*__import__('math').exp(cv/d.thermodynamics.R_v))
 dm=w.dm;ddm=w.dm1*z.ps
 def shift(S,C,m):return S+cd*m+(cv-cd)*C[...,0]-cd*C[...,1:].sum(-1)
 s2=MoistState(s.u_e,s.ps,shift(s.entropy_content,s.water_content,dm),s.water_content,s.mesh,s.column)
 z2=MoistState(z.u_e,z.ps,shift(z.entropy_content,z.water_content,ddm),z.water_content,s.mesh,s.column)
 w2=MoistPressureWave(MoistConservativeDynamics(thermodynamics=t),s2)
 torch.testing.assert_close(_MoistInner(w).norm(_Vector.pack(z)),_MoistInner(w2).norm(_Vector.pack(z2)),rtol=2e-13,atol=1e-15)
 torch.testing.assert_close(w.wind(z.entropy_content,z.water_content,z.ps),w2.wind(z2.entropy_content,z2.water_content,z2.ps),rtol=2e-9,atol=1e-15)


def test_moist_native_order_against_independent_dop853():
 d,s,z=setup();keys=('u_e','ps','entropy_content','water_content')
 shapes=[getattr(s,k).shape for k in keys];sizes=[int(np.prod(x)) for x in shapes]
 def pack(z):return np.concatenate([getattr(z,k).detach().numpy().ravel() for k in keys])
 def unpack(y):
  parts=np.split(y,np.cumsum(sizes)[:-1])
  return MoistState(*(torch.from_numpy(x.reshape(shape).copy()) for x,shape in zip(parts,shapes)),s.mesh,s.column)
 with torch.no_grad():
  ref=solve_ivp(lambda t,y:pack(d.tendency(unpack(y))),[0,3600],pack(s),method='DOP853',rtol=2e-12,atol=1e-11)
  assert ref.success
  ref=unpack(ref.y[:,-1]);inner=_MoistInner(MoistPressureWave(d,s));errors=[]
  for dt in [600,300,150]:
   stepper=MoistPEIMEXARK(d,IMEXSolverConfig(preconditioner='hydrostatic_schur',rtol=1e-12,max_iterations=100))
   out=s
   for _ in range(3600//dt):out=stepper.step(out,dt)
   errors.append(float(inner.norm(_Vector.pack(out)-_Vector.pack(ref))))
  print("ark4",errors)
  order=4
  assert min(errors[0]/errors[1],errors[1]/errors[2])>2**(order-.4)


def test_stage_guard_and_input_preservation():
 d,s,z=setup();before=[v.clone() for v in _Vector.pack(s).values];seen=[]
 def guard(state,dt):seen.append(dt)
 stepper=MoistPEIMEXARK(d,stage_validator=guard)
 with torch.no_grad():stepper.step(s,600.)
 assert seen==[600.]*7  # Six ARK stages and the returned state.
 for a,b in zip(_Vector.pack(s).values,before):assert torch.equal(a,b)
 def reject(state,dt):raise ValueError('limiter required')
 with pytest.raises(ValueError,match='limiter required'):
  MoistPEIMEXARK(d,stage_validator=reject).step(s,600.)
 for a,b in zip(_Vector.pack(s).values,before):assert torch.equal(a,b)

@pytest.mark.parametrize('dt',[0.,-1.,float('nan'),float('inf')])
def test_reject_bad_dt(dt):
 d,s,z=setup()
 with pytest.raises(ValueError,match='dt'):MoistPEIMEXARK(d).step(s,dt)


def test_reject_nonfinite_wind():
 d,s,z=setup();s.u_e[0,0]=float('nan')
 with pytest.raises(ValueError,match='nonfinite'):MoistPEIMEXARK(d).step(s,600.)


def test_absent_condensates_remain_absent():
 d,s,z=setup();s.water_content[...,1:]=0
 with torch.no_grad():out=MoistPEIMEXARK(d).step(s,1800.)
 assert torch.equal(out.water_content[...,1:],s.water_content[...,1:])



def test_returned_state_is_validated(monkeypatch):
 d,s,z=setup();before=s.water_content.clone();real=d.tendency;calls=0
 def tendency(stage):
  nonlocal calls
  calls+=1
  rhs=real(stage)
  if calls==6:
   rhs.water_content=rhs.water_content.clone()
   rhs.water_content[...,2]=-100.
  return rhs
 monkeypatch.setattr(d,'tendency',tendency)
 with pytest.raises(ValueError,match='nonnegative condensates'):
  MoistPEIMEXARK(d).step(s,1.)
 assert calls==6
 assert torch.equal(s.water_content,before)


def test_zero_vapor_is_rejected():
 d,s,z=setup();s.water_content[...,0]=0
 with pytest.raises(ValueError,match='positive layer mass/vapor'):
  MoistPEIMEXARK(d).step(s,600.)


def test_unsupported_wave_block_rejected():
 d,s,z=setup()
 for cfg in [IMEXSolverConfig(preconditioner='multilevel_hydrostatic_schur')]:
  with pytest.raises(ValueError,match='supports eager analytic waves'):
   MoistPEIMEXARK(d,cfg)
