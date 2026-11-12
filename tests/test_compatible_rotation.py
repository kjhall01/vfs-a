"""Compatible geometry, PV transport, work, derivatives and component use."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from vfs import VoronoiMesh, MoistAtmosphere, DryIMEX
from vfs.ocean import OceanState, OceanColumn
from vfs.ocean.conservative import ConstantDensityDynamics
from vfs.geometry.mesh.grid import _save_mesh as save_mesh_netcdf, _load_mesh as load_mesh_netcdf

spec=importlib.util.spec_from_file_location('pv_reference',Path(__file__).parents[1]/'tools/audit_rotation_pv.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def grid(n=32):
    return VoronoiMesh(n_cells=n,lloyd_iters=0,rotation_scheme='compatible')


@pytest.mark.parametrize('n',[16,64,256])
def test_compatible_geometry_and_skew(n):
    m=grid(n);D,C,_,J=a.operators(m);area,P,_,_=a.transfers(m)
    assert a.relative(C@J,P@D)<2e-12
    x=torch.randn(n,3,dtype=torch.float64)
    np.testing.assert_allclose(a.array(m.cell_to_dual(x)),P@a.array(x),rtol=1e-12,atol=1e-12)
    W=a.sparse.diags(a.array(m.dc_edge*m.dv_edge))@J
    assert a.norm(W+W.T)/a.norm(W)<1e-13
    torch.testing.assert_close(m.cell_to_dual(torch.ones_like(x)),torch.ones(m.nVertices,3,dtype=x.dtype),rtol=1e-12,atol=1e-12)


def test_uniform_pv_variable_thickness_and_independent_flux_oracle():
    m=grid();_,P,_,_=a.transfers(m);_,C,_,J=a.operators(m)
    # This invokes production ConstantDensityDynamics with synthetic f and
    # checks exact initial uniform PV; f is restored by the audit helper.
    r=a.uniform_pv(m,P,J,J,.2)
    assert r['initial_pv_relative_max']<1e-14
    p=r['variants']['production']
    assert p['relative_vorticity_tendency_defect']<1e-11
    assert abs(p['normalized_rotation_work'])<1e-13


def test_general_pv_work_and_derivative():
    m=grid(16);torch.manual_seed(6)
    u=torch.randn(m.nEdges,1,dtype=torch.float64,requires_grad=True)
    h=(1000+100*torch.randn(m.nCells,1,dtype=torch.float64)).requires_grad_()
    r=m.compatible_rotation(u,h,7.292115e-5)
    metric=(m.dc_edge*m.dv_edge)[:,None]*m.cell_to_edge(h)
    work=metric*u*r
    assert abs(float(work.sum().detach()))<1e-13*float(work.abs().sum().detach())
    # Compare reverse AD with independent centered differences, including mass.
    assert torch.autograd.gradcheck(lambda uu,hh:m.compatible_rotation(uu,hh,7.292115e-5), (u,h),eps=1e-4,atol=1e-9,rtol=1e-5)
    with pytest.raises(ValueError): m.compatible_rotation(u,-h,0.)


def test_mesh_cache_roundtrip(tmp_path):
    m=grid();path=tmp_path/'mesh.nc';save_mesh_netcdf(m,path)
    other=load_mesh_netcdf(path)
    assert other.rotation_scheme=='compatible'
    u=torch.randn(m.nEdges,2,dtype=torch.float64);h=torch.full((m.nCells,2),1000.,dtype=u.dtype)
    torch.testing.assert_close(other.compatible_rotation(u,h,1e-4),m.compatible_rotation(u,h,1e-4),rtol=0,atol=0)


def test_atmospheric_work_and_imex():
    from vfs import make_hybrid_coefficients, CharneyPhillipsColumn, DryAtmosphere
    from vfs.atmosphere.initial_conditions.jablonski_williamson import JWInitialConditions
    m=grid();A,B=make_hybrid_coefficients(3,p_top=5000.)
    col=JWInitialConditions.make_column_for_mode(m,3,A,B,mode='constant_ps_surface_phi',phi_full_mode='direct_u_gl2_adjoint')
    dry=DryAtmosphere(m,col,stepper=DryIMEX())
    s=JWInitialConditions(m,col,mode='constant_ps_surface_phi').make_state(dry.dynamics,perturb=False)
    r=dry.dynamics.rotation_work_terms(s)
    assert float(r['work_relative'])<1e-12
    out=dry(s,1.);assert torch.isfinite(out.u_e).all()
    m2=grid();col2=CharneyPhillipsColumn(3,A,B,phi_full_mode='direct_u_gl2_adjoint')
    moist=MoistAtmosphere(m2,col2)
    s=moist.state_from_fields(temperature=280.,surface_pressure=100000.,normal_wind=1.,vapor=.01)
    rhs=moist.dynamics.tendency(s)
    assert torch.isfinite(rhs.u_e).all()
    out=moist(s,1.);assert torch.isfinite(out.water_content).all()
    from vfs import CenteredIMEX
    m3=grid();col3=CharneyPhillipsColumn(3,A,B,phi_full_mode='direct_u_gl2_adjoint')
    implicit=MoistAtmosphere(m3,col3,stepper=CenteredIMEX())
    s3=implicit.state_from_fields(temperature=280.,surface_pressure=100000.,normal_wind=1.,vapor=.01)
    out3=implicit(s3,1.);assert torch.isfinite(out3.water_content).all()



@pytest.mark.parametrize('coast',[False,True])
def test_stratified_ocean_steps_and_remap(coast):
    from vfs.ocean import OceanDomain
    from vfs.ocean.stratified import StratifiedOcean, AdaptiveThermalRemapper
    m=grid();column=OceanColumn(3,total_depth=3000.)
    domain=OceanDomain(m,m.cell_xyz[:,2]<.6) if coast else None
    model=StratifiedOcean(m,column,domain=domain,remapper=AdaptiveThermalRemapper())
    h=torch.tensor([500.,1000.,1500.],dtype=torch.float64).expand(m.nCells,3).clone()
    T=torch.tensor([17.,10.,4.],dtype=torch.float64)[None,:]+.4*m.cell_xyz[:,:1]
    S=torch.tensor([34.,35.,35.5],dtype=torch.float64)[None,:]+.05*m.cell_xyz[:,1:2]
    s=model.state(u_e=torch.zeros(m.nEdges,3,dtype=torch.float64),h=h,temperature=T,salinity=S,tracer=torch.ones(m.nCells,3,1,dtype=torch.float64))
    before=model.budgets(s)
    with torch.no_grad():
        for _ in range(5): s=model.step(s,10.)
        s,report=model.remap_with_report(s,s.h.sum(-1,keepdim=True)*torch.tensor([.2,.3,.5],dtype=torch.float64))
    after=model.budgets(s)
    for key in ('salt','volume','tracers'):
        torch.testing.assert_close(after[key],before[key],rtol=2e-13,atol=1e-8)
    assert torch.isfinite(s.u_e).all()
    if coast: assert torch.count_nonzero(s.u_e[~domain.wet_edge])==0


def test_moist_energy_derivative_pairing():
    from vfs import make_hybrid_coefficients, CharneyPhillipsColumn
    m=grid();A,B=make_hybrid_coefficients(3)
    model=MoistAtmosphere(m,CharneyPhillipsColumn(3,A,B,phi_full_mode='direct_u_gl2_adjoint'))
    ps=100000.+500.*m.cell_xyz[:,2:3]
    s=model.state_from_fields(temperature=280.,surface_pressure=ps,normal_wind=2.,vapor=.01)
    rhs=model.dynamics.tendency(s);g=model.dynamics.energy_gradients(s)
    works=[getattr(g,k)*getattr(rhs,k) for k in ('u_e','ps','entropy_content','water_content')]
    assert abs(float(sum(w.sum() for w in works))) < 1e-12*float(sum(w.abs().sum() for w in works))
