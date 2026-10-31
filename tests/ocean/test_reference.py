"""Archive migration checks, not scientific acceptance thresholds."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from vfs.ocean import Ocean, OceanColumn, OceanDomain, OceanState
from vfs.ocean.dynamics import MultilayerOceanDynamics
from vfs.ocean.physics import OceanSurfaceForcing
from vfs.ocean.io import save_snapshot, load_snapshot
from vfs.integrators.explicit import ssprk3
from vfs.geometry.mesh import VoronoiMesh
from vfs.field import Field

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('capture_ocean', ROOT/'tools/capture_ocean.py')
CAPTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAPTURE)
FIXTURES = ROOT/'tests/fixtures/ocean'


def case():
    torch.set_default_dtype(torch.float64)
    mesh, fields = CAPTURE.load_inputs(FIXTURES/'inputs.npz')
    column = OceanColumn(3, total_depth=3000.)
    return Ocean(mesh,column), OceanState(**fields,mesh=mesh,column=column)


def test_archive_capture(fixture_mode):
    torch.set_default_dtype(torch.float64)
    actual = CAPTURE.capture(FIXTURES/'inputs.npz')
    with np.load(FIXTURES/'archive_capture.npz',allow_pickle=False) as z:
        assert actual.keys() == set(z.files)
        for key,value in actual.items():
            expected=z[key]
            if fixture_mode=='exact' or value.dtype.kind not in 'fc':
                assert np.array_equal(value,expected),key
            else:
                # Portable migration allowance: a roundoff-scaled bound per
                # array. Same-environment reproduction uses exact comparison.
                scale=max(float(np.max(np.abs(expected))),1e-300)
                np.testing.assert_allclose(value,expected,rtol=512*np.finfo(value.dtype).eps,
                    atol=512*np.finfo(value.dtype).eps*scale,err_msg=key)


def test_tensor_identity_and_gradient_survive_one_step():
    ocean,state=case()
    u=state.u_e.detach().requires_grad_()
    field=Field(u,location='edge',vertical_location='full')
    active=OceanState(field,state.h,state.temp_h,state.salt_h,state.tracer_h,
                      mesh=state.mesh,column=state.column)
    assert active.u_e is u
    out=ocean.step(active,30.)
    objective=out.u_e.square().sum()
    grad,=torch.autograd.grad(objective,u)
    direction=torch.sin(torch.arange(u.numel(),dtype=u.dtype)).reshape_as(u)
    def f(a):
        s=active.with_tensors(a,active.h,active.temp_h,active.salt_h,active.tracer_h)
        return ocean.step(s,30.).u_e.square().sum()
    eps=1e-5
    fd=(f(u+eps*direction)-f(u-eps*direction))/(2*eps)
    torch.testing.assert_close((grad*direction).sum(),fd,rtol=1e-7,atol=1e-9)


def test_facade_matches_ordered_sum_and_shared_stages():
    ocean,state=case()
    source=OceanSurfaceForcing()
    ocean=Ocean(state.mesh,state.column,sources=(source,))
    expected=ssprk3(lambda s:ocean.dynamics.tendency(s)+source.tendency(s),state,30.)
    actual=ocean.step(state,30.)
    for name in CAPTURE.FIELDS:
        assert torch.equal(getattr(actual,name),getattr(expected,name))
    assert 'reference_mechanical_energy' in ocean.budgets(state)
    assert len(ocean.tendency_terms(state)['sources'])==1


def test_physical_initialization_and_no_tracer():
    ocean,state=case()
    actual=ocean.state(u_e=state.u_e,h=state.h,temperature=state.temperature(),salinity=state.salinity())
    assert actual.h is state.h
    assert actual.u_e is state.u_e
    assert actual.tracer_h is None
    assert ocean.step(actual,30.).tracer_h is None
    with pytest.raises(ValueError,match='dtype'):
        ocean.state(u_e=state.u_e.float(),h=state.h,temperature=state.temperature(),salinity=state.salinity())


def test_wrong_geometry_and_domain_are_rejected():
    ocean,state=case()
    different=OceanColumn(3,total_depth=3000.)
    with pytest.raises(ValueError,match='exact mesh and column'):
        ocean.step(OceanState(state.u_e,state.h,state.temp_h,state.salt_h,
                             mesh=state.mesh,column=different),30.)
    mesh=VoronoiMesh(n_cells=32,lloyd_iters=0,verbose=False)
    domain=OceanDomain.global_ocean(mesh)
    with pytest.raises(ValueError,match='exact mesh'):
        Ocean(state.mesh,state.column,domain=domain)
    domain=OceanDomain.global_ocean(state.mesh)
    with pytest.raises(ValueError,match='same domain'):
        Ocean(state.mesh,state.column,domain=domain,dynamics=MultilayerOceanDynamics())


@pytest.mark.parametrize('dt',[0.,-1.,float('nan'),float('inf')])
def test_bad_step_rejected_without_mutation(dt):
    ocean,state=case();old=state.h.clone()
    with pytest.raises(ValueError,match='dt'):
        ocean.step(state,dt)
    assert torch.equal(state.h,old)


def test_snapshot_restart_and_explicit_regrid(tmp_path):
    ocean,state=case()
    save_snapshot(state,tmp_path/'ocean.nc')
    restored=load_snapshot(tmp_path/'ocean.nc',mesh=state.mesh,column=state.column)
    for name in CAPTURE.FIELDS:
        assert torch.equal(getattr(restored,name),getattr(state,name))
        assert torch.equal(getattr(ocean.step(restored,30.),name),getattr(ocean.step(state,30.),name))
    from vfs.geometry.remap import LatLonGrid, MeshToLatLonKNN
    from vfs.ocean.regrid import state_to_latlon
    grid=LatLonGrid.regular(6,12)
    remapper=MeshToLatLonKNN.from_mesh(state.mesh,grid)
    output=state_to_latlon(remapper,state,fields='all')
    assert output['h'].shape[:2]==(remapper.nlat,remapper.nlon)
    assert torch.isfinite(output['temperature'].values).all()


def test_capture_detects_changed_equation_output():
    original={'rhs':np.array([1.,2.])}
    changed={'rhs':np.array([1.,2.+1e-8])}
    assert CAPTURE.compare(changed,original)['differences']
