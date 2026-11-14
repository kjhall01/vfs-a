"""Diagnostic harness checks, not qualification of float32 ocean dynamics."""
import sys
from pathlib import Path
import torch
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'validation/temperature_coordinate'),str(Path(__file__).resolve().parents[2]/'validation/ocean_column')]
from precision_campaign import construct,audit,bind
from complete_campaign import make
from vfs.ocean.stratified import TKEState


@pytest.fixture
def case():
    previous=torch.get_default_dtype();torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    source=make(16,4,False)
    try:yield construct(source,torch.float32,False,'cpu')
    finally:torch.set_default_dtype(previous)


def test_geometry_state_and_oracle_ownership(case):
    model,state,forcing,oracle=case
    assert state.ocean.mesh is model.ocean.mesh
    assert state.ocean.column is model.ocean.column
    assert all(x.dtype==torch.float32 for x in model.ocean.mesh.buffers() if x.is_floating_point())
    assert oracle.ocean.mesh is not model.ocean.mesh
    torch.testing.assert_close(oracle.ocean.mesh.cell_area,model.ocean.mesh.cell_area.double(),rtol=0,atol=0)
    model.check(state)


def test_audit_promotes_before_multiplication_and_reduction(case):
    model,state,forcing,oracle=case
    result=audit(oracle,state)
    area=state.ocean.mesh.cell_area.double()[:,None]
    expected=(area*model.ocean.column.rho0*state.ocean.h.double()).sum()
    assert result['mass']==float(expected)
    # Deliberate positive salt-content change must remain visible in the audit.
    s=state.ocean
    changed=s.with_tensors(s.u_e,s.h,s.temp_h,s.salt_h*1.001,s.tracer_h)
    new=audit(oracle,TKEState(changed,state.tke_h))
    assert new['salt']>result['salt'] and new['energy']!=result['energy']
    assert new['mass']==result['mass'] and new['tracer']==result['tracer']


def test_cast_after_construction_requires_cache_rebuild(case):
    model,state,forcing,oracle=case
    model.ocean.mesh.double()
    model.ocean.column.double()
    transport=model.ocean.base._stratified_transport
    with pytest.raises(ValueError,match='rebuild the model'):
        transport._indices(state.ocean.h.shape[-1],state.ocean.h.device)
