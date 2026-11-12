"""Independent production match and read-only guards for JW-A rotation audit."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy import sparse
import torch
from vfs import VoronoiMesh

spec=importlib.util.spec_from_file_location('jwa_audit',Path(__file__).resolve().parents[1]/'tools/audit_jwa_rotation_balance.py')
a=importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def test_pairing_against_dense_oracle():
    rng=np.random.default_rng(23)
    J=rng.normal(size=(7,7));u=rng.normal(size=(7,3));q=rng.normal(size=(7,3))
    metric=np.exp(rng.normal(size=(7,3)))
    actual=a.paired_rotation(sparse.csr_matrix(J),u,q,metric)
    for k in range(3):
        A=-.5*(q[:,k,None]+q[None,:,k])*J
        M=np.diag(metric[:,k]);expected=.5*(A-np.linalg.solve(M,A.T@M))@u[:,k]
        np.testing.assert_allclose(actual[:,k],expected,atol=3e-14,rtol=1e-13)
        assert abs(np.dot(metric[:,k]*u[:,k],actual[:,k]))<1e-12


def test_jwa_measure_matches_production_and_keeps_geometry():
    mesh=VoronoiMesh(n_cells=32,lloyd_iters=0)
    before={n:t.clone() for n,t in mesh.named_buffers()}
    r=a.measure(mesh,levels=4)
    for name,value in mesh.named_buffers(): assert torch.equal(value,before[name]),name
    assert not r['initial_condition']['perturb']
    assert not r['initial_condition']['project_mass_flux']
    variants=r['variants']
    assert variants['stored_reassembled']['rotation_change']['max_abs_m_s2']<1e-16
    for item in variants.values():
        assert abs(item['rotation_work_relative'])<1e-13
        assert abs(item['closed_energy_rate_relative'])<1e-12
        assert len(item['full_momentum_residual']['per_level_area_rms_m_s2'])==4
