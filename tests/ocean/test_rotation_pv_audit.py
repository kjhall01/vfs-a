"""Independent oracles and corruption sensitivity for read-only PV diagnostics."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from vfs import VoronoiMesh

spec=importlib.util.spec_from_file_location('pv_audit',Path(__file__).resolve().parents[2]/'tools/audit_rotation_pv.py')
a=importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


@pytest.mark.parametrize('icosa',[False,True])
def test_geometric_partition_and_sparse_operators(icosa):
    m=VoronoiMesh.refined_icosahedron(2) if icosa else VoronoiMesh(n_cells=32,lloyd_iters=0)
    area,P,_,_=a.transfers(m)
    np.testing.assert_allclose(P@np.ones(m.nCells),1,atol=3e-14,rtol=0)
    np.testing.assert_allclose(P.T@a.array(m.dual_area),a.array(m.cell_area),rtol=3e-14)
    assert area.min()>=0
    D,C,raw,stored=a.operators(m)
    rng=np.random.default_rng(73)
    u=torch.from_numpy(rng.normal(size=(m.nEdges,3)))
    np.testing.assert_allclose(D@a.array(u),a.array(m.div(u)),rtol=1e-13,atol=1e-20)
    np.testing.assert_allclose(C@a.array(u),a.array(m.curl(u)),rtol=1e-13,atol=1e-20)
    rebuilt=a.path_rotation(m,a.array(m.thuburn_corner_fraction),shift=0)
    assert a.relative(rebuilt,raw)<1e-13


def test_compatibility_ablation_and_wrong_index_sensitivity():
    m=VoronoiMesh(n_cells=32,lloyd_iters=2)
    D,C,raw,stored=a.operators(m)
    area,P,Ps,Pshift=a.transfers(m)
    J=a.path_rotation(m,area/a.array(m.cell_area)[:,None],shift=-1)
    assert a.relative(C@J,P@D)<1e-12
    assert a.relative(C@raw,Pshift@D)<1e-12
    # The deliberately wrong association must not pass the same oracle.
    wrong=a.path_rotation(m,area/a.array(m.cell_area)[:,None],shift=0)
    assert a.relative(C@wrong,P@D)>1e-3
    W=a.sparse.diags(a.array(m.dc_edge*m.dv_edge))@J
    assert a.norm(W+W.T)/a.norm(W)<1e-12


def test_uniform_pv_oracle_uses_production_and_restores_f():
    m=VoronoiMesh(n_cells=32,lloyd_iters=0)
    area,P,_,_=a.transfers(m)
    D,C,raw,stored=a.operators(m)
    J=a.path_rotation(m,area/a.array(m.cell_area)[:,None],shift=-1)
    original=m.coriolis_unit_vertex.clone()
    report=a.uniform_pv(m,P,stored,J,.2)
    assert torch.equal(m.coriolis_unit_vertex,original)
    assert report['initial_pv_relative_max']<1e-14
    assert report['production_pairing_relative_error']<1e-13
    assert report['full_vs_rotation_curl_relative']<1e-12
    trial=report['variants']['pv_flux_compatible_geometry']
    assert trial['relative_vorticity_tendency_defect']<1e-12
    assert abs(trial['normalized_rotation_work'])<1e-13
    assert abs(report['variants']['production']['normalized_rotation_work'])<1e-13
