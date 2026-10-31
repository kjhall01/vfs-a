"""Version selection and integrity are independent of numerical tolerances."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import pytest

ROOT = Path(__file__).resolve().parents[1]


def policy():
    spec = importlib.util.spec_from_file_location('references', ROOT/'tools/regression_references.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_versions_are_explicit_and_historical_bytes_are_preserved():
    p = policy()
    with pytest.raises(ValueError, match='Unknown'):
        p.reference_path('latest')
    assert hashlib.sha256(p.reference_path('historical-v1').read_bytes()).hexdigest() == '00219b583824ef19c15ba8ffae4a2394942c072a260f73337ab22a3b418d7716'
    assert p.reference_path('corrected-gibbs-v1').name == 'corrected_gibbs_v1.pt'


@pytest.mark.parametrize('name', ['provenance.json','corrected_gibbs_v1.pt'])
def test_corrupted_reference_rejected_before_comparison(tmp_path, name):
    p = policy()
    shutil.copytree(p.ROOT, tmp_path/'fixtures')
    p.ROOT = tmp_path/'fixtures'
    path = p.ROOT/'corrected-gibbs-v1'/name
    path.write_bytes(path.read_bytes()+b'corruption')
    with pytest.raises(ValueError, match='hash differs'):
        p.load_reference('corrected-gibbs-v1')


def test_provenance_archives_are_intact():
    p = policy()
    folder = p.ROOT/'corrected-gibbs-v1'
    record = json.loads((folder/'provenance.json').read_text())
    for name, expected in record['files'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == expected
