"""Infrastructure checks, not scientific acceptance thresholds."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from vfs import VoronoiMesh

spec = importlib.util.spec_from_file_location(
    "verification_driver", Path(__file__).resolve().parents[1] / "tools/run_verification.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_metrics_are_reproducible_and_zero_denominator_is_explicit(tmp_path):
    torch.set_default_dtype(torch.float64)
    mesh = VoronoiMesh(n_cells=16, lloyd_iters=0, verbose=False)
    before = {name: value.clone() for name, value in mesh.state_dict().items()}
    first, inputs = driver.measure_geometry(mesh, 71)
    second, repeated = driver.measure_geometry(mesh, 71)
    assert first == second
    for name in inputs:
        assert torch.equal(inputs[name], repeated[name])
    for name, value in mesh.state_dict().items():
        assert torch.equal(value, before[name])
    assert driver.normalized_residual(0, 0) is None
    driver.write_json(tmp_path / "metrics.json", first)
    assert json.loads((tmp_path / "metrics.json").read_text()) == first


def test_nonfinite_measurements_fail():
    mesh = VoronoiMesh(n_cells=16, lloyd_iters=0, verbose=False)
    mesh.cell_area[0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        driver.measure_geometry(mesh, 71)


def test_runtime_provenance_rejects_changed_missing_and_extra_files(tmp_path, monkeypatch):
    expected = b"# frozen runtime\n"
    monkeypatch.setattr(driver, "git", lambda repo, *args: b"vfs/__init__.py\n" if args[0] == "ls-tree" else expected)
    package = tmp_path / "vfs"
    package.mkdir()
    with pytest.raises(ValueError, match="differs"):
        driver.verify_package(tmp_path, package)
    (package / "__init__.py").write_bytes(expected)
    assert driver.verify_package(tmp_path, package) == {"vfs/__init__.py": driver.sha256(expected)}
    (package / "unexpected.py").write_text("# extra\n")
    with pytest.raises(ValueError, match="unexpected"):
        driver.verify_package(tmp_path, package)
    (package / "__init__.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="differs"):
        driver.verify_package(tmp_path, package)


def test_only_pinned_interface_blob_is_allowed(tmp_path, monkeypatch):
    original, extracted = b'original', b'extracted stage sequence'
    def git(repo, *args):
        if args[0] == 'ls-tree':
            return (driver.INTERFACE_PATH + '\n').encode()
        return extracted if args[1].startswith(driver.INTERFACE_COMMIT) else original
    monkeypatch.setattr(driver, 'git', git)
    package = tmp_path / 'vfs'
    path = package / Path(driver.INTERFACE_PATH).relative_to('vfs')
    path.parent.mkdir(parents=True)
    for content in (original, extracted):
        path.write_bytes(content)
        assert driver.verify_package(tmp_path, package)[driver.INTERFACE_PATH] == driver.sha256(content)
    path.write_bytes(extracted + b' unapproved change')
    with pytest.raises(ValueError, match='differs'):
        driver.verify_package(tmp_path, package)


def test_structural_mapping_rejects_unknown_missing_changed_extra_and_tampered(tmp_path, monkeypatch):
    structural = driver.structural
    old = b'old owner\n'
    new = b'new owner\n'
    name = 'vfs/atmosphere/state.py'
    row = {'structural_commit': structural.COMMIT, 'before_structure': 'pinned-old',
           'runtime': {name: {'sha256': driver.sha256(new),
                             'old_sha256': {'vfs/core/state.py': driver.sha256(old)}}}}
    mapping = tmp_path/'mapping.json'
    mapping.write_text(json.dumps(row))
    monkeypatch.setattr(structural, 'MAP', mapping)
    monkeypatch.setattr(structural, 'MAP_SHA256', driver.sha256(mapping.read_bytes()))
    def git(repo, *args):
        if args[0] == 'ls-tree': return (name+'\n').encode()
        return old if args[1].startswith('pinned-old:') else new
    monkeypatch.setattr(driver, 'git', git)
    package = tmp_path/'vfs'
    target = package/'atmosphere/state.py'
    target.parent.mkdir(parents=True)
    with pytest.raises(ValueError, match='Unknown structural'):
        driver.verify_package(tmp_path, package, structural_revision='HEAD')
    with pytest.raises(ValueError, match='differs'):
        driver.verify_package(tmp_path, package, structural_revision=structural.COMMIT)
    target.write_bytes(new)
    assert driver.verify_package(tmp_path, package, structural_revision=structural.COMMIT)[name] == driver.sha256(new)
    target.write_bytes(new+b'changed numerical expression')
    with pytest.raises(ValueError, match='differs'):
        driver.verify_package(tmp_path, package, structural_revision=structural.COMMIT)
    target.write_bytes(new)
    extra = package/'unexpected.py'
    extra.write_text('')
    with pytest.raises(ValueError, match='unexpected'):
        driver.verify_package(tmp_path, package, structural_revision=structural.COMMIT)
    extra.unlink()
    mapping.write_bytes(mapping.read_bytes()+b' ')
    with pytest.raises(ValueError, match='mapping hash'):
        driver.verify_package(tmp_path, package, structural_revision=structural.COMMIT)
