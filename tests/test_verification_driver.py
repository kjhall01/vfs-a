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
