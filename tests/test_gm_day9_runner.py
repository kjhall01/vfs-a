"""Regressions for checkout identity and restarting the parcel-energy experiment."""
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from tools import run_gm_day9 as runner


def test_import_from_symlinked_checkout_rejects_a_different_gm(tmp_path, monkeypatch):
    pytest.importorskip("pyam21")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    native = runner.load_campaign()
    alias = tmp_path / "checkout"
    alias.symlink_to(runner.ROOT, target_is_directory=True)
    monkeypatch.setattr(runner, "ROOT", alias)
    assert runner.load_campaign() is native

    class DifferentGM:
        pass

    monkeypatch.setattr(native, "GMRedi", DifferentGM)
    with pytest.raises(RuntimeError, match="Imported .*Expected"):
        runner.load_campaign()


def test_output_can_retry_after_early_failure_and_preserves_real_run(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    with runner.reserve_output(output):
        with pytest.raises(ValueError, match="Another replay"):
            with runner.reserve_output(output):
                pytest.fail("Concurrent replay acquired the output directory")

    with runner.reserve_output(output):
        (output / "run.log").write_text("existing run")
    with pytest.raises(ValueError, match="Output contains files"):
        with runner.reserve_output(output):
            pytest.fail("Nonempty run directory was acquired")
    assert (output / "run.log").read_text() == "existing run"


def test_restart_bridge_accepts_only_known_gm_transition():
    parent = {runner.EDDY: runner.PARENT_SHA256, "second source": "unchanged"}
    candidate = {**parent, runner.EDDY: runner.PARCEL_SHA256}
    audit = runner.verify_fork(parent, candidate)
    assert audit["source_change"][runner.EDDY]["candidate"] == runner.PARCEL_SHA256

    for other in (parent, {**candidate, "second source": "changed"},
                  {**candidate, runner.EDDY: "other GM implementation"}):
        with pytest.raises(ValueError, match="Expected only"):
            runner.verify_fork(parent, other)
