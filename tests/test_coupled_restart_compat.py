"""Audited restart bridge checks, including strict rejection of unknown source drift."""
import importlib.util
from pathlib import Path
import pytest

source = Path(__file__).resolve().parents[1] / "workflows" / "coupled" / "restart_compat.py"
spec = importlib.util.spec_from_file_location("restart_compat_standalone", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def old_source():
    return dict(module.APPROVED_PRE_TRACE_HASHES, **{"vfs/ocean/jmd95.py": "original"})


def new_source():
    return dict(module.APPROVED_PRE_TRACE_HASHES,
                **{"vfs/ocean/jmd95.py": "original",
                   "workflows/coupled/restart_compat.py": "added"})


def test_approved_instrumentation_only_transition():
    before, after = old_source(), new_source()
    after["validation/sea_ice/snow.py"] = "updated tracing callback"
    after["workflows/coupled/campaign.py"] = "new trace CLI"
    result = module.verify_instrumentation_only_delta(before, after)
    assert set(result["modified_instrumentation_files"]) == {
        "validation/sea_ice/snow.py", "workflows/coupled/campaign.py"
    }
    assert result["added_verifier_files"] == ["workflows/coupled/restart_compat.py"]


def test_reject_modified_physics():
    before, after = old_source(), new_source()
    after["vfs/ocean/jmd95.py"] = "modified"
    with pytest.raises(ValueError, match="non-instrumentation changes"):
        module.verify_instrumentation_only_delta(before, after)


def test_reject_unknown_old_trace_version():
    before, after = old_source(), new_source()
    before["workflows/coupled/energy_trace.py"] = "unknown old baseline"
    with pytest.raises(ValueError, match="unknown pre-trace sources"):
        module.verify_instrumentation_only_delta(before, after)


def test_reject_removed_files():
    before, after = old_source(), new_source()
    del after["vfs/ocean/jmd95.py"]
    with pytest.raises(ValueError, match="source inventory changed"):
        module.verify_instrumentation_only_delta(before, after)
