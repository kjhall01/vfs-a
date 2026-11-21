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

def test_radiation_manifest_json_roundtrip_preserves_strict_identity():
    import json
    live = dict(
        files={"optical.dat": {"bytes": 123, "sha256": "abcdef"}},
        boxes=[((0.1, 0.2), (-0.4, -0.3))],
        reference_pressures_Pa=[[100.0, 200.0]],
        namelist={"radiation_driver_nml": {"solar_constant": 1361.0}},
    )
    persisted = json.loads(json.dumps(live))
    assert live != persisted  # tuple/list mismatch formerly rejected valid resume
    assert module.canonical_manifest_json(live) == module.canonical_manifest_json(persisted)
    persisted["files"]["optical.dat"]["sha256"] = "different"
    assert module.canonical_manifest_json(live) != module.canonical_manifest_json(persisted)


def test_canonical_radiation_manifest_preserves_precision():
    live = {"boxes": [((0.12345678901234567, 0.2), (0.3, 0.4))]}
    changed = {"boxes": [[[0.12345678901234568, 0.2], [0.3, 0.4]]]}
    assert module.canonical_manifest_json(live) != module.canonical_manifest_json(changed)
