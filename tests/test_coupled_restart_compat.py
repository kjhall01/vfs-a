"""Audited restart bridge checks, including strict rejection of unknown source drift."""
import importlib.util
import json
from pathlib import Path
import pytest

source = Path(__file__).resolve().parents[1] / "workflows" / "coupled" / "restart_compat.py"
spec = importlib.util.spec_from_file_location("restart_compat_standalone", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def new_source():
    fixture = Path(__file__).parent / "fixtures/coupled_restart_sources-6e68cb3.json"
    return json.loads(fixture.read_text())


def old_source():
    old = {**new_source(), **module.APPROVED_PRE_TRACE_HASHES}
    del old[module.VERIFIER]
    return old


def test_approved_instrumentation_only_transition():
    before, after = old_source(), new_source()
    result = module.verify_instrumentation_only_delta(before, after)
    assert result["modified_instrumentation_files"] == sorted(module.APPROVED_PRE_TRACE_HASHES)
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


def test_reject_unknown_new_trace_and_identically_modified_physics():
    before, after = old_source(), new_source()
    with pytest.raises(ValueError, match="audited trace revision"):
        module.verify_instrumentation_only_delta(before, {**after, "validation/sea_ice/snow.py": "changed physics"})
    for manifest in (before, after):
        manifest["vfs/ocean/jmd95.py"] = "same unaudited change"
    with pytest.raises(ValueError, match="audited trace revision"):
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


def endpoint_sources():
    previous = new_source()
    fixture = Path(__file__).parent / "fixtures/coupled_endpoint_sources-ed46ac5.json"
    return previous, {**previous, **json.loads(fixture.read_text())}


def test_audited_endpoint_restart_and_earlier_trace_restart():
    previous, current = endpoint_sources()
    assert module.verify_compatible_source_delta(previous, current) == {
        "endpoint_fix_files": sorted(module.PRE_ENDPOINT_HASHES)
    }
    pinned_trace = {**previous, module.VERIFIER: module.PINNED_TRACE_VERIFIER}
    assert module.verify_compatible_source_delta(pinned_trace, current) == {
        "endpoint_fix_files": sorted(module.PRE_ENDPOINT_HASHES)
    }
    assert module.verify_compatible_source_delta(current, current) == {"endpoint_fix_files": []}
    # A checkpoint from before the trace instrumentation also needs its
    # original source changes accounted for.
    old = {**previous, **module.APPROVED_PRE_TRACE_HASHES}
    del old["workflows/coupled/restart_compat.py"]
    report = module.verify_compatible_source_delta(old, current)
    assert report["modified_instrumentation_files"] == sorted(module.APPROVED_PRE_TRACE_HASHES)
    assert report["added_verifier_files"] == ["workflows/coupled/restart_compat.py"]


@pytest.mark.parametrize("path", ["workflows/pyam21/adapter.py", "vfs/ocean/jmd95.py"])
def test_unknown_source_fails_on_either_side_of_endpoint_restart(path):
    previous, current = endpoint_sources()
    with pytest.raises(ValueError, match="restart"):
        module.verify_compatible_source_delta({**previous, path: "unknown"}, current)
    with pytest.raises(ValueError, match="audited endpoint revision"):
        module.verify_compatible_source_delta(previous, {**current, path: "unknown"})


def day15_source():
    """Reconstruct pinned day-15 manifest from archived source fixtures.

    Changes since the 6e68cb3 fixture are themselves pinned by the day-15
    checkpoint manifest. Unknown changes must never qualify for a restart.
    """
    source, endpoint = endpoint_sources()
    updated = dict(endpoint)
    updated.update({
        "workflows/coupled/campaign.py":
            "08553df13411dc7fe416711c3fecc2cbcf41b00f1540d01d11f0aba509da0754",
        "workflows/coupled/restart_compat.py":
            module.DAY15_ORIGINAL_VERIFIER_SHA,
        "validation/sea_ice/category_phase.py":
            "31270fef6cc80a3fa06d23d2304fc9e713b21eb181e98429b3799e8c6d3a56b6",
        "validation/sea_ice/ocean_ice.py":
            "48e078d70601e741a68e81a859f99059500d351ee172aa0a85c2eb3e9ac42963",
        "validation/temperature_coordinate/eddy.py":
            "b9ab1403e93ef90ad49d755f8cdce9bbad004d0b76fa243fa348d599c60473b2",
    })
    return updated


def test_day15_restarts_only_into_pinned_flux_fix():
    original = day15_source()
    fixed = dict(original)
    fixed[module.DAY15_ENERGY_PATH] = module.DAY15_FIXED_ENERGY_SHA
    fixed[module.VERIFIER] = "current verifier source SHA256"  # excluded from digest
    assert module._without_verifier_digest(original) == module.DAY15_ORIGINAL_SOURCES_DIGEST
    assert module._without_verifier_digest(fixed) == module.DAY15_FIXED_SOURCES_DIGEST
    assert module.verify_compatible_source_delta(original, fixed) == {
        "endpoint_fix_files": [],
        "shared_flux_fix_files": ["workflows/pyam21/energy.py"],
    }


def test_day15_restart_rejects_source_drift_and_wrong_origin():
    original = day15_source()
    fixed = dict(original)
    fixed[module.DAY15_ENERGY_PATH] = module.DAY15_FIXED_ENERGY_SHA
    fixed[module.VERIFIER] = "current verifier source SHA256"
    with pytest.raises(ValueError, match="audited pair"):
        module.verify_compatible_source_delta(
            original, {**fixed, "vfs/ocean/transport.py": "unreviewed"}
        )
    with pytest.raises(ValueError, match="audited pair"):
        module.verify_compatible_source_delta(
            {**original, "vfs/ocean/transport.py": "unreviewed"}, fixed
        )
    with pytest.raises(ValueError, match="pinned fixed"):
        module.verify_compatible_source_delta(
            original, {**fixed, module.DAY15_ENERGY_PATH: "unreviewed"}
        )


def vapor_sources():
    import hashlib
    root = Path(__file__).resolve().parents[1]
    original = json.loads((root / "tests/fixtures/coupled_vapor_sources-dcda5cd.json").read_text())
    fixed = dict(original)
    for path in (module.VERIFIER, "workflows/coupled/campaign.py",
                 "workflows/pyam21/adapter.py", "workflows/pyam21/state.py",
                 "workflows/pyam21/vapor.py"):
        fixed[path] = hashlib.sha256((root / path).read_bytes()).hexdigest()
    return original, fixed


def test_vapor_restart_pins_the_implemented_source_transition():
    before, after = vapor_sources()
    result = module.verify_compatible_source_delta(before, after)
    assert result["numerical_change"] == "funded_zero_vapor_v1"
    assert result["vapor_correction_files"] == sorted(
        path for path in before.keys() | after.keys() if before.get(path) != after.get(path)
    )
    assert module._without_verifier_digest(before) == module.VAPOR_ORIGINAL_SOURCES_DIGEST
    assert module._without_verifier_digest(after) == module.VAPOR_FIXED_SOURCES_DIGEST


@pytest.mark.parametrize("path", ["workflows/pyam21/vapor.py", "vfs/atmosphere/gibbs_means.py"])
def test_vapor_restart_rejects_unreviewed_sources(path):
    before, after = vapor_sources()
    with pytest.raises(ValueError, match="audited"):
        module.verify_compatible_source_delta(before, {**after, path: "unknown"})
    with pytest.raises(ValueError, match="audited"):
        module.verify_compatible_source_delta({**before, path: "unknown"}, after)
    with pytest.raises(ValueError, match="audited"):
        module.verify_compatible_source_delta(before, {k: v for k, v in after.items() if k != path})
