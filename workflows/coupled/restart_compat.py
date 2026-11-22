"""Audited one-way source transitions for explicit campaign restarts."""

import hashlib
import json

# Source entries replaced by the endpoint fix and its restart bridge. These
# are the values in the 6e68cb3 campaign manifest, before either change.
PRE_ENDPOINT_HASHES = {
    "workflows/coupled/campaign.py":
        "02c1c66a39e2ddd003f28ac338be3d2105418a1d8405c674521e2c7b80525d98",
    "workflows/coupled/restart_compat.py":
        "c7339dea3e6eac295de024ffae0f62fdd51fa02bfdfce29f210c0d66e9851611",
    "workflows/pyam21/adapter.py":
        "0e477172300daa543c72ce49aed2d8e2d55c3652341931a7dde34c22806bc447",
}
# Digest of the complete fixed source manifest apart from this verifier itself.
# Embedding the verifier's own SHA256 in this constant would be recursive.
ENDPOINT_TARGET_DIGEST = "252b123027243310ce854c287bdef3042ad82b8a852b3490d64c5e549bb5ad85"

# SHA256 entries from the full 2.25-day traced experiment's source manifest.
APPROVED_PRE_TRACE_HASHES = {
    "workflows/coupled/campaign.py":
        "e9d71ceda77d44c4e1a7a6ae24035a46f63c1331a06772c3c803e47c9e5a102e",
    "workflows/coupled/energy_trace.py":
        "5784b947e4969804277c14a3649058dbd3288237228d7541126af57c7af0350b",
    "workflows/coupled/ice_temperature.py":
        "daa35937ce4853716e4836038d68758694969fb2063d4c3d4000ec5778f0d9b8",
    "validation/sea_ice/snow.py":
        "2bbe85c0d70c266d019b653a11866dcb4ed397c89a9125414562fed648c4c045",
}

# This module itself was added after the old run. Its contents are a verifier,
# not an operator used by the integrator.
ALLOWED_NEW_FILES = {"workflows/coupled/restart_compat.py"}


def canonical_manifest_json(value):
    """Stable, strict JSON value equality across tuple/list round trips.

    The radiation constructor builds each box as a tuple; JSON reloads the
    same array as a list. Canonical serialization normalizes this harmless
    representation difference while retaining numeric values and all table
    SHA256s. Do not round floats, omit paths, or compare only selected fields.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def verify_instrumentation_only_delta(old_hashes, current_hashes):
    """Return the audited diff, failing closed for unknown source differences."""
    if not isinstance(old_hashes, dict) or not isinstance(current_hashes, dict):
        raise ValueError("source SHA256 manifests must be dictionaries")
    old_keys, new_keys = set(old_hashes), set(current_hashes)
    added = new_keys - old_keys
    removed = old_keys - new_keys
    if removed or added - ALLOWED_NEW_FILES:
        raise ValueError(
            "restart source inventory changed outside approved trace bridge: "
            f"added={sorted(added)} removed={sorted(removed)}"
        )
    changes = {
        path: (old_hashes[path], current_hashes[path])
        for path in old_keys & new_keys
        if old_hashes[path] != current_hashes[path]
    }
    unknown = set(changes) - set(APPROVED_PRE_TRACE_HASHES)
    if unknown:
        raise ValueError(
            "restart has non-instrumentation changes: " + ", ".join(sorted(unknown))
        )
    # Even if an approved path happens to be unchanged, it must be from the
    # same audited old version; otherwise a crafted mixed-version manifest
    # might pass only because no diff was observed.
    bad_known = sorted(
        path for path, expected in APPROVED_PRE_TRACE_HASHES.items()
        if old_hashes.get(path) != expected
    )
    if bad_known:
        raise ValueError("restart uses unknown pre-trace sources: " + ", ".join(bad_known))
    return {
        "modified_instrumentation_files": sorted(changes),
        "added_verifier_files": sorted(added),
    }


def verify_compatible_source_delta(old_hashes, current_hashes):
    """Check the endpoint transition, optionally preceded by the trace bridge."""
    verifier = "workflows/coupled/restart_compat.py"
    if not isinstance(old_hashes, dict) or not isinstance(current_hashes, dict):
        raise ValueError("source SHA256 manifests must be dictionaries")
    if old_hashes == current_hashes:
        return {"endpoint_fix_files": []}
    if verifier not in current_hashes:
        raise ValueError("missing compatible restart verifier in current source manifest")
    target = {k: v for k, v in current_hashes.items() if k != verifier}
    digest = hashlib.sha256(canonical_manifest_json(target).encode()).hexdigest()
    if digest != ENDPOINT_TARGET_DIGEST:
        raise ValueError("current sources differ from the audited endpoint revision")

    # Undo only the known source transition on paper. All other files must
    # then match the previous manifest, unless the old trace bridge checks out.
    previous = {**current_hashes, **PRE_ENDPOINT_HASHES}
    trace = {} if previous == old_hashes else verify_instrumentation_only_delta(
        old_hashes, previous
    )
    return {"endpoint_fix_files": sorted(PRE_ENDPOINT_HASHES), **trace}
