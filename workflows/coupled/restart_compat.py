"""Conservative one-way restart bridge for the audited trace-only revision.

This bridge accepts ONLY the observed November 21, 2026 traced run before
internal ocean/ice stage callbacks were added. It does not establish arbitrary
cross-version compatibility. The old source hashes are pinned from that run's
manifest, and all non-trace files must match byte-for-byte by SHA-256.
"""

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
    import json
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
    mismatched_baseline = sorted(
        path for path in changes
        if old_hashes[path] != APPROVED_PRE_TRACE_HASHES[path]
    )
    if mismatched_baseline:
        raise ValueError(
            "restart trace source hashes do not match the audited 2.25-day run: "
            + ", ".join(mismatched_baseline)
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
