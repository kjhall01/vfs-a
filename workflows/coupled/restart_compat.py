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
# Replacing the historical trace verifier with its target-pinned version
# changed only this verifier on #93. Both prior versions are identifiable.
PINNED_TRACE_VERIFIER = "48547e17f1b7044153a8d971387d1951dd443c32b5afb69c576f9353b3b34241"

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

# Complete 6e68cb3 numerical source inventory. Exclude this verifier's own
# bytes to avoid a self-referential digest.
VERIFIER = "workflows/coupled/restart_compat.py"
TRACE_TARGET_DIGEST = "45db058e852bcce15dce1f5f562f1848817b71422a78acd54a1651da27453b46"

# Exact source transition for the 2026-11-23 4096-cell campaign checkpoint
# (fingerprint 8f6e7bfd...). Both digests cover ALL 258 Python source paths
# other than this verifier. The old values come from the uploaded, checked
# campaign manifest; the fixed energy hash is the reviewed source in this PR.
# Neither digest includes this verifier to avoid a recursive source hash.
DAY15_ORIGINAL_VERIFIER_SHA = "8835ab1a51c5748399c839508160df8da31ca6947b26304455fe154b9901b76a"
DAY15_ENERGY_PATH = "workflows/pyam21/energy.py"
DAY15_ORIGINAL_ENERGY_SHA = "00b0a55d6d0679e32d338c09423770af95922a6f27b411f97d01d2e6c71820e9"
DAY15_FIXED_ENERGY_SHA = "5535ef4e12ecb4a1155d2c780dd9bda5f8045db522a6247ec7290951a69748b9"
DAY15_ORIGINAL_SOURCES_DIGEST = "25ca65f87f86b7f419fd2aea20d5074b90fb363bd78c8878216769c5d259e56b"
DAY15_FIXED_SOURCES_DIGEST = "8e39d515effc25df2e8fc2cc4b6e9a3c214720963b9d8141ba750fcad835a405"

# Exact numerical transition from main dcda5cd. State layout, configuration
# and upstream PyAM21 are unchanged. Inventories are reproducible from Git.
# Only the host vapor correction and its reporting are introduced. Both full
# source inventories are pinned, excluding this verifier's recursive hash.
VAPOR_ORIGINAL_VERIFIER_SHA = "5a39bd00299ac9bb631bb82282d98f1011ae0e86ac56b7af6bb7e9d2b819dbf2"
VAPOR_ORIGINAL_SOURCES_DIGEST = "60c3cce216b15d9a2e983e2d6b4345ced921f95bd6fb6b924a776b12664844e9"
VAPOR_FIXED_SOURCES_DIGEST = "a0861e8909afe2d32309ec196b19f2d883eb5b599f5571967c2918c90f84b767"
# PR #113 already contains the ice, GM and mixing changes on both sides
# of its vapor transition. Pair the inventories so a restart cannot skip
# any of those numerical changes. The original verifier is shared.
VAPOR_TARGET_BY_ORIGIN = {
    VAPOR_ORIGINAL_SOURCES_DIGEST: VAPOR_FIXED_SOURCES_DIGEST,
    "1762a52b61ae4f11d4f8a25b2073af4d8224179c25bcd53bf76c958ccb31b05c":
        "f7e7c56cafb64857884f2572f884e853902ce2bd548a647344a150bad49d0860",
}


# Lossless storage transition from d342be9. All numerical sources are pinned.
IO_ORIGINAL_VERIFIER = "2233c54f59ec51d6006a075a92e1dad9f23bfcb684a8eceda7ec7efcfc59734c"
IO_ORIGINAL_SOURCES = "fe39c1ebc41181f1ab8bcd58d9a5fbc76d6d54e5c0e06207fd645ff621038312"
IO_NETCDF_SOURCES = "5d8e9954b0b7b8dafd5d7a4b846ac68bb5f2e1dcc1fd07ac6b2ae03bd3b577fb"


# Package reorganization from 64b56ef. Hash every source except this verifier.
CLEANUP_ORIGIN = "5d8e9954b0b7b8dafd5d7a4b846ac68bb5f2e1dcc1fd07ac6b2ae03bd3b577fb"
CLEANUP_ORIGINAL_VERIFIER = "1542d7834224de0041d83339ac9e881836c90492fafd921b5968ca82635e26ca"
CLEANUP_TARGET = "996c37939a2c848e11a732faa54203fdf387bad8515511c97ca964ad1fd5e322"


# Both permitted origins have the same campaign physics before vapor repair.
COLUMN_VAPOR_TARGET = "79486890e847cdf214472eb6f8ac954735d206c0ea378de19f0932152070088e"
COLUMN_VAPOR_ORIGINS = {
    (CLEANUP_ORIGIN, CLEANUP_ORIGINAL_VERIFIER),
    (CLEANUP_TARGET, "e6b31c07fe75bd37a6ec00b7b937982a5c350b99bad95d71a1301291ebe96468"),
}


# The direct Roquet model preserves the NetCDF payload and numerical operators.
# Pin the experimental source tree exactly, including the parent verifier.
DIRECT_OCEAN_TARGET = "6159f0a4cbe73624a514736b8b5706023c2e5bc41d6e5c449da7d63c6026c9c2"
DIRECT_OCEAN_PARENT_VERIFIER = "ad3e8c4bf2d7fa6e1857ecc205c37ae06bb7321830a3ab262c7c8c91dec9f8de"


def _without_verifier_digest(sources):
    other = {path: sha for path, sha in sources.items() if path != VERIFIER}
    return hashlib.sha256(canonical_manifest_json(other).encode()).hexdigest()


def _verified_vapor_transition(old_hashes, current_hashes):
    if old_hashes.get(VERIFIER) != VAPOR_ORIGINAL_VERIFIER_SHA:
        return None
    target = VAPOR_TARGET_BY_ORIGIN.get(_without_verifier_digest(old_hashes))
    if target is None:
        return None
    if _without_verifier_digest(current_hashes) != target:
        raise ValueError("vapor correction restart target differs from the audited revision")
    changed = sorted(k for k in old_hashes.keys() | current_hashes.keys()
                     if old_hashes.get(k) != current_hashes.get(k))
    return {"vapor_correction_files": changed,
            "numerical_change": "funded_zero_vapor_v1"}


def _verified_day15_shared_flux_transition(old_hashes, current_hashes):
    """Allow only the pinned exchange-check correction on one verified run.

    An unknown path, differing source, or changed numerical configuration is
    never accepted by this bridge. The caller additionally checks the full
    campaign fingerprint, mesh, radiation provenance and execution metadata.
    """
    if (old_hashes.get(VERIFIER) != DAY15_ORIGINAL_VERIFIER_SHA
            or old_hashes.get(DAY15_ENERGY_PATH) != DAY15_ORIGINAL_ENERGY_SHA):
        return None
    if current_hashes.get(DAY15_ENERGY_PATH) != DAY15_FIXED_ENERGY_SHA:
        raise ValueError("day15 shared-flux restart requires the pinned fixed energy.py")
    if (set(old_hashes) != set(current_hashes)
            or _without_verifier_digest(old_hashes) != DAY15_ORIGINAL_SOURCES_DIGEST
            or _without_verifier_digest(current_hashes) != DAY15_FIXED_SOURCES_DIGEST
            or {k for k in old_hashes if old_hashes[k] != current_hashes[k]}
               != {VERIFIER, DAY15_ENERGY_PATH}):
        raise ValueError("day15 shared-flux restart sources differ from the audited pair")
    return {"endpoint_fix_files": [], "shared_flux_fix_files": [DAY15_ENERGY_PATH]}


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
    if old_keys - new_keys or added != {VERIFIER}:
        raise ValueError("restart source inventory changed outside approved trace bridge")
    changes = {path for path in old_keys if old_hashes[path] != current_hashes[path]}
    unknown = changes - set(APPROVED_PRE_TRACE_HASHES)
    if unknown:
        raise ValueError("restart has non-instrumentation changes: " + ", ".join(sorted(unknown)))
    bad_old = [path for path, digest in APPROVED_PRE_TRACE_HASHES.items()
               if old_hashes.get(path) != digest]
    if bad_old:
        raise ValueError("restart uses unknown pre-trace sources: " + ", ".join(sorted(bad_old)))
    target = {path: digest for path, digest in current_hashes.items() if path != VERIFIER}
    if hashlib.sha256(canonical_manifest_json(target).encode()).hexdigest() != TRACE_TARGET_DIGEST:
        raise ValueError("restart target differs from the audited trace revision")
    return {
        "modified_instrumentation_files": sorted(changes),
        "added_verifier_files": sorted(added),
    }


def verify_compatible_source_delta(old_hashes, current_hashes):
    """Check the endpoint transition, optionally preceded by the trace bridge."""
    verifier = VERIFIER
    if not isinstance(old_hashes, dict) or not isinstance(current_hashes, dict):
        raise ValueError("source SHA256 manifests must be dictionaries")
    if old_hashes == current_hashes:
        return {"endpoint_fix_files": []}
    if verifier not in current_hashes:
        raise ValueError("missing compatible restart verifier in current source manifest")
    if _without_verifier_digest(current_hashes) == DIRECT_OCEAN_TARGET:
        origin = (_without_verifier_digest(old_hashes), old_hashes.get(verifier))
        if origin == (COLUMN_VAPOR_TARGET, DIRECT_OCEAN_PARENT_VERIFIER):
            return {"direct_roquet_ocean": True, "numerical_change": "none"}
        if origin in COLUMN_VAPOR_ORIGINS:
            return {"direct_roquet_ocean": True, "numerical_change": "same_column_vapor_redistribution"}
        raise ValueError("ocean refactor restart origin differs from the audited revisions")
    if _without_verifier_digest(current_hashes) == COLUMN_VAPOR_TARGET:
        if (_without_verifier_digest(old_hashes), old_hashes.get(verifier)) in COLUMN_VAPOR_ORIGINS:
            return {"numerical_change": "same_column_vapor_redistribution"}
        raise ValueError("column vapor restart origin differs from the audited revisions")
    if _without_verifier_digest(current_hashes) == CLEANUP_TARGET:
        origin = (_without_verifier_digest(old_hashes), old_hashes.get(verifier))
        if origin == (CLEANUP_ORIGIN, CLEANUP_ORIGINAL_VERIFIER):
            return {"package_cleanup": True, "numerical_change": "none"}
        if origin == (IO_ORIGINAL_SOURCES, IO_ORIGINAL_VERIFIER) and CLEANUP_ORIGIN == IO_NETCDF_SOURCES:
            return {"io_format_change": "coupled_netcdf_v1", "package_cleanup": True,
                    "numerical_change": "none"}
    if (old_hashes.get(verifier) == IO_ORIGINAL_VERIFIER
            and _without_verifier_digest(old_hashes) == IO_ORIGINAL_SOURCES):
        if _without_verifier_digest(current_hashes) != IO_NETCDF_SOURCES:
            raise ValueError("NetCDF restart target differs from the audited revision")
        return {"io_format_change": "coupled_netcdf_v1", "numerical_change": "none"}
    vapor = _verified_vapor_transition(old_hashes, current_hashes)
    if vapor is not None:
        return vapor
    roundoff = _verified_day15_shared_flux_transition(old_hashes, current_hashes)
    if roundoff is not None:
        return roundoff
    target = {k: v for k, v in current_hashes.items() if k != verifier}
    digest = hashlib.sha256(canonical_manifest_json(target).encode()).hexdigest()
    if digest != ENDPOINT_TARGET_DIGEST:
        raise ValueError("current sources differ from the audited endpoint revision")

    # Undo only the known source transition on paper. All other files must
    # then match the previous manifest, unless the old trace bridge checks out.
    previous = {**current_hashes, **PRE_ENDPOINT_HASHES}
    if old_hashes.get(verifier) == PINNED_TRACE_VERIFIER:
        previous[verifier] = PINNED_TRACE_VERIFIER
    trace = {} if previous == old_hashes else verify_instrumentation_only_delta(
        old_hashes, previous
    )
    return {"endpoint_fix_files": sorted(PRE_ENDPOINT_HASHES), **trace}
