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


# Coupled Redi energy funding changes only trials that previously failed.
REDI_ENERGY_TARGET = "a96c4e0e5f7258897b6d81fc3a9165a857d3eebbe66edf5558902438566130d9"
REDI_PARENT_VERIFIER = "ad3e8c4bf2d7fa6e1857ecc205c37ae06bb7321830a3ab262c7c8c91dec9f8de"


COUPLED_LIMITERS_TARGET = "61eb298b44bcbb2eecfaf94a500f6e519f6b3403be6e9775b34e6cf9af32d022"
REDI_VERIFIER = "a4cc034ea7a6f763a3537c1920e0ba39268e2b779a0cf72f7f7aa8d7038d0c7d"


NATIVE_VAPOR_TARGET = "0bb330a01200f82c35e58321bd2bdf54fa42d89f2e82f935b714e54e0c01e120"
COUPLED_LIMITERS_VERIFIER = "ad17d1544336c7ed052c32ffa6e044af3d850c98314d174ce2fd2ecdf3ec2952"


# Extending the campaign horizon changes only campaign.py and this verifier.
EXTENDED_HORIZON_TARGET = "8360f64bae59925327942036f22d33495fe5d1df3fa2f749e366ddec27482daa"
NATIVE_VAPOR_VERIFIER = "2ba588ac1ac86ca9b5df87aac5b09ca9cb4d4eebb6cf27f97cd398ebfde87dc8"


ATMOSPHERE_SSPRK43_TARGET = "3f5001cc3d9a5c4da76c12458ae19a62d536cec2a4c21399e7b29c82e421a33f"
EXTENDED_HORIZON_VERIFIER = "f0c46c0fd0e7cba19d18b225ae21757ae4b334bb82cbfba4fe694542db7f49cc"


# One-way bridge from the original SSPRK(4,3) campaign. Only the adapter
# and this verifier changed, and the checkpoint state layout is identical.
CLOUD_ROUNDOFF_TARGET = "92b94ebf0c13820e666624855fbde20b3257d01d96708bb4cd34f9d7cc21683f"
PRE_CLOUD_ADAPTER_SHA = "04f1cb0579116bb4710ca188812294da7acaae74d4fb33e7cac683e121d703f5"
PRE_CLOUD_VERIFIER_SHA = "69f304aaf1706f0cf538a3ce0d0bcc700e6133867da8d40830de4033fdf9a7d7"


# The main-branch checkpoint uses the historical shelf metric. The campaign
# verifies that the saved and requested metric settings agree before this check.
CLOSED_SHELF_TARGET = "10414cfe5cee8dbc89af83cab14f47568c88d7382eb1cf9cccc9d63be6d117e6"
PRE_SHELF_HASHES = {
    "vfs/ice/ocean_ice.py": "ac9a81a713ef90bba5b87bb57b7e247a084fdcdd402d535bc7a9831df40633ab",
    "vfs/ocean/energy.py": "3f93240575d825cf392874a9a397f92acdfe4390326cee91accbd4ca5c4f0adf",
    "vfs/ocean/mixing/lagged.py": "0013da6598ff19f89199241d82df17ded7a1fe1a5e83af39798c10f116d7a9e4",
    "vfs/ocean/stratified/energy_columns.py": "a493ef5aa96107004465ae5eaae9e3a8f70aae343b79de9ad9763b570ca475ef",
    "vfs/ocean/temperature/ale.py": "4e65a97ddbaee349c3cffe0797e4751f2bb97a7e05b1420ddda13568b9eb83e3",
    "vfs/ocean/temperature/backscatter.py": "d16857f28dc506f17bc585c9650d03550d52203e63dcafb4b2777fcc1b38fb46",
    "vfs/ocean/temperature/model.py": "0f3bfbc51594268d64cf2e376f64c14ca35310b43ea1268f5996a1b0ca7fcd5b",
    "vfs/ocean/temperature/shelf_transport.py": "2f3f2b7b6385c0e6dc40d33b1b912981224c9ee9e6c38d00f7720b0dfbcb3392",
    "workflows/coupled/campaign.py": "56c0c7b69030f1b4ab6c7918e6fb0bd187993a4b108cd85450925e538040ddf0",
    "workflows/coupled/restart_compat.py": "9698e118340437353a2dd544867996de4b4f073a0f24db73a10660f27459aa11",
}


# Source transition for selecting the drag law in coupled runs. Older
# manifests are compared using their original midpoint setting.
NATIVE_EDGE_DRAG_TARGET = "55e845087c9a05b11052c9dfd80e52790f02c22d515c201acdca589f80c2db65"
PRE_DRAG_HASHES = {
    "vfs/ice/snow.py": "cc04ab01fe1fe66da31e454b91a5bccb90ac4ffe4e18df89323002b8d025e1d7",
    "validation/shelf_momentum/operators.py": "96d6b36b923365d200b077a6e19671fc73665befd0aae02108b3826ba0a96acc",
    "vfs/ice/ice_drag.py": "f528f9a7ccf220de3ef5e259aeebcc0a79d105dbea43d439c8a91f6af7e4e146",
    "vfs/ice/ocean_ice.py": "c23c3388be4610550ef1f8d80dc94d78e818bd812736d1ab2b84ffdf0774a43c",
    "workflows/coupled/campaign.py": "06e22517e911914389554df9a227ab086ece458b3614c3966460c99101a6b2da",
    "workflows/coupled/restart_compat.py": "255aeb62e73c2f4ca7d1cac2e5da3bef1381b4cffe2a74842ac40423a1740b01",
}


# Only the explicit --resume-compatible-manifest path uses this historical
# bridge. Ordinary development restarts validate the grid stored in NetCDF.
CAMPAIGN_CLI_TARGET = "18b9c70f0719045f58d4dde6408e708c84e55e167d5f7496516c4bfe2c62b0ca"
PRE_CLI_HASHES = {
    "workflows/coupled/campaign.py": "e4947ff455544edb126089c12d60dfc53da1c04db0b7d089f84613d21f1088f4",
    "workflows/coupled/restart_compat.py": "16cb8e5a3b0952274a6ad85a7f99c6487d030861798ea085b99f9fd1a6345a9b",
}
SURFACE_OUTPUT_TARGET = "becbb1963c0c5a7366b60b2380be2d7d7355ffd2d1affaa1bbe06d175d798f22"
PRE_OUTPUT_HASHES = {
    "workflows/coupled/campaign.py": "425885aebe969a21adcf9ae9d8aae380c57b91b61e119141086b95cd2381b299",
    "workflows/coupled/grid_io.py": "905b78c65a8b130aba4883c0bee3bf67bfbde040f8076b7f018905ffaf9b1cbc",
    "workflows/coupled/state_io.py": "ab2c568a7ac1f369ef1a77d7e6c28f29f00b64a5d342aa3f5634f27f43619462",
    "workflows/pyam21/adapter.py": "0b0f63cdb0f1288e5857ac91af433da1c2155d6632f4f14c7a196100afdcdfb0",
    "workflows/coupled/restart_compat.py": "94bd671c03971c2cf062d76ce6cd524edcc6e778a0abf6dea0290495baab1dd0",
}


def pre_output_sources(current):
    parent = current.copy()
    for name in ("workflows/coupled/continuation.py", "workflows/coupled/surface_output.py"):
        parent.pop(name, None)
    return {**parent, **PRE_OUTPUT_HASHES}


def pre_shelf_sources(current):
    """Reconstruct the parent source inventory for its existing audited bridges."""
    parent = {key: value for key, value in current.items()
              if not key.startswith("validation/shelf_momentum/")
              and key != "vfs/ocean/temperature/shelf_momentum.py"}
    parent.update(PRE_SHELF_HASHES)
    return parent


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
    target = _without_verifier_digest(current_hashes)
    if target == CAMPAIGN_CLI_TARGET:
        parent = {**current_hashes, **PRE_CLI_HASHES}
        if _without_verifier_digest(parent) != SURFACE_OUTPUT_TARGET:
            raise ValueError("campaign CLI restart parent differs from the audited revision")
        return verify_compatible_source_delta(old_hashes, parent)
    if target == SURFACE_OUTPUT_TARGET:
        parent = pre_output_sources(current_hashes)
        if _without_verifier_digest(parent) != NATIVE_EDGE_DRAG_TARGET:
            raise ValueError("surface output restart parent differs from the audited revision")
        return verify_compatible_source_delta(old_hashes, parent)
    if target == NATIVE_EDGE_DRAG_TARGET:
        parent = {**current_hashes, **PRE_DRAG_HASHES}
        if _without_verifier_digest(parent) != CLOSED_SHELF_TARGET:
            raise ValueError("ice drag restart parent differs from the audited revision")
        prior = verify_compatible_source_delta(old_hashes, parent)
        return {**prior, "ice_ocean_drag_configurable": True}
    if target == CLOSED_SHELF_TARGET:
        parent = pre_shelf_sources(current_hashes)
        if _without_verifier_digest(parent) != CLOUD_ROUNDOFF_TARGET:
            raise ValueError('shelf restart parent differs from the audited revision')
        prior = verify_compatible_source_delta(old_hashes, parent)
        return {**prior, 'shelf_momentum_configurable': True}
    if target == CLOUD_ROUNDOFF_TARGET:
        parent = {**current_hashes, "workflows/pyam21/adapter.py": PRE_CLOUD_ADAPTER_SHA,
                  verifier: PRE_CLOUD_VERIFIER_SHA}
        prior = verify_compatible_source_delta(old_hashes, parent)
        prior["cloud_area_boundary_roundoff"] = True
        if prior.get("numerical_change") in (None, "none"):
            prior["numerical_change"] = "pyam21_cloud_area_boundary_roundoff"
        return prior
    if target in (NATIVE_VAPOR_TARGET, EXTENDED_HORIZON_TARGET, ATMOSPHERE_SSPRK43_TARGET):
        origin = (_without_verifier_digest(old_hashes), old_hashes.get(verifier))
        if target == ATMOSPHERE_SSPRK43_TARGET and origin == (EXTENDED_HORIZON_TARGET, EXTENDED_HORIZON_VERIFIER):
            return {"atmosphere_integrator_configurable": True}
        if target in (EXTENDED_HORIZON_TARGET, ATMOSPHERE_SSPRK43_TARGET) and origin == (NATIVE_VAPOR_TARGET, NATIVE_VAPOR_VERIFIER):
            return {"campaign_horizon_update": True, "numerical_change": "none"}
        if origin in {(COUPLED_LIMITERS_TARGET, COUPLED_LIMITERS_VERIFIER),
                      (REDI_ENERGY_TARGET, REDI_VERIFIER), (COLUMN_VAPOR_TARGET, REDI_PARENT_VERIFIER)}:
            return {"numerical_change": "coupled_reservoir_limiters_and_native_interface_vapor_v1"}
        raise ValueError("native vapor restart origin differs from the audited revisions")
    if _without_verifier_digest(current_hashes) == COUPLED_LIMITERS_TARGET:
        origin = (_without_verifier_digest(old_hashes), old_hashes.get(verifier))
        if origin == (REDI_ENERGY_TARGET, REDI_VERIFIER):
            return {"numerical_change": "coupled_reservoir_limiters_v1"}
        if origin == (COLUMN_VAPOR_TARGET, REDI_PARENT_VERIFIER):
            return {"numerical_change": "locally_funded_redi_and_coupled_reservoir_limiters_v1"}
        raise ValueError("coupled limiter restart origin differs from the audited revisions")
    if _without_verifier_digest(current_hashes) == REDI_ENERGY_TARGET:
        origin = (_without_verifier_digest(old_hashes), old_hashes.get(verifier))
        if origin == (COLUMN_VAPOR_TARGET, REDI_PARENT_VERIFIER):
            return {"numerical_change": "locally_funded_redi_v1"}
        raise ValueError("Redi energy restart origin differs from the audited revision")
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
