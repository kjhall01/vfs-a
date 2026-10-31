# Separate output metadata correction

The parent structural stack retained an ocean output label of K. The recovered
ocean temperature and EOS convention is degrees Celsius; no numerical conversion
was performed by that output path. This separate commit changes only the ocean
lat/lon temperature Field's units label to `degC` and checks it in the existing
snapshot/regrid test. Field values and all model equations remain unchanged.

This touches the ocean-owned output adapter introduced by PR #22. It does not
change shared remapping or atmospheric output. The energy experiment source
hashes retain the exact pre-correction files used to generate those measurements
(local numerical commit `25c57ce`, published with identical tree as
`2d1ca581fd701468ba0a1cb7f5550c967c09d4ea`); that driver never calls the output
adapter. See [publication provenance](PR_DRAFTS.md).

The adapter's density field still uses the legacy linear-EOS state helper.
For the selected dynamics' authoritative density, use `model.physical_fields`.
A general model-aware output contract is deferred instead of silently replacing
the inherited state-output behavior in this metadata correction.
