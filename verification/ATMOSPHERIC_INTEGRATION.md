# Atmospheric stack integration and reference-policy review

Starting main: **6c5bf938503b357ba01c50ab99ae7e2cc770e90a** (#25 merge).
GitHub status and local ancestry were both checked. #25 head
0bedd0212dd41a846e16c8d9a1e1356e5c6382ea is an ancestor of main.
#23 merged into the reconciliation branch *after* #25 merged; neither its
head/merge (54783444f7ce85074cabe0c97b8e738dda81d4a3 /
0a9de626d6ba7bb1fc9ac27ab4a8ed0a9ffcbe5b), nor #24 head
8936334f027e5660987a15daa000685746ff5b47, is an ancestor of starting main.
Main has only `ocean/__init__.py` and `ocean/regrid.py`. No ocean development
branch or worktree is changed by this atmospheric integration.

Original #18 head: **93b54b3630ba7d17afac87e38e84b8b0080b0d50**.
Original #19 head: **24594d55e32d487987fed22c5af0f39b0c9a0426**; numerical
commit **a7facc1bcdb88f7dc9c10f4fb6059f29b0a9bac7**. Integration uses merges,
not history rewriting. These original commits and all committed experiment
artifacts stay reachable and unchanged. Merge order remains #18 then #19;
both stay unmerged for review.

## Explicit structural provenance

Scientific baseline **8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648** and approved
interface extraction **ae132773e621671f70bfa6a698eebb1a82cffa94** stay identifiable.
`tools/runtime_relocations_v1.json` maps every runtime path in the exact #25
merge tree to explicit pre-structure source paths, with hashes on both sides.
The mapping follows the reviewed ownership table and 457-definition AST audit.
It includes compatibility modules and the pre-existing ocean output extraction,
not ocean physics. No file is admitted merely because its import path changed.

Use `--structural-revision 6c5bf938503b357ba01c50ab99ae7e2cc770e90a` to opt in.
Without it the historical source policy is unchanged. The map's hash and commit
are pinned in `structural_sources.py`; the verifier checks complete file coverage,
source/origin hashes, exact installed bytes, and absence of extra Python modules.
Unknown revisions and tampered maps/files fail. Run manifests archive the map,
verifier and source hashes separately from the historical baseline.

No scientific acceptance thresholds, widened regression tolerances or expanded
campaigns are introduced. Current integration/reference validation results will
be recorded here separately for the investigation and corrected implementation.

## Integrated investigation validation

On macOS ARM64, Python 3.12.13, PyTorch 2.15.0, CPU float64 (Accelerate),
the installed wheel passes **577 tests, one accelerator skip**. All **419**
captured tensors and scalar/solver records agree exactly with original #18
in the same environment: states, tendencies, energies, bounded ledgers, limiter
records, diagnostic and full-step gradients, dry/moist ARK4 stages and solves.
Both portable comparison tools and all six documented explicit/IMEX/forcing
example commands pass. The small 16-cell/120-step constant-composition C5
case and face investigation complete with the explicit structural source map.
These are local checks; GitHub Actions results are reported separately.
The runtime tree is byte-for-byte main; this integration adds no numerical fix.
