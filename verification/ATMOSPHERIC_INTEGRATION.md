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

## Corrected regression policy and integration

Integrated investigation commit: `6e2dcfd` (original #18 remains its ancestor).
Integrated corrected runtime: **d972f55683bf40b4edf35d1f5193949ff420cb0d**
(original #19 remains its ancestor). The sole runtime delta between them is
`vfs/atmosphere/gibbs_means.py`, byte-identical to the original numerical commit.
The source verifier admits that one exact pinned change only when both numerical
and structural revisions are selected; all other files retain strict mapping.

`tests/fixtures/corrected-gibbs-v1/` is a separately identified regression version,
not a regeneration of checkpoint-54. Its provenance records the exact runtime
commit, all runtime hashes, CPU/backend/software environment, configuration,
initial tensors and geometry, archived capture scripts, and artifact hashes.
`tools/regression_references.py` pins its provenance hash and verifies the fixture
and historical hash. The old fixture, dry fixture, quantity mapping and tolerance
constants are unchanged. Unknown versions and corrupted fixtures are rejected.

Current pytest and CI explicitly use this corrected reference; the standalone
baseline tool retains its historical default so historical commands keep their
meaning. Current commands, outside the source directory with the wheel installed:

```sh
python -I -m pytest tests --fixture-mode=portable -q
python -I tools/check_baseline.py --reference-version corrected-gibbs-v1 --mode portable
python -I tools/check_dry_helpers.py --mode portable
# Historical evidence: intentional disagreement, still exits nonzero.
python -I tools/check_baseline.py --reference-version historical-v1 --mode portable
# Exact is a distinct mode, never silently converted to portable success.
python -I tools/check_baseline.py --reference-version corrected-gibbs-v1 --mode exact
```

The quantity-by-quantity comparison is
[`quantity_comparison.json`](../tests/fixtures/corrected-gibbs-v1/quantity_comparison.json):
**49 of 142 tensors change** between old and corrected operators in this same
environment; **93 are exact**. Changes comprise 7 RHS, 6 Euler, 9 final-state,
12 limiter/ledger, 12 final-budget and 3 energy-rate tensors. Initial states,
initial energy gradients/budgets, all dry outputs, and CFL values are unchanged.
The change in candidate flux requires matched momentum/entropy work, then
propagates through Euler and SSPRK3. Final budget diagnostics evaluate that changed
state. Energy-rate differences reflect cancellation, independently checked below.
No unexplained output was admitted. Per-species errors remain separate in the
comparison. Scalar and discrete records are also listed, rather than omitted.
The original corrected 142-tensor capture agrees exactly with the integrated
capture; this isolates integration from those intentional operator differences.

To reproduce reference creation, build/install wheels from original #18
`93b54b3630ba7d17afac87e38e84b8b0080b0d50`, original #19
`24594d55e32d487987fed22c5af0f39b0c9a0426`, and integrated runtime `d972f556...`
in that order in the **same** environment. Copy tests/tools/examples outside all
checkouts. For each original wheel, save `check_baseline.capture(check_baseline.imports())`
with `torch.save` as `old.pt` and `original-corrected.pt` respectively (load the tool
by `importlib.util.spec_from_file_location`). Install the integrated wheel and run:

```sh
python -I tools/build_corrected_reference.py --repo /path/to/repo \
  --old-capture old.pt --original-corrected-capture original-corrected.pt \
  --output /new/nonexistent/reference-directory
```

The tool refuses to overwrite an existing directory, requires exact runtime
source hashes, checks old versus historical using unchanged portable tolerances,
requires exact integrated versus original-corrected results, and rejects changes
outside the explicitly justified output groups. Both same-environment input
captures are archived in the version directory. Fixture-file serialization hashes
are provenance identifiers, not claims of cross-platform bitwise reproducibility.
Historical reproduction commands and original reports remain available unchanged.

## Local corrected results and limits

macOS ARM64 CPU float64, Python 3.12.13, PyTorch 2.15.0/Accelerate:
**585 passed, one CUDA skip** in the installed-wheel portable suite. The 235
quantity corruption tests now compare against stored versioned references,
not a copy of the current result. Independent shape, finiteness, positivity,
conservation, limiter, exact-ledger and energy checks remain unchanged. Additional
version/hash-corruption checks guard reference selection. Both portable tools
pass; corrected moist exact passes locally. Historical moist portable fails as
expected and retains its failure report. All six supported example commands pass.

Both old and corrected branches match their own original **419** tensors exactly,
including states, tendencies, energy terms, ledgers, limiter diagnostics,
diagnostic/RHS gradients, smooth/active full-step gradients, and dry/moist ARK4
stage and solver records. This does not qualify IMEX backward or long rollouts.

The repeated 116-face checks give maximum absolute Gibbs residual
**1.834450333503908e-10 J/kg**, cancellation-normalized **1.5036986470991033e-14**.
Actual Euler/SSPRK3 tests include zero condensates, one-sided support, unequal
pressure/temperature, nearly equal states and active limiting. C2–C4 and the
zero-cloud one-sided derivative probe are archived under
[data/atmospheric-integration](data/atmospheric-integration). These retain measured
residuals rather than imposing new scientific acceptance thresholds.

One inexpensive paired C5 case (16 cells, 120 steps, NL2010 Case 4, constant
composition) confirms the integrated harness/source manifests: final drift is
**5.422871183259799e-6** old versus **5.360295540768334e-16** corrected. Full
inventories, errors and bounds are archived. This is not a new 22-case campaign;
the original 22 paired results remain unchanged. Nonmonotonic deformational
errors remain unresolved; constant-composition correction does not establish
transport convergence or coupled energy conservation.

Merge order: **#18, then #19**, both left open for review. Historical scientific
baseline remains `8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648`; `d972f556...` is the
proposed corrected runtime, not an automatically adopted scientific baseline.
GitHub Actions results will be reported separately from these local checks.
