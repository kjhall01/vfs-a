# Ocean integration onto corrected atmospheric main

> Status reconciled against main `e959b7005832a5041e082b6e300838e15bfbaa9e`:
> #26 and #27 are merged and their heads are ancestors of main. Draft/unmerged
> language below describes the original publication, not current status.
> The reference solver remains default; the constant-density candidate remains
> explicit. Current investigation: [rotating-flow diagnosis](ROTATING_BALANCE.md).

Starting main: `4ec610ed25c818e60863a8f8b772107b33f9251c` (#18/#19/#25).
Ocean history is merged from `0a9de626d6ba7bb1fc9ac27ab4a8ed0a9ffcbe5b`:
#24 had merged into the ocean extraction branch, then #23 into the structural
reconciliation branch after that branch's earlier merge into main. Neither
ocean implementation had reached main. This merge retains all those commits.
The only textual merge conflict was the handoff; both histories are retained
beneath a current summary. No other development branch was changed.

## Runtime and verification ownership

Every non-ocean runtime file is byte-identical to starting main. Every ocean
runtime file is byte-identical to reviewed #24 head
`8936334f027e5660987a15daa000685746ff5b47`, including model-aware density output.
`integration_results/results.json` lists the exact source commit and SHA-256
for each runtime file. No operator, default, state variable, fixture, regression
tolerance, or scientific baseline is changed. The archived ocean remains the
default; `ConstantDensityDynamics` remains an explicit constant-density option.

The shared **verification infrastructure**, but no shared runtime, needs an
explicit source-provenance extension. The existing atmospheric structural map
rejects extra Python modules and the two changed ocean adapter files. New
`--ocean-revision 8936334f027e5660987a15daa000685746ff5b47` admits only the exact
`vfs/ocean` tree at that commit, alongside the existing structural/numerical
pins. It does not admit atmospheric or shared files from the ocean branch.
Default commands retain their prior strict behavior. Manifests record the ocean
commit and every installed source hash. Six corruption/selection cases cover
unknown revisions, missing/changed/extra ocean files and changed atmospheric code.
This is the explicit coordination boundary with atmospheric development.

CI retains the corrected atmospheric reference and its original-to-integrated
comparisons. It adds a separate wheel from the reviewed ocean head and compares
both ocean equation sets, global/coastal states, fields, tendencies, terms,
budgets, steps and step gradients. Existing source checks still run with all
three explicit pins; there is no wildcard or skipped atmospheric gate.

## Local installed-wheel evidence

CPU float64, PyTorch 2.15.0+cpu, one numerical thread. Tests and tools were copied
outside all source trees and the wheel installed into an isolated task venv.
This reuses the available dependency environment, not a fresh dependency solve.

| Check | Result |
| --- | --- |
| Full portable pytest | 612 passed, 1 CUDA skip, 46.12 s |
| Starting-main atmosphere capture | 419 tensors and accompanying records exactly equal |
| Starting-main output capture | 74 tensors exactly equal; only the already reviewed ocean density-convention metadata differs |
| Reviewed ocean capture | 164 tensors exactly equal, including complete-step gradients |
| Original archive extraction fixture | 1,416 arrays exactly equal |
| Corrected Gibbs v1 and dry portable tools | Both pass, unchanged references and tolerances |
| Structural executable-definition audit | 457 preserved |
| Pinned atmospheric Gibbs/energy drivers | Complete with the explicit ocean source pin |

The tensor/fixture comparison logs, raw portable reports and runtime manifest
are in `integration_results/`. Hosted CI is reported on the PR separately.
Historical atmospheric fixture disagreements and exact-versus-portable policy
remain as documented in `verification/ATMOSPHERIC_INTEGRATION.md`.

## Reproduction

Build/install the starting-main wheel and run the copied tools outside the
checkout with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`:

```sh
python -I tools/capture_atmospheric_integration.py --save main-before.pt
python -I tools/capture_io_ownership.py --save output-before.pt
```

Install the reviewed ocean wheel and save its separate comparison:

```sh
python -I tools/capture_ocean_integration.py --save ocean-before.pt
```

Install the integrated wheel in that same environment:

```sh
python -I tools/capture_atmospheric_integration.py --compare main-before.pt
python -I tools/capture_io_ownership.py --compare output-before.pt
python -I tools/capture_ocean_integration.py --compare ocean-before.pt
python -I tools/capture_ocean.py --inputs tests/fixtures/ocean/inputs.npz \
  --compare tests/fixtures/ocean/archive_capture.npz
python -I -m pytest tests --fixture-mode=portable -q
python -I tools/check_baseline.py --reference-version corrected-gibbs-v1 --mode portable
python -I tools/check_dry_helpers.py --mode portable
```

For atmospheric scientific drivers use the existing numerical/structural pins
plus the ocean pin (full commands are in `.github/workflows/ci.yml`). This
integration adds no scientific qualification, GPU/compile support, long-run
claim, or new stratified ocean dynamics. Rotating balance is measured separately.
