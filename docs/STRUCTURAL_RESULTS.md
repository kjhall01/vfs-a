# Structural migration evidence

Merged-main structural baseline: `3f806a947e2ff8a5947953137a96af02861d083f`.
Numerical #18/#19 changes are excluded. Fixtures and tolerances are unchanged.
The [ownership map](PACKAGE_OWNERSHIP.md) precedes the structural edits.

## Slice 1: atmospheric model, state and pressure column

Moved dry model/state and CP column to atmosphere; old public paths re-export
the identical objects. Internal callers use the owner paths. Core exports are
lazy to avoid a state/configuration import cycle; no state is cloned or wrapped.

Installed wheel outside checkout, macOS 14.6.1 ARM64, Python 3.12.13, Torch
2.15.0/Accelerate, CPU float64: 551 existing tests pass, one CUDA skip; the new
alias-identity test passes separately. Exactly 298 capture tensors plus all
scalar/solver records agree with the pre-edit baseline, including dry/moist
ARK4 states and stages, bounded ledgers, diagnostic derivatives and RHS VJPs.
Existing complete-step bounded directional-gradient tests pass. Historical
portable tools pass; informational exact outcomes are recorded separately.
No GPU, full-rollout or new scientific claim is made. CI runs the same structural
baseline capture; C1/C2–C4 measurement jobs are paused during this stack rather
than relaxing their frozen-source guard to accept moved files.

## Slice 2: atmospheric IMEX and shared integration

Atmospheric ARK4 waves, configuration, scaling, stage systems and Schur solvers
move together to `atmosphere.imex`. Shared `integrators` owns unchanged ARK4
coefficients, explicit sequences and GMRES arithmetic; only generic typing
replaces dry-state annotations there. Coarse-state assembly moves to
`atmosphere.imex.transfer.restrict_state`; mesh weights remain shared. This
formerly non-exported geometry helper now requires the owner import. Tests
which patch private implementation globals target the owner module. Public
`vfs.imex` and `vfs.integrators.imex` imports retain identical class objects.

Same installed-wheel environment: 552 tests passed, one CUDA skip; the added
shared-import check passes separately (two ownership checks total). All 298
capture tensors and complete solver records match the same pre-edit baseline
exactly, including multilevel Schur. Both portable tools and six examples pass.
Historical moist and dry exact fixture tools fail separately on this environment,
as they did before these moves; no tolerance or fixture changes.

## Slice 3: component output/initialization and shared mechanics

Atmospheric earth/boundary profiles, moisture conventions, ERA5 interpolation
and initial conditions move into atmosphere. Shared elevation acquisition and
NetCDF coordinates/metadata/mechanics remain independent of component states.
Generic field remapping is `geometry.remap`; explicit atmosphere/ocean output
adapters preserve field interpretation. The old regrid class alone retains its
inference as a compatibility convenience. Ocean has an output adapter only;
scope tests still reject ocean state, column, dynamics and physics modules.

Final local installed wheel: **559 passed, one CUDA skip**. All 298 dynamics
capture tensors/scalar records and **74 output/initialization tensors** compare
exactly to the unchanged baseline wheel. Output captures include gradients,
all dry and ocean-contract fields, snapshots/reload, prescribed terrain/earth
initialization and boundary-file round trip. Both portable tools and all six
examples pass. Historical exact moist and dry comparisons fail separately.
The AST audit preserves **457 relocated executable definitions/declarations**,
including defaults and statement order; imports/types/docstrings are excluded.
Four output-adapter boundaries are explicitly excluded and numerically captured.
No fixture/tolerance or Gibbs/limiter/RK arithmetic changed.

An initial direct-owner import exposed a cycle through the legacy data facade;
lazy compatibility exports resolve it, covered by fresh-process tests. The new
output-only ocean namespace required revising the deliberate-removal tests to
continue rejecting ocean numerical modules rather than rejecting the adapter.
These issues were fixed before final validation, not reclassified as passes.

[Machine-readable evidence](STRUCTURAL_EVIDENCE.json) records local environments,
fixture outcomes, capture logs/hashes and executable-definition audit. The local
sandbox blocked CPU model-name querying; architecture and Torch Accelerate
backend are recorded rather than inferring hardware support. All tests use CPU
float64. No expanded atmospheric experiments, GPU or rollout qualification.

Hosted #20 run 41225650886 and #21 run 41225837208 completed successfully.
Downloaded #21 artifacts confirm 553 passes, one CUDA skip, and all 298 capture
tensors/solver records exact on Linux; this is same-environment equivalence,
not cross-platform bitwise reproducibility. The third-slice hosted result is
pending at delivery preparation and must be distinguished from these runs.

Reproduce using an isolated installed wheel and `tools/capture_readability.py`
(`--save` before, `--compare` after), then `tools/capture_io_ownership.py` similarly.
Use new output paths; never overwrite fixture data. Run
`python -I tools/check_structural_moves.py --repo /path/to/checkout --report /tmp/structural-ast.json`
for the pinned merged-main AST audit. CI executes both old/new installed-wheel
captures in one environment. Scientific driver guards remain frozen and their
measurement jobs remain paused during this structural phase.
