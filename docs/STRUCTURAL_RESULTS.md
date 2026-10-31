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
