# Pass 4 — physical-field workflow

> Historical report. For current APIs and supported configurations, read
> [retained timestep paths](CURRENT_PATHS.md) and [readability results](READABILITY_RESULTS.md).

Added a small MoistAtmosphere facade using composition around the existing
MoistConservativeDynamics and BoundedGibbs. It holds mesh/column, constructs the
existing MoistState from physical fields, projects edge-sampled east/north wind,
recovers interface fields, advances bounded SSPRK3 and returns tensor budgets.

No numerical kernel file was changed. Only root/atmosphere exports and local
package version changed in existing package sources. New implementation:
vfs/atmosphere/model.py. Guide: PHYSICAL_FIELDS_API.md.
Runnable example: examples/moist_atmosphere.py.

## Validation

Installed wheel tested outside checkout, PyTorch 2.15.0+cpu, CPU float64.
78 tests passed in 23.47 s (see tests_pass4.log).
142 checkpoint-54 tensors match exactly (equivalence_pass4.json).
The suite also compares 193 dry helper fixture tensors exactly and checks all
99 extracted dry helper ASTs. Thus all 335 existing fixture tensors still match.

New tests cover physical-field round trips, exact agreement with manual state
construction and the underlying bounded step, wind projection, invalid shape/
dtype/nonfinite/physical inputs, explicit rejection of external numpy arrays,
geometry mismatch, zero condensates, zero-vapor rejection, and finite-difference
agreement of energy gradients with respect to temperature, pressure and vapor.

The 64-cell, 6-layer example completed 10 steps of 30 s:
- Relative total mass change: -6.217176653065638e-16.
- Species inventory changes: [-24, -1, 0] kg, on planetary total inventories.
- Energy change: -1.542650926958044e-5 J/m².
- Minimum vapor/liquid/ice: [0.009999999995771525, 1.4474766504931322e-50, 0].
These are illustrative diagnostics, not new timestep guidance or a balanced
benchmark; no scientific acceptance thresholds are asserted.

## Existing limitation exposed

The first new dry-limit test failed because the unchanged moist adapter
requires vapor strictly greater than zero. Initialization now requires an
explicit vapor input and rejects zeros early with an explanatory error.
No floor, numerical change or automatic dry-core substitution was introduced.
Exact-zero-vapor dry equivalence has NOT been established. Liquid/ice zeros
are supported. This remains a separate numerical question for later work.

## Scope

The model is a thin Python facade, not an nn.Module with automatic device moves.
Autograd is retained in tensor initialization and diagnostics; this pass does
not qualify full backward rollouts, GPU behavior or every limiter branch.
Temperature/composition must already be at CP interfaces; winds at layer-center
edges. No interpolation, regridding or phase equilibration is hidden.

Xarray/DataArray adapters, output/restarts and the friendly dry facade remain
future work. Legacy optional modules still ship. This snapshot is suitable for
moving development to GitHub; public release metadata/license remain undecided.
Original checkpoint 54 and passes 1–3 are unchanged.

## Subsequent CI regression policy

The [portable policy](PORTABLE_REGRESSION.md) adds a separate installed-wheel
regression mode without changing pass-4 numerical code or fixtures. The
[CI investigation](CI_911AADA_INVESTIGATION.md) retains the CPU/backend evidence;
[validation results](PORTABLE_REGRESSION_RESULTS.md) distinguish portable and
exact outcomes. Historical exact results above describe their original environment,
not a cross-platform bitwise guarantee.

The later [scope-reduction PR](SCOPE_REDUCTION.md) deliberately removes non-atmospheric
and ML APIs while preserving this physical-field workflow and its numerical baseline.

## Subsequent module-boundary work

The [module API](MOIST_MODULE_API.md) owns mesh, column and dynamics once,
with explicit non-owning steppers delegating to the existing bounded SSPRK3,
centered SSPRK3 or experimental centered IMEX algorithms. See the separate
[validation results](MOIST_MODULE_RESULTS.md) and [bounded gradient limitation](BOUNDED_GRADIENT_LIMITATION.md).
These do not turn historical CPU forward results into accelerator or rollout
qualification. Centered IMEX does not provide bounded species transport.

The subsequent [tensor-tolerance fix](BOUNDED_GRADIENT_LIMITATION.md) preserves
forward results and adds scoped CPU float64 one-step directional derivative
checks. It fixes the allclose argument error without qualifying full rollouts
or differentiability at limiter switches.

## Subsequent State/Field ownership cleanup

See [the ownership guide and validation](STATE_OWNERSHIP.md). Dry prognostic
attributes now return tensors; standalone Field retains its mesh/regridding role.
Dry geometry mismatches and implicit tracer casts are rejected, explicit
conversion preserves gradients, and model methods expose uncached physical fields.
Local installed-wheel portable tests: 600 passed, 1 CUDA skip. Both comparison
tools and examples pass; 382 same-environment capture tensors agree exactly.
Frozen-fixture exact failures remain reported separately. No numerical operator,
fixture, moist stepper or private entropy-adapter body changed.

PR #8 review follow-up: coarse Schur tracer restriction now uses tensor axis -3
instead of Field metadata. Three new cases verify restricted values and shape
with real transfers and up to two leading batch axes. Local installed-wheel
portable suite: 603 passed, 1 CUDA skip; both standalone comparisons and examples
pass. No operators, transfer weights, fixtures or tolerances changed. See the
ownership guide for the audit and the limited scope of batched coverage.

## Subsequent supported-path scope slice

Merged PR #8 is the baseline for [the scientific-readability plan](SIMPLIFICATION_MAP.md).
ARK4-only public IMEX selection preserves 298 captured tensors and all stage solver
diagnostics exactly. Installed-wheel portable suite: 601 passed, 1 CUDA skip;
335 frozen fixture tensors and both examples pass. See CURRENT_PATHS.md for
explicit/implicit guarantees and the precise representative-case evidence.

## Subsequent retained-equation consolidation

See [readability changes and validation](READABILITY_RESULTS.md). The dry function
binding/recipe stack is replaced by ordinary conservative/energy methods and
DryAtmosphere with explicit steppers. A borrowed scalar transport view replaces
entropy-as-mtheta; bounded ledgers and formulas remain. ARK4 wave/stage/solver
owners are separated without new methods. Installed-wheel portable: 519 passed,
1 CUDA skip; both tools/examples pass, 298 same-environment pre/post tensors and
solver diagnostics exactly equal. Frozen exact failures remain separate. The
100 removed canonical-core fixture paths are retired explicitly, with fixture
bytes and retained tolerances unchanged. No IMEX backward qualification.

## Current guide reconciliation

Start with [retained paths](CURRENT_PATHS.md), [ownership examples](STATE_OWNERSHIP.md)
and [current regression evidence](READABILITY_RESULTS.md). Earlier reports and
readability/ownership measurements are preserved under [history](HISTORY.md).
The final guide slice changes no executable expressions; both usage snippets ran
against the installed wheel. PR #10 final-head hosted CPU checks passed; its push
artifact records 519 portable passes, 1 skip and separate exact failures.

## Standalone retained dry forcing

Before merging the simplification stack, preserve existing Held–Suarez thermal
relaxation and Rayleigh drag in [a standalone source](HELD_SUAREZ.md). No legacy
framework is restored. Original method ASTs and 132 same-environment comparisons
are exact, including forced-stage states and representative energy gradients.

## Final bounded cleanup and verification handoff

The [cleanup report](FINAL_CLEANUP_RESULTS.md) records removal of obsolete IMEX
configuration, intrinsic bounded heat release and redundant tensor wrappers,
with exact same-environment comparisons and unchanged fixtures. The committed
Held–Suarez archive tool and five supported scenario types are runnable. Freeze
implementation for [verification](VERIFICATION_HANDOFF.md); Conda is deferred.
