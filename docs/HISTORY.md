# Historical reports and measurements

For current usage start with [the verification handoff](VERIFICATION_HANDOFF.md),
[cleanup results](FINAL_CLEANUP_RESULTS.md), [retained paths](CURRENT_PATHS.md),
[ownership](STATE_OWNERSHIP.md) and [readability results](READABILITY_RESULTS.md).
The reports below record earlier source trees, APIs and validation. Their old
next-step instructions and supported-option lists are not current guidance.

- [Pass 1](PASS1_RESULTS.md), [pass 2](PASS2_RESULTS.md),
  [pass 3](PASS3_RESULTS.md), [pass 4](PASS4_RESULTS.md): checkpoint extraction.
- [Initial scope reduction](SCOPE_REDUCTION.md): domain/ML removals.
- [Module ownership proposal](MOIST_INTEGRATOR_OWNERSHIP_PROPOSAL.md) and
  [module results](MOIST_MODULE_RESULTS.md): PR #6, before ARK2/3 removal.
- [CI investigation](CI_911AADA_INVESTIGATION.md),
  [initial CI results](CI_SETUP_RESULTS.md),
  [portable results](PORTABLE_REGRESSION_RESULTS.md): reproducibility evidence.
- [Bounded derivative limitation/fix](BOUNDED_GRADIENT_LIMITATION.md): measured
  one-step derivatives and remaining limits; not IMEX backward qualification.
- [Pre-edit simplification map](SIMPLIFICATION_MAP.md): retained dependencies.
- [Handoff](../handoff/index.md): current section followed by labeled snapshots.

## Earlier State/Field ownership validation (PR #8)


Local installed-wheel checks ran outside the checkout on macOS 14.6.1 arm64,
Python 3.12.13, PyTorch 2.15.0, CPU float64, Accelerate BLAS/LAPACK (NumPy 2.5.4,
SciPy 1.18.1). OMP/MKL/OpenBLAS thread environment variables were set to 1.
The sandbox denied the CPU model sysctl query; no accelerator execution is
claimed for this pass.

- Portable pytest: **600 passed, 1 CUDA skip**, with the existing scalar-conversion
  warning from bounded donor subcycle selection. Includes ownership tests, moist
  module/stepper tests, dry IMEX checks, and existing one-step gradient tests.
- Standalone portable comparisons: **142 moist + 193 dry tensors passed**.
- Both existing moist examples and both usage snippets above completed.
- Same-environment pre/post captures against merged PR #7 (`9021af2`): all
  **335 fixture-capture tensors** and **47 additional tensors** are exactly equal
  (`rtol=atol=0`). Extras cover centered moist terms/energies, diagnostic gradients,
  RHS VJPs, bounded facade stepping, dry tracer SSPRK3, dry IMEX and dry energy
  gradients. Private adapter aliases were checked. Captures stayed outside the
  repository; fixtures were not rewritten.
- Frozen-fixture **exact** mode remains a failure on this host: first reported
  moist mismatch `smooth/initial/u_e` (max abs 4.440892098500626e-15), dry mismatch
  `canonical/primitives/dp_e` (max abs 1.0913936421275139e-11). These are distinct
  from exact same-environment pre/post agreement; portable success does not
  reinterpret exact failures. See the existing CI investigation/portable policy.

GitHub Actions results will be recorded separately in the PR after it runs.
This is ownership regression coverage, not a new scientific verification campaign,
full-rollout qualification, or expansion of supported precision/accelerators.

### PR #8 review follow-up: coarse tracer restriction

The coarse Schur builder still read `anchor.tracer_m.horizontal_axis`, which
fails for the new tensor payload. It now passes `horizontal_axis=-3`, matching
`[..., cell, full, tracer]`; no transfer weights or numerical operators changed.
The new regression fails with the old access and passes after this correction
for no batch axes, one leading axis `(2,)`, and two leading axes `(2, 3)`.
It executes real hierarchy transfers and coarse linear-operator construction,
checks the complete tracer shape and values against an independent weighted
donor sum, and verifies the fine input is unchanged. The test isolates coarse
construction from modal fitting; it does not qualify a complete batched solver.

An audit of retained Python consumers found no further Field-only metadata
accesses on state tensors. Remaining placement/axis accesses are in standalone
Field mesh/regridding paths, guarded Field input conversion, and StateSpec metadata.
Local rebuilt-wheel validation on the same macOS CPU environment: **603 passed,
1 CUDA skip**, both portable tools (142 + 193 tensors) and both examples pass.
The existing subcycle scalar-conversion warning remains. Fixtures are unchanged;
this follow-up makes no new exact cross-environment or accelerator claim.

## Earlier bounded-wrapper readability validation (PR #4)


This PR adds source explanations and renames locals only. No helper extraction,
method reordering, cached/shared intermediates or algebraic simplification is
proposed. In particular, deduplicating repeated graph/gradient evaluations or
reconstructing Euler water from the RHS would need separate review for operation
order, tensor/gradient ownership and roundoff; neither is part of this change.

Executable ASTs of all four edited modules match the base after removing
docstrings and reversing the local-name map. Expressions, calls, branch order,
defaults, dictionary keys, state conventions and ledger operations are preserved.
Fixtures and the portable/exact policies are unchanged.

Local validation: Apple M3, macOS 14.6.1 arm64, Python 3.12.13, torch 2.15.0
with Accelerate, NumPy 2.5.4, SciPy 1.18.1, one thread. A freshly staged wheel
was installed into site-packages and checked outside the checkout with `-I`:
**527 portable tests passed**; standalone comparisons passed **142 moist/control
and 193 dry tensors**; both examples completed 300 simulated seconds. A separate
same-environment capture before and after this edit matched **all 335 tensors
exactly** using the original exact comparators. Those temporary captures did
not replace or modify repository fixtures. This exact pre/post result is
separate from comparison to the historical cross-backend fixtures.

AST normalization confirmed the four edited modules preserve executable syntax
relative to `9f82a0c`; wheel Python files match the current source, and
`git diff --check` passed.

Actual Linux [push run 41003311864](https://github.com/kjhall01/vfs-a/actions/runs/41003311864)
at `4046f3b` passed **527 portable tests**, both standalone portable tools and
both examples. Runner: AMD EPYC 7763 / AVX2, Python 3.12.15, torch 2.15.0+cpu
with MKL, CPU float64. Separately, both exact fixture tests and both exact tools
failed against the historical fixtures; these remain explicit informational
failures, not portable passes. Full CPU/backend details and mode-specific reports
are in the run's `cpu-checks` artifact. Subsequent PR/final-head runs are linked
in [PR #4](https://github.com/kjhall01/vfs-a/pull/4). CPU float64 checks do not
qualify GPU, long-run or full backward behavior.

## Centered moist readability validation

This follow-up formats only `moist_conservative.py`, expands packed statements,
renames ambiguous locals and explains the existing execution/adapter boundary.
No new helper file, class, cache, registry or functionality is introduced.
Executable AST comparison with `850dd21`, excluding docstrings and reversing
local renames, preserves all expressions, calls, defaults, branches, in-place
updates and evaluation order. Fixtures and comparison policies remain unchanged.

Local validation used Apple M3 / macOS 14.6.1, Python 3.12.13, torch 2.15.0 /
Accelerate, CPU float64 and one thread. A fresh staging wheel was installed into
site-packages; tests/tools/examples ran outside the checkout with `-I`:

- **527 portable tests passed**, including both fixture tests and existing
  derivative/invariant checks; standalone tools passed **142 + 193 tensors**.
- Both examples completed 300 simulated seconds.
- Same-environment captures before/after matched all **335 fixture tensors**
  exactly, plus **33 additional tensors**: component tendencies (15), energy
  terms (6), energy autograd derivatives (4), a fixed RHS vector–Jacobian product
  (4) and combined RHS (4). Additional comparisons required finite values and
  zero absolute/relative tolerance. The fixed contraction is a connectivity
  check, not a physical objective or full backward-rollout qualification.
- Adapter wind/pressure/entropy tensor identities and shared geometry were
  checked in both captures. Temporary captures did not touch repository fixtures.
- Executable AST comparison, wheel/source byte checks, single-file formatter
  check and `git diff --check` passed. Black was temporary tooling only; no
  dependency or formatting-policy change was added to the package.

Actual Linux [push run 41005300530](https://github.com/kjhall01/vfs-a/actions/runs/41005300530)
at implementation commit `4058876` passed **527 portable tests**, both standalone
portable comparisons and both examples. It used AMD EPYC 9V45 / AVX512,
Python 3.12.15, torch 2.15.0+cpu / MKL and CPU float64. **Both exact fixture tests
and both exact standalone tools also passed on this runner**; that is not a
cross-platform guarantee. Full environment and separate mode reports are in its
`cpu-checks` artifact. Subsequent PR/final-head runs are linked in
[PR #5](https://github.com/kjhall01/vfs-a/pull/5). These are regression checks,
not a scientific verification campaign.
