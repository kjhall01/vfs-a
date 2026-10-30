# Portable regression validation — 2026-10-29

The [policy](PORTABLE_REGRESSION.md) implements the separately reviewed
[investigation proposal](CI_911AADA_INVESTIGATION.md). Its original evidence
archive and all-tensor summaries remain available. No fixtures or numerical
package code changed. This report concerns CPU float64 forward behavior only.

## Local macOS

Apple M3, macOS 14.6.1 arm64, Python 3.12.13, PyTorch 2.15.0, NumPy 2.5.4,
SciPy 1.18.1 and pytest 9.2.0. PyTorch, NumPy and SciPy use Apple Accelerate;
PyTorch reports DEFAULT CPU capability and OpenMP. OMP/MKL/OpenBLAS thread
controls were all 1. [Full environment](portable_regression/macos/environment.json).

Tests/tools/examples ran from `/tmp/vfs-ci-check` against `site-packages/vfs`
in a clean standalone environment. A newly rebuilt wheel was installed and its
138 Python package files verified byte-for-byte against the unchanged source.

| Check | Portable | Exact |
| --- | --- | --- |
| pytest | 511 passed (full suite, including both fixture tests) | 2 failed (fixture tests only) |
| checkpoint-54 tool | 142 tensors passed | Failed at smooth/initial/u_e |
| dry helper tool | 193 tensors passed | Failed at canonical/primitives/dp_e |
| MoistAtmosphere example | Passed | Not mode-dependent |

Both exact tools exit 1. The first moist mismatch has maximum absolute error
4.440892098500626e-15; the first dry mismatch 1.0913936421275139e-11.
Portable budget usage peaks at 0.34305 (JW thermal) for checkpoint-54 and
0.14622 (dry helper) for dry helpers; all independent invariants pass.
[Portable moist](portable_regression/macos/baseline-portable.json),
[portable dry](portable_regression/macos/dry-helpers-portable.json),
[exact moist](portable_regression/macos/baseline-exact.json) and
[exact dry](portable_regression/macos/dry-helpers-exact.json) retain the outcomes.

The suite adds 433 tests, including every one of the 335 tensor mappings,
budget boundaries, per-species/zero scales, malformed structure, nonfinite
data and deliberate invariant violations. Original exact comparison function
bodies were verified unchanged against 911aada. Workflow YAML and embedded
Bash syntax checks passed locally. No fixture was regenerated.

## GitHub Actions Linux

Pending the implementation push. Prior diagnostic Linux runs in the
investigation are evidence for the budgets, not completed tests of this new
required portable gate. This section will record the actual implementation run,
CPU/backends, portable and exact outcomes separately.

No GPU, long-run, full backward or broader dependency-version qualification
is claimed. Portable success does not establish bitwise reproducibility.
