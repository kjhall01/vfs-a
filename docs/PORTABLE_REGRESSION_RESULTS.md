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

Both implementation runs at `6c8b79b` completed successfully:
[push 40993224721](https://github.com/kjhall01/vfs-a/actions/runs/40993224721)
and [PR 40993229024](https://github.com/kjhall01/vfs-a/actions/runs/40993229024).
These are actual installed-wheel runs of the new workflow, not the earlier
diagnostic campaign. GitHub timestamps are 2026-10-30 UTC (October 29 locally).

Both used Intel Xeon Platinum 8573C, Ubuntu 24.04 x86_64, Python 3.12.15,
PyTorch 2.15.0+cpu / MKL 2024.2 / oneDNN 3.13.3 / AVX512, NumPy 2.5.4 and
SciPy 1.18.1 with OpenBLAS, and pytest 9.2.0. Thread controls were all 1;
ATEN_CPU_CAPABILITY and MKL_CBWR were unset. Both recorded identical installed
versions; CPU feature lists differ despite the same model designation. Linux
netCDF4 was 1.7.5.1.post1 versus macOS 1.7.4. Full configurations and dependency
lists are retained for [PR](portable_regression/linux-pr/environment.json) and
[push](portable_regression/linux-push/environment.json).

| Check (both runs) | Portable | Exact |
| --- | --- | --- |
| pytest | 511 passed, zero skipped | Both fixture tests failed |
| checkpoint-54 tool | 142 tensors passed | Failed at smooth/rhs/u_e |
| dry helper tool | 193 tensors passed | Failed at canonical/wind_fields/cell_wind_vector |
| MoistAtmosphere example | Passed | Not mode-dependent |

Exact tools exited 1; failure JSON and JUnit artifacts are retained. The job is
successful because portable checks are required and exact checks are explicitly
informational. The first exact mismatches have maximum absolute errors
1.0842021724855044e-18 (moist) and 3.552713678800501e-15 (dry). They remain exact
failures. Portable maximum budget fractions were 0.003163 (gradient, moist)
and 0.000516 (dry helper), with all independent invariants passing. Both runs
produced the same comparison metrics.

Reports and pytest summaries are retained under
[linux-pr](portable_regression/linux-pr/) and
[linux-push](portable_regression/linux-push/); full JUnit reports remain in each
run's `cpu-checks` artifact. The implementation needed no Linux-specific budget
changes. The original investigation still provides broader CPU/dispatch evidence.

No GPU, long-run, full backward or broader dependency-version qualification
is claimed. Portable success does not establish bitwise reproducibility.
