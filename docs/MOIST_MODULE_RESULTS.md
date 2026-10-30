# Moist module integration results

> Historical report. For current APIs and supported configurations, read
> [retained timestep paths](CURRENT_PATHS.md) and [readability results](READABILITY_RESULTS.md).

Scope: single module ownership, explicit supported steppers, checked collaborator
binding, explicit state conversion and two device-aware integer graph allocations.
BoundedGibbs, moist_conservative, IMEX algorithms and fixtures are unchanged.

## Local installed-wheel checks

macOS 14.6.1, Apple M3 arm64, Python 3.12.13, PyTorch 2.15.0, NumPy 2.5.4,
SciPy 1.18.1, Accelerate BLAS/LAPACK; CPU float64 and one numerical thread.
Built a wheel from staged package sources and force-installed it into
`/tmp/vfs-ci-standalone`. Tests/tools/examples were copied into
`/tmp/vfs-module-check`; Python `-I` imports the installed package outside the
checkout. No fixture regeneration or tolerance changes occurred.

| Check | Result |
| --- | --- |
| Portable full pytest, including both fixture tests | 561 passed, 1 skipped (no CUDA hardware) |
| Portable checkpoint-54 tool | 142 tensors pass; limiter/ledger/invariant checks retained |
| Portable dry helpers tool | 193 tensors pass |
| Both examples | Complete 10 × 30 s; finite diagnostics |
| Same-environment pre/post capture | All 335 fixture-capture tensors exactly equal |
| Additional pre/post centered values/gradients + facade step | 37 tensors exactly equal |
| Direct stepper delegation | Exact field equality for bounded SSPRK3, centered SSPRK3, centered ARK2/3/4 |
| Exact frozen checkpoint-54 comparison | FAIL, retained as an exact failure |
| Exact frozen dry comparison | FAIL, retained as an exact failure |

The pre-change capture used the PR #5 installed wheel before replacement. The
additional 37 tensors comprise 15 tendency terms, 6 energy terms, 4 diagnostic
energy derivatives, 4 centered-RHS vector-Jacobian products, 4 combined RHS fields,
and 4 nongradient facade-step fields. Private adapter tensor aliases were checked.
These are same-environment implementation comparisons, not cross-platform bitwise
claims or validation of long-run derivatives.

Frozen fixture exact failures remain consistent with the separate
[CPU/backend investigation](CI_911AADA_INVESTIGATION.md): the first reported moist
mismatch is initial wind (max absolute difference 4.44e-15); dry is edge pressure
thickness (1.09e-11). Portable passes do not convert these exact failures to passes.

New tests cover unique module/buffer ownership, state exclusion from state_dict,
call/step hooks, explicit conversion and gradient connectivity, dtype/device and
geometry mismatch errors, integer connectivity, checkpoint round-trip, component
replacement/reuse rejection, supported delegation and IMEX rejection/failure
propagation. Existing IMEX tests still cover all three methods, wave blocks,
output/stage admissibility and unsupported solver configurations.

The first full run exposed an eager IMEX import that loaded optional mesh hierarchy
code for the default path. Experimental imports were moved to explicit IMEX
selection and the import-boundary tests and full suite passed afterward.

## Gradients and accelerator limits

Diagnostic gradients and conversion gradients pass. The separate
[gradient reproduction](BOUNDED_GRADIENT_LIMITATION.md) fails on the pre-existing
allclose scalar argument in bounded timesteps; that assertion is unchanged.
The ordinary forward pre/post probe uses independently initialized nongradient
state, rather than detaching a failed gradient-enabled state or disabling its
graph. Centered one-step entropy-gradient probes were finite for each input field;
no IMEX backward or full-rollout gradient qualification was performed.

Apple MPS was available when tested outside the process sandbox. A manual
installed-wheel smoke moved a 16-cell/3-layer model before initialization to MPS
float32 (MPS does not support float64), initialized T=280 K, ps=100000 Pa,
wind=1 m/s and vapor/liquid/ice=.01/.002/.001, and completed one bounded dt=1 s
step with finite prognostic fields. This is only an execution smoke, with no
reference-accuracy, conservation, backward or long-run qualification in float32.
CUDA was unavailable; its execution test was skipped honestly. CPU float32
conversion/index tests establish ownership behavior, not numerical accuracy.

## GitHub Actions

[Installed-wheel Linux run 41009154563](https://github.com/kjhall01/vfs-a/actions/runs/41009154563)
completed on implementation commit `a272947` for
[PR #6](https://github.com/kjhall01/vfs-a/pull/6). This is separate hosted evidence,
not the local macOS run. Ubuntu runner: AMD EPYC 7763, AVX2, Python 3.12.15,
PyTorch 2.15.0+cpu, Intel oneAPI MKL 2024.2 / MKL-DNN 3.13.3, CPU float64;
OMP/MKL/OpenBLAS thread limits were 1.

- Required portable pytest: **561 passed, 1 skipped** (CUDA unavailable).
- Required portable tools: **142 moist + 193 dry tensors pass**, including the
  existing ledger, limiter and independent invariant checks.
- Both installed-wheel examples completed successfully.
- Informational exact pytest: **2 failed**. Both exact standalone tools also
  **failed**, with exit status 1 and failing JSON reports. The first moist
  mismatch is initial wind (max absolute 1.78e-15), dry edge pressure thickness
  1.09e-11. No fixture or tolerance changes were made.

The overall job is green because exact steps use the existing continue-on-error
policy; the downloaded XML/JSON artifacts and logs establish their failures.
Step conclusions alone are not evidence that exact comparisons passed. The run's
`cpu-checks` artifact contains the environment, portable and exact results.
