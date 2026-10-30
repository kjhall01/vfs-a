# CI discrepancy at 911aada: evidence and portable-mode proposal

This is the investigation and proposal recorded before implementation. The
subsequently authorized [portable policy](PORTABLE_REGRESSION.md) and
[validation results](PORTABLE_REGRESSION_RESULTS.md) supersede implementation
status statements below; the original evidence and proposed budgets are retained.

**Finding:** the original jobs have identical recorded dependency versions and
source trees. CPU/library dispatch changes numerical results with those versions
held fixed. New Intel jobs reproduce both first-mismatch fingerprints of the
original failing job. The original CPU models were not recorded, so attributing
the original pass to a particular processor or backend is not possible.

**No portable mode or tolerance changes have been implemented.** The original
exact comparators, workflow on the contributor branch, numerical package,
fixtures and integrators are unchanged. The new audit tool only records evidence;
its successful completion is not a regression pass. Proposed tolerances below
are for review, not scientific acceptance thresholds.

## Original runs

| Item | Passing push run | Failing PR run |
| --- | --- | --- |
| Run | [40957129456](https://github.com/kjhall01/vfs-a/actions/runs/40957129456) | [40957173935](https://github.com/kjhall01/vfs-a/actions/runs/40957173935) |
| Recorded head | 911aada26967a44534cb76a82eb26a86dfc1a8cc | Same |
| Checked-out commit | 911aada | fbd056e, GitHub's PR merge commit |
| Git tree | fafb0daf8c0a57c1446adac42fa303c6fc1d2faf | Same |
| Suite | 78 passed, 16.28 s | 76 passed, 2 failed, 22.70 s |
| Exact tools | 142 moist/control + 193 dry tensors pass | Both stop at first mismatch |
| Azure region | westus3 | centralus |
| Runner image | ubuntu-24.04, 20261025.353.1 | Same |
| OS / architecture | Ubuntu 24.04.5, x86_64, glibc 2.39 | Same |
| Kernel | 7.0.0-1017-azure | Same |
| Python | 3.12.15 | Same |
| PyTorch | 2.15.0+cpu | Same |
| numpy / scipy / pytest | 2.5.4 / 1.18.1 / 9.2.0 | Same |
| netCDF4 / xarray | 1.7.5.1.post1 / 2026.9.0 | Same |
| OPENBLAS / OMP / MKL thread env vars | All 1 | All 1 |
| CPU model, flags, torch dispatch, MKL dispatch | Not recorded | Not recorded |

All **28 versioned entries** in `pip freeze` match, including build tools;
this is not just a torch-version comparison. The local VFS wheel URL/hash differs
between builds, as expected for separately built wheels; the Git trees match.
Dependency wheel hashes and backend configurations were not recorded originally,
so identical version strings do not prove byte-identical dependency installations.

The failing run's first errors are:

- `smooth/rhs/u_e`: 936/1116 unequal, maximum absolute error
  1.0842021724855044e-18 m/s²; maximum relative error 0.04851404255084217.
- `canonical/wind_fields/cell_wind_vector`: 127/384 unequal, maximum absolute
  error 3.552713678800501e-15 m/s; maximum relative error 1.3907541538645057e-14.

Both smoke tests finished. Energy changes were -1.542650926958044e-5 and
-1.54922939572972e-5 J/m² respectively. Neither original run archived all actual
fixture tensors; the original failing run's remaining errors cannot be recovered
retroactively. The complete measurements below are explicitly **new executions**.

## Controlled CPU investigation

[Diagnostic run 40988452748](https://github.com/kjhall01/vfs-a/actions/runs/40988452748)
uses branch `audit/ci-911aada`, commit `b7f39ad`. Only its diagnostic workflow,
collector and dependency constraints differ from 911aada. It builds the unchanged
package, installs the wheel, copies checks outside the checkout, and pins all
28 recorded versions, including Python 3.12.15. No fixture capture flags are used.

Four jobs each run four fresh processes: native, `ATEN_CPU_CAPABILITY=avx2`,
`ATEN_CPU_CAPABILITY=default`, and `MKL_CBWR=COMPATIBLE`. Each process visits all
335 tensors, even after a mismatch, and invokes the original exact comparators.
The separate original pytest suite retains its failure exit status.

| CPU (4 visible logical CPUs each) | Native torch capability | Native exact moist / dry | ATen AVX2 | ATen DEFAULT | MKL COMPATIBLE |
| --- | --- | --- | --- | --- | --- |
| Intel Xeon Platinum 8573C | AVX512 | 121 / 191 | 121 / 191 | 120 / 159 | 142 / 191 |
| Intel Xeon 6973P-C | AVX512 | 121 / 191 | 121 / 191 | 120 / 159 | 142 / 191 |
| AMD EPYC 7763 | AVX2 | 60 / 67 | 60 / 67 | 60 / 58 | 60 / 67 |
| AMD EPYC 9V74 | AVX2 | 60 / 67 | 60 / 67 | 60 / 58 | 60 / 67 |
| Local Apple M3, macOS ARM64 | Native ARM build | 52 / 68 | Not run | Not run | Not run |

Denominators are 142 and 193. All four Linux suites completed with 76 passes
and the same two exact-fixture tests failing; none is claimed as a green CI run.
Both Intel native executions reproduce the original failing run's first error
paths, counts, maximum absolute errors and maximum relative errors exactly.
Changing only MKL dispatch on those machines restores **all 142** moist/control
comparisons, including scalar CFL and stage counts. It does not restore the two
dry wind-vector comparisons. Forcing ATen AVX2 alone does not restore either.
Thus pinning torch, or blindly forcing one instruction set, is insufficient.
The [PyTorch 2.15 dispatch source](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/DispatchStub.cpp)
documents the environment override used in the ATen experiments. The evidence
here does not isolate every affected operation or identify the original passing
runner's dispatch configuration.

The new Linux torch builds report GCC 13.3, MKL 2024.2 (build 20240605), and
MKL-DNN 3.13.3. numpy reports OpenBLAS 0.3.34.106.0; scipy reports OpenBLAS
0.3.31.dev. Runtime SIMD availability differs across the CPUs. Torch intra-op
threads are 1 in every audit; inter-op threads are 2 on Intel and 4 on AMD.
The within-machine dispatch experiments hold that inter-op setting fixed.
The raw configurations, CPU flags and dependency lists are archived below.

The macOS check uses the previously validated installed wheel, Python 3.12.13
and torch 2.15.0 (without `+cpu`), with numpy/scipy/pytest matching Linux.
Its netCDF4 is 1.7.4. It supplies additional portability evidence, not an isolated
CPU-only comparison against Linux. GPU, long-run and backward-rollout behavior
were not tested.

## Complete tensor evidence and invariants

[Full evidence archive](ci_911aada/evidence.zip) contains the original logs/job
metadata/artifacts and all 17 audit reports (16 Linux configurations and one
macOS), including every tensor's shape, dtype, finite counts, unequal-element
count, absolute/RMS error, reference scale, relative error, zero-reference
behavior, and component-wise species metrics. It also includes the executed
workflow/constraints. SHA256:
`bcac65ef808005adaa66bd4c2fcf518df5ad2fd4513577ecd619c8b709643ac7`.

[335-row tensor summary](ci_911aada/tensor_summary.csv) aggregates the **5,695
fixture-tensor evaluations**. [Quantity summary](ci_911aada/quantity_summary.json)
identifies each category's worst path/environment, including species components.
No tensor is omitted because an earlier comparison failed. The summaries do
not imply that values within a tensor have uniform relative error.

Across all 17 reports:

- All 335 shapes and dtypes match; every actual and reference tensor is finite.
  Container structure also matches. Scalar CFL differs at roundoff level;
  integer counts and stage-count lists match exactly.
- Initial, Euler and final moist states have positive pressure, layer thickness,
  dry fraction and vapor. Liquid/ice are nonnegative. All four species' ledger
  amounts are nonnegative, and heat is nonnegative. This checks the captured
  endpoints, not unrecorded intermediate states.
- Euler water contents equal `amounts[:, 1:] / cell_area` **exactly**, including
  reshaping. The bounded ledger remains the update source.
- `beta` is in [0, 1]. The recorded initial active masks match exactly, as do
  limited-face counts 0/105/119 and all nine recorded SSPRK3 stage counts per
  case. The frozen fixtures do not contain the later stages' full masks.
- Maximum moist final-minus-initial relative mass change: 2.0724e-16;
  per-species change: 3.2382e-16; Euler ledger inventory change: 1.7096e-16.
  Initially absent ice remains exactly zero. Species are assessed separately.
- Existing conservation, diagnostic-gradient, AST and API tests are among the
  76 passing tests in every new Linux job. The two failures remain exact
  migration comparisons, not hidden skips or xfails.

A small residual is not a useful denominator: reverse-cloud energy-rate error
is 2.6693 W against a reference residual of -0.029602 W (90.17 relative error).
Its underlying sum of absolute gradient–tendency products is 1.4455e16 W.
Across all cases/environments, the maximum energy-rate difference divided by
that reference work scale is **1.8467e-16**. Maximum absolute actual rate divided
by the reference work scale is 1.8262e-16. Maximum normalized dry rotation-work
residual is 2.3376e-17. These indicate cancellation sensitivity; they are not
90-fold errors in the underlying energy budget.

## Proposed separate portable mode — not implemented

Keep exact mode (`rtol=atol=0`) unchanged and the default. A future explicit
portable mode should retain the same captures and produce a separate result;
it must not relabel an exact failure as a pass. Existing physical invariant and
AST tests remain enabled. A stable exact lane needs a recorded CPU/backend
configuration as well as package versions; the generic Ubuntu label does not
supply that. Do not change production MKL settings based on this experiment.

For ordinary quantities, propose a **quantity-scaled maximum absolute error**:
`max(abs(actual - reference)) <= tau * S`, with `S = max(abs(reference))` for
that specific field/component. This is an absolute tolerance in that field's
units, not pointwise relative error around zeros. Split dry/vapor/liquid/ice
and tracer components before computing S: dry air must never set a condensate's
allowance. If S is zero, require exact zero unless the field is explicitly one
of the cancellation residuals below. Never clamp S to a dimensionless 1.

| Quantity group | Largest observed error / S | Proposed tau | Rationale |
| --- | ---: | ---: | --- |
| Ordinary prognostic states and geopotential | 1.291e-13 | 5e-13 | Accumulated geometry/step roundoff; about 4× observed maximum |
| Other dry helper fields, norms and force terms | 7.311e-14 | 5e-13 | Geometry, local solves and reductions; excludes residuals below |
| Moist/other ordinary tendencies | 4.272e-13 | 2e-12 | Divergence and nearly balanced subtraction |
| Analytic energy gradients, per input field/species | 1.683e-14 | 2e-13 | Separate derivative units and quadrature/reduction scale |
| Mass/species/tracer inventories | 2.024e-16 | 128 eps = 2.842e-14 | Float64 reductions; separate conservation checks still required |
| Ledger amounts, integrated fluxes, beta | 1.766e-15 | 128 eps | Matches the existing amount/flux roundoff scale; no positivity allowance |
| Energy totals/components | 1.233e-14 | 2e-13 | Extensive energies; do not mix with geopotential or near-zero work |
| JW control initial/final `mtheta_i` only | 3.431e-12 | 1e-11 | Analytic geopotential differenced with eta increment 1e-5 |
| JW control `rhs/u_e`, `rhs/mtheta_i` only | 6.236e-11 | 2e-10 | Propagation of that differenced initialization into balanced tendencies |
| Limiter force correction only | 1.633e-10 | 1e-9 | Tiny correction formed from differences; absolute allowance ~1.63e-22 m/s² in reverse cloud |
| Limiter heat and entropy source only | 6.031e-12 | 2e-11 | Near-cancelling Gibbs work; heat allowance ~8.71e-4 W in cloud |
| Floating CFL scalars | Roundoff-level changes | 128 eps | Dimensionless positive scalar; integer limiter metadata remains exact |

These are **provisional regression budgets**, not proven all-platform error
bounds. The JW path explicitly differences geopotential anomalies before
forming theta, so epsilon/1e-5 amplification (~2.2e-11 before conditioning)
explains why a generic few-ULP allowance is inappropriate there. Limiter-force
and heat allowances are narrowly scoped to their own small corrections; they
must not relax the full momentum tendency or total energy checks. An explicit
reviewed path-to-quantity mapping should cover every fixture leaf, with unknown
paths rejected rather than assigned a broad fallback tolerance.

For cancellation residuals, propose:

- Moist `energy_rate`: use `W = sum(abs(gradient[k] * rhs[k]))` over all cells,
  levels and fields, in watts. Require both an independent closure check
  `abs(rate_actual) <= 128*eps*W_actual` and a regression difference bound
  `abs(rate_actual-rate_reference) <= 128*eps*W_reference`. The latter permits
  ~411–420 W for these global fixtures, or ~8e-13 W/m²; observed differences
  are at most 2.67 W. First compare the gradients/tendencies themselves so
  inflated operands cannot make a broken closure appear acceptable.
- Dry rotation `work`: use its existing `work_abs_scale`, separately for each
  formulation (their native units differ). Use the same 128-eps normalized
  closure and difference bounds. For already normalized `work_relative`, use
  absolute tolerance 128 eps and zero relative tolerance. Do not compare these
  residuals with a relative tolerance based on the near-zero residual itself.

The proposed portable result must also require, independently of those bounds:

1. Exact structure, shapes, dtypes, integer/bool metadata, stage counts, recorded
   active masks and the 99 helper ASTs. Every float must be finite; do not allow
   matching NaNs to pass this mode.
2. Strictly positive pressure/layer mass, dry fraction and vapor; nonnegative
   condensates and bounded-fixture ledger amounts; beta in [0,1]; nonnegative
   heat for `release_as_heat=True`. No moisture floor, clipping or relaxed
   positivity derived from the regression tolerance. Preserve the existing
   separate tests of the kernel's documented condensate-roundoff policy.
3. Exact Euler water recovery from the species ledger. For mass and each
   species, check `abs(final-initial) <= 128*eps*sum(abs(initial amounts))`;
   an initially absent species has a zero bound. Apply the same separate check
   to the Euler ledger, and retain existing dry/tracer conservation tests.
4. Report finite-step energy drift separately. SSPRK3 is not exactly energy
   conserving; do not confuse total-energy regression, instantaneous work
   cancellation, and finite-step drift.

Before adopting this mode, review the per-path mapping and run deliberate
perturbation tests that break each numerical budget and each invariant. Confirm
that wrong shape/dtype, a single NaN, negative/zero vapor, negative condensate,
changed limiter mask/count, and a broken amount ledger cannot pass. Broader
Python/library versions and additional architectures require new evidence;
this sample does not qualify them automatically.

## Reproduce the audit

With the wheel installed in a clean environment, from outside the checkout:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -I /path/to/vfs-a/tools/audit_fixture_differences.py \
  --report /tmp/fixture-audit.json
```

The collector patches `torch.autograd.grad` to reject forward autodiff during
capture, calls the original exact comparison functions, and verifies fixture
hashes before/after. It reports failure evidence without changing test gates.
Synthetic checks cover nonfinite reporting, shape mismatch, multiple tensor
mismatches in one traversal, and discrete scalar differences; a complete local
audit also ran against the installed wheel. `git diff --check` passed.
The original tools still fail exactly where they did. No baseline fixture was
regenerated, no numerical expression/default changed, and no portable mode was
added. The diagnostic branch is separate; no investigation PR has been opened.
