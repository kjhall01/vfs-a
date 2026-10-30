# Increment 1b — initial measurements and validation

Runtime source remains `8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648`.
[Methods and commands](INCREMENT_1B.md) define the experiments and units.
These measurements are not new scientific pass criteria or completed paper claims.

## Local CPU float64

Apple M3, macOS 14.6.1 arm64, Python 3.12.13, Torch 2.15.0/Accelerate,
one Torch thread. Installed baseline wheel outside the checkout; its source files
were verified against the pinned merge before each run. Manifests include actual
backend/library configuration. Both 16-cell/3-layer and 32-cell/4-layer runs used
seed 20261030, dt=1 s, two directions per C2 block, epsilon 1e-2 through 1e-7.

C2 recorded 360 moist and 180 dry rows per run: six and three prognostic blocks,
two directions, six epsilons, five energy components. Maximum total-energy
|analytic−AD|/|AD| over these directions was 1.43e-13 (moist) and 2.87e-14 (dry)
at 16 cells, and 1.88e-14 / 1.29e-14 at 32 cells. These are directional checks,
not an exhaustive componentwise gradient bound. All FD and Taylor rows are
retained. Some weak ice-direction responses suffer cancellation: for example,
the first 16-cell ice direction's best relative FD error over the sweep is
1.31e-6, despite much closer analytic/AD agreement. Do not select a favorable
epsilon to imply uniform accuracy; inspect complete curves before a paper claim.

C3 signed total work divided by the sum of absolute component total-energy work:

| Cells/layers | Smooth M-B | Sharp M-B | Reversed M-B | D-E |
|---|---:|---:|---:|---:|
| 16/3 | 6.59e-15 | 1.55e-14 | −1.79e-14 | −7.01e-14 |
| 32/4 | 1.58e-14 | 1.88e-15 | 9.22e-15 | 1.87e-14 |

The matrices retain internal, potential, kinetic and lid work separately,
including heat allocation residuals. Finite-step energy differences are also
stored, not identified with instantaneous closure. These differences subtract
planetary energies and are sensitive to float64 cancellation; no drift order or
physical timestep accuracy is established by one step.

C4 actual limited-face counts for the three production Euler evaluations:

| Cells/layers | Smooth | Sharp | Reversed |
|---|---|---|---|
| 16/3 | 0, 0, 0 | 31, 65, 95 | 17, 59, 80 |
| 32/4 | 0, 0, 0 | 51, 150, 206 | 44, 124, 192 |

All recorded Euler outputs matched the authoritative water ledger exactly.
Independent species incidence residuals, normalized separately by each initial
species' maximum node amount, were at most 3.60e-16 in these samples. Absolute
kg residuals and their scales remain available: planetary quantities make
small relative residuals compatible with tens of kg in the dry species. Zero
ice has a null relative denominator, not an invented scale. Observed fractions
were nonnegative, vapor/dry positive; sharp-case liquid and ice minima were zero.
These observations do not qualify arbitrary timesteps or all limiter branches.

Local directories `/tmp/vfs-1b-final16` and `/tmp/vfs-1b-final32` contain manifests,
metrics and raw arrays and are ephemeral. CI archives the 16/3 case as part of
`cpu-checks`; reproduce with the documented commands rather than relying on /tmp.

## Checks and limitations

* Installed-wheel portable suite: **548 passed, one CUDA skip**. Nine added tests
  verify observer equivalence/nonmutation, independent derivative references,
  corrupt work-decomposition detection, exact-ledger/count/beta/finiteness
  rejection, corrupted-flux residual sensitivity, and failed-run provenance.
* Both portable comparison tools pass. The C1 metrics remain bitwise equal to
  the local increment 1a metrics after sharing the manifest lifecycle.
* Existing bounded donor-subcycle scalar conversion warnings remain visible.
  Running the new measurement alone can emit a scalar-conversion warning at the
  JSON reporting boundary; no prognostic graph is detached to perform a step.
* No exact fixture rerun, accelerator, IMEX split/backward, long-run budgets,
  restart equivalence or timestep-backward campaign is claimed here.
* Hosted increment 1b validation is pending at initial publication. Increment
  1a's updated PR and push runs passed independently:
  [PR run](https://github.com/kjhall01/vfs-a/actions/runs/41122657847),
  [push run](https://github.com/kjhall01/vfs-a/actions/runs/41122650432).

Initial driver-only setup failures (NumPy coefficient handling and dry State
reconstruction) were corrected without changing the package. No numerical defect
was diagnosed. Prescribed transport and terrain-rest construction remain queued.
