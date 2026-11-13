# Ocean mixing cost and direct energy prediction

## Scope and conclusion

This is a two-hour CPU/float64 cost/accuracy experiment, following the 48-hour
[sustained campaign](SUSTAINED_MIXING.md). It does not qualify a larger timestep,
entrainment, a global ocean, GPU throughput, long-run behavior or end-to-end
backpropagation. The 18.75-second comparison is a numerical reference, not an
analytic solution or a demonstrated converged limit.

The supported closure reuses sweep-invariant thermodynamic calculations without
changing the numerical update. A separate research predictor reduces the cost
of finding an affordable exchange. It does **not** resolve timestep sensitivity.
The predictor remains in `validation/ocean_column`, outside the installed package,
and is not the default closure.

## What changed

Within each disjoint pair sweep, the original-state enthalpy-to-temperature
inversion, entropy evaluation, active-layer mask, geometric allocation weights
and available pair budget are independent of the trial coefficient. They now
run once per sweep. In particular, the original inverse/evaluate round trip is
retained: replacing it with the original entropy would change cancellation at
very small exchanges. No state-dependent cache persists across calls.

The private `_mixing_proposal` hook permits the experiment to choose an initial
coefficient between zero and the original implicit exchange. The default returns
the original coefficient. Nonfinite, negative and excessive proposals reject.
All proposals still undergo the original full nonlinear Hamiltonian work check,
local TKE affordability check, entropy check and backtracking fallback.
Defaults, fixtures, tolerances and physical source terms are unchanged.

## Analytic predictor

Let c [m] be an exchange coefficient; it transfers c times the lower-minus-upper
salinity and reference potential-enthalpy contrasts. For a layer,

    dtheta = (dh0 - h_s dS) / h_theta
    dPsi = a dh0 + b dS
    a = Psi_theta / h_theta
    b = Psi_S - a h_s

Here Psi and its derivatives use the same vertical JMD95 quadrature as the
Hamiltonian. The reference enthalpy contributions cancel in the pair total.
Therefore, at zero exchange,

    dW/dc = rho0 A [(a_upper-a_lower) Delta(h0)
                    + (b_upper-b_lower) Delta(S)].

With alpha = h_upper/(h_upper+h_lower), the pair can spend at most

    B = min(R_upper/alpha, R_lower/(1-alpha)),

where each R is its current local TKE reservoir in joules. When the linear
prediction exceeds B, the research trial is

    c_trial = min(c_implicit, margin B / (dW/dc)).

Nonpositive linear cost leaves the implicit proposal unchanged. The derivative
is only a predictor: nonlinear EOS work need not be monotone or linear, and
roundoff matters near exhausted reservoirs. Actual work always decides acceptance.
No forward autograd, energy floor or nonlocal borrowing is introduced. Independent
autograd tests check the analytic slope for stable and unstable unequal layers.
The margin is a numerical research parameter, not a physical mixing efficiency.

## Cost evidence

The initial 15-minute, 32-layer profile spent 3.32 of 3.84 seconds in reference
thermodynamic evaluation (inclusive timings). The initial baseline made 35 full
exchange evaluations. With cached baselines, the conservative margin-0.5 predictor
made 18; the aggressive margin-0.95 predictor made 46. Attempting to spend nearly
all predicted available TKE can trigger extra fallback trials.

The fixed-input benchmark uses identical 32-layer inputs, a 150-second update,
one CPU thread, one warmup per method, and five interleaved timed repetitions.
It includes `mix_with_report` validation and diagnostics. The 128-column case uses
a native spherical mesh; it is a frozen-input microbenchmark, not a trajectory
or a forecast of whole-model throughput. Other experiments shared the host.

| Columns | Parent median (s) | Cached default (s) | Predictor, margin 0.5 (s) | Predictor reduction vs parent |
|---:|---:|---:|---:|---:|
| 1 | 0.4808 | 0.4517 | 0.3776 | 21.5% |
| 128 | 1.5696 | 1.5221 | 1.3765 | 12.3% |

The cached default matches the parent exactly in all 31 state/report comparisons
for each mesh. Its complete two-hour column profiles and limiter totals also
match exactly. The predictor is a numerical change and is not claimed bitwise
identical. See [raw benchmark](mixing-cost/benchmark.json) and profiler text files.
Campaign wall times are retained, but concurrent workload variation makes them
unsuitable for precise speed ratios.

## Accuracy evidence

All trajectories use the existing 100 m, 32-layer f-plane column and prescribed
forcing definitions from [the column harness](../../validation/ocean_column/README.md).
Wind runs compare 150, 75, 37.5 and 18.75 seconds. Positive initial TKE is 1e-4
m2/s2; prescribed mixing length is 2 m. Predictor margins 0.5, 0.8 and 0.95 are
numerical sensitivity probes. Cooling and rain/heating have two-hour 150-second
smokes for both default and margin-0.5 predictor, not separate convergence studies.

Against the **default method's** 18.75-second endpoint:

| Method | dt (s) | Velocity relative L2 (%) | TKE relative L2 (%) | Temperature RMS (deg C) |
|---|---:|---:|---:|---:|
| Cached default | 150 | 4.2126 | 23.8620 | 0.005365 |
| Predictor, margin 0.5 | 150 | 4.2126 | 23.8620 | 0.005330 |
| Cached default | 75 | 1.7579 | 10.7389 | 0.002928 |
| Predictor, margin 0.5 | 75 | 1.7579 | 10.7389 | 0.002927 |
| Cached default | 37.5 | 0.5805 | 3.6583 | 0.000797 |
| Predictor, margin 0.5 | 37.5 | 0.5805 | 3.6583 | 0.000796 |

The predictor buys cheaper updates, not permission to increase dt. Final analysis
also records successive timestep differences, each method's own fine reference,
same-dt method differences, temperature maxima and local weak-TKE comparisons.
Bulk norms do not establish local diffusivity/entrainment insensitivity. At
150 seconds, the margin-0.5 predictor differs from the default by only 0.00205%
in bulk TKE, but the 15–30 m layer TKE ratios range from 0.812 to 1.200. Margins
0.8 and 0.95 have much larger relative discrepancies in some weak-TKE layers.
These local differences are retained in the analysis; the predictor is not a
replacement for the unresolved local mixing qualification.

## Next numerical experiment

Keep expensive subcycling as a short reference only. The next candidate is one
predictor/corrector iteration for the coupled momentum/TKE/scalar update:

1. Compute an uncommitted provisional step to estimate end-of-step TKE.
2. Form nonnegative time-centred diffusivity coefficients from initial and
   provisional TKE, then redo the update **from the original state**. Reuse the
   affordable-exchange predictor, but retain all actual work and entropy checks.
3. Commit only the corrected candidate and its ledger. Never add both passes'
   production or dissipation to the physical budget.

This costs about two closure solves per step. It is worthwhile only if a larger
step reaches the same profile accuracy with lower total runtime. Compare a fixed
number of corrections against the short reference; do not claim second order or
convergence merely because a corrector is present. An unstable or unconverged
iteration must remain explicit in the evidence. Shared edge momentum and cell
TKE must use synchronized coefficients; independent per-cell subcycling is not
safe at shared edges.

Separately, thermodynamic inversions and repeated physical-state validation are
still dominant costs. Passing already-validated physical quantities through
private routines or a safeguarded warm-start inversion deserves its own numerical
and derivative regression. Changing inversion tolerances to obtain speed is not
part of this experiment. No surface TKE source is added to compensate for numerical
sensitivity.

## Reproduction and validation

See [commands](../../validation/ocean_column/README.md#mixing-cost-experiment).
Results, profiler call counts and source hashes are in `mixing-cost/`.

All 18 two-hour cases completed without physical rejection.
Maximum absolute energy-minus-input residual was 8.55e-11 J/kg;
maximum salt/passive relative changes were 1.3e-16
and 3.55e-16. These are finite-step budget checks,
not accuracy measures. All individual evidence files are below 128 KiB.

The final installed-wheel portable suite passed **998 tests, one CUDA skip**;
35 focused closure/predictor checks passed. Corrected-Gibbs and dry-helper portable
fixture comparisons passed. Earlier test setup failures (missing copied examples
and a target installation outside the interpreter's purelib directory) are retained
in separate logs; the final complete run used a normally installed wheel outside
the checkout. Runtime source bytes match the separately installed campaign wheel.
