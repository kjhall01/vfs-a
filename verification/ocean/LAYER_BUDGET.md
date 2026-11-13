# Rejected-step local TKE budget diagnostic

This diagnostic observes the unchanged research solver from the conservative-path
work experiment. It captures the pre-mixing state at rejection, then replays that
same input with controlled solver settings. It does not change the supported
runtime, closure equations, energy allocation or acceptance tolerances.

## Ledger and reference

For layer mass M, specific TKE k, native momentum energy loss P, buoyancy work B,
dissipation coefficient d and time-integrated upward TKE flux F, the residual is

    R = M*(k-k0) - P + B + d*k - D(F).

Positive P and incoming flux fund TKE; positive B and d*k consume it. Negative B
releases energy into TKE. `D(F)` is incoming minus outgoing, with zero flux at the
top and bottom. Every ledger entry is in joules for the one-square-metre test
column. The acceptance scale, unchanged from the parent solver, is

    scale = M*1e-13 + 1e-9*(M*k0 + P + abs(B)).

The reported normalized residual is abs(R)/scale, with acceptance at <=1.
This is a discrete equation residual, not an error against an analytic ocean
solution. A small column-summed energy residual can conceal a local failure.
Both the signed direct solve (including negative candidates) and the positive
iterate are recorded. Neither is clipped by the observer.

The observer uses Python tracing around the existing `CoupledClosure.apply`
function. Regression tests verify exact successful-result agreement, signed
ledger accounting, transport telescoping, and rejected-call input preservation.
The trace retains the last 12 iteration ledgers. The compact evidence preserves
all final layer terms and scalar iteration trends; oversized raw traces are
excluded from upload.

## Reproduction

Use a noneditable installed wheel and copies of the validation scripts outside
the checkout:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python validation/ocean_column/diagnose_budget.py --output diagnostic.json --checkpoint rejected-mix.pt
python validation/ocean_column/summarize_budget.py --input diagnostic.json --output summary.json
python -I -m pytest tests/ocean/test_budget_diagnostic.py -q
```

The checkpoint is locally generated, trusted Python/Torch state. Do not load an
untrusted pickle. Reusing it skips the six-hour prefix. All replay cases use the
same pre-mixing state, so the half-dt case is only a mixing-operator sensitivity:
it does not constitute a forced trajectory with a smaller timestep.

Controls: unchanged 80-iteration accelerated solve, 320 accelerated iterations,
320 cautious iterations (relaxation .3, no acceleration), order-5 path quadrature,
and half mixing timestep. Input tensors are checked for exact preservation.

## Findings

The pinned input reproduces the rejection at six hours. Its largest positive-
iterate budget residual is in the third layer, centered at 15.625 m, not in the
almost exhausted layer that invalidates the direct solve.

| Third-layer term | J per 1 m2 column |
|---|---:|
| TKE storage change | 1.609951642e-5 |
| Shear/momentum energy production | 0.01102252909 |
| Buoyancy work | 0.01752843077 |
| Dissipation | 2.633529188e-6 |
| Net TKE transport into layer | 0.006524638240 |
| Signed residual | -3.514577087e-9 |
| Allowed residual scale | 6.693189959e-10 |

This residual is 5.25097 times the limit. It is still declining: normalized
residuals for iterations 75–80 are 10.786, 9.340, 8.087, 7.003, 6.064, 5.251.
The solve has not stagnated. Extending the iteration cap to 320 accepts at
**iteration 92**, without changing tolerances, positivity or physics.

The direct linear solve closes the frozen-coefficient ledger but gives
k=-6.06555e-36 m2/s2 in layer six, centered at 34.375 m. That single negative
entry invalidates the whole direct candidate. In that layer, buoyancy work and
incoming transport are both approximately 2.25046e-19 J, while starting TKE is
1.62806e-36 m2/s2. The sign of the tiny remainder is sensitive to the calculation.
The positive iterate there has negligible budget residual (about 1.04e-32 J).
This does not justify clipping a physical state or declaring the nonlinear
problem infeasible from the frozen signed solution.

## Fresh nonlinear residual versus the acceptance ledger

The parent solver checks its candidate against coefficients/work evaluated at
the preceding guess, and separately bounds the TKE iterate change. Its absolute
iterate tolerance (1e-12 m2/s2) can permit very large relative changes in weak-TKE
layers. Thus its accepted frozen-coefficient budget need not be the budget
obtained by **recomputing all transport, scalar mixing, buoyancy work and
production from the candidate end TKE**.

`probe_budget_candidate.py` independently recomputes that complete trial from
the same initial state for each accepted candidate. The acceptance scale formula
is unchanged, with newly evaluated P and B in the refreshed check.

| Replay | Iterations | Accepted frozen budget / limit | Refreshed budget / limit |
|---|---:|---:|---:|
| Order 3, cap 320, accelerated | 92 | 0.93315 | 0.76221 |
| Order 3, cautious | 185 | 4.81e-8 | 6397.93 |
| Order 5, accelerated | 80 | 7.49e-8 | 4688.18 |
| Order 3, half mixing dt | 65 | 7.45e-8 | 2183.41 |

The 92-iteration candidate passes both checks for this input. The other accepted
replays demonstrate that the existing iterate-change criterion does not ensure
end-coefficient consistency at the local energy-budget scale. These are not
new forced trajectories or comparisons of timestep accuracy.

A diagnostic candidate that retains the positive iterate only where the direct
candidate is negative closes the frozen budget (8.20e-8 times the limit), but
fails the refreshed check by 4688.18. Its largest refreshed residual is in layer
four at 21.875 m: 3.00338e-6 J. **A local splice is therefore not a sufficient
solver correction.** It is never committed as a physical update.

These refreshed residuals do not demonstrate an energy leak in the accepted
state: its actual full-H/entropy guards still run. They demonstrate incomplete
self-consistency of the intended end-TKE diffusivities and mixing work. A lagged
coefficient update may close its own energy ledger while differing from the
fully coupled backward-Euler equations.

## Next implementation target

Require a fresh end-state residual evaluation before acceptance, retaining full
H, entropy and positivity checks. Then improve the constrained nonlinear solve
using that residual, so weak-layer equations are not hidden by a global absolute
TKE-change scale. An adaptive extra-iteration allowance is a useful control,
not a general cure: higher quadrature order and cautious iteration already
produce frozen-budget acceptance with large refreshed residuals.

Qualify the resulting solver on this pinned input, the fixed one-hour matrix,
and longer trajectories. Compare timestep convergence and cost before supported
integration. No default, acceptance gate or physical source is changed here.

## Validation and evidence

Installed unchanged wheel outside checkout: **1020 passed, one CUDA skip**, one
existing tensor-to-scalar warning, 221.41 s. Final observer refinements (deduplicated
line events and successful-return capture) pass all **10 focused tests**, including
exact output/residual agreement and rejected-input preservation. Parent #63's
hosted installed-wheel CI passed. CPU float64 only; no GPU/backward qualification.

`layer-budget/summary.json` retains every final signed layer term, successful
candidate TKE, scalar convergence trends and a hash of the full local trace.
The compact summary deduplicates the initial observer's repeated line events.
`candidate-probe.json` retains the frozen/refreshed comparisons. All uploaded
files are below 128 KiB; no checkpoint pickle is uploaded.
