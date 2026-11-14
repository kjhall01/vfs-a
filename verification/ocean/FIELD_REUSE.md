# Validated ocean field reuse

The ocean carries a checked physical reconstruction between consumers of the
same state inside one call. Every changed SSPRK stage and convex combination is
validated; standard boundary operators return their checked endpoint internally.
Velocity-only stress preserves temperature and salinity. Public state checks
still reconstruct independently. No state-dependent cache survives a public step.
Custom ocean subclasses retain the public composition path.

Layer transport indices, neighbor lists and node degrees are prepared once per
model. Connectivity and active-mask tensor identities and mutation versions are
checked before use. Inference tensors without mutation counters use the ordinary
uncached indexing path. Geometry/device/layer changes require a new model. Repeated
tracer gathers and signed flux arrays are shared locally. Neither flux formulas
nor the Hamiltonian, EOS, guards, fixtures or tolerances change. Prepared indices
retain memory proportional to edges × layers (plus cell-layer degrees); this trades
persistent geometry storage for repeated allocation and indexing work.

## Optional warm inversion

`StratifiedOcean(..., warm_start=True)` seeds each changed dynamics stage with a
previous checked temperature. The solver retains domain endpoint checks and the
original entropy residual threshold, `128*eps*(abs(entropy)+20)`. It Newton-polishes
an accepted guess (including an initially exact guess), checks the full reference
residual, and falls back to the established solver on failure. Target-entropy and
salt derivative tests cover exact initial guesses. The default remains `False`.
Warm starts are not applied to all forcing, remap or closure inversions.

Warm inversion is residual-equivalent, not bitwise-equivalent. Roundoff-sensitive
limiter diagnostics may change substantially even where the resulting state
changes only at roundoff. Report values in extensive joules also magnify tiny
specific-energy differences; endpoint/budget differences are reported separately.

## Measurement scope

Independent parent and candidate wheels are unpacked outside the checkout and
loaded in separate processes. Parent: `0d3c9c131d087be4ac138f5eea011067b22da0d1`
(PR #71). CPU float64, one Torch thread, no concurrent timing jobs. Each sample
warms an evolving trajectory, times a fresh trajectory, then independently checks
endpoint budgets and counts calls on another trajectory. Setup and diagnostics
are excluded from the timer. These short samples do not qualify long integrations,
Mac throughput, GPU execution, or full-model reverse-mode differentiation.

The full cases use the existing opt-in research fast mixing closure, not a new
supported closure. This pass does not address mixing substep requirements or
change the integrator/timestep. No atmospheric kernels change.

## Reproduction

Build both revisions as wheels and unpack them in distinct directories. Use the
same interpreter/dependencies for both:

```sh
python -I validation/performance/field_reuse.py \
  --package /tmp/parent-wheel --case ocean --cells 1024 --layers 64 \
  --steps 1 --output /tmp/parent.json
python -I validation/performance/field_reuse.py \
  --package /tmp/candidate-wheel --case ocean --cells 1024 --layers 64 \
  --steps 1 --output /tmp/candidate.json
# Repeat the candidate command with --warm for optional warm inversions.
```

Repeat with alternating order. JSON includes source hashes, configurations, call
counts, independent inventories and energy residuals. `.pt` captures contain every
state/report tensor for local comparisons. Oversized tensor captures are omitted
from the repository rather than split; every uploaded file is at most 128 KiB.

## Large-mesh dynamics measurements

At 1024×64, dt=60 s, one timed step per process (two samples):

| Path | Samples, s/step | Median, s/step | Parent / path |
|---|---|---:|---:|
| PR #71 parent | 14.047, 15.259 | 14.653 | 1.00 |
| Default reuse | 9.973, 9.139 | 9.556 | 1.53 |
| Reuse + optional warm inversion | 6.486, 6.357 | 6.421 | 2.28 |

The first large candidate sample preceded the final custom-`TKEOcean` subclass
fallback guard. That guard does not change the standard class path measured
here; both fingerprints are retained. After timings, a compatibility fallback was added for inference tensors without
mutation counters; timed standard geometry does not enter that branch. The full
installed-wheel suite was rerun for this final change. Source and wheel hashes
distinguish these revisions. These are exploratory timing
samples, not confidence intervals. No comparative timing job ran alongside the test suite. CLI smoke checks ran
with the suite and their timings are not used for speed comparisons.

Per dynamics step, physical reconstruction calls fall 13 → 6; Gibbs reference
evaluations fall 204 → 99 (49 with warm starts). JMD95 gradient evaluations stay
at three. The remaining mixing/forcing/remap inversions limit whole-model gains.
The benefit of prepared connectivity and repeated-array reuse is included in
these combined numbers; no isolated speedup is attributed to either component.

## Endpoint agreement and composed workloads

Final wheel comparisons match every captured default state/report tensor exactly:
122 tensors for 128×16 dynamics, 232 for full ocean, 280 for full+ALE, and 64 for
1024×64 dynamics. The small cases evolve two steps; the large case evolves one.
Warm-start endpoint relative L2 differences are at most 3.16e-15 across these
cases (TKE in full+ALE); large dynamics velocity differs by 1.81e-16 relative L2.
Thickness and salt agree exactly in all four comparisons. Small-case independent
energy-minus-prescribed-input residuals remain around 1e-11 J/kg or below. All
endpoint positivity checks and operator energy guards pass. These short smooth
trajectories do not establish long-run equivalence of warm and cold inversions.

Final 128×16 samples (two timed steps, seconds per step):

| Case | Parent | Default reuse | Warm reuse |
|---|---:|---:|---:|
| Dynamics | 0.566 | 0.234 | 0.168 |
| Full forced ocean | 2.505 | 1.683 | 2.021 |
| Full forced ocean + ALE | 2.271 | 1.957 | 1.824 |

Small timings are noisy: the final warm full-ocean sample is slower than default
reuse despite fewer evaluations. Preliminary samples, including a default ALE
sample slower than its parent, are retained rather than discarded. Use call
counts, repeated larger measurements and local hardware runs when deciding which
path to enable. These measurements do not justify a uniform whole-model speedup.

Raw timings and source hashes are in `field-reuse/samples/`; per-tensor exactness
and differences are in `field-reuse/comparisons/`, with endpoint-only summaries in
`comparison-summary.json`. Portable atmospheric fixture comparisons pass (142
corrected-Gibbs tensors and 93 dry-helper tensors). They retain historical portable
roundoff differences; no fixture or tolerance is changed.

## Regression validation

Final installed wheel, tested outside the checkout: **1074 passed, one CUDA skip,
one existing tensor-to-scalar warning**. The 13 new tests cover inversion domain
and residuals, exact-guess entropy/salt/guess derivatives, invalid inputs,
prepared/unprepared ledger equality, stale connectivity rejection, warm/cold
step agreement, state freshness, custom ocean/composition hooks, and inference
geometry fallback. Existing frozen sharp-front/coastal and step/remap regressions
remain unchanged. Both portable fixture comparisons pass. Three `--warm-start`
CLI smoke cases complete (dynamics, full, full+ALE); smoke timings are excluded
from throughput comparisons. Full-model backward, GPU and long-rollout behavior
remain unqualified.
