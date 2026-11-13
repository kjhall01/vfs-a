# Sustained column mixing and limiter sensitivity

**Campaign in progress.** Seven 48-hour cases are complete; the remaining
requested runs are still executing. Current evidence is partial, not a completed
convergence qualification.

This campaign evaluates the energy-limited TKE closure from draft #58. The
production algorithm's default numerical behavior is unchanged: this branch
adds tensor diagnostics and exposes the existing 0.5 trial-reduction factor as
an explicit validated sensitivity parameter. It does not add TKE forcing,
change mixing lengths, retune coefficients, alter the EOS, or weaken guards.

## Experiments

All cases request 48 h with active scalar mixing. The primary matrix is
16/32/64 layers × 300/150 s × wind/cooling/rain-heating (18 cases). Five additional
32-layer wind cases examine dt=75 s, initial k=1e-6/1e-3 m²/s², and trial-reduction
factors 0.25/0.75. Primary k=1e-4 m²/s², L=2 m, factor=0.5. All other conditions
match [the original campaign](COLUMN_CAMPAIGN.md). The
[runner and analysis instructions](../../validation/ocean_column/README.md)
reproduce the matrix. A case that rejects retains its partial trajectory and
is excluded from equal-duration final-profile comparisons.

Seed sensitivity changes a physical initial energy reservoir: the 100 m,
1025 kg/m³ column starts with 0.1025, 10.25 or 102.5 J/m² of TKE for
k=1e-6, 1e-4 or 1e-3 m²/s², respectively. Differences between seed cases are
therefore not automatically numerical errors; they test how much initial-state
memory remains under the same forcing.

The initial warm/fresh surface remains a centre-sampled profile. Cooling is a
surface forcing experiment, not an independently established convective benchmark.
The mixed-layer diagnostic uses a 0.2-degree surface temperature threshold and
must not be interpreted as density-based entrainment accuracy. There is no
horizontal dycore, ALE, bottom stress, atmosphere coupling or wave model here.

## What the limiter diagnostics mean

Pair counts include the even-half/odd-full/even-half sweeps. A limited pair has
accepted exchange coefficient below the original implicit-pair proposal. The
reported salt/enthalpy acceptance ratios sum absolute accepted transfers and
divide by the corresponding proposed transfers at the same actual stage.
These quantify suppression along the limited trajectory; they do not estimate
a separate unlimited run. Counts include active pairs with negligible scalar
contrast, so flux-weighted ratios are more useful for mixing strength.

Temperature/salinity comparisons use absolute thickness-weighted RMS differences.
Velocity errors use both components and the finer/reference norm. Fine fields
are conservatively averaged onto the coarse layer intervals; rain expansion is
included. Refinement changes the centre-sampled initial profile as well as the
numerics, so spatial differences do not establish formal convergence order.

## Regression and provenance

Installed-wheel CPU float64 suite outside checkout: 990 passed, one CUDA skip,
using the existing portable fixture mode. The separate final five-test column
harness run includes conservative-overlap analysis and exact checkpoint/restart
equivalence, including rejection of mismatched restart parameters.
Focused closure/harness tests: 32 passed. No fixtures or tolerances changed.
GPU behavior and derivatives at limiter switching points are unqualified.

The instrumented 16-layer/dt=300 s wind run exactly reproduces every final-profile
field of the prior 48-hour run from #58. See
[instrumentation equivalence](sustained-mixing/instrumentation-equivalence.json).
Default physical behavior is unchanged; nondefault reduction factors are
explicitly labelled sensitivity experiments, not revised defaults.

## Startup recommendation

See [the proposed surface TKE boundary flux](TKE_STARTUP_DESIGN.md). It would
supply an explicit external energy input to start wind-driven turbulence from
zero. It is not implemented here and does not resolve wind-free convective
onset or establish a calibrated mixing-length formulation.

## Execution interruptions

Seven final cases were preserved when the initial worker sessions disappeared.
Unfinished attempts without restart states were excluded from final-profile
comparisons and are recorded in
[the interruption manifest](sustained-mixing/interrupted-attempts.json).
The remaining cases were restarted with atomic hourly tensor checkpoints and
short execution segments; a test verifies exact restarted/uninterrupted output.
Execution interruption is distinct from physical guard rejection. Completed
cases are never shortened or substituted with an earlier transient.
