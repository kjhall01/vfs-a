# Integrated ocean evaluation — 2026-11-14

**Energy accounting is encouraging; spatial behavior of backscatter is not yet
qualified.** Seventeen integrations completed (50,400 steps, 58 aggregate model
days). Two deliberately excessive-timestep attempts rejected as intended.
No model numerical methods or coefficients were changed to rescue a run.

Model checkpoint: main `8415094fed295e9a36f2d85e67ae6f664bac7754` (PR #83).
The new driver uses the installed checkpoint package plus this checkout's Roquet
research modules. Per-campaign provenance records source hashes and environment.
These are small CPU float64, uncompiled screening runs, not reference-solution
validation or a demonstration of realistic ocean circulation. Concurrent runs
make their wall times unsuitable for performance comparisons.

## Completed experiments

All cases use eight layers unless specified. Channel counts include land: the
128-cell mesh has only 42 wet cells. R/G/B mean resolved, GM–Redi, and GM–Redi
plus instantaneous backscatter. All three retain reservoir decay to heat.

| Experiment | Cells | Days per variant | dt (s) | Variants | Final max speed (m/s) | Final H-minus-work error (J/kg) |
|---|---:|---:|---:|---|---|---|
| Uniform bathymetric rest | 128 | 2 | 120 | R | 3.29e-13 | 0 |
| Stratified bathymetric rest | 128 | 2 | 120 | R | 3.42e-4 | -3.37e-9 |
| Same, 16 layers | 128 | 2 | 120 | R | 9.86e-5 | -3.38e-9 |
| Channel | 128 | 10 | 120 | R/G/B | .314 / .313 / 3.697 | -5.18e-7 / -5.19e-7 / -5.60e-7 |
| Channel timestep refinement | 128 | 2 | 60 | R/G/B | .218 / .218 / 1.860 | -1.98e-8 / -1.98e-8 / -1.99e-8 |
| Channel mesh sensitivity | 256 | 2 | 60 | R/G/B | .187 / .187 / 3.648 | -5.60e-8 / -5.60e-8 / -5.68e-8 |
| Weak backscatter, nu=100 m²/s | 128 | 2 | 120 | B | .222 | -1.09e-7 |
| Strong backscatter, nu=1e8 m²/s | 128 | 2 | 120 | B | 2.632 | +3.12e-7 |
| Wind/heat-forced global ocean | 128 | 2 | 120 | R/G/B | .03651 / .03651 / .03665 | approximately -5.9e-9 |

Full endpoint numbers are in `metrics.json`; histories, accepted endpoint
states, initial states, mesh generators, and configuration are retained per run.
The maximum absolute final volume/salt/passive-inventory relative error across
completed runs is 4.43e-13. Local closure energy residuals are at most
6.20e-12 J/kg; paired backscatter power residuals at most 4.32e-23 W/kg.
Finite-step total-H error is distinct from these instantaneous/local identities.

## Findings

### Backscatter needs spatial-quality work

At two days on 128 cells with dt=60 s, backscatter increases peak speed from
0.218 to 1.860 m/s. The highest-energy approximately 1% of active edges contains
22.1% of native resolved KE versus 5.23% in the resolved baseline. Here there
are only 93 active edges, so this statistic means **one edge**. Its velocity
maximum is at 145.75° W, 40.87° N, in the channel interior. Its center distance
is 444 km but its dual face length is only 5.14 km. The curl-based length
proxy decreases from 124 to 34.7 km. These are warning signs of sensitivity
to the shortest mesh features, not evidence of useful mesoscale eddies.

Halving dt does not remove this behavior: at two days, dt=120 and dt=60 yield
peak backscatter speeds 1.859708 and 1.859698 m/s. Energy error falls from
1.10e-7 to 1.99e-8 J/kg. Thus timestep error in energy is not explaining the
large velocity amplification. At 256 cells the top approximately 1% holds
38.3% of KE versus 7.43% without backscatter, and peak speed rises to 3.65 m/s.
This is mesh sensitivity, not a formal convergence series.

Changing viscosity produces a large response. The weaker setting nearly
matches baseline; the stronger setting spreads activity differently. Neither
is a calibrated recommendation. A funded reservoir limits energy extraction,
but does not ensure that energy reaches physically useful spatial scales.
Next work should examine scale-selective/regularized forcing, compatible
small-scale dissipation, and sensitivity to mesh face lengths. Preserve the
instantaneous Hamiltonian pairing while evaluating those alternatives.

### Resting stratification is not exactly balanced

Uniform reference water stays at rest over smooth 800–1200 m bathymetry to
roundoff. With CT=10 and SA=35-.001*z, maximum spurious speed after two days is
0.342 mm/s with eight layers and 0.0986 mm/s with sixteen. Surface displacement
falls from 1.70 to 0.429 mm. This is consistent with decreasing vertical
pressure-gradient/truncation imbalance, but two layer counts are not a formal
order-of-accuracy claim. Keep this case as a separate balance diagnostic.

### Forced coupling works in a short integration

Wind work, surface/penetrating heating, bottom drag, vertical TKE mixing,
mesoscale transport, decay and backscatter run together for two days. Their
full energy-minus-external-work residual stays small. The ocean starts at rest,
is very coarse, and is nowhere near equilibrium. These runs establish short
integration compatibility, not credible gyres, overturning, or climate skill.
GM–Redi changes are modest over this duration; that does not establish that
GM–Redi is unnecessary. The channel is also only approximately balanced and
far too coarse to demonstrate resolved baroclinic eddies.

## Rejection behavior and next runs

At dt=400000 s, the resolved channel rejects carrier overdraw; the backscatter
variant rejects its source positivity bound. Both retain zero accepted steps,
unchanged initial/last-accepted arrays (checked exactly), and zero accumulated
work. No retry, clipping, or altered physics conceals these failures.

Use `validation/ocean_evaluation/README.md` for installation and higher-resolution
commands. Start at 1024×16 with matching ablations, then consider the supplied
16384 mesh. Timesteps in the examples are starting points, not qualified limits.
The principal next scientific test is a better-balanced, adequately resolved
channel with mesh/scale diagnostics, alongside the stratified-rest refinement.

Analysis-only changes were exercised by these campaigns and analyzers. The
checkpoint's prior 1217-pass/2-skip suite was not rerun for this driver-only work.
