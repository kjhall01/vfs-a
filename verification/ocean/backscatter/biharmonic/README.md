# Budgeted rotational biharmonic sink — 2026-11-14

The optional sink removes resolved KE through a compatible squared rotational
Laplacian and deposits its nonnegative dissipation in the eddy reservoir. It
reduces the short-edge KE concentration exposed by the integrated evaluation,
while retaining the full instantaneous Hamiltonian pairing. No clipping,
finite-step energy repair, or secondary energy functional was introduced.

## API and accounting

```python
closure = EddyBackscatter(
    viscosity=1e4, timescale=86400., fraction=.25,
    biharmonic_viscosity=1e13,  # m^4/s; experimental coefficient
)
model = RoquetOcean(mesh, column, backscatter=closure)
```

The default `biharmonic_viscosity=0` retains the existing numerical path. Sink
only is available with `viscosity=0`. For the current stage, with the kinetic
metric M and masked, thickness-weighted positive rotational Laplacian L,

    F4 = -nu4 L(Lu),
    <u,F4>_M = -nu4 <Lu,Lu>_M.

The positive quadratic loss is allocated to adjacent cells with the existing
interpolation transpose and added to their reservoir tendencies. Signed local
mechanical work also includes viscous energy redistribution. That redistribution
is explicitly reported and sums to zero; it is not a reservoir debit or an
unreported correction. Thus local accounting includes transport and the global
full-H derivative vanishes to roundoff. Coasts are projected between both
Laplacian applications. The sink does not require a nonempty reservoir.

See `validation/temperature_coordinate/BACKSCATTER.md` for the full identities,
report fields, and the separate explicit timestep restriction. No automatic
coefficient calibration or stiffness cap is provided. This is rotational
biharmonic damping, not a complete strain-stress viscosity or a guarantee of
angular-momentum conservation.

## Verification

- Installed wheel outside the checkout: **1227 passed, 2 skipped**, CPU float64,
  portable fixture policy; full log retained. Research modules were explicitly
  copied into the external test harness; the installed package supplied VFS.
- Both frozen comparisons pass: 142 corrected-Gibbs tensors and 93 dry helpers.
- 90 old/new tensors are bitwise identical with the sink disabled, across six
  coastal/open and limited/unlimited/zero-viscosity cases, including coupled
  steps. Reference revision is recorded in `default-equivalence.json`.
- New independent full-H autodiff contractions test pure sink and the combined
  closure over variable thickness with/without coasts. Forward code never
  calls autodiff. The four recorded contraction residuals are <=5.55e-29 W/kg.
- Positive reservoir deposition, coastal masks, empty reservoirs, finite stencil
  locality, backward scalar sensitivity and all three coupled RK stages pass.
- Independent matrix eigenmodes confirm sink damping proportional to squared
  discrete eigenvalue, rather than merely testing the implementation twice.

For an isolated combined closure over 400 s, with nu4=1e17 m^4/s:

| dt (s) | KE + reservoir error (J/kg) |
|---:|---:|
| 100 | 5.2703e-12 |
| 50 | 5.2689e-13 |
| 25 | 5.7987e-14 |
| 12.5 | 6.5708e-15 |

These changing terms belong to the same full H; unchanged thermal/PE terms
are excluded only to avoid subtractive cancellation in this diagnostic.
Refinement approaches SSPRK3 behavior in this smooth test. Finite timesteps
are not exactly energy conserving.

## Coupled two-day experiments

All six cases below completed (14,400 timesteps). Final reservoir minima stayed
positive, and volume/salt/passive-inventory relative errors were <=1.75e-13.
Channel uses eight layers, dt=60 s unless noted,
and the same deterministic initializations as the prior integrated campaign.
Unchanged reference runs remain under `verification/ocean/integrated`.

| Channel mesh | Configuration | Max speed (m/s) | KE in highest-energy ~1% of edges |
|---|---|---:|---:|
| 128 | GM–Redi | .2181 | 5.23% |
| 128 | GM–Redi + backscatter | 1.8597 | 22.11% |
| 128 | Both + sink nu4=1e13 | .2160 | 5.24% |
| 128 | Both + sink nu4=1e14 | .2082 | 5.39% |
| 256 | GM–Redi | .1870 | 7.43% |
| 256 | GM–Redi + backscatter | 3.6479 | 38.26% |
| 256 | Both + sink nu4=1e13 | .1936 | 7.58% |

At 128 cells that concentration statistic denotes one of 93 active edges.
The curl-based length proxy grows from 34.7 km with backscatter alone to
127.0 km with the sink, near 124.1 km without backscatter. The speed maximum
moves away from the previously problematic 5.14 km dual face.

The sink does not simply disable energy return. At 128 cells/nu4=1e13,
backscatter supplies 2.0948e-4 J/kg and the sink returns 3.3663e-5 J/kg to the
reservoir: net positive transfer into resolved motion. Resolved KE is
.002213 J/kg versus .002027 with the sink alone (whose max speed is .2063 m/s).
This is encouraging spatial behavior, not validation of mesoscale statistics.

The 128-cell energy error is -1.9922e-8 J/kg. Repeating at dt=120 s gives
-1.1042e-7 J/kg and nearly identical peak speed (.21598698 vs .21598711 m/s).
The 256-cell error is -5.6814e-8 J/kg. A two-day 128-cell forced global run
with wind/heat, TKE mixing, GM–Redi, decay and both momentum terms completes
with -5.8671e-9 J/kg energy-minus-work error and max speed .03665 m/s.

Initial/accepted arrays, streaming CSV, provenance and summaries are retained
for all six runs. `metrics.json`, `grid_metrics.json`, and `comparison.png`
provide aggregated evidence. The first three runs began before the driver
added its global-power check to the maximum diagnostic; their full energy
histories and the independent contraction tests still cover global closure.

## Reproduce

```sh
python -m pip install .
python -I validation/ocean_evaluation/run.py --case channel \
  --cells 128 --layers 8 --days 2 --dt 60 --variants backscatter \
  --biharmonic-viscosity 1e13 --output channel-with-sink
python -I validation/ocean_evaluation/grid_diagnostics.py channel-with-sink
python -I validation/temperature_coordinate/backscatter_campaign.py \
  --mode refinement --biharmonic-viscosity 1e17 --output refinement.json
```

For a larger test, increase cells/layers while retaining matched no-sink and
no-backscatter comparisons. The example coefficient and timestep are not
qualified at arbitrary resolution. No GPU, compiled, long-run climate,
resolved eddy spectrum, or calibrated scale-aware coefficient claim is made.

## Measured cost

Separate sequential processes, 16384 cells × 40 layers, one CPU thread, eager
float64, three samples after warmup on the same wave initialization:

| Measurement | Backscatter alone | Backscatter + sink |
|---|---:|---:|
| Median isolated tendency | .4311 s | .6390 s |
| Median complete SSPRK3 model step | 15.7968 s | 16.8242 s |
| Full-process peak RSS | 2.684 GB | 2.622 GB |

The measured full-step increase is about **1.03 s (6.5%)**. Three samples are
noisy (15.40–16.27 s without the sink, 15.91–16.88 s with it); do not interpret
this as a precise throughput guarantee. Peak RSS is comparable, and the small
decrease is not evidence of a memory optimization. Isolated-process peak RSS
increased from .780 to .849 GB. These figures are not additive to model RSS.

Reproduce with `backscatter_campaign.py --mode time` or `--mode step-time`,
`--cells 16384 --layers 40`, and `--biharmonic-viscosity 0` versus `1e13`.
Full-step timing uses a fixed initial state and dt=1 s, not a long integration
or a high-resolution timestep qualification. No other evaluation simulations
were running during these measurements.
