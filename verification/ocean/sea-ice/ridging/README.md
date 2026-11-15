# Compression/ridging mechanics — 2026-11-14

Branch `feature/ice-compression-ridging`, based on stationary ice checkpoint
`fa1a370`. The [component guide](../../../../validation/sea_ice/RIDGING.md) specifies
the full state, extended common Hamiltonian, constitutive law and limitations.
No production package source or defaults change.

## Independent verification

Thirteen new tests (28 total ice tests) pass on CPU float64, PyTorch 2.15.0+cpu,
one thread. They cover:

- Independent derivatives of the actual full ocean/ice H contracted with all
  tendencies, coastal/open geometry, with zero/positive friction and viscosity.
  The full volume conjugates contain a large common latent datum; the roundoff
  bound uses absolute products before category-transfer cancellation. Separate
  stress-work identities use the smaller mechanical scale and close at roundoff.
- An omitted ridge PE term is detectable; heating is not a computed residual.
- Local volume conservation, nonnegative PE production and dissipation, decreasing
  category footprint under the isolated redistribution, exact dry-cell masks.
- Exact reduction to slab H for one fully covered category, increased gravitational
  energy at reduced footprint for the same mass, and one-sided category-birth PE.
- Rest, separating-cell inactivity, ice-free limit, rejection of invalid states
  and an excessive timestep without clipping or hidden retries.
- Unchanged ocean mass, salt, passive amounts, velocity, TKE and eddy stores;
  only surface thermal content changes during this isolated operator.
- Smooth-branch backward derivative versus finite differences; no forward AD.
- RK4 finite-step energy-error refinement, plus comparison between direct H
  subtraction and an independent algebraically factored difference.

## Experiments

All nine runs cover **1000 seconds of isolated stress relaxation and category
redistribution**, on a sphere of radius 100 km. They do not include advection,
wind, drag, ocean dynamics or freezing/melting. The initial ice velocity has
converging and separating regions. Ice categories initially occupy 0.70/0.15
of each wet cell at 0.8/4 m thickness. Ocean layers are 20/60/120 m; CT=0 is
chosen for sensitive tiny-work diagnostics, not phase-equilibrium realism.

For the 32-cell open case:

| Timestep | Final energy discrepancy, J/m² |
|---:|---:|
| 500 s | 3.94e-7 |
| 250 s | 2.19e-8 |
| 125 s | 1.33e-9 |
| 20 s | 8.29e-13 |
| 10 s | -1.99e-14 |
| 5 s | -1.11e-13 |

These are factored full-H endpoint differences, not accumulated report residuals
or corrections to the state. The two finest values are at subtraction roundoff;
they are not evidence of further convergence. Direct subtraction of full H has
~6.5e-8 J/m² resolution in this case because the initial datum is ~-3.55e8 J/m².
Both diagnostics are retained in every summary.

At dt10 s, ice KE decreases ~0.217844 J/m², ridge PE increases ~0.108922 J/m²,
and heat increases ~0.108922 J/m². Equal partition follows the specified
`friction=1`, not a prediction that nature partitions work equally. Cell ice
volume is bitwise unchanged in these particular runs; tests elsewhere allow
floating-point inventory roundoff. Maximum coverage remains 0.85.

Two coastal tests use dt20/10 s; final errors are 1.74e-12/8.49e-14 J/m².
The 128-cell open test at dt10 s has 2.87e-12 J/m² error. Across the nine cases,
the peak relative instantaneous stress-work residual is below 1.2e-16.
The small CPU experiments take roughly 6–8 ms per update including driver
work/diagnostics; this is not a measurement of full-ocean coupling overhead.

Each directory retains CSV trajectories and JSON metadata, including source
hashes and actual elapsed timings. Run commands are in the component guide.

## Interpretation

This demonstrates that a simple compressive constitutive law and thickness
redistribution can share an energy-consistent implementation. It does not
qualify sea-ice rheology, concentration evolution, realistic ridge heights,
angular momentum, climate behavior, GPU performance, or backward differentiation
through category birth.

Coverage decreases here because only the redistribution part runs; the missing
transport part would concentrate ice in convergence. This operator alone is not
a solution for transport overshooting 100% coverage. It also cannot further
ridge an exhausted thin category. The next integration must match transport,
ocean loading, reciprocal drag and phase momentum to the same H before applying
this alongside the ocean dynamics or old single-slab thermodynamics.

## Regression

Installed-wheel regression outside the checkout: **1255 passed, 2 skipped**
under the established portable fixture policy (200.15 s). The existing installed
wheel supplies VFS; current research modules/tests were explicitly copied into
that external harness. No packaged source changed. `wheel-tests.log` retains
the full output; `focused.log` retains the 28-test ice run. Frozen comparisons
pass for 142 corrected-Gibbs tensors and 93 dry helpers; JSON reports include
the installed package source path and numerical budgets. CPU float64 only.
