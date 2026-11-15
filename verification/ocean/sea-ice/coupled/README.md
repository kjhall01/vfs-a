# Coupled ocean–ice verification (2026-11-15)

The opt-in `OceanIce` wrapper runs the actual Roquet ocean dynamics with two-category
ice, wind and thermal forcing, transport/loading, compression, reciprocal drag,
and phase momentum transfer. [Equations and scope](../../../../validation/sea_ice/COUPLED.md).
No packaged source/default changed. The common Hamiltonian includes ocean thermal,
mechanical, TKE and mesoscale energy, ice latent/gravitational energy, category
subcell gravitational energy and ice kinetic energy.

## Reproducible integrations

CPU float64, one torch thread, Earth-radius mesh, three liquid layers (20/60/120 m),
SA=34 g/kg, initially at the empirical liquidus. Initial thin/thick ice fractions
are 0.5/0.1 modulated spatially, with thickness 0.3/1 m. Positive wind is eastward,
8 m/s at the equator. Cooling is -30 W/m² except the unforced and melt cases.
Coastal cases use static land and initially hemispheric ice coverage. The melt
stress test supplies +5000 W/m² to force complete disappearance; it is intentionally
unrealistic forcing for testing a difficult numerical transition.

All nine integrations complete: 2010 steps and 57 aggregate model hours. The
reported residual is `(H(t)-H(0)-accumulated external heat/work)/wet area`.
It is a finite-step error, not an instantaneous closure residual.

| Case | Cells | Hours | dt (s) | Peak energy residual (J/m²) |
|---|---:|---:|---:|---:|
| [pilot](pilot/summary.json) | 32 | 1 | 120 | 9.16302e-05 |
| [refine60](refine60/summary.json) | 32 | 1 | 60 | 1.10333e-05 |
| [refine30](refine30/summary.json) | 32 | 1 | 30 | 1.1853e-06 |
| [day32](day32/summary.json) | 32 | 24 | 120 | 0.00182056 |
| [coast-patch](coast-patch/summary.json) | 32 | 6 | 120 | 0.000468537 |
| [coast128](coast128/summary.json) | 128 | 6 | 120 | 0.000493653 |
| [unforced](unforced/summary.json) | 32 | 6 | 120 | 5.26278e-07 |
| [melting](melting/summary.json) | 32 | 6 | 120 | 0.000187125 |
| [melting60](melting60/summary.json) | 32 | 6 | 60 | 2.31891e-05 |

Water and salt relative residuals stay below 1.10e-15 and 1.19e-15 respectively.
Both melting runs end with exactly zero ice stores. Energy errors decrease with
timestep refinement in these cases; that is not a proof of third-order coupled
trajectory convergence (phase/drag splitting is second order on smooth branches).
CSV files retain each step, and JSON summaries record forcing, timing and hashes
of the sea-ice source files. The git checkpoint supplies the rest of the source.

Observed time is about 0.08–0.11 s/step for these very small grids; runs shared the
host with verification jobs, so these are run timings, not an overhead benchmark
or a high-resolution performance estimate. No GPU/compile/float32 claim.

Example reproduction from the repo root with VFS installed:

```sh
python -I validation/sea_ice/run_coupled.py --hours 24 --output coupled-day
python -I validation/sea_ice/run_coupled.py --cells 128 --hours 6 --coast --patch --output coupled-coast
python -I validation/sea_ice/run_coupled.py --hours 6 --heat 5000 --dt 60 --output coupled-melt
```

## Independent checks and development findings

The final focused suite has **66 passing tests**, including 38 new coupled tests.
These independently contract analytic tendencies with autograd derivatives of
full H, check the reconstruction transpose, reciprocal edge momentum, nonnegative
drag TKE, phase derivatives and actual phase energy changes, open water, coasts,
new-category births, exhausted thin ice, complete melting, existing mixing and
backscatter composition, and selected smooth-branch backward sensitivities.
Forward tendencies do not use autodiff.

The aggressive melting case was necessary: early explicit drag became unstable
as ice vanished. Midpoint implicit drag with a diagonally preconditioned Newton/CG
solve resolves that stiffness. A raw inverse-mass PV rotation also produced a
margin velocity spike and large budget error; bounded harmonic-mass antisymmetric
rotation replaces it. The same masked reconstruction is used in instantaneous
and implicit drag, preventing fictitious work from ice-free edges. These are
explicit changes to the new research coupling, not adjustments to the existing
ocean kernels. Invalid stages still raise; no clipping, mass floor, hidden retry,
or post-step energy repair is used.

## Regression and limits

Frozen portable comparisons pass: 142 corrected-Gibbs tensors and 93 dry helpers;
`baseline.json` and `dry-helpers.json` identify the installed package and numerical
budgets. The unmodified exact fixture check fails at platform-level roundoff in
its initial state, as already documented for this environment; the exact fixture
and tolerances were not changed. `initial-exact-suite.log` records that run.

The full portable suite passed **1292 tests, with 2 skipped** (202.83 s).
The final partial-ice-margin drag correction and its additional regression test
were then covered by the **66-test final focused suite** (4.68 s), bringing
combined distinct passing coverage to 1293 tests. No frozen tolerance changed.

The external harness explicitly copies research modules/tests while importing
VFS from its installed wheel outside the checkout. The full portable-suite log
and final focused-suite log are retained alongside these reports.

This establishes numerical coupling in small idealized examples, not a validated
sea-ice climate. Shear/elastic rheology, snow, brine, internal ice enthalpy,
calibrated category physics and dynamic wetting/drying remain outside scope.
Energy consistency alone does not establish correct ice drift or a complete
Hamiltonian Poisson structure. Long runs and realistic forcing still need evaluation.
