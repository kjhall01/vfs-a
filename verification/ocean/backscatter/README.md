> Historical finite-step prototype, superseded by the [instantaneous closure](instantaneous/README.md).

# First deterministic backscatter verification

Reference defaults: commit `9dce3b5`. See [equations and usage](../../../validation/temperature_coordinate/BACKSCATTER.md).

This opt-in research addition uses the existing mesh curl/adjoint and the
mechanical metric inside the full Roquet Hamiltonian. It caps local reservoir
withdrawals by an explicit fraction/timescale and accounts for finite quadratic
kinetic work. A common amplitude per layer guarantees local affordability but
can be restrictive when a cell is depleted. No production defaults change.

Focused tests cover open/coastal geometries, active and inactive funding caps,
large timesteps, patchy reservoirs, zero energy, zero flow, disabled coefficients,
invalid inputs, independent KE differences, direct full-H comparisons, unchanged
tracers/TKE, the positive-work adjoint identity, and backward finite differences
in both limited and unlimited regimes. Forward calculations do not use autograd.
Repeated composition includes GM, Redi, backscatter and reservoir decay-to-heat.

All 486 existing default-path tensors (states, tendencies, full ledgers and
backward sensitivities) match the reference bitwise. Frozen-fixture comparisons
cover 142 corrected-Gibbs and 93 dry-helper tensors. No fixtures or tolerances
were changed. Tests use the installed wheel outside the source checkout, with
an explicit copy of the research modules; these modules are not package exports.

`budgets.json` records 20-step channel experiments with weak and strongly
funding-limited coefficients, both with and without resolved dynamics. The
finite-work local residual is the precise exchange diagnostic. Direct total H
subtraction also includes cancellation against the large thermal datum; the
resolved dynamics has its own time-discretization drift.

This is CPU float64, short-step verification, not long-run stability, spectral
fidelity, spatial convergence, angular-momentum conservation, or GPU/compiled
qualification. In particular, negative viscosity can energize grid-scale noise;
this pass does not implement the full biharmonic/scale-aware literature closure.

The maximum local backscatter residual across the four channel cases is
**7.106e-18 J/kg**. The most depleted reservoir remains **0.0999219 J/kg**,
from an initial 0.1 J/kg, with positive cumulative resolved KE gain in every
case. These short runs are energy-budget checks, not calibration.

Installed-wheel portable regression: **1218 passed, 2 skipped** in 201.10 s,
including 15 new focused backscatter cases. Both frozen comparisons pass.
Full test log, wheel/research source hashes, raw channel budgets and default
path equivalence are retained here.

At 16384×40, single-thread eager CPU float64, the isolated backscatter stage
has median **0.4755 s** (samples 0.4755, 0.5298, 0.4619 s; one warm-up).
This is about 3.2% of the preceding 15.063 s dynamics median, an estimate of
incremental cost rather than a timed combined step. Isolated-process peak RSS
is 0.802 GB, including mesh/runtime; it is not an additive full-model RAM cost.
