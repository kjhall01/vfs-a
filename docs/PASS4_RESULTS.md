# Pass 4 — physical-field workflow

Added a small MoistAtmosphere facade using composition around the existing
MoistConservativeDynamics and BoundedGibbs. It holds mesh/column, constructs the
existing MoistState from physical fields, projects edge-sampled east/north wind,
recovers interface fields, advances bounded SSPRK3 and returns tensor budgets.

No numerical kernel file was changed. Only root/atmosphere exports and local
package version changed in existing package sources. New implementation:
vfs/atmosphere/model.py. Guide: PHYSICAL_FIELDS_API.md.
Runnable example: examples/moist_atmosphere.py.

## Validation

Installed wheel tested outside checkout, PyTorch 2.15.0+cpu, CPU float64.
78 tests passed in 23.47 s (see tests_pass4.log).
142 checkpoint-54 tensors match exactly (equivalence_pass4.json).
The suite also compares 193 dry helper fixture tensors exactly and checks all
99 extracted dry helper ASTs. Thus all 335 existing fixture tensors still match.

New tests cover physical-field round trips, exact agreement with manual state
construction and the underlying bounded step, wind projection, invalid shape/
dtype/nonfinite/physical inputs, explicit rejection of external numpy arrays,
geometry mismatch, zero condensates, zero-vapor rejection, and finite-difference
agreement of energy gradients with respect to temperature, pressure and vapor.

The 64-cell, 6-layer example completed 10 steps of 30 s:
- Relative total mass change: -6.217176653065638e-16.
- Species inventory changes: [-24, -1, 0] kg, on planetary total inventories.
- Energy change: -1.542650926958044e-5 J/m².
- Minimum vapor/liquid/ice: [0.009999999995771525, 1.4474766504931322e-50, 0].
These are illustrative diagnostics, not new timestep guidance or a balanced
benchmark; no scientific acceptance thresholds are asserted.

## Existing limitation exposed

The first new dry-limit test failed because the unchanged moist adapter
requires vapor strictly greater than zero. Initialization now requires an
explicit vapor input and rejects zeros early with an explanatory error.
No floor, numerical change or automatic dry-core substitution was introduced.
Exact-zero-vapor dry equivalence has NOT been established. Liquid/ice zeros
are supported. This remains a separate numerical question for later work.

## Scope

The model is a thin Python facade, not an nn.Module with automatic device moves.
Autograd is retained in tensor initialization and diagnostics; this pass does
not qualify full backward rollouts, GPU behavior or every limiter branch.
Temperature/composition must already be at CP interfaces; winds at layer-center
edges. No interpolation, regridding or phase equilibration is hidden.

Xarray/DataArray adapters, output/restarts and the friendly dry facade remain
future work. Legacy optional modules still ship. This snapshot is suitable for
moving development to GitHub; public release metadata/license remain undecided.
Original checkpoint 54 and passes 1–3 are unchanged.
