# Timestepping

| Model | Stepper | Transport and limits |
| --- | --- | --- |
| `DryAtmosphere` | `DrySSPRK3()` (default) | Conservative unbatched, tracer-free dry core |
| `MoistAtmosphere` | `BoundedSSPRK3()` (default) | Three bounded Euler species-amount budgets |
| `DryAtmosphere` | `DryIMEX()` | ARK4 analytic wave split with residual-checked solves |
| `MoistAtmosphere` | `CenteredIMEX()` | ARK4 centered mixture transport; no bounded-species guarantee |
| `StratifiedOcean` | Checked sharp-front SSPRK3 | Entropy/salt work pairing; optional explicit post-step ALE |
| `Ocean` | `ssprk3` (default) | Behavior depends on the supplied dynamics and sources |

Atmospheric models advance through `model(state, dt)` or `model.step(state, dt)`.
Ocean facades use `model.step(state, dt)`. Timesteps are seconds. Invalid states
are rejected rather than clipped; no universal timestep is prescribed.

## Dry atmosphere

`DryAtmosphere` calls its stepper and conservative dynamics. Continuity advances
surface pressure and determines vertical pressure-coordinate mass transport.
Analytic derivatives of thermal and kinetic energy supply matched pressure,
transport and momentum work. The energy diagnostic includes the fixed-lid
reservoir. SSPRK3 introduces finite-timestep error even when instantaneous work
closes. [Source](../vfs/atmosphere/conservative.py).

## Moist atmosphere

`MoistAtmosphere` delegates bounded steps to `BoundedGibbs`. Each Euler stage
constructs carrier/species budgets, limits face transfers and pairs their energy
work with momentum and intrinsic heating. Water contents are installed from the
integrated amount ledger, not reconstructed by subtracting large tendencies.
The three Euler maps use the full timestep and are combined in SSPRK3 order.
See the [detailed reading guide](BOUNDED_MOIST_READING_GUIDE.md).

## Atmospheric IMEX

ARK4 treats an analytic wave linearization implicitly and the nonlinear residual
explicitly. The linearization is fixed at the start of each outer timestep. Full
stage residuals determine convergence; failure raises `IMEXConvergenceError`.
Solver records are available through `model.stepper.solver.last_step_info`.

```python
from vfs import DryIMEX
from vfs.imex import IMEXSolverConfig

stepper = DryIMEX(solver_config=IMEXSolverConfig(
    preconditioner="hydrostatic_schur", rtol=1e-11, max_iterations=100,
))
```

Use a fresh stepper when constructing each atmospheric model. Moist IMEX uses
centered transport with stage admissibility checks; it is not an SSP or bounded
transport extension. Long backward trajectories and arbitrary solver settings
are not qualified. See [moist configuration](MOIST_MODULE_API.md).

## Stratified ocean

`StratifiedOcean.step_with_report` returns the new entropy state and three stage
work ledgers. Each stage applies the Gibbs/entropy work pairing and local front
correction, then checks physical admissibility. An optional `target_h` requests
a remap after the step; otherwise the layers remain material layers.

The remapper conservatively projects potential enthalpy, salt, passive contents
and edge momentum, then adjusts layer temperatures to restore the column
Hamiltonian subject to an entropy check. Remap energy closure does not compensate
for timestep drift. See [ocean API and limits](OCEAN_INTEGRATION.md).

## Forcing and diagnostics

[Held–Suarez forcing](HELD_SUAREZ.md) is explicitly summed into each dry SSPRK3
stage. `Ocean` sums supplied source tendencies; `StratifiedOcean` currently has
no source interface. Diagnostics report tensor budgets. Neither a successful
example nor a converged implicit solve establishes physical accuracy; see
[verification](../verification/README.md).
