# Timestepping

| Model | Stepper | Transport and limits |
| --- | --- | --- |
| `DryAtmosphere` | `DrySSPRK3()` (default) | Conservative unbatched, tracer-free dry core |
| `MoistAtmosphere` | `BoundedSSPRK3()` (default) | Three bounded Euler species-amount budgets |
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
