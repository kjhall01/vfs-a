# Retained atmospheric timestep paths

This is current guidance; pass reports and CI investigations describe historical
snapshots. Start from tensor states on model-owned spherical mesh/column geometry.
Terrain is fixed `column.phi_s`. Move geometry/model before initialization; use
explicit state conversion rather than casting inside a tendency.

| Path | Entry | Guarantees and limits |
|---|---|---|
| Dry conservative SSPRK3 | `ssprk3(DryHydrostaticConservativeDynamics(), state, dt)` | Matched semidiscrete energy/continuity; time integration is not exactly energy conserving. Unbatched, tracer-free, positive fixed lid, direct_u_gl2_adjoint. |
| Bounded moist SSPRK3 | `MoistAtmosphere(mesh, column, stepper=BoundedSSPRK3(release_as_heat=True))` then `model(state, dt)` | Bounded Euler species amounts are authoritative; three dt-sized Euler budgets and convex combinations. Positive vapor required, no floor. |
| Dry ARK4 | `DryPEIMEXARK(dynamics, solver_config).step_with_info(state, dt)` | Frozen analytic conservative wave plus nonlinear explicit residual. No scalar bound/SSP guarantee. |
| Moist ARK4 | `MoistAtmosphere(mesh, column, stepper=CenteredIMEX())` | Explicit opt-in to centered mixture transport; admissibility rejection, not repair or bounded transport. No bounded IMEX scheme. |

For the dry ARK4 path, import `DryPEIMEXARK, IMEXSolverConfig` from `vfs.imex`.
Use an explicit configuration:

```python
solver_config = IMEXSolverConfig(
    implicit_operator="gravity_wave", preconditioner="hydrostatic_schur",
    rtol=1e-11, max_iterations=100,
)
```

This is a documented tested selection, not a change to the solver defaults.
Moist CenteredIMEX's existing defaults are analytic gravity waves, Schur,
rtol=1e-9, 80 iterations, restart=30. ARK4 is the only retained implicit tableau;
ARK2/3 selectors and pressure-prototype classes were deliberately removed.
`CenteredSSPRK3` is no longer a model option; the centered moist RHS remains an
essential ingredient of bounded stepping and ARK4.

## Follow the execution

Dry explicit: [SSPRK3](../vfs/core/integrators.py) calls
[conservative tendency/_assemble](../vfs/atmosphere/conservative.py). First build
layer/interface pressure masses and continuity fluxes; then pair thermal energy
gradients with transport transposes for pressure work, and skew-pair rotation and
vertical momentum in the diagonal kinetic-energy metric. The three RK updates
act on pressure-weighted potential temperature, surface pressure and edge wind.

Bounded moist explicit: [model](../vfs/atmosphere/model.py) →
[stepper](../vfs/atmosphere/steppers.py) →
[bounded SSPRK3/Euler/evaluate](../vfs/atmosphere/bounded_moist.py).
See [the detailed bounded reading guide](BOUNDED_MOIST_READING_GUIDE.md).
Centered candidates, donor fluxes and antidiffusive limiter corrections are
necessary stages of one retained algorithm, not selectable alternative physics.
The Euler water update installs integrated species amounts directly; it must not
be reconstructed from `water + dt * tendency`. Gibbs/limiter work is allocated as
heat under the default configuration.

Dry ARK4: [dry_ark](../vfs/integrators/imex/dry_ark.py) `_step` freezes
[conservative_wave](../vfs/integrators/imex/conservative_wave.py), then loops over
the six [ARK4 stages](../vfs/integrators/imex/tableau.py). At anchor y0, solve with
L(y-y0) implicitly and F(y)-L(y-y0) explicitly. Schur eliminates wind and solves
the thermodynamic block; multilevel adds the existing geometric preconditioner
using [mesh transfers](../vfs/geometry/mesh/multilevel.py). Full residual checks
and IMEXConvergenceError remain authoritative for solver acceptance.

Moist ARK4: [moist_ark](../vfs/integrators/imex/moist_ark.py) `step_with_info` freezes
[moist_wave](../vfs/integrators/imex/moist_wave.py). Its analytic wave differentiates
mixture enthalpy/pressure transport at fixed entropy/water contents. Advection,
rotation and kinetic forces remain explicit. Each actual stage and final state
is validated; the validator sees outer dt, not an Euler limiter budget. No
implicit stage is silently converted to bounded transport.

## Evidence boundary

The merged-PR-8 capture uses the existing 16-cell, 3-level dry/moist test states,
CPU float64, dt=600 s, analytic waves, rtol=1e-11, max_iterations=100. Dry full,
Schur and multilevel Schur converged in 29, 13 and 23 total iterations; moist full
and Schur in 30 and 20. All stage states and residual/iteration records were
captured. This is representative regression evidence, not qualification of every
solver flag, timestep, resolution, batched run or precision.

Moist ARK4 already had independent DOP853 refinement, conservation and stage
rejection tests. Dry tests previously selected ARK2; they now exercise ARK4.
The tiny old refinement interval reached roundoff under ARK4, so the retained
independent check uses the moist test's 3600 s interval and 600/300/150 s steps.
Dry wind RMS errors were 3.0452565571e-8, 1.9060099373e-9, 1.1889634595e-10.
No IMEX backward, accelerator or full-rollout support is established here.
