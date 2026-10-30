# Retained atmospheric timestep paths

This is current guidance; pass reports and CI investigations describe historical
snapshots. Start from tensor states on model-owned spherical mesh/column geometry.
Terrain is fixed `column.phi_s`. Move geometry/model before initialization; use
explicit state conversion rather than casting inside a tendency.

| Path | Entry | Guarantees and limits |
|---|---|---|
| Dry conservative SSPRK3 | `DryAtmosphere(mesh, column, stepper=DrySSPRK3())` then `model(state, dt)` | Matched semidiscrete energy/continuity; time integration is not exactly energy conserving. Unbatched, tracer-free, positive fixed lid, direct_u_gl2_adjoint. |
| Bounded moist SSPRK3 | `MoistAtmosphere(mesh, column, stepper=BoundedSSPRK3(release_as_heat=True))` then `model(state, dt)` | Bounded Euler species amounts are authoritative; three dt-sized Euler budgets and convex combinations. Positive vapor required, no floor. |
| Dry ARK4 | `DryAtmosphere(mesh, column, stepper=DryIMEX(solver_config=...))` | Frozen analytic conservative wave plus nonlinear explicit residual. No scalar bound/SSP guarantee. |
| Moist ARK4 | `MoistAtmosphere(mesh, column, stepper=CenteredIMEX())` | Explicit opt-in to centered mixture transport; admissibility rejection, not repair or bounded transport. No bounded IMEX scheme. |

Import the model and steppers from `vfs`; import `IMEXSolverConfig` from
`vfs.imex`. For dry ARK4, use `DryIMEX(solver_config=solver_config)` below.
Direct `DryPEIMEXARK(dynamics, solver_config).step_with_info(state, dt)` remains
available for stage diagnostics.
Use an explicit configuration:

```python
solver_config = IMEXSolverConfig(
    implicit_operator="gravity_wave", preconditioner="hydrostatic_schur",
    rtol=1e-11, max_iterations=100,
)
```

This is a documented tested selection. `DryIMEX()` defaults to analytic waves
and Schur with the original config tolerance/restart/iteration defaults; the
explicit tolerance above is the one used in the pre/post stage capture. Low-level
dry solver construction requires an explicit gravity_wave configuration.
Moist CenteredIMEX's existing defaults are analytic gravity waves, Schur,
rtol=1e-9, 80 iterations, restart=30. ARK4 is the only retained implicit tableau;
ARK2/3 selectors and pressure-prototype classes were deliberately removed.
`CenteredSSPRK3` is no longer a model option; the centered moist RHS remains an
essential ingredient of bounded stepping and ARK4.

## Follow the execution

### Dry explicit

1. [DryAtmosphere.forward](../vfs/model.py) checks geometry, dtype/device and dt;
   [DrySSPRK3.advance](../vfs/atmosphere/steppers.py) delegates to the existing
   [ssprk3](../vfs/core/integrators.py) formula. Each of three stages evaluates
   `DryHydrostaticConservativeDynamics.tendency` and updates tensor contents.
2. [conservative._assemble](../vfs/atmosphere/conservative.py) builds pressure
   masses and continuity primitives. Pressure increases downward: p=A+B ps,
   dp=p[k+1]-p[k]>0. CP interface mass gives each adjacent layer half its dp,
   with half-weights at the lid and surface. State stores mtheta=dm*theta (Pa K),
   not temperature or a mass in kg. ps is (cell,1) Pa; u is (edge,full) m/s.
3. `_layer_divergence` takes horizontal divergence of dp_edge*u (Pa/s).
   `ps_dot=-sum(layer_divergence)` and `_vertical_mass_flux` integrate the layer
   continuity residual down the column. Vertical flux is zero at the fixed lid
   and surface. `_omega_full` averages interface fluxes for CP scalar exchange.
   `centered_scalar_transport` supplies the horizontal part; vertical adjacent
   interface fluxes cancel in the column sum.
4. `thermal_gradients` returns energy covectors with respect to mtheta and ps.
   Their pairing with transport is pressure work. Explicit transpose operations
   return the opposite work to edge momentum; area and pressure mass factors
   belong to the pairing, not an empirical force correction. Rotation is skew
   paired in the diagonal kinetic metric. Vertical wind exchange includes the
   changing-mass metric correction; Bernoulli work completes the kinetic pairing.
5. Divide summed momentum covectors by the edge kinetic metric to obtain m/s².
   Assemble mtheta (Pa K/s) and ps (Pa/s) tendencies and apply the RK update.
   `tendency_terms` instead solves components separately for inspection; that
   distinct rounding path is intentionally retained. `boundary_total_energy`
   adds the fixed-pressure lid reservoir to atmospheric energy.

[DryEnergy](../vfs/atmosphere/dry_energy.py) owns thermal energy and its analytic
content/pressure derivatives plus budget diagnostics; the conservative class
supplies the diagonal kinetic metric. Methods are ordinary class definitions,
not externally installed alternatives. [ScalarTransportState](../vfs/atmosphere/scalar_transport.py)
shares the pressure-coordinate transport calculation with moist entropy while
keeping the physical quantities explicitly distinguished. No forward tendency
is generated by autograd.

For an explicitly forced dry step, see [Held–Suarez](HELD_SUAREZ.md) and
[the runnable example](../examples/held_suarez.py). The source retains existing
thermal relaxation and drag, summed with the conservative RHS at each SSPRK3
stage. Its thermal/drag energy work is reported separately. This does not alter
the conservative operators or configure forcing for ARK4.

### Bounded moist explicit

[model](../vfs/atmosphere/model.py) →
[stepper](../vfs/atmosphere/steppers.py) →
[bounded SSPRK3/Euler/evaluate](../vfs/atmosphere/bounded_moist.py).
See [the detailed bounded reading guide](BOUNDED_MOIST_READING_GUIDE.md).
Centered candidates, donor fluxes and antidiffusive limiter corrections are
necessary stages of one retained algorithm, not selectable alternative physics.
The Euler water update installs integrated species amounts directly; it must not
be reconstructed from `water + dt * tendency`. Gibbs/limiter work is allocated as
heat under the default configuration.

### Dry ARK4

[dry_ark](../vfs/integrators/imex/dry_ark.py) `_step` freezes
[conservative_wave](../vfs/integrators/imex/conservative_wave.py), then loops over
the six [ARK4 stages](../vfs/integrators/imex/tableau.py). At anchor y0, solve with
L(y-y0) implicitly and F(y)-L(y-y0) explicitly. Schur eliminates wind and solves
the thermodynamic block; multilevel adds the existing geometric preconditioner
using [mesh transfers](../vfs/geometry/mesh/multilevel.py). [solvers.py](../vfs/integrators/imex/solvers.py) owns the physically scaled
inner product, GMRES residual checks, Schur factorization and multilevel cycles.
Full residual checks and IMEXConvergenceError remain authoritative for solver
acceptance; preconditioning changes the solve, not the equation. The final
weighted sum uses all six explicit and implicit stage tendencies. Stage records
expose residuals, convergence, iterations and preconditioner applications through
`model.stepper.solver.last_step_info`; `step_with_info` also returns them.

### Moist ARK4

[moist_ark](../vfs/integrators/imex/moist_ark.py) `step_with_info` freezes
[moist_wave](../vfs/integrators/imex/moist_wave.py). Its analytic wave differentiates
mixture enthalpy/pressure transport at fixed entropy/water contents. Advection,
rotation and kinetic forces remain explicit. Each actual stage and final state
is validated; the validator sees outer dt, not an Euler limiter budget. No
implicit stage is silently converted to bounded transport. Wind is edge/full;
pressure is cell/1; entropy and each water content are cell/interface. The wave
maps perturbations in those stored contents, rather than treating temperature or
fractions as independent prognostics. A stage validator receives the outer dt;
it is not a bounded Euler flux budget. Full and Schur solves retain rejection
and failure behavior. Multilevel moist and compiled paths are not supported.

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

## What to inspect next

Use [ownership examples](STATE_OWNERSHIP.md) to initialize and call both models,
then inspect the equation owners above. External [Field](../vfs/field.py)
placement/regridding and dry State diagnostic/I/O helpers remain meaningful
boundaries; neither changes the tensor-only state payload. Current regression
and intentional removal details are in [readability results](READABILITY_RESULTS.md).
[Historical reports](HISTORY.md) retain earlier evidence without advertising
removed APIs as current configurations.
