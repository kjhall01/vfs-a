# Handoff to scientific verification

Verification has begun against merged PR #12, commit
`8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648`. See the current
[claim-to-evidence tracker and run contract](../verification/README.md).
The readiness inventory below remains useful; it is not a list of completed
scientific experiments.

Freeze the retained implementation after this cleanup is reviewed. Conda packaging
is deferred. Treat any numerical correction or missing benchmark capability as a
separate, explicit change; do not restart stylistic restructuring or quietly tune
fixtures/tolerances. Regression agreement is not scientific validation.

## Supported entry points and executable smoke scenarios

All commands run from an installed-wheel check directory containing examples.
Use CPU float64 initially, a fixed positive pressure lid and
`phi_full_mode="direct_u_gl2_adjoint"`. Move models before initializing states.

| Path | Entry | Runnable scenario |
|---|---|---|
| Conservative dry explicit | `DryAtmosphere(mesh, column)` / `DrySSPRK3()` | `python -I examples/dry_atmosphere.py` |
| Bounded moist explicit | `MoistAtmosphere(mesh, column)` / `BoundedSSPRK3()` | `python -I examples/moist_atmosphere.py`; low-level ledger example: `python -I examples/bounded_moist.py` |
| Dry ARK4 | `DryAtmosphere(..., stepper=DryIMEX())` | `python -I examples/dry_atmosphere.py --imex` |
| Centered moist ARK4 | `MoistAtmosphere(..., stepper=CenteredIMEX())` | `python -I examples/moist_atmosphere.py --imex` |
| Forced dry explicit | `HeldSuarezForcing` added to the dry RHS at every SSPRK3 stage | `python -I examples/held_suarez.py` |

These are short API smoke runs, not JW/HS climate or mountain-wave qualifications.
The dry core is unbatched/tracer-free. Moist vapor must be positive; condensates
may vanish. Only bounded explicit stepping has the authoritative species ledger.
Centered ARK4 rejects inadmissible stages; it does not bound species or provide an
implicit heat/limiter scheme. No forced-ARK4 composition is currently provided.

The retained ARK4 operator is analytic; there is no coupled/fast-wave selector or
compiled option. Dry full, hydrostatic Schur and multilevel Schur have representative
forward evidence; moist full and Schur are supported, not multilevel. Preserve
stage validation and solver-failure reporting. Record config, iterations and
residuals with `model.stepper.solver.last_step_info` at every assessed step.

## Benchmark readiness and remaining setup

| Case | Existing ingredients/evidence | Required before a verification claim |
|---|---|---|
| JW-A (unperturbed) | `JWInitialConditions.make_state(..., perturb=False)`; dry control capture in `tools/check_baseline.py` | Add a reproducible resolution/timestep driver and drift/error diagnostics against the chosen JW specification. Record `mode`, theta/wind source, optional mass-flux projection and `ICBuildInfo`. Explicitly select the retained column mode: the JW helper's own default is `interface_mean`, which this core rejects. Analytic initialization is not guaranteed discrete balance. |
| JW-B (perturbed) | Same builder with `perturb=True` provides the existing Gaussian wind perturbation | Add evolution/output schedules, reference comparison, perturbation/error norms and refinement runs. No mature JW-B campaign runner or verified growth/error envelope currently exists. |
| Held–Suarez | Standalone unchanged relaxation/drag; `held_suarez_equilibrium_state` and `isothermal_rest_state`; forced-step example; pinned archive comparison | Specify initialization, grids, timestep, spin-up and averaging periods, zonal/time-mean diagnostics and sampling uncertainty. No equilibrated climate, averaging campaign or forced IMEX path is qualified. Keep thermal and drag energy sources separate from dynamics. |
| Flat rest | Dry `isothermal_rest_state(..., wind_mode="none")`; moist uniform-rest regression in `tests/test_moist_conservative.py` | Add explicit rest-scenario drivers across grids/levels/timesteps, velocity/pressure/energy drift and residual measurements. Current one-step regression tolerances are not scientific thresholds. |
| Topographic rest | Model-owned `column.phi_s`; pressure-gradient/energy derivatives and nonzero-terrain regression cases | Construct a physically consistent hydrostatic resting atmosphere with terrain-dependent surface pressure. The existing isothermal initializer uses uniform ps: applying it unchanged over terrain is not a demonstrated rest state. Add independent balance residuals and convergence measurements; no qualified topographic-rest campaign yet. |
| Transport | Bounded face transfers, donor/subcycle correction, limiter and exact amount ledgers; sharp cloud/reversed-flow fixtures and active-limiter/derivative checks | Specify smooth and sharp transport protocols, prescribed-flow or coupled-dynamics choice, return-time errors, positivity, conservation and limiter activity. A generic dry passive-tracer dycore/standalone tracer-advection runner was removed and is not available. Do not restore it silently or substitute centered ARK4 for bounded transport. |
| Mountain wave | Hydrostatic core, prescribed topography, explicit/ARK4 stepping | Candidate: small-amplitude, broad smooth ridge in a stably stratified zonal background, in the hydrostatic regime. Select a matching analytic/reference solution and spherical geometry/scale specification first. Build terrain/background initialization and phase/amplitude/flux diagnostics; assess fixed-lid reflection and domain recirculation. There is no dedicated mountain-wave initializer, absorbing upper layer or open lateral boundary. A short-wavelength nonhydrostatic benchmark is inappropriate for this core. If the selected reference requires missing boundary physics, that is a separate capability blocker, not cleanup work. |

JW builders and Held–Suarez initializers are explicit CPU initialization boundaries,
not end-to-end differentiable benchmark constructors. Their NumPy use, prescribed
data and initialization choices must be stated separately from tensor diagnostics.
No new benchmark physics or drivers are implemented by this cleanup.

## Measurement contract

Record source commit, mesh generation/seed, vertical coefficients, terrain,
initialization choices, physical constants, stepper/solver settings, dtype,
Torch/CPU/backend versions, duration, and output cadence. Establish references
and refinement sequences before interpreting metrics. Report measured errors,
convergence and uncertainty without inventing subjective pass thresholds.

Separate instantaneous conservative energy closure (including the fixed-pressure
lid reservoir) from finite-step energy drift. For forcing, report thermal and drag
work independently; the existing drag does not return its kinetic loss as heat.
For bounded runs, inspect actual Euler species amounts, carrier closure, minima
and limiter activity. Never reconstruct the authoritative ledger from the RHS.

Existing independent tests cover conservative derivatives, analytic wave actions,
DOP853 ARK4 refinement on small cases, bounded one-step directional derivatives
away from switches, and selected active-branch derivatives. Limiter switches are
nonsmooth. IMEX backward, arbitrary long-rollout gradients, GPU/reduced precision
and broad timestep/resolution stability are unqualified. Keep these as separate
questions rather than extrapolating from `.to()` or finite gradients.

## Starting evidence

Read [cleanup results](FINAL_CLEANUP_RESULTS.md), [the timestep trace](CURRENT_PATHS.md),
[bounded gradients and limits](BOUNDED_GRADIENT_LIMITATION.md), and
[portable regression policy](PORTABLE_REGRESSION.md). The original fixture archive
and quantity tolerances remain unchanged. Use installed-wheel portable checks for
migration, exact pre/post comparison on the same environment for refactors, and
the pinned [Held–Suarez archive tool](HELD_SUAREZ.md#reproduce-the-archive-comparison).
Scientific references and campaign artifacts belong to the next phase.
