> Historical design checkpoint. Superseded by the accepted explicit-stepper
> [module API](MOIST_MODULE_API.md) and [results](MOIST_MODULE_RESULTS.md).
> The string-selection sketch below was not implemented.

# Moist ownership and stepping proposal — not implemented

The module-boundary work is paused for this design correction. PR #5 is merged;
the working branch is `api/moist-module-ownership`. The existing uncommitted
module/state-conversion edits are preserved, but their transport-owned dynamics
and hard-wired forward path are superseded by this proposal, not accepted API.
No new structural changes or numerical algorithms have been implemented here.

## What the current implementations require

- [BoundedGibbs](../vfs/atmosphere/bounded_moist.py), `step_with_info`:
  three bounded Euler maps, each receiving the full outer `dt`. The second
  and final states use the existing 3/4–1/4 and 1/3–2/3 content combinations.
  `euler_with_info` installs water directly from `info['amounts'] / area`;
  reconstructing it as state plus tendency loses the cancellation-safe ledger.
  Each Euler map has its own ledger and limiter diagnostics; these are not a
  single ledger for the final RK state. The facade default releases limiter
  work as heat; standalone BoundedGibbs still defaults to False.
- [MoistConservativeDynamics.step](../vfs/atmosphere/moist_conservative.py):
  existing centered SSPRK3 using `MoistState.combine`, with adapter validation.
  It has no bounded species ledger or heat-release correction.
- [Generic explicit functions](../vfs/core/integrators.py): Euler, RK4 and
  SSPRK3 use State arithmetic and transient `_current_tendency_dt` context.
  MoistState has `combine`, not that arithmetic protocol. BoundedGibbs also
  requires an explicit `dt` argument and its direct Euler map. These functions
  are not drop-in moist steppers; do not adapt them in this PR.
- [MoistPEIMEXARK](../vfs/integrators/imex/moist_ark.py): existing experimental
  centered smooth-mixture ARK2/3/4 (default ARK4). A
  [MoistPressureWave](../vfs/integrators/imex/moist_wave.py) is frozen at each
  outer-step anchor; its coefficients are step-local, not persistent model
  buffers. The solver splits the unchanged nonlinear centered RHS into
  `F(y) - L(y-y0)` and `L(y-y0)`, solves stage corrections, and assembles the
  existing weighted output. The implicit budget is `dt * a_ii`; neither this
  nor the validator's outer `dt` is a bounded forward-Euler budget.
  The stage validator can reject, never repair. Full-system residual checks,
  stage/output admissibility checks and convergence errors must remain intact.
  Eager analytic gravity waves with `none` or `hydrostatic_schur`
  preconditioning are supported; compiled kernels and other wave/preconditioner
  configurations are rejected by the existing constructor.

The ARK tables are not SSP positivity extensions. There is no existing bounded
moist IMEX scheme to expose. Dry ARK and legacy pressure prototypes operate on
the dry State API and are outside this moist facade change.

## Recommended ownership

Register numerical ownership independently of the stepping choice:

```text
MoistAtmosphere (nn.Module)
  mesh
  column
  dynamics: MoistConservativeDynamics
    kernel
```

`model.transport` is either a BoundedGibbs collaborator referring to this exact
`model.dynamics`, or None for centered transport. `model.integrator` is a
non-owning stepping collaborator; neither stores additional geometry or state
tensors as persistent model data. Geometry and dynamics register exactly once.

For the smallest design, I recommend keeping BoundedGibbs a plain collaborator,
as in the released code, rather than making it an nn.Module with a deliberately
unregistered base module. This revises the earlier request to convert both
classes: the atmosphere's module conversion, geometry buffers, conversion API
and hook behavior are retained. It avoids duplicate registration, weak references,
conditional ownership flags and surprising standalone `.to()` semantics.
Its existing constructor, base reference and numerical methods remain intact.
If BoundedGibbs must itself be a module, that ownership decision remains open;
do not silently bypass module registration for its base.

## Small stepping interface

Extend the existing facade constructor with explicit keyword choices:

```python
MoistAtmosphere(mesh, column, *, release_as_heat=True,
                transport="bounded", integrator="ssprk3",
                solver_config=None, stage_validator=None)
```

These strings select a finite set of existing configurations with ordinary
conditionals, not a registry or arbitrary provider protocol. Keep `dynamics`
and the configured stepping collaborators fixed after construction so aliases
cannot become stale. Unknown or incompatible choices fail at construction.

| Transport | Integrator | Existing implementation to delegate to |
| --- | --- | --- |
| bounded (default) | ssprk3 (default) | BoundedGibbs.step / step_with_info |
| centered | ssprk3 | MoistConservativeDynamics.step |
| centered | ark2, ark3, ark4 | MoistPEIMEXARK with the selected method |
| bounded | any ARK | Reject: no compatible bounded stage algorithm exists |

Selecting ARK must also explicitly select centered transport; never silently
drop the limiter. Other combinations, arbitrary callables, generic RK4/Euler,
dry steppers and external tendency additions are outside the proposed facade.
`solver_config` and `stage_validator` apply only to ARK; reject them for SSPRK3.
`release_as_heat` retains its current default and only configures bounded
transport; document it as inapplicable for centered configurations.

`forward(state, dt)` validates geometry/dtype/device and positive finite `dt`,
then calls the configured `integrator.step(state, dt)`. `step` calls
`self(state, dt)` so atmosphere forward hooks run. For explicit SSPRK3, use
one small non-owning adapter whose `step` delegates to the selected existing
implementation. IMEX already supplies this method. Do not copy RK expressions
into the facade or rewrite the Euler maps to fit a generic tendency interface.

Keep detailed results in their native APIs: bounded `transport.step_with_info`
returns the three stage ledgers; IMEX `integrator.step_with_info` returns solver
stage records. Do not invent a common ledger or pretend their budgets coincide.
Centered SSPRK3 has no existing step-info result to synthesize.
Tendencies remain explicit: `dynamics.tendency(state)` is centered, while
`transport.tendency(state, stage_dt)` is budgeted and is not a substitute for
the bounded Euler map. Calling the atmosphere always advances a timestep.

## Preserved work and validation boundary

Retain the explicit state conversion and mismatch checks, model-owned geometry
buffers and integer graph-index allocation fixes already drafted. MoistState
stays a tensor dataclass. Move the model before initialization; conversion of
existing states requires the same mesh/column objects and preserves gradients.
No fixtures, limiter policy, positive-vapor restriction, private entropy adapter,
kernel equations or operation order change. No new moisture floor is introduced.

After the ownership decision, validate registration/state_dict, hooks, conversion
gradients, rejected combinations, and exact delegation for each supported path.
Keep the existing IMEX rejection/convergence checks and bounded ledger tests.
Run same-environment pre/post comparisons and installed-wheel portable checks
and examples; report exact separately. A model state_dict contains registered
buffers, not prognostic states, integrator configuration or a complete restart.
Loading requires matching geometry layout and separately reconstructed numerical
and stepping configuration.

This proposal is based on source inspection, including
[existing moist IMEX tests](../tests/test_moist_imex.py), not new runtime results.
Those tests cover ARK2/3/4, stage guards, conservation, wave blocks and rejected
solver configurations; they do not establish bounded IMEX support.
No new ownership tests or installed-wheel validation have completed in this
paused task. The attempted pre-change capture failed on a gradient-enabled
bounded step: `torch.allclose` rejected a tensor-valued `atol` in
`bounded_transfer`. This occurred with the pre-change installed package and
must be reported separately, not fixed by changing operators or detaching state.
No accelerator or reduced-precision numerical qualification is claimed.
