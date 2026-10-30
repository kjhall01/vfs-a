# Moist module ownership and stepping

MoistAtmosphere is an nn.Module. It registers `mesh`, `column`, and `dynamics`
once; `dynamics` registers its existing `kernel`. BoundedGibbs and steppers are
plain collaborators, not modules or alternate owners. MoistState remains a tensor
dataclass referencing the same geometry. Default numerical behavior is unchanged.

```python
from vfs import MoistAtmosphere, BoundedSSPRK3, CenteredIMEX
from vfs.integrators.imex import ARK4_TABLEAU, IMEXSolverConfig

model = MoistAtmosphere(mesh, column)  # bounded SSPRK3, release_as_heat=True
# Alternatively, construct ONE of these choices with a fresh stepper:
model = MoistAtmosphere(mesh, column, stepper=BoundedSSPRK3(release_as_heat=True))
model = MoistAtmosphere(mesh, column, stepper=CenteredIMEX(method=ARK4_TABLEAU))
```

Existing tableau objects avoid new ARK configuration classes. CenteredIMEX defaults
to ARK4, the only retained implicit method. Custom tableaus, strings,
arbitrary callable steppers and subclasses are rejected. Solver configuration is
an existing IMEXSolverConfig object. Its usual moist defaults apply when omitted;
not every dry solver configuration is supported by the moist solver.

| Stepper | Delegation | Transport guarantee |
| --- | --- | --- |
| BoundedSSPRK3 | BoundedGibbs.step | Three bounded Euler species ledgers |
| CenteredIMEX | MoistPEIMEXARK.step | Centered only; rejection, not repair |

There is no bounded IMEX option or independent transport selector. Selecting
CenteredIMEX explicitly selects centered transport; no option silently removes
the limiter. ARK methods are not SSP positivity extensions. Dry IMEX and generic
RHS integrators are not accepted through this facade.

## Binding and advancement

Each stepper binds once, during model construction, to the model's dynamics.
`stepper.advance(state, dt)` returns a new MoistState by delegating to the existing
algorithm. It requires a bound stepper; the normal public boundary is
`model(state, dt)`, which also checks geometry, tensor dtype/device and finite
positive dt. `model.step(state, dt)` calls the module, so forward hooks execute.
Calling `.forward` directly follows ordinary PyTorch hook-bypass semantics.

Construct a fresh stepper for another model. Model component replacement/deletion
is rejected, stepper configuration is frozen, and advancement checks that borrowed
transport/solver dynamics references still match. Do not mutate collaborator
internals to reconfigure a model; construct a new model instead. This contract
supports `.to()` and ordinary `load_state_dict`, not arbitrary component swapping.

The IMEX solver is constructed once at binding, not rebuilt per call. Its wave
coefficients are still frozen anew at each outer-step anchor. Solver failure and
stage/output rejection propagate unchanged. Its optional `stage_validator(state,
dt)` receives the outer timestep, not a bounded forward-Euler budget. Solver
stage records remain on `model.stepper.solver.step_with_info` and `last_step_info`.

For bounded transport, `model.transport` is the existing BoundedGibbs; centered
configurations have `transport=None`. Each of its three Euler maps uses the full
dt budget and installs water directly from `info['amounts'] / area`. Keep the
three stage ledgers separate from the final convex combination. Detailed results
remain available through `model.transport.step_with_info`; no common IMEX/bounded
ledger is invented. The `release_as_heat` constructor keyword must be True
when the stepper is omitted; the non-heat bounded configuration was removed. With an explicit stepper, configure heat on
BoundedSSPRK3; a conflicting nondefault facade keyword raises.

Tendencies remain separate: `model.dynamics.tendency(state)` is centered;
`model.transport.tendency(state, stage_dt)` is budgeted but is NOT a substitute
for the direct bounded Euler map. The atmosphere call always advances time.

## Moving geometry and state

Construct geometry on CPU, then move the whole model **before initialization**:

```python
model = MoistAtmosphere(mesh, column).to(dtype=torch.float64, device="cpu")
state = model.state_from_fields(temperature=280., surface_pressure=100000.,
                                normal_wind=0., vapor=.01)
```

Mesh-derived geometry is already in registered buffers, including connectivity,
masking and reconstruction weights; column coefficients and terrain are buffers.
Geometry constructors compute these before movement. There is no new lazy cache.
Bounded graph indices now allocate explicitly on the carrier device as int64;
other bounded-path temporary allocations use input-derived dtype/device. Integer
and boolean buffers retain their types when floating buffers change dtype.

Moving a model changes shared geometry but does not move an existing state's
prognostic tensors. Use `state = model.state_to_model(state)` explicitly. It
requires the exact same mesh/column objects, uses Tensor.to without detach, and
returns a new dataclass; no-op conversions retain the original tensor objects.
It never regrids or accepts unrelated geometry. Public state-taking model methods
reject dtype/device mismatches. Diagnostic and initialization gradients are
retained. Strictly positive vapor remains required; there is no new state floor.

## Checkpoints and limits

`model.state_dict()` contains registered geometry buffers, not MoistState, time,
timestep, RNG state, stepper choice, heat policy, thermodynamic constants or solver
configuration. It is **not a complete simulation restart format**. Construct a
model with matching mesh topology/buffer shapes, column levels/mode and numerical
configuration, then load with the usual `load_state_dict` (default `assign=False`).
Move the model and initialize or explicitly convert a compatible state afterward.
Do not load state_dict as a substitute for reconstructing the chosen configuration.

See [current validation results](READABILITY_RESULTS.md) and the
[bounded gradient fix and remaining limits](BOUNDED_GRADIENT_LIMITATION.md). Module ownership,
forward numerical equivalence, diagnostic gradients and timestep backpropagation
are distinct claims. Device movement alone does not establish accelerator or
reduced-precision numerical support.
