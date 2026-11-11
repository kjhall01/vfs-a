# Moist atmosphere API

`MoistAtmosphere` is a PyTorch `nn.Module` owning mesh, column and dynamics.
`MoistState` holds prognostic tensors and borrows the same geometry. A stepper
binds once to a model; construct a fresh stepper for each model.

```python
from vfs import MoistAtmosphere, BoundedSSPRK3, CenteredIMEX

model = MoistAtmosphere(mesh, column)  # bounded SSPRK3
# Explicit alternatives, used when constructing a model:
bounded_stepper = BoundedSSPRK3()
centered_implicit_stepper = CenteredIMEX()
```

| Method | Purpose |
| --- | --- |
| `state_from_fields(...)` | Initialize from interface temperature, pressure, velocity and water fractions |
| `model(state, dt)` / `step(state, dt)` | Advance by dt seconds and validate inputs/outputs |
| `physical_fields(state)` | Recover temperature and composition |
| `diagnostics(state)` | Tensor mass, species, entropy and energy budgets |
| `state_to_model(state)` | Explicit dtype/device conversion on the same geometry |

## Stepping

`BoundedSSPRK3` uses three bounded Euler species-amount ledgers and intrinsic
heat release. `model.transport.step_with_info` exposes its detailed ledgers.
`model.dynamics.tendency` is the centered RHS; it is not a replacement for the
bounded Euler update. The timestep-dependent transport tendency alone also does
not replace installing the authoritative integrated water amounts.

`CenteredIMEX` uses ARK4 and centered transport. Each stage is checked for
admissibility, but there is no bounded-species or SSP guarantee. Configure solves
with `vfs.imex.IMEXSolverConfig`; inspect
`model.stepper.solver.last_step_info` for residuals and iteration counts.
There is no bounded IMEX option. Model calls advance time; they do not return RHS
values. See [timestep paths](CURRENT_PATHS.md).

## Geometry, movement and checkpoints

Move the model with `.to(dtype=..., device=...)` before state initialization.
Existing state tensors require explicit `state_to_model` conversion afterward.
No-op conversions share tensors; conversions retain gradient connections and do
not regrid. Runtime methods reject unrelated geometry or mismatched dtype/device.

Model components and stepper configuration are fixed after construction.
`state_dict` records registered buffers, not simulation time, prognostic state or
the full physical/stepper configuration. Recreate matching topology, column mode
and settings before `load_state_dict`; a state dict alone is not a restart.

Positive vapor is required; no moisture floor is added. CPU float64 is the
verified baseline. One-step derivative checks do not establish reliable gradients
through every limiter switch, IMEX solve or long trajectory. See
[physical-field initialization](PHYSICAL_FIELDS_API.md) and
[measured gradient limitations](BOUNDED_GRADIENT_LIMITATION.md).
