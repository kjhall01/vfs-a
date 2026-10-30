# State and Field ownership

Models own fixed mesh, column (including prescribed `phi_s` topography), dynamics,
and stepping configuration. States are ordinary tensor containers with borrowed
mesh/column references, not modules, parameters, or model buffers. Move the model
before initialization. Sharing a tensor does not imply copying it: initialization,
algebra and explicit conversions preserve autograd connections.

`MoistState` already satisfies this boundary: `u_e`, `ps`, `entropy_content` and
`water_content` are tensors. `MoistAtmosphere.physical_fields(state)` computes
interface temperature and composition on demand. Its class-instance stepper API,
module registration, positive-vapor requirement, and bounded species ledger are
unchanged; the default still releases Gibbs correction as heat.

Dry `State` now exposes tensors consistently through its `PrognosticFields`
payload: `u_e` (edge/full, m/s), `ps` (cell/surface, Pa), `mtheta_i`
(cell/interface, pressure-weighted potential temperature, Pa K), and optional
`tracer_m` (cell/full/tracer, pressure-weighted mixing ratios). Leading batch axes
are retained; surface pressure has a trailing singleton level axis. StateSpec
continues to describe supported tracers. No cached derived fields were added.

## Public API changes

- Use `state.u_e`, `state.ps`, etc. directly; remove `.values` from dry-state
  access. Compatible legacy `Field` inputs are validated and unwrapped by identity,
  including assignments to the dry payload. Their metadata is not stored in state.
- `Field` itself remains available for placement-aware mesh/regridding operations;
  its `.values` API is unchanged. It is not the state payload type.
- Dry `bind_state` rejects unrelated geometry, even with matching sizes. An
  unbound dry state can still be explicitly bound, sharing its payload without
  mutating the input. Moist states must already reference model geometry.
- `model.state_to_model(state)` explicitly converts tensors on the **same**
  mesh/column objects to model dtype/device, without detaching or regridding.
  A no-op conversion shares tensors. Dry stepping/tendencies/physical fields
  reject dtype/device mismatches. Existing explicit `State.to` and `move_state`
  remain available; neither moves geometry.
- `State.from_theta` no longer silently casts tracer inputs to pressure
  dtype/device. Convert those inputs explicitly with `.to(...)` first.
- Prefer `DryHydrostaticModel.physical_fields(state)` for derived dry fields.
  Dry temperature is on full levels; moist temperature is on interfaces.

## Short usage examples

Both examples use this shared setup (CPU float64):

```python
import torch
from vfs import (
    State, DryHydrostaticModel, MoistAtmosphere,
    VoronoiMesh, CharneyPhillipsColumn, make_hybrid_coefficients,
)
mesh = VoronoiMesh(n_cells=16, lloyd_iters=0, verbose=False)
A, B = make_hybrid_coefficients(3, p_top=5000.0)
column = CharneyPhillipsColumn(
    3, A, B, phi_s=0.0, phi_full_mode="direct_u_gl2_adjoint",
)
```

Dry potential-temperature initialization and a single step:

```python
dry = DryHydrostaticModel(mesh=mesh, column=column).to(dtype=torch.float64)
ref = dry.mesh.cell_area
state = State.from_theta(
    ref.new_zeros((mesh.nEdges, 3)),
    ref.new_full((mesh.nCells, 4), 300.0),
    ref.new_full((mesh.nCells, 1), 100000.0),
    mesh=dry.mesh, column=dry.column,
)
next_state = dry(state, 1.0)
temperature = dry.physical_fields(next_state)["temperature"]
assert isinstance(next_state.u_e, torch.Tensor)
```

Moist physical-field initialization and the existing default bounded step:

```python
moist = MoistAtmosphere(mesh, column).to(dtype=torch.float64)
state = moist.state_from_fields(
    temperature=280.0, surface_pressure=100000.0, normal_wind=0.0,
    vapor=0.01, liquid=0.002, ice=0.001,
)
next_state = moist(state, 1.0)
temperature = moist.physical_fields(next_state)["temperature"]
assert isinstance(next_state.entropy_content, torch.Tensor)
```

These are API demonstrations, not balanced initial conditions or timestep advice.
The examples intentionally share fixed geometry. Do not mutate shared geometry
independently after setup. For an existing state after a model dtype/device move,
use `state = model.state_to_model(state)` explicitly.

## Remaining legacy boundaries

Dry State's geometry/diagnostic methods remain uncached compatibility helpers for
existing dynamics and I/O; the model method delegates to them. StateSpec/attrs,
explicit restart/NetCDF methods and the unloaded restart shell remain supported.
Xarray objects enter only at explicit external I/O/initialization boundaries.
The older dry recipe's stepping configuration is unchanged in this pass.

The private moist adapter still places entropy content in dry `State.mtheta_i`:
`adapter.mtheta_i is moist_state.entropy_content`. Its construction and numerical
method bodies are unchanged. This borrowed transport view must never be treated
as physical dry potential temperature or passed to dry physical-field output.
See the [execution reading guide](BOUNDED_MOIST_READING_GUIDE.md).
The model state_dict still is not a complete simulation restart format.

## Validation for this change

Local installed-wheel checks ran outside the checkout on macOS 14.6.1 arm64,
Python 3.12.13, PyTorch 2.15.0, CPU float64, Accelerate BLAS/LAPACK (NumPy 2.5.4,
SciPy 1.18.1). OMP/MKL/OpenBLAS thread environment variables were set to 1.
The sandbox denied the CPU model sysctl query; no accelerator execution is
claimed for this pass.

- Portable pytest: **600 passed, 1 CUDA skip**, with the existing scalar-conversion
  warning from bounded donor subcycle selection. Includes ownership tests, moist
  module/stepper tests, dry IMEX checks, and existing one-step gradient tests.
- Standalone portable comparisons: **142 moist + 193 dry tensors passed**.
- Both existing moist examples and both usage snippets above completed.
- Same-environment pre/post captures against merged PR #7 (`9021af2`): all
  **335 fixture-capture tensors** and **47 additional tensors** are exactly equal
  (`rtol=atol=0`). Extras cover centered moist terms/energies, diagnostic gradients,
  RHS VJPs, bounded facade stepping, dry tracer SSPRK3, dry IMEX and dry energy
  gradients. Private adapter aliases were checked. Captures stayed outside the
  repository; fixtures were not rewritten.
- Frozen-fixture **exact** mode remains a failure on this host: first reported
  moist mismatch `smooth/initial/u_e` (max abs 4.440892098500626e-15), dry mismatch
  `canonical/primitives/dp_e` (max abs 1.0913936421275139e-11). These are distinct
  from exact same-environment pre/post agreement; portable success does not
  reinterpret exact failures. See the existing CI investigation/portable policy.

GitHub Actions results will be recorded separately in the PR after it runs.
This is ownership regression coverage, not a new scientific verification campaign,
full-rollout qualification, or expansion of supported precision/accelerators.
