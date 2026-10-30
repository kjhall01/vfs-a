# Tensor-first moist workflow

This provisional nn.Module API owns geometry and dynamics and delegates to
explicitly selected existing steppers. Bounded SSPRK3 remains the default. See
[module ownership and stepping](MOIST_MODULE_API.md); start with
examples/moist_atmosphere.py. Numerical equations are unchanged.

## Construct and run

```python
import torch
from vfs import (
    MoistAtmosphere, VoronoiMesh,
    CharneyPhillipsColumn, make_hybrid_coefficients,
)

mesh = VoronoiMesh(n_cells=64, lloyd_iters=0, verbose=False)
A, B = make_hybrid_coefficients(6, p_top=5000.0, sigma_power=1.0)
column = CharneyPhillipsColumn(
    6, A, B, phi_s=0.0, phi_full_mode="direct_u_gl2_adjoint",
)
model = MoistAtmosphere(mesh, column).to(dtype=torch.float64)  # bounded, heat=True

p = model.interface_pressure(100_000.0)
state = model.state_from_fields(
    temperature=280.0 * (p / 100_000.0)**0.08,
    surface_pressure=100_000.0,
    normal_wind=model.normal_wind(eastward=10.0, northward=0.0),
    vapor=0.01, liquid=0.0, ice=0.0,
)
with torch.no_grad():  # forward-only example; see bounded gradient limitation below
    before = model.diagnostics(state)
    state = model.step(state, dt=30.0)
    after = model.diagnostics(state)
fields = model.physical_fields(state)
```

This is a workflow illustration, not a balanced JW or rest initialization.
The coarse mesh and 30 s step do not establish safe timesteps at other resolutions.

## Locations, units, ownership

| Input | Units | Tensor location/shape |
|---|---|---|
| temperature | K | cells × (levels+1), CP interfaces |
| vapor, liquid, ice | kg/kg total mixture | cells × (levels+1) |
| surface_pressure | Pa | cells, or cells × 1 |
| normal_wind | m/s | edges × levels |
| eastward, northward for normal_wind() | m/s | already sampled at edges × levels |
| column.phi_s | m²/s² | scalar or cells |

All vertical indices run from top to bottom. Temperature is NOT supplied at
layer centers. Fractions are NOT dry-air mixing ratios, relative humidity, or
specific contents. Vapor is required and strictly positive for this kernel;
liquid and ice may be exactly zero. Dry air must have positive mass fraction.

Scalars and vertical profiles broadcast. Use (cells,1) or (edges,1) for a
horizontally varying, vertically constant field. Except for surface pressure,
one-dimensional arrays mean vertical profiles. Supply tensors with the same
dtype/device as mesh.cell_area; the interface does not silently move or cast
them. Mesh generation and this pass's tests use CPU float64.

The model holds mesh/column geometry; terrain stays in column.phi_s. States
reference that geometry and own prognostic pressure, wind, entropy and species
contents. Treat shared geometry as fixed after setup. No new state wrapper is
introduced. MoistState includes geometry references but its
prognostic members are torch tensors.

## What happens inside

1. Compute pressure from A + B ps and reject nonpositive layer thicknesses.
2. Check finite, physical T and fractions; reject rather than clip invalid data.
3. Use the existing mixture entropy function and interface pressure-mass weights
   to form entropy_content and water_content. These contents carry Pa weights,
   not kg/m² weights; the existing operators apply gravity and cell areas.
4. Advance through BoundedGibbs.step, including its amount ledger and SSPRK3
   stages. Do not replace this with a generic step of the dt-aware tendency.
5. Recover interface fields by dividing by those weights and inverting entropy.
   This differs from the layer Gauss-point reconstruction used in energy.

normal_wind() projects local east/north vectors onto mesh normals. It does not
interpolate cell winds to edges or regrid external data. It rejects exact poles,
where longitude-dependent east/north directions are ambiguous.

## Diagnostics and gradients

model.diagnostics(state) returns scalar mass (kg), species vector (vapor,
liquid, ice; kg), scalar entropy inventory (J/K), and scalar energy (J).
Energy includes the top-pressure boundary reservoir. These are tensors;
no detach, numpy conversion, or .item() occurs in their calculation.

Initialization retains tensor gradients. Tests compare diagnostic energy
gradients with respect to T, ps and vapor against finite differences. This
does not qualify every limiter branch or long rollout backward computation.
The model does not implicitly disable autograd; use torch.no_grad for forward
verification runs to avoid retaining graphs.

dynamics and the selected stepper are exposed for inspection; transport is
BoundedGibbs in the default configuration and None for centered configurations.
The facade is an nn.Module; use the explicit movement workflow documented below.

## Scope and known limitation

The exact-zero-vapor state is thermodynamically definable but rejected by the
retained moist composition validator. This pass exposes that limitation early, with
no moisture floor and no automatic switch to the separate dry formulation.
The new API therefore does not demonstrate zero-water/dry-core equivalence.

Xarray/DataArray initialization, conservative regridding, cell-centered wind
conversion, restart/output datasets and averaged trajectory writing remain
future work. External arrays are rejected at this tensor boundary; a later
state_from_xarray adapter will make those conversions explicit. No implicit
solver promotion, phase adjustment, precipitation or column physics is added.

For the implementation call sequence and ledger/heat units, see the
[bounded moist reading guide](BOUNDED_MOIST_READING_GUIDE.md).

The module now supports explicit state conversion and configurable existing
steppers; see [ownership, movement and checkpoint requirements](MOIST_MODULE_API.md).
Diagnostic gradients remain supported. The separate
[allclose fix and one-step derivative checks](BOUNDED_GRADIENT_LIMITATION.md)
remove the documented tensor-atol failure; limiter switches remain nonsmooth.
Removing no_grad from a forward example does not establish rollout support.
