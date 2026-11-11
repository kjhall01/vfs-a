# States, geometry and units

Models own fixed mesh/column geometry and numerical configuration. States hold
prognostic tensors and borrow those exact geometry objects. State arithmetic and
explicit conversions preserve autograd connections. Derived physical fields are
computed on request; xarray is restricted to initialization and output boundaries.

## Atmospheric states

Let C be cells, E edges and L full vertical levels. Vertical indices increase
from the lid toward the surface.

| State | Field | Shape | Meaning |
| --- | --- | --- | --- |
| Dry `State` | `u_e` | [E, L] | Edge-normal velocity, m/s |
| Dry `State` | `ps` | [C, 1] | Surface pressure, Pa |
| Dry `State` | `mtheta_i` | [C, L+1] | Interface pressure mass times potential temperature, Pa K |
| `MoistState` | `u_e`, `ps` | As above | Velocity and surface pressure |
| `MoistState` | `entropy_content` | [C, L+1] | Interface pressure mass times specific entropy |
| `MoistState` | `water_content` | [C, L+1, 3] | Pressure-weighted vapor, liquid and ice fractions |

The dry container can hold additional axes/tracers; the conservative dry model
supports unbatched, tracer-free states. Moist temperature inputs are at CP
interfaces. Dry `physical_fields` reports full-level temperature. Moist fractions
are relative to total mixture mass, not dry-air mixing ratios; vapor and dry-air
fractions must be positive, while liquid and ice may be zero.

Move an atmospheric model before initializing its state:

```python
import torch
from vfs import MoistAtmosphere, VoronoiMesh, CharneyPhillipsColumn, make_hybrid_coefficients

mesh = VoronoiMesh(n_cells=16, lloyd_iters=0, verbose=False)
a, b = make_hybrid_coefficients(3, p_top=5000.)
column = CharneyPhillipsColumn(3, a, b, phi_full_mode="direct_u_gl2_adjoint")
model = MoistAtmosphere(mesh, column).to(dtype=torch.float64)
state = model.state_from_fields(temperature=280., surface_pressure=100000.,
                                normal_wind=0., vapor=.01)
```

Moving the model does not move existing state tensors. Use
`model.state_to_model(state)` explicitly for the same mesh/column; it does not
regrid or detach. Direct state attributes are tensors, not `Field` wrappers.

## Ocean states

| State | Fields | Shape | Meaning |
| --- | --- | --- | --- |
| Both ocean states | `u_e` | [E, L] | Edge-normal velocity, m/s |
| Both ocean states | `h` | [C, L] | Layer thickness, m |
| `OceanState` | `temp_h` | [C, L] | Thickness times temperature, m °C |
| `EntropyState` | `entropy_h` | [C, L] | Thickness times specific entropy |
| Both ocean states | `salt_h` | [C, L] | Thickness times practical salinity |
| Both ocean states | `tracer_h` | [C, L, N] or None | Thickness times passive concentrations |

Ocean thermodynamic variables are layer-centered. `StratifiedOcean.state` takes
potential temperature in IPTS-68 degrees Celsius and practical salinity; it stores
entropy contents. `from_temperature(OceanState)` explicitly converts coordinates.
The facades require matching geometry, dtype and device and unbatched tensors.
See [ocean initialization and remapping](OCEAN_INTEGRATION.md).

## Fields and checkpoints

`Field` supplies placement metadata for mesh and regridding operations; it is not
the prognostic state payload. Model `physical_fields` methods return derived
physical tensors. Component I/O adapters interpret those tensors for output.

An atmospheric `state_dict()` contains registered geometry/model buffers, not a
complete simulation restart: state, time, stepper configuration and physical
constants must also be retained. Reconstruct matching geometry and configuration
before loading. Ocean snapshot helpers require the corresponding mesh and column;
a snapshot alone does not establish their identity. Do not pass an entropy state
to an `OceanState` snapshot writer without an explicit physical-state conversion.
