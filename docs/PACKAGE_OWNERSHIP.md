# Package structure

| Package | Responsibility |
| --- | --- |
| `vfs.geometry.mesh` | Spherical mesh geometry, connectivity and horizontal operators |
| `vfs.geometry.regrid`, `vfs.geometry.remap` | Field remapping and mesh transfers |
| `vfs.field` | Tensor fields with placement metadata |
| `vfs.atmosphere` | Dry/moist states, pressure columns, thermodynamics and dynamics |
| `vfs.ocean.temperature` | Coupled CT/SA dynamics, shelf transport, Roquet EOS, GM/Redi and eddy energy |
| `vfs.ocean.mixing` | Column closure and thermodynamic work calculations |
| `vfs.ice` | Moving ice, ice drag, category thermodynamics, ridging and snow |
| `vfs.ocean` | Layer geometry, wet domains, temperature-content states and configurable dynamics |
| `vfs.ocean.stratified` | Entropy-state JMD95 ocean, matched transport and adaptive thermal ALE |
| `vfs.integrators` | Explicit Runge–Kutta formulas |
| `vfs.io` | Output schedules, metadata and file mechanics |
| Component `io` and `regrid` modules | Physical field interpretation and component snapshots |
| `vfs.data` | External elevation data acquisition and caching |

Models own geometry and equation/stepper objects. States hold tensors and borrow
geometry. Derived fields are calculated explicitly; xarray is an initialization
or output boundary, not part of a prognostic state.

Shared geometry and numerical utilities do not depend on component state types.
Atmosphere and ocean own their thermodynamics, vertical staggering and physical
adapters. There is no universal state or coupled-model registry.

## Entry points

```python
from vfs import DryAtmosphere, MoistAtmosphere, VoronoiMesh, CharneyPhillipsColumn
from vfs.ocean import Ocean, OceanState, OceanColumn, OceanDomain
from vfs.ocean.stratified import StratifiedOcean, EntropyState, AdaptiveThermalRemapper
```

Atmospheric models are PyTorch modules; move their geometry with `.to()` before
state initialization. Ocean facades are ordinary Python objects; construct mesh,
column and state tensors with matching dtype and device. State conversion is
explicit and does not detach tensors or interpolate between unrelated meshes.

Use owner-module imports for specialized sources, diagnostics and initial
conditions. Top-level compatibility modules re-export selected implementations.
See [state conventions](STATE_OWNERSHIP.md) and [timestepping](CURRENT_PATHS.md).
