# VFS — a coupled atmosphere, ocean, sea-ice and land model in PyTorch

VFS advances hydrostatic flow on a spherical Voronoi C-grid. The coupled
campaign links a moist atmosphere and PyAM2.1 column physics to a Roquet
ocean, moving sea ice, snow and an optional land surface with runoff.
The component models live in `vfs/`. The campaign, external physics adapter
and experiment configurations live in `workflows/` and run from a checkout.

| What do you want to do? | Start here |
| --- | --- |
| Run with continents, topography and shallow shelves | [Earth pilot, required inputs and restart instructions](workflows/coupled/EARTH_SHELF.md) |
| Run a coupled aquaplanet | [Default aquaplanet](workflows/coupled/CAMPAIGN.md) or [4,096-cell full-depth configuration](workflows/coupled/FULL_DEPTH_4096.md) |
| Run atmosphere and physics with prescribed SST | [PyAM2.1 column and aquaplanet drivers](workflows/pyam21/README.md) |
| Explore standalone dynamics | [Dry Held–Suarez atmosphere](examples/held_suarez.py), [moist atmosphere](examples/moist_atmosphere.py), [stratified ocean](examples/ocean.py) |
| Inspect or plot model output | [NetCDF fields and masks](workflows/coupled/EARTH_SHELF.md#existing-inputs-and-netcdf-output) and [native mesh plots](docs/NATIVE_MESH_PLOTTING.md) |
| Benchmark the coupled workflow | [Throughput and operator profiling](tools/PROFILE_COUPLED.md) |

## Install

Python 3.10 or newer is required. Regression verification uses Python 3.12 and CPU float64.
Run these commands from the repository root.

```bash
python -m pip install .
python examples/moist_atmosphere.py
python examples/ocean.py
```

For an editable installation, use `python -m pip install -e '.[test]'`.
Optional extras include `plotting`, `data`, `diagnostics` and `validation`.
See [contributing](CONTRIBUTING.md) for the complete test environment.

## Standalone models

| Model | Main features | Example |
| --- | --- | --- |
| `vfs.DryAtmosphere` | Conservative dry dynamics; SSPRK3 | [Dry atmosphere](examples/dry_atmosphere.py) |
| `vfs.MoistAtmosphere` | Entropy and water contents; bounded SSPRK3 with matched energy work | [Moist atmosphere](examples/moist_atmosphere.py) |
| `vfs.ocean.Ocean` | Roquet CT/SA dynamics, bounded transport, optional mixing, surface forcing, EKE and ALE | [Forced ocean](examples/ocean.py), [partial-column Earth campaign](workflows/coupled/EARTH_SHELF.md) |

The atmosphere uses a hybrid pressure Charney–Phillips column. The ocean uses
layer thicknesses and layer-centered thermodynamic contents. Both store horizontal
normal velocity on mesh edges. Ocean components do not import atmospheric code.

## A moist atmospheric step

```python
import torch
from vfs import MoistAtmosphere, VoronoiMesh, CharneyPhillipsColumn, make_hybrid_coefficients

mesh = VoronoiMesh(n_cells=16, lloyd_iters=0, verbose=False)
a, b = make_hybrid_coefficients(3, p_top=5000.)
column = CharneyPhillipsColumn(3, a, b, phi_full_mode="direct_u_gl2_adjoint")
model = MoistAtmosphere(mesh, column).to(dtype=torch.float64)
state = model.state_from_fields(temperature=280., surface_pressure=100000.,
                                normal_wind=0., vapor=.01)
state = model(state, dt=1.)
budgets = model.diagnostics(state)
```

The [ocean example](examples/ocean.py) constructs the campaign ocean on three
layers and applies a prescribed surface heat flux for one minute. The flat
state exposes layer thickness, edge velocity, temperature and salt inventories,
tracers and turbulent energy. Supplying `eddy_energy` adds a mesoscale reservoir.
The coupled workflow gives the same ocean to the ocean–ice integrator, which
evaluates both components at each Runge–Kutta stage.

## Numerical scope

- Moist SSPRK3 requires positive vapor and uses bounded species-amount ledgers.
- The Earth preset selects partial-column transport when constructing the ocean.
  The aquaplanet uses matching layer indices across wet edges.
- Semidiscrete energy closure does not imply exact finite-step conservation.
  The atmospheric energy budget includes the fixed-pressure lid reservoir.
- Tensor operations retain autograd connections, but limiter switches, iterative
  solves and long backward trajectories require separate qualification.
- The coupled campaign supports CPU and CUDA float64. Device-specific checks,
  precision limits and experimental MPS support are described in the
  [device guide](workflows/coupled/GPU.md). Short runs establish throughput
  and budget behavior for their particular configuration. Climate fidelity
  requires longer integrations and comparison with observations.
- For a forced ocean without atmospheric physics, see the examples for
  [wind stress](examples/ocean_boundary_stress.py), [radiative heating](examples/ocean_radiation.py)
  and [surface fluxes](examples/ocean_surface_fluxes.py).

## Documentation

- [Documentation index](docs/README.md)
- [State shapes and ownership](docs/STATE_OWNERSHIP.md)
- [Timestepping](docs/CURRENT_PATHS.md) and [package structure](docs/PACKAGE_OWNERSHIP.md)
- [Atmospheric initialization](docs/PHYSICAL_FIELDS_API.md) and [moist API](docs/MOIST_MODULE_API.md)
- [Roquet ocean and historical reference configurations](docs/OCEAN_INTEGRATION.md)
- [Verification](verification/README.md) and [ocean measurements](verification/ocean/README.md)

- [Compatible atmosphere/ocean rotation](docs/COMPATIBLE_ROTATION.md)
- [Equiangular grid diagnostics and operators](docs/EQUIANGULAR_GRID.md)
