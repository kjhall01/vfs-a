# VFS documentation

## Using the models

| Topic | Guide |
| --- | --- |
| Installation and examples | [Repository README](../README.md) |
| Tensor shapes, units and geometry | [State ownership](STATE_OWNERSHIP.md) |
| Atmospheric initialization and output | [Physical fields](PHYSICAL_FIELDS_API.md) |
| Moist model and steppers | [Moist API](MOIST_MODULE_API.md) |
| Stratified ocean and vertical remapping | [Ocean API](OCEAN_INTEGRATION.md) |
| Ocean wind stress and bottom drag | [Boundary stress](OCEAN_BOUNDARY_STRESS.md) |
| Optional prognostic ocean turbulence | [TKE closure](OCEAN_TKE.md) |
| Explicit and implicit stepping | [Timestep paths](CURRENT_PATHS.md) |
| Atmospheric thermal relaxation and drag | [Held–Suarez forcing](HELD_SUAREZ.md) |
| Horizontal rotation | [Compatible rotation](COMPATIBLE_ROTATION.md) |
| Latitude–longitude diagnostics | [Equiangular grid](EQUIANGULAR_GRID.md) |
| Modules and dependencies | [Package structure](PACKAGE_OWNERSHIP.md) |
| Bounded moist numerical operations | [Step reading guide](BOUNDED_MOIST_READING_GUIDE.md) |

## Checking results

[Verification](../verification/README.md) distinguishes regression checks from
physical accuracy measurements. [Ocean verification](../verification/ocean/README.md)
covers transport, balance, waves and remapping. The [portable regression policy](PORTABLE_REGRESSION.md)
defines the fixture comparison budgets. [Contributing](../CONTRIBUTING.md) gives
installed-wheel test commands.

Dated results, proposals and source provenance are indexed in [historical records](HISTORY.md).
They describe particular configurations; the guides above describe the package API.

The [equiangular dataset qualification](EQUIANGULAR_VALIDATION.md) covers
cell-centered and pole-sampled coordinates, vector diagnostics and polar errors.
