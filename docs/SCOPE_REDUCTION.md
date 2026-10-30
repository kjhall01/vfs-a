# Atmospheric package scope reduction

This PR narrows the distributed package from 138 to 70 Python modules. It removes
68 modules and selected out-of-scope declarations from shared files. There is no
second legacy package. Git history at `35cf019` retains the previous source.
This is scope reduction, not a kernel rewrite or scientific verification campaign.

## Deliberate removals and migration

| Removed scope | APIs / modules | Retained alternative or boundary |
| --- | --- | --- |
| Ocean dycore and drivers | `vfs.ocean`, `vfs.ocean_model`, `vfs.ocean_diagnostics`; associated root exports | No ocean replacement in this package |
| Land, ice and surface processes | `vfs.land`, `vfs.ice`, `vfs.surface` | Atmospheric terrain, boundary data and solar forcing remain; they are not surface-process solvers |
| Coupling | `vfs.coupling`, its surface/ARK orchestration and root exports | Atmospheric explicit/IMEX integrators remain in their existing owners |
| Coupled component graph | `vfs.core.graph`, including its atmospheric adapters/observers | Use existing `VFSModel`/`HeldSuarezModel`, direct initializers and I/O; this is an intentional graph-API removal, not a drop-in configuration conversion |
| Learned pressure/damping/transport | `vfs.atmosphere.pgf_correction`, `LearnedAdaptiveDivergenceDamping`, `MthetaMUSCLActivatorMLP`, `DryHydrostaticLearnedSoftMUSCLMthetaTransportDynamics`, learned recipe/feature constant | Ordinary strategies/recipes remain; `learned_potential` now raises an explicit removal error |
| ML mesh kernels | `vfs.geometry.mesh.kernels` and geometry/mesh exports | Geodesic geometry, mesh generation, mesh I/O and multilevel transfer/hierarchy remain |
| ML transport harness | `neural_reconstruction`, `shadow_replay`, `reversible_advection` under `vfs.transport` and their exports | Synthetic tracer initialization, shadow tracer transport and differentiable shadow diagnostics remain |
| Ocean initialization | Ocean-domain constructors, wet-component filtering and ocean-depth smoothing in `vfs.data.topography` | Shared ETOPO reading/regridding and atmospheric topographic column helpers remain |

The reversible-advection module is the coupled classical/neural benchmark and
training harness, not a dycore dependency. Its benchmark API is deliberately
removed with the ML workflow. Production tracer strategies and diagnostic
functions are retained. No replacement benchmark campaign is added here.

The dry constructor keeps `pgf_correction_mode='none'` so old learned selections
fail clearly. ML-only checkpoint/model/architecture options are removed from its
signature. Ordinary default/custom pressure-gradient behavior is unchanged.
Unknown removed model/component registry names fail instead of importing deleted
domains. No silently substituted model is provided.

## Dependency trace and retained support

[SCOPE_DEPENDENCIES.json](SCOPE_DEPENDENCIES.json) records all deleted files,
the incoming static/lazy references before deletion and the changed retained
files/functions. Imports were traced before removal, then rechecked against
the installed wheel. The remaining lazy exports are exercised explicitly.

- `vfs.model` imported `AirSeaMomentumCoupler` only for registration and lazily
  loaded ocean/surface registries. Removing those hooks preserves atmospheric
  composition and the Held–Suarez initializer's model dependency.
- `vfs.data.topography` mixed atmospheric ETOPO ingestion with ocean-domain
  construction. Only the ocean functions/constants and their exports are removed.
  `earth.py` and `boundary.py` still consume the unchanged atmospheric helpers.
- `geometry.mesh.multilevel` supports `integrators.imex.ark2`; it is not an ML
  module. All integrators, mesh numerical code and transfer routines stay intact.
- `core.state`, `Field`, configuration/registry, process/physics infrastructure,
  moisture helpers and diagnostic contracts still support dry state, model
  composition, initialization and I/O. Their simplification is deferred, not
  hidden in this deletion. Existing SCM adapters remain atmospheric support;
  they are not expanded or newly qualified here.
- Torch remains required for tensors, gradients and dycores, independently of
  ML removal. The obsolete `surface` extra is removed. `test` stays pytest-only;
  data/plotting/diagnostics/workflows/dev/docs/book extras remain for retained
  ingestion, rendering and development. No new runtime dependency is added.

59 retained Python modules are byte-identical to the base. Retained declarations
in the mixed files were checked by AST: only `DryHydrostaticDynamics.__init__`,
`_build_component` and `VFSModel.from_config` changed, for removed selection and
registry paths. Numerical expressions/defaults, the 99 dry helper functions,
all integrators and both baseline fixtures remain unchanged. BoundedGibbs still
uses the Euler species amount ledger; MoistAtmosphere still selects
`release_as_heat=True` and requires strictly positive vapor without a floor.

## Validation

Local installed-wheel results and actual hosted results are recorded below.
The wheel is built in a fresh staging directory to exclude stale build output,
installed into site-packages, then tested from `/tmp/vfs-scope-check` with `-I`.
Scope tests inspect distribution contents, import every retained module/lazy
export, reject removed APIs and reject dangling imports. Earlier compatibility
tests for learned PGF and ML mesh kernels are explicitly revised for removal.

Local: Apple M3, macOS 14.6.1 arm64, Python 3.12.13, torch 2.15.0 with Accelerate,
NumPy 2.5.4, SciPy 1.18.1, pytest 9.2.0; OMP/MKL/OpenBLAS thread controls are 1.
Final installed-wheel portable suite: **527 passed** (17 seconds), including
both fixture tests, all 99 helper AST checks and the scope tests. Standalone
portable tools passed **142 checkpoint-54 + 193 dry tensors**. Both examples
completed 300 simulated seconds. Wheel inspection confirms exactly 70 Python
modules, each matching current retained source bytes. `git diff --check` passed;
fixtures and regression policy are unchanged. The suite also exercises explicit
exact CLI status reporting; it does not turn exact failures into passes.

GitHub Actions has not yet run this scope-reduction commit. Its required
installed-wheel gate now runs both examples, in addition to the full portable
suite and both standalone comparisons. Hosted results will be linked in the PR.

Only CPU float64 forward regression is qualified by these checks. Portable and
exact remain separate modes; known cross-backend exact failures are not passes.
No GPU, long-run or full backward qualification is claimed.

## Next specific readability work

1. Document the dry helper method index by physical tendency and trace one
   complete dry RHS evaluation, without changing its expressions or binding.
2. Give the retained dry State/Field/configuration boundary a concise ownership
   guide before proposing smaller data contracts; preserve tensor/gradient flow.
3. Explain bounded Euler transport, Gibbs correction and heat allocation in
   call order, with explicit units and shapes for each intermediate ledger.

These are follow-up readability tasks. No deep inheritance rewrite, new registry,
broad internal move or scientific verification campaign is bundled here.
