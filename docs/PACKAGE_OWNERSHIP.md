# Component ownership and structural migration

Structural baseline: merged main `3f806a947e2ff8a5947953137a96af02861d083f`,
confirmed through GitHub before branching. This is distinct from the historical
scientific baseline. PRs #18/#19 and their evidence stay on their own branches;
no candidate Gibbs formula, fixture change or reference-policy decision enters
this stack. Atmospheric verification campaigns are paused.

## Old path → implementation owner

| Existing implementation | Destination / owner |
|---|---|
| `model.DryAtmosphere`, `core.state` | `atmosphere.dry_model`, `atmosphere.state` |
| `geometry.column` | `atmosphere.column`: hybrid pressure and CP hydrostatics |
| `geometry.mesh.multilevel.restrict_state` | `atmosphere.imex.transfer`; tensor/Field mesh transfers stay shared |
| `integrators.imex` dry/moist waves, ARK systems, state norms, Schur/preconditioners/configuration | `atmosphere.imex` |
| `integrators.imex.tableau`, generic GMRES | `integrators.tableau`, `integrators.krylov` |
| `core.integrators` | `integrators.explicit`; no concrete state import; bounded SSPRK3 remains atmospheric |
| `geometry.regrid` field weights/remapping | stays shared; state interpretation goes to explicit component adapters |
| `core.io` coordinates/metadata/schedules/file mechanics | `io`; atmospheric snapshot field construction in `atmosphere.io` |
| `earth`, `era5`, `moisture`, `initial_conditions` | atmosphere: prescribed profiles, pressure interpolation and tracer semantics |
| `boundary` | atmosphere: retained atmospheric boundary prescription; generic data requirements remain `core.config` |
| `data.topography` | shared elevation acquisition/cache stays data; pressure/column/geopotential interpretation belongs to atmosphere |
| `field`, `geometry.mesh`, `geometry.geodesic` | shared horizontal geometry/tensor metadata; no component-state dependency |
| `core.config`, `core.diagnostic_contracts` | generic field/data validation and typing only |
| `atmosphere.bounded_transport` | keep atmospheric: CP graph plus dry/vapor/liquid/ice partition assumptions |

Three dependent slices: (1) state/column/model ownership; (2) integration and
coarse-state restriction; (3) initialization, remapping and I/O boundaries.
Documented old imports remain thin compatibility re-exports, with one underlying
implementation. Compatibility surfaces are explicitly not shared implementation.

## Dependency rules

Shared geometry, integration algorithms, I/O mechanics and field metadata do not
import atmospheric or ocean state/column implementations. Components import
shared operations; their adapters own physical field meanings and state layout.
Top-level public exports and named compatibility modules may expose components.
No universal state, dycore inheritance, registry or coupling system is introduced.
Steppers remain class instances; model geometry/dynamics ownership, tensor states,
autograd connections and explicit dtype/device conversion are preserved.
Distinct explicit RK sequences are not algebraically rewritten. Atmospheric
ARK4 scaling, wave systems, stage acceptance and failure behavior stay together.

## Ocean inventory (read-only)

Available archive: `VFS_native_conservative_moist/VFS-main/vfs/ocean` beside
this repository. It includes independent `OceanState`, `OceanColumn`, domain,
EOS, dynamics, forcing/mixed-layer physics, helpers and initial conditions.
The old model facade inherits `VFSModel` and uses a registry; dynamics/physics
also import that removed registry. These are historical source, not a supported
installed ocean configuration. No ocean solver/state is restored in this stack.
Any future ocean implementation owns its thickness/temperature/salinity physics
and column independently. Existing shared regrid ocean-field interpretation is
an output adapter boundary only, not evidence of an operational ocean model.
The moist amount limiter assumes nonnegative species partitioning a carrier;
signed heat/salt transports cannot inherit that guarantee.

## Validation and #19 rebase

Capture the unchanged installed baseline before edits. For each slice compare
all dry/moist states, tendencies, energies, bounded ledgers, ARK4 stage/solver
records and diagnostic gradients exactly in the same environment. Run installed
portable tests, both fixture tools, bounded directional-gradient tests and six
examples. Historical exact fixture comparisons remain separate. These are
migration checks, not new scientific qualification or expanded campaigns.

Keep `atmosphere/gibbs_means.py`, bounded transport formulas and fixture bytes
unchanged. After structural review, rebase #19's numerical commit separately;
it should still touch only Gibbs means. Rebase its tests/drivers with the new
owner imports, preserving historical artifacts and explicitly reviewing source
manifest path mappings. Do not treat a moved-file hash as numerical evidence or
accept arbitrary source changes in the scientific baseline verifier.
