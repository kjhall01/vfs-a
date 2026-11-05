# Component ownership and structural migration

> Historical structural record. Main now includes the independent ocean state,
> column, reference solver and explicit constant-density candidate through merged
> #26/#27. See [current ocean ownership](../verification/ocean/README.md).
> Statements below that ocean is not installed describe the earlier extraction.

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

## Resulting package map and migration notes

- `atmosphere/`: tensor state/model, pressure column, conservative dry and moist
  physics, forcing, thermodynamics, initial conditions, ERA5/terrain adapters,
  state output; `atmosphere/imex/` owns atmospheric waves and ARK4 systems.
- `geometry/`: horizontal meshes/connectivity/operators, transfer weights and
  generic `remap` tensor/Field operations. `field.py` remains shared metadata.
- `integrators/`: explicit arithmetic sequences, ARK4 coefficients, Krylov
  algorithms supplied with caller-owned inner products and vector arithmetic.
- `io.py`, `data/elevation.py`: coordinates/metadata, NetCDF mechanics,
  external elevation acquisition/cache. No component state or column imports.
- `ocean/regrid.py`: existing ocean output contract only, separated from shared
  remapping. No OceanState, OceanColumn, model, EOS or solver is installed.
- `core/`: generic field/data requirements and diagnostic typing; old state,
  integrator and I/O modules are compatibility import surfaces only.

Top-level imports still work. Prefer owner imports when reading implementations:

```python
from vfs.atmosphere import DryAtmosphere, MoistAtmosphere, State, CharneyPhillipsColumn
from vfs.atmosphere.imex import IMEXSolverConfig
from vfs.geometry.mesh import VoronoiMesh
from vfs.geometry.remap import LatLonGrid, MeshToLatLonKNN
from vfs.atmosphere.regrid import state_to_latlon

# Given an existing dry atmospheric state and grid:
# remapper = MeshToLatLonKNN.from_mesh(state.mesh, grid)
# output = state_to_latlon(remapper, state, fields="all")
# state.to_latlon(remapper, fields="all") is also supported.
```

`geometry.regrid.MeshToLatLonKNN` is a thin compatibility subclass preserving the
old `state_to_latlon` convenience and its historical component inference. **Only
that compatibility surface** infers the component; `geometry.remap` has no state
method or component import. The explicit ocean adapter takes the existing ocean
output contract; it neither imports nor creates an atmospheric state. Moist
physical fields continue to come from `MoistAtmosphere.physical_fields`; this
change does not invent a new moist lat/lon or restart format.

Old public imports under `model`, `core.state`, `geometry.column`, `imex`,
`integrators.imex`, `initial_conditions`, `earth`, `era5`, `moisture`, `boundary`
and `data.topography` remain aliases. Private callers of mesh-level
`restrict_state` must import `atmosphere.imex.transfer.restrict_state`.
Code monkeypatching implementation globals must patch the new owner module,
not its re-export facade. Old snapshot metadata strings and values are preserved.

The boundary audit split prescribed atmospheric surface profiles/pressure from
generic boundary-file writers/loaders (now shared I/O). ETOPO elevation is shared;
clipping elevation to atmospheric land height and converting it to geopotential
are atmospheric conventions, not ocean bathymetry operations. These pre-existing
formulas/defaults are unchanged. Archived ocean state stores h/temp_h/salt_h with
legacy Field wrappers and its own column; adapting that archive is future work,
not a reason to generalize current atmospheric state or limiter abstractions.

## Rebasing #19 after structural review

1. Keep the original #18/#19 commits and artifacts available. Rebase the
   investigation branch onto the reviewed structural stack first, then rebase
   the numerical-fix branch onto that rebased investigation. Do not merge its
   numerical patch into these structural PRs.
2. The numerical patch still belongs solely in `atmosphere/gibbs_means.py`;
   that file and both bounded transport files are byte-identical to merged main
   throughout this stack. Review any other proposed runtime change separately.
3. Update implementation imports in probes/tests to owner paths. Preserve the
   22 historical old/new run archives, manifests and reference failures. The
   original scientific baseline stays identifiable, distinct from structural
   and numerical commits.
4. The scientific source verifier intentionally still rejects moved runtime
   files. A separately reviewed, exact structural-source mapping is needed
   before using it on the reorganized package. Do not remove source checks,
   accept arbitrary trees or regenerate fixtures. Rerun same-environment #19
   pre/rebase capture and existing checks when authorized; no new experiment
   or reference-policy decision is implied here.
