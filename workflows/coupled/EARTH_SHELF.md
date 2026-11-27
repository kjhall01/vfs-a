# Coupled Earth shelf pilot

The runner advances the native moist atmosphere and PyAM21 physics, seasonally
forced SeaEsfRad radiation, land and runoff, ocean, moving sea ice and snow.
With `shelf_redi: true`, resolved ocean transport intersects the left and
right water columns with the staggered velocity layers at every RK stage.
Only the portion of a face wet on both sides can carry water. One bounded
flux ledger carries CT, SA, passive tracers, TKE and EKE. The pressure force
(including ice loading) uses the transpose of that volume-flux map. With the
Roquet caloric law, a cell's bounded CT update also preserves at least the
entropy brought by the donor water.

GM constructs its streamfunction on merged reference partial cells, evaluates
it at the physical boundaries of the prognostic wet openings and closes the
circulation with a vertical return flow. Redi averages native concentrations
onto the partial cells and returns fluxes through the transpose of that map.
Both use the prognostic inventories for their local eddy-energy settlement.

Both Earth presets enable `partial_cell_layers: true`. Each wet column uses the
reference interfaces above its seabed. The last layer ends at the bed. A bottom
sliver thinner than one quarter of its nominal layer merges into the layer above.
In trenches the last nominal layer extends down to the bed. For example, at
initialization a 40 m column has a 10 m surface layer and 30 m of water below
it. An 8 m column has one layer. The deepest columns have all 40. The count is
fixed by bathymetry as layer thicknesses and sea level evolve.

### Experimental shelf momentum

For a fresh Earth-shelf campaign, these configuration keys enable the
opening-weighted ocean Hamiltonian and the closed-face treatment.

```yaml
shelf_redi: true
partial_cell_layers: true
matched_shelf_momentum: true
```

The opening can close even while both adjacent cells still contain water in
that layer. Rotation and backscatter use only velocities supported by the
instantaneous opening. If a velocity slot closes or reopens during dynamics,
overlapping water supplies its normal momentum. Mixing donor velocities
transfers the lost shear kinetic energy to ocean heat. ALE uses the same
opening for momentum and the column kinetic-energy budget. The option defaults
to `false`. An in-place restart uses the saved configuration by default.

`max_ocean_speed_m_s` is the largest absolute normal velocity on an actually
open face. Use `max_ocean_edge` and `max_ocean_level` to locate its zero-based
edge and layer in a snapshot or restart.

### Run the coupled shelf and native-edge ice drag together

The [395-day preset](earth_shelf_4096_395d_native_drag.yaml) enables
`matched_shelf_momentum: true` and `ice_ocean_drag: native_edge`. It uses the
4,096-cell Earth mesh, 40 ocean levels, seasonal SeaEsfRad radiation, CPU
float64 and the timesteps of the 395-day Earth pilot. Use the same mesh,
40 m-floor geography, PyAM21 source and radiation tables as the pilot.
From the repository root with the coupled environment active:

```bash
cfg=workflows/coupled/earth_shelf_4096_395d_native_drag.yaml
export VFS_GEOGRAPHY=input-data/earth_4096_geography_40m.nc
python -u tools/run_earth_shelf.py --config "$cfg" --preflight --output runs/native-drag-preflight
python -u tools/run_earth_shelf.py --config "$cfg" --output runs/native-drag-395d
```

Set `VFS_GEOGRAPHY` to the actual archive path if it differs.
The preflight runs one coupled interval twice and checks restart equality.
The second command initializes at day zero. To run the first day alone, add
`--stop-after-days 1`. Continue in the same directory with

```bash
python -u tools/run_earth_shelf.py --restart --output runs/native-drag-395d
```

Use `--resume FILE --output NEW_DIR` to fork a separate segment.
The [surface output and restart guide](CAMPAIGN.md#outputs-and-restarting)
describes configurable means, field units, input overrides and grid checks. Both operators are recorded in the startup log and manifest.
The saved configuration records the drag law and shelf metric of the run that wrote it.

The campaign applies native-edge drag for both half steps around ocean–ice
dynamics. It exchanges equal and opposite impulses and credits the kinetic
energy loss to surface-ocean TKE. For a paired control using the same shelf
geometry, set `ice_ocean_drag: midpoint` in a copy of the preset.

Snapshots store `ocean_active_layer_count` for each cell (zero on land).
Checkpoints retain rectangular tensors. Slots below a wet column's bed have
exactly zero thickness and zero inventories of heat, salt, tracers, TKE and EKE.
Normal velocities and transport close at shelf walls. Vertical mixing and
backscatter act on the open layers and faces. Sea ice and runoff exchange with
the top wet layer. Start a fresh run when switching from the old stretched
40-layer state.

ALE targets each column's own reference profile scaled to its current liquid
depth. It conserves edge-integrated momentum and checks column Hamiltonian
and entropy after remapping. If the thermal correction is infeasible, the
column and its incident wet neighbors can keep their current interfaces for that
event. This holds the shared face momentum fixed while the rest of the grid
rezones. `ale_changed_columns` and `ale_deferred_columns` count completed and
deferred column rezones across events. Both groups continue advancing dynamics
and physics.

From the repository root, with the coupled environment active and a CUDA device available, run

```bash
python -u tools/run_earth_shelf.py --device cuda:0 --preflight \
    --output runs/earth-shelf-preflight
python -u tools/run_earth_shelf.py --device cuda:0 --stop-after-days 1 \
    --output runs/earth-shelf-day1
```

Fresh runs and preflights require a new or empty output directory. The preflight checks one coupled
interval and its restart replay. The day-one command starts a fresh run.
The preset has a 30-day horizon, 600 s atmospheric dynamics, 2,400 s physics,
600 s ocean dynamics, and 7,200 s radiation and ALE. It defaults to CPU
float64 eager execution. Use the same device and configuration for restart.

The Python launcher supplies defaults to the same `workflows.coupled.campaign`
entry point. To use scattered inputs, pass `--mesh` and `--geography` as file
paths and `--source` and `--radiation-data` as directory paths. Explicit flags
override the environment variables below. Relative paths you supply are
resolved from your current working directory. Built-in defaults are relative
to the checkout. For the benchmark schedule, pass
`--config workflows/coupled/earth_shelf_4096_benchmark.yaml` from the repo root.

| Input | Default | Environment override |
|---|---|---|
| Voronoi generators | `input-data/mesh_4096_exact_3000.nc` (also accepts `.npy`) | `VFS_MESH` |
| Geography archive | `input-data/earth_4096_geography.nc` (also accepts `.npz`) | `VFS_GEOGRAPHY` |
| SeaEsfRad tables | `input-data/sea_esf` | `PYAM21_RADIATION_DATA` |
| PyAM21 source | `../PyAM21` | `PYAM21_SOURCE` |

### Existing inputs and NetCDF output

See [native mesh plotting](../../docs/NATIVE_MESH_PLOTTING.md) to draw a field on a Robinson map or 3D globe.

You can keep using your original `.npy` generators and `.npz` geography.
The launcher prefers the NetCDF files when both versions of a default input
exist. To convert a matched pair once, from the checkout:

```bash
python tools/convert_earth_inputs.py \
    --mesh input-data/mesh_4096_exact_3000.npy \
    --mesh-output input-data/mesh_4096_exact_3000.nc \
    --geography input-data/earth_4096_geography_40m.npz \
    --geography-output input-data/earth_4096_geography_40m.nc
```

This preserves generator order, mesh identity, bathymetry floor, raw depths and
land fractions. A VFS `VoronoiMesh.save` operator cache can also supply its
ordered centers. The campaign builds the configured operators from those centers.

A new run writes snapshots and complete restarts as separate NetCDF4 datasets.
Both carry native mesh connectivity, cell and edge positions, hybrid pressure
coefficients, nominal and per-column reference thickness, and the number of
active ocean layers. `land_cell_index` maps compact land fields back to mesh
cells. `soil_layer` is the soil midpoint in metres below ground. Geometry
coordinates have no time axis. Instantaneous pressure and wet ocean midpoint
depth in snapshots follow the evolving state.

```python
from pathlib import Path
import xarray as xr

run = Path("runs/earth-shelf-40m-day1")
with xr.open_dataset(run / "snapshots/state_00000000.nc") as ds:
    wet_temperature = ds.ocean_temperature.where(ds.ocean_layer_depth.notnull())
    shelf = (ds.ocean_depth_m > 0) & (ds.ocean_depth_m <= 40)
    print(wet_temperature.isel(time=0, ocean_level=0).where(shelf, drop=True).load())

with xr.open_dataset(run / "diagnostics.nc") as diag:
    print(diag.numerical_energy_residual_J_m2.to_series())
```

**Reading SST from a restart.** A snapshot contains `ocean_temperature` in
°C. A restart stores `ocean_temp_h`, layer thickness times conservative
temperature (m·°C). Dry land cells can have finite placeholder thicknesses
and temperatures. Use the active layer count for the wet mask.

```python
with xr.open_dataset(run / "restarts/restart_00086400.nc") as ds:
    wet = ds.ocean_active_layer_count > 0
    h = ds.ocean_h.isel(ocean_level=0).where(wet)
    sst_degC = (ds.ocean_temp_h.isel(ocean_level=0) / h).load()
```

This includes water beneath sea ice. For a full-depth field, the active slots
satisfy `ds.ocean_level < ds.ocean_active_layer_count`. The atmospheric
`air_surface_t_surf` covers the whole mesh, including land.

To continue a one-day segment, pass the same inputs and configuration with
`--resume runs/earth-shelf-40m-day1/restarts/restart_00086400.nc`,
a new output directory and the new end day. For example,
`--stop-after-days 14` ends at model day 14 counted from initialization.
The end must fall on a whole second within the configured campaign horizon.
Omit the flag to run to the configured final day. The restart holds native prognostic
inventories, the coupled clock, budget ledgers, and PyAM21 physics and radiation
memory. Use xarray directly for inspection.

To continue a legacy `.pt` from the immediately preceding `main` (`d342be9`),
pass both `--resume /path/to/old-run/restarts/restart_00086400.pt` and
`--resume-compatible-manifest /path/to/old-run/manifest.json`. This checks the
original fingerprint and the audited source transition. The next restart is
NetCDF. Earlier revisions remain subject to their own source-compatibility checks.

### Continue the 30-day Earth pilot through day 395

The [395-day preset](earth_shelf_4096_395d.yaml) uses four-stage, third-order SSPRK(4,3) for the bounded atmosphere at 600 s. Set `atmosphere_dynamics_integrator: ssprk43` to select it. The ocean–ice stepper is independently set to `ocean_dynamics_integrator: ssprk43` at 1,200 s. Physics runs every 1,200 s, with five mixing substeps and two GM/Redi substeps per 1,200 s. Radiation and ALE run every 4,800 s. The preset uses CPU float64 with compilation and four threads.

From the repository root, set `VFS_MESH`, `VFS_GEOGRAPHY`, `PYAM21_SOURCE` and `PYAM21_RADIATION_DATA` to the paths used for the pilot. Select the segment that **wrote** the day-30 checkpoint. For example:

```bash
pilot=runs/earth-final-segment
python -u tools/run_earth_shelf.py \
  --config workflows/coupled/earth_shelf_4096_395d.yaml \
  --resume "$pilot/restarts/restart_02592000.nc" \
  --resume-compatible-manifest "$pilot/manifest.json" \
  --output runs/earth-day030-to-395
```

Day 395 is absolute elapsed time, so this advances another 365 simulated days. The manifest check uses the checkpoint's original configuration and records both the longer horizon and the time of the atmospheric integrator switch in `compatible_resume_audit.json`. It also verifies model sources, input identities, runtime, compilation, and numerical settings. If the original run used different settings, copy its `configuration.json` and set `days: 395` and `atmosphere_dynamics_integrator: ssprk43`. Use that file as `--config`. To initialize a fresh 395-day run, omit both resume arguments and choose a new output directory. The PyAM21 calendar is Julian and the configured seasonal forcing repeats every 365 simulated days.

### Generate the mesh-matched geography

From the checkout, use a Python environment with NumPy, SciPy, netCDF4 and
Shapely 2. The script downloads and verifies the GMT SRTM15 relief grid and
Natural Earth land polygons, then caches them beside the output archive.
Supply the same mesh file to the generator and the campaign.

```bash
python tools/build_earth_geography.py \
    --mesh input-data/mesh_4096_exact_3000.npy \
    --min-wet-depth-m 40 \
    --output input-data/earth_4096_geography_40m.nc
```

Use this archive explicitly for a fresh 40 m shelf pilot with 600 s atmospheric
dynamics, 1,200 s ocean SSPRK3, five mixing substeps, two GM/Redi substeps, and
4,800 s radiation and ALE. With the source and radiation paths already set:

```bash
python -u tools/run_earth_shelf.py --device cuda:0 \
    --config workflows/coupled/earth_shelf_4096_benchmark.yaml \
    --mesh input-data/mesh_4096_exact_3000.npy \
    --geography input-data/earth_4096_geography_40m.nc \
    --stop-after-days 1 --output runs/earth-shelf-40m-day1
```

The generator defaults to a 600 m depth floor for reproducibility. An existing
600 m archive has already deepened the shallower cells. Use the 40 m archive
for this experiment.
To use existing source files offline, pass `--relief /path/to/earth_relief_10m_p.grd`
and `--coastline /path/to/ne_50m_land.geojson`. The archive includes the
unfloored wet depth and the fractional land area alongside the campaign fields.

The geography supplies land mask, grid-mean terrain and wet-column depth in
matching mesh order. The loader verifies its embedded mesh digest when present.
For fresh runs the atmosphere computes terrain pressure by integrating
`R_mix T` in log pressure from sea level (100,000 Pa) using the same local
temperature and vapor profile used to construct the atmospheric state. The
zero-wind atmosphere still adjusts to its meridional temperature gradient.
The historical comparison below used the 600 m floor and the original stretched columns.

At the 600 s atmosphere step, the following six-hour comparison used the same
4,096-cell mesh, geography, CPU float64 configuration, disabled radiation and
1,200 s ocean step. Both runs completed 36 GM/Redi updates and four ALE remaps.

| Numerical energy residual (J/m²) | Previous 288 K pressure estimate | Moist hydrostatic initialization |
|---|---:|---:|
| Atmosphere dynamics, first 40 minutes | −49.29 | −1.79 |
| Atmosphere dynamics, six hours | −277.38 | −16.39 |
| Whole coupled system, six hours | −279.47 | −18.48 |

The earlier six-hour run at an 800 s atmosphere step used the 288 K pressure
estimate and gave a whole-system residual of −426.66 J/m².

In the earlier stretched-column test, at rest with constant salinity of 35 g/kg, constant CT of
12 °C gives a maximum resolved ocean acceleration of 5.12e-17 m/s². For
CT = 12 + 0.001 z °C, with z in meters positive upward, the maximum is
4.73e-9 m/s² and the 99th percentile among open velocity slots is 1.95e-9 m/s².
These measurements exclude ice loading and external forcing.
