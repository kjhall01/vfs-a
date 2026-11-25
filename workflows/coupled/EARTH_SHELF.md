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

On an allocated Derecho node, activate the VFS/PyAM21 environment, then run

```bash
cd /glade/work/khall/vfs-a
python -u tools/run_earth_shelf.py --device cuda:0 --preflight \
    --output /glade/work/khall/earth-shelf-preflight
python -u tools/run_earth_shelf.py --device cuda:0 --stop-after-days 1 \
    --output /glade/work/khall/earth-shelf-day1
```

Each output directory must be new or empty. The preflight checks one coupled
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

You can keep using your original `.npy` generators and `.npz` geography.
The launcher prefers the NetCDF files when both versions of a default input
exist. To convert a matched pair once, from the checkout:

```bash
python tools/convert_earth_inputs.py \
    --mesh input-data/mesh_4096_exact_3000.npy \
    --mesh-output /glade/work/khall/mesh_4096_exact_3000.nc \
    --geography /glade/work/khall/earth_4096_geography_40m.npz \
    --geography-output /glade/work/khall/earth_4096_geography_40m.nc
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

run = Path("/glade/work/khall/earth-shelf-40m-day1")
with xr.open_dataset(run / "snapshots/state_00000000.nc") as ds:
    wet_temperature = ds.ocean_temperature.where(ds.ocean_layer_depth.notnull())
    shelf = (ds.ocean_depth_m > 0) & (ds.ocean_depth_m <= 40)
    print(wet_temperature.isel(time=0, ocean_level=0).where(shelf, drop=True).load())

with xr.open_dataset(run / "diagnostics.nc") as diag:
    print(diag.numerical_energy_residual_J_m2.to_series())
```

To continue a one-day segment, pass the same inputs and configuration with
`--resume /glade/work/khall/earth-shelf-40m-day1/restarts/restart_00086400.nc`,
a new output directory and the new end day. The restart holds native prognostic
inventories, the coupled clock, budget ledgers, and PyAM21 physics and radiation
memory. Use xarray directly for inspection.

To continue a legacy `.pt` from the immediately preceding `main` (`d342be9`),
pass both `--resume /path/to/old-run/restarts/restart_00086400.pt` and
`--resume-compatible-manifest /path/to/old-run/manifest.json`. This checks the
original fingerprint and the audited source transition. The next restart is
NetCDF. Earlier revisions remain subject to their own source-compatibility checks.

### Generate the mesh-matched geography

From the checkout, use a Python environment with NumPy, SciPy, netCDF4 and
Shapely 2. The script downloads and verifies the GMT SRTM15 relief grid and
Natural Earth land polygons, then caches them beside the output archive.
Supply the same mesh file to the generator and the campaign.

```bash
python tools/build_earth_geography.py \
    --mesh input-data/mesh_4096_exact_3000.npy \
    --min-wet-depth-m 40 \
    --output /glade/work/khall/earth_4096_geography_40m.nc
```

Use this archive explicitly for a fresh 40 m shelf pilot with 600 s atmospheric
dynamics, 1,200 s ocean SSPRK3, five mixing substeps, two GM/Redi substeps, and
4,800 s radiation and ALE. With the source and radiation paths already set:

```bash
python -u tools/run_earth_shelf.py --device cuda:0 \
    --config workflows/coupled/earth_shelf_4096_benchmark.yaml \
    --mesh input-data/mesh_4096_exact_3000.npy \
    --geography /glade/work/khall/earth_4096_geography_40m.nc \
    --stop-after-days 1 --output /glade/work/khall/earth-shelf-40m-day1
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
