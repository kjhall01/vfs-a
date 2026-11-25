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

The campaign checkpoint still stores the same number of terrain-following
layers in every wet column. Vertical mixing, the rotational velocity stencil,
surface drag and the ALE target use that layout. The standalone shelf
experiment exercises variable counts, merged bottom cells and moving-face
momentum with zero-thickness padding.

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
| Voronoi generators | `input-data/mesh_4096_exact_3000.npy` | `VFS_MESH` |
| Geography archive | `input-data/earth_4096_geography.npz` | `VFS_GEOGRAPHY` |
| SeaEsfRad tables | `input-data/sea_esf` | `PYAM21_RADIATION_DATA` |
| PyAM21 source | `../PyAM21` | `PYAM21_SOURCE` |

The geography supplies land mask, grid-mean terrain and wet-column depth in
matching mesh order. The loader verifies its embedded mesh digest when present.
Terrain enters the atmosphere through surface geopotential and a height-adjusted
initial pressure. The zero-wind, latitude-dependent initialization undergoes
adjustment. The supplied geography has a 600 m minimum wet-column depth.

The 4,096-cell Earth benchmark preset completed six simulated hours on CPU
in float64 with radiation disabled. With 600 s atmospheric dynamics and
1,200 s ocean dynamics, this covered 36 GM/Redi updates and four ALE remaps.
The whole-system numerical energy residual was −279.47 J/m², of which
−277.38 J/m² accrued during atmospheric dynamics. The earlier 800 s
atmospheric run with the same mesh, terrain and ocean schedule gave
−426.66 J/m². Both runs used the same terrain-pressure initialization.

At rest on the same grid with constant salinity of 35 g/kg, constant CT of
12 °C gives a maximum resolved ocean acceleration of 5.12e-17 m/s². For
CT = 12 + 0.001 z °C, with z in meters positive upward, the maximum is
4.73e-9 m/s² and the 99th percentile among open velocity slots is 1.95e-9 m/s².
These measurements exclude ice loading and external forcing.
