# Coupled Earth shelf pilot

The runner advances the native moist atmosphere and PyAM21 physics, seasonally
forced SeaEsfRad radiation, land and runoff, ocean, moving sea ice and snow.
The bathymetry-aware Redi operator averages native concentrations onto
partial cells and returns fluxes with the transpose of the same overlap map.
The positive-inventory limiter and local eddy-energy settlement operate on
the prognostic inventories. Resolved dynamics, GM, vertical mixing, ice drag
and ALE use the native ocean layers. This arrangement requires no auxiliary
face-momentum state in a checkpoint.

On an allocated Derecho node, activate the VFS/PyAM21 environment, then run

```bash
cd /glade/work/khall/vfs-a
bash tools/run_earth_shelf.sh --device cuda:0 --preflight \
    --output /glade/work/khall/earth-shelf-preflight
bash tools/run_earth_shelf.sh --device cuda:0 --stop-after-days 1 \
    --output /glade/work/khall/earth-shelf-day1
```

Each output directory must be new or empty. The preflight checks one coupled
interval and its restart replay. The day-one command starts a fresh run.
The preset has a 30-day horizon, 800 s atmospheric dynamics, 2,400 s physics,
600 s ocean dynamics, and 7,200 s radiation and ALE. It defaults to CPU
float64 eager execution. Use the same device and configuration for restart.

Default paths are relative to the checkout. Each can be overridden independently.

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

On the supplied 4,096-cell geography, a CPU float64 two-hour radiation-off run
completed 60 GM/Redi updates and one ALE remap. The maximum local GM/Redi
energy residual was 4.0e-12 J/m². Whole-system numerical energy changed by
−202.1 J/m². A separate preflight replayed bitwise. Prescribed-radiation tests
exercise the coupled radiation interface; a real-table run on the chosen
device is the next qualification.
