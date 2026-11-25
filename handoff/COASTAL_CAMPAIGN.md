# Running the coupled model with continents

Use `--island` on the campaign or throughput profiler for a smooth equatorial
island up to 1,000 m high in a flat-bottom ocean. Both commands default to the
aquaplanet when invoked without geography. Run the pair on the same A100
allocation and software environment. This example uses the 4,096-cell mesh,
32 atmospheric and 40 ocean levels, a 2,400 s physics interval, 800 s atmospheric
dynamics and 7,200 s radiation. Each timed segment spans 16 simulated hours.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
run_dir=$(mktemp -d ./coastal-profile-XXXXXX)
common=(--config workflows/coupled/aquaplanet_multirate_4096_40.yaml
        --mesh input-data/mesh_4096_exact_3000.npy
        --source ../PyAM21
        --radiation-data input-data/sea_esf
        --device cuda:0 --dtype float64 --threads 4 --no-compile
        --set atmosphere_dt=800 --set radiation_dt=7200
        --warmup-intervals 4 --measure-intervals 24)
python -u -I tools/profile_coupled.py "${common[@]}" --output "$run_dir/aquaplanet"
python -u -I tools/profile_coupled.py "${common[@]}" --output "$run_dir/island" --island
```

For a campaign, set the timesteps in the YAML and use the existing
`workflows/coupled/campaign.py` command with `--island` or
`--geography coastline.npz`. Add `--preflight` to exercise one interval and
restart replay. For continuation supply the same geometry, mesh and configuration
with `--resume` and a fresh output directory.

## Supplied terrain and bathymetry

Use a NumPy archive with one entry per mesh cell, in generator order.

| Array | Meaning |
|---|---|
| `land` | Boolean, true on land. Both surfaces must be present. |
| `height_m` | Land surface elevation above sea level in metres, zero at sea. |
| `depth_m` | Resting liquid ocean depth in metres, positive at sea. Dry entries can be zero. |

The atmosphere starts from an idealized pressure-dependent thermal profile with
hydrostatically adjusted surface pressure. The Earth presets truncate the reference layers at the bed and merge thin bottom
cells. Other presets retain stretched fractions of local depth by default. Check that the shallowest wet cells support the ice
solver's minimum liquid thickness and that the timestep suits the chosen terrain.
The coastal geometry is static. Nearest-wet runoff is delivered within one physics
interval. This setup supports land at or above the receiving sea level.

## Exchanges, diagnostics and restarts

Each atmospheric column meets grassland or a mosaic of open water and two ice
categories. Land evaporation draws on canopy, snow and available soil moisture.
Runoff enters the active ocean top layer as freshwater carrying its liquid
enthalpy and source geopotential. Salt inventory is preserved during delivery.
The ocean/ice Hamiltonian accounts for thickness, dilution, and mechanical
energy before the thermal increment is solved. The stationary coast takes the
blocked impulse at zero work. Closed edges carry no normal ocean or ice transport.

Diagnostics include wet-only ocean extrema, combined water, energy and salt
budgets, land water storage, and cumulative runoff and downhill heat per unit
global area. Snapshots include the land index, four-layer water and temperature,
canopy, snow, skin temperature, surface elevation and ocean depth. Restarts
include land state and identify the mask, bathymetry, terrain, and mesh.

## Verification

CPU float64 tests exercised coastal ice convergence, saturated-soil runoff on
a sloping shelf, the receiving Hamiltonian, synthetic radiation and bitwise
checkpoint replay. In an 18-interval, 16-cell island smoke run with active atmospheric
and ocean dynamics and with radiation disabled, the total-water
residual was −1.46e−10 kg/m², relative salt change −8.26e−16 and numerical
energy residual −1.05e−5 J/m². A100 performance and long-term climate behavior
require separate measurements.

On the saved `mesh_4096_exact_3000.npy` (SHA256 starting `3e1d9200`),
the island covers 717 of 4,096 cells. A CPU float64 radiation-off smoke
profile with the 32/40-level preset completed a warmup and two timed
intervals. The two land-column calls took 0.002 s in total. The six runoff
routing calls took 0.001 s in total. A separate 128-cell smoke profile
completed four warmup and 24 timed intervals with the same levels and
800 s atmosphere timestep.

Use the saved mesh for the 4,096-cell preset above. On an unrelaxed generated
mesh, both the aquaplanet and island experiments failed the ocean carrier
outflow check in the first interval at the same cell and layer.
