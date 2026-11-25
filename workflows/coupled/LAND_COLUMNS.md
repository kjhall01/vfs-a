# Land columns and shared surface exchange

Run this experiment to exercise rain, snow, signed vapor, radiation, heat and
stress between PyAM21 and grassland or open water. Each cell has one surface.
Wet cells have a finite, thermally active, stationary 50 m liquid slab. For a restartable run with active ocean and sea ice, use the
[coastal campaign](../../handoff/COASTAL_CAMPAIGN.md).

With VFS installed from this checkout and the pinned PyAM21 environment active,
run a one-day CPU experiment from the repository root.

```bash
python -u -I workflows/coupled/run_land_columns.py \
  --source /glade/work/khall/PyAM21 --steps 36 --dt 2400 \
  --atmosphere-dt 800 --soil-water 135
```

The default 32-cell mesh has a meridional sector of grassland. Supply
`--mesh input-data/mesh_4096_exact_3000.npy` and optionally
`--land-mask land_mask.npy` to use your geometry. The mask is a 1-D Boolean
array in mesh generator order with both land and water present.
Use `--surface-height height_m.npy` for terrain. Supply one elevation in metres
per mesh cell, including the wet outlets. The default elevation is zero.
Initial surface pressure follows the hydrostatic integral of each column's prescribed
temperature and specific humidity from sea level to the terrain.
`--radiation-data input-data/sea_esf --radiation-step 7200` enables SeaEsfRad.
Without tables, radiation is off. Omit `--atmosphere-dt` for a column-physics
experiment. Both modes use the real PyAM21 column solver. The example uses
its configurable 0.1 m/s minimum gust speed for initially calm, stable cells.
The experiment starts with no snow or canopy water. The initial 135 kg/m² of
soil water is intentionally above field capacity to exercise drainage.

## Land physics and budgets

The source is the original VFS archive at `62095de`, particularly its
`vfs/land/hydrology.py`, `soil_thermal.py`, `snow.py` and `vegetation.py`.
The port retains the four soil thicknesses (0.10, 0.30, 0.60, 1.00 m),
grassland canopy and root parameters, root-water extraction above wilting,
downward soil heat conduction, snow insulation and albedo, infiltration and
gravity drainage. The initial experiment uses liquid canopy interception,
a snow water store, fixed soil/skin heat capacities and one land-use class.

A paired 2,400 s grassland column check against the original archive reproduced
all four layer water amounts and temperatures, runoff, albedo and roughness
exactly in CPU float64. Moisture availability differed by 1.11e-16.

The skin and snow determine the land albedo and roughness supplied to PyAM21.
Root-zone moisture, canopy wetness and snow cover control the surface humidity
available to the bulk flux calculation. When the implicit air solve requests
more evaporation than the column could supply at the beginning of the step,
the adapter repeats that humidity solve with the available flux. Land and air
then book the same signed water amount. Canopy, snow and rooted soil fund
evaporation in that order. Dew enters liquid storage; frost enters snow.

Land uses the common VFS atmospheric enthalpy datum. Its energy inventory is
the sensible heat of skin and soil, minus the fusion energy of stored snow,
plus `surface_geopotential * (soil_water.sum(-1) + canopy + snow)`.
All stores share the atmospheric surface elevation. Precipitation and dew carry
this potential energy into storage, and evaporation carries it back to air.
Infiltration and redistribution within the column occur at one elevation.
The skin receives the shared exchange energy after accounting for the actual
soil heat increment over all diffusion substeps, snow phase change and outgoing
runoff enthalpy. Runoff is assigned liquid enthalpy at the exporting skin
temperature and carries its source geopotential. Mass and total exported energy
move to the nearest wet cell, weighted by donor and recipient area. The slab
retains the water at its outlet elevation and receives the potential drop
`runoff_mass * (phi_land - phi_outlet)` as heat. The driver reports this heat
per unit global area. A nearest outlet above its donor is rejected at setup. There is no external runoff sink.
Both surfaces are stationary, so the reported support impulse has zero work.
The atmospheric upstream energy residual stays visible.

The command prints separate budgets per unit land area and per unit global
area. On the land-area basis, precipitation minus vapor to air minus runoff
equals storage change. The reported global numerical energy residual includes
atmospheric dynamics when enabled. The driver also measures the energy change
across those dynamics substeps independently.
