# 4,096-cell full-depth coupled aquaplanet (experimental)

This preset is built on `feat/coupled-cuda` and changes **vertical geometry** without turning off any of the existing coupled components: moist atmosphere and PyAM21 clouds/convection/radiation, prognostic ocean dynamics, TKE mixing, GM–Redi, eddy dissipation, funded backscatter and biharmonic sink, adaptive ALE, moving sea ice, snow and prognostic ice temperature.

| Item | Setting |
| --- | --- |
| Horizontal grid | 4,096 generated Voronoi cells (10 Lloyd iterations) |
| Atmosphere | 32 levels, lid at 100 Pa (1 hPa) |
| Ocean | 60 levels, 5,000 m constant depth, uppermost 10 m |
| Precision | CUDA float64 |
| Timesteps | 240 s atmosphere, 1,200 s ocean / coupling, 3,600 s radiation |
| Ocean closures | All existing switches enabled (see YAML) |
| Output | Six-hour snapshots through day 14, daily thereafter; six-hour diagnostics, **daily** restart |
| Calendar | January 1, 2001 to January 1, 2002; prescribed seasonal and diurnal insolation |

The ocean uses the existing analytic initial temperature field with an exponential 250 m thermocline e-folding scale, and constant 34 g/kg salinity. **This is not a spun-up, full-depth equilibrium state.** The 5,000 m bottom has not been validated for the existing mixing/bathymetry-free configuration. The atmosphere initialization uses the existing 205 K upper-temperature profile, extended to 1 hPa. No land, bottom topography or geographic basin boundaries exist.

## Derecho GPU: preflight first

On a CUDA-equipped compute node, after activating the same environment used for the working 16,384-cell 2-day campaign:

```bash
cd /glade/work/khall/vfs-a
git fetch origin
git switch experiment/full-depth-aquaplanet-4096
python -m pip install -e .
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p /glade/derecho/scratch/khall
python -u -I workflows/coupled/campaign.py \
  --config workflows/coupled/aquaplanet_full_depth_4096.yaml \
  --source /glade/work/khall/PyAM21 \
  --radiation-data /glade/work/khall/vfs-a/input-data/sea_esf \
  --device cuda:0 --dtype float64 --threads 4 --no-compile \
  --output /glade/derecho/scratch/khall/vfs-4096-full-depth-preflight \
  --preflight --verbose
```

No `--mesh` is provided: the preset generates 4,096 cells. **Do not pass the old 16,384-cell mesh**, which overrides the cell count. Retain the new run's `manifest.json`, which records the generated mesh/configuration identity for restart validation.

Check that `status.json` says `preflight_passed`. This proves one interval and bitwise restart replay only; it is not a long-run stability test.

## Initial one-day and two-day runs

For a first complete day, use a **new output directory**:

```bash
python -u -I workflows/coupled/campaign.py \
  --config workflows/coupled/aquaplanet_full_depth_4096.yaml \
  --source /glade/work/khall/PyAM21 \
  --radiation-data /glade/work/khall/vfs-a/input-data/sea_esf \
  --device cuda:0 --dtype float64 --threads 4 --no-compile \
  --stop-after-days 1 \
  --output /glade/derecho/scratch/khall/vfs-4096-full-depth-day1 --verbose
```

After examining the diagnostic history, resume to model day 2 from
`vfs-4096-full-depth-day1/restarts/restart_00086400.pt`, adding:
```bash
  --resume /glade/derecho/scratch/khall/vfs-4096-full-depth-day1/restarts/restart_00086400.pt \
  --stop-after-days 2 \
  --output /glade/derecho/scratch/khall/vfs-4096-full-depth-day1-to2
```
with the same config, source, radiation, device, dtype, threads, and compilation flags.

Do not launch a 365-day job before a two-day run and a comparison of numerical budgets and basic physics. Monitor `numerical_energy_residual_J_m2`, `upstream_residual_J_m2`, column temperature/salinity, minimum thickness, atmospheric velocity, ice area, and ocean TKE/EKE. A longer run requires a production queue and walltime/checkpoint planning; the develop queue is only for short tests.

## Important constraints

- The new 1 hPa atmospheric lid is passed into `make_hybrid_coefficients` through `example(top_pressure_pa=...)`; its defaults remain 10 hPa for existing users. The coupled YAML default is also unchanged.
- The 60-layer distribution uses the preexisting `rest_thickness` rule (10 m first layer, logarithmic stretching beneath), not a calibrated climate-model vertical coordinate. Deep stratification and timestep stability must be checked.
- Changing level counts, mesh, atmospheric lid, dtype or source invalidates ordinary restart identity; do not reuse the 16,384-cell checkpoints.
- All physics remain enabled; `--no-compile` changes **only compilation**, not physical parameterizations. Once eager CUDA is validated, try `--compile` using a compatible host C++ compiler and fresh output directory.
- No GPU runtime or long-term physical validation was performed in preparing this preset.
