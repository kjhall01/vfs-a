# 4,096-cell full-depth coupled aquaplanet (experimental)

## Fresh 30-day run with two GM/Redi substeps

Run the following from the repository root in the coupled environment, with PyAM21 installed. The example uses the saved 4,096-cell mesh, the PyAM21 source checkout and the Sea ESF tables on the Desktop.

```sh
python -u -I workflows/coupled/campaign.py \
  --config workflows/coupled/aquaplanet_full_depth_4096_gm2.yaml \
  --mesh "$HOME/Desktop/mesh_4096_exact_3000.npy" \
  --source "$HOME/Desktop/PyAM21" \
  --radiation-data "$HOME/Desktop/sea_esf" \
  --output "$HOME/Desktop/vfs-4096-gm2-fresh" --verbose
```

The campaign initializes at day zero. The preset uses 32 atmospheric levels up to 1 hPa, 60 ocean layers to 5,000 m, seasonal Sea ESF radiation, and all ocean and ice processes. Atmosphere and ocean dynamics run every 600 s. The coupling and radiation period is 1,200 s. Within each ocean step, GM/Redi runs twice at 300 s and vertical mixing retains ten substeps. The preset selects the locally funded donor-parcel GM scheme, CPU float64, four threads and regional compilation. The `gm_limiting_passes` and `gm_limited_interfaces` diagnostic columns accumulate the number of retry passes and the number of interfaces limited in accepted updates. An interface limited on successive substeps counts once per substep.

## Original full-depth preset

The original `aquaplanet_full_depth_4096.yaml` preset changes the vertical geometry while retaining moist atmosphere and PyAM21 clouds/convection/radiation, prognostic ocean dynamics, TKE mixing, GM–Redi, eddy dissipation, funded backscatter and biharmonic sink, adaptive ALE, moving sea ice, snow and prognostic ice temperature.

| Item | Setting |
| --- | --- |
| Horizontal grid | 4,096 generated Voronoi cells (10 Lloyd iterations) |
| Atmosphere | 32 levels, lid at 100 Pa (1 hPa) |
| Ocean | 60 levels, 5,000 m constant depth, uppermost 10 m |
| Precision | CUDA float64 |
| Timesteps | 240 s atmosphere, 300 s ocean, 1,200 s coupling, 3,600 s radiation |
| Ocean closures | All existing switches enabled (see YAML) |
| Output | Six-hour snapshots through day 14, daily thereafter; six-hour diagnostics, **daily** restart |
| Calendar | January 1, 2001 to January 1, 2002; prescribed seasonal and diurnal insolation |

The ocean uses the existing analytic initial temperature field with an exponential 250 m thermocline e-folding scale, and constant 34 g/kg salinity. **This is not a spun-up, full-depth equilibrium state.** The 5,000 m bottom has not been validated for the existing mixing/bathymetry-free configuration. A local first attempt with ocean_dt=1200 s rejected at model second 2400 with 'actual carrier outflow exceeds Euler budget'; the configuration now uses 300-second ocean substeps as an unverified conservative trial. This error checks layerwise volume outflow, not gravity-wave CFL directly. Do not loosen the carrier limiter. The atmosphere initialization uses the existing 205 K upper-temperature profile, extended to 1 hPa. No land, bottom topography or geographic basin boundaries exist.

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
`vfs-4096-full-depth-day1/restarts/restart_00086400.nc`, adding:
```bash
  --resume /glade/derecho/scratch/khall/vfs-4096-full-depth-day1/restarts/restart_00086400.nc \
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

## Carrier-CFL failure investigation (2026-11-21)

A user-executed local CPU integration reached 2,400 simulated seconds, then
rejected an ocean RK stage: `actual carrier outflow exceeds Euler budget`.
It was using the initial 1,200 s ocean substep. The check is a strict
layerwise carrier-volume bound, not a direct external gravity-wave CFL test.

The experimental preset has been changed to `ocean_dt: 300`, with the same
1,200 s physics/coupling interval and **all physics enabled**. This is an
unverified timestep hypothesis, not a demonstrated numerical correction.
The program now reports the worst cell/layer, layer thickness, actual integrated
outflow, permitted outflow, outflow ratio, stage-local permissible dt and global
peak edge speed upon rejection. A synthetic regression test checks this error
report. These diagnostics do not alter accepted fluxes or the positivity guard.

The unmodified 60-level thickness rule produces an uppermost layer of 10 m,
then approximately 21.5, 22.4, 23.3, 24.2 m, followed by increasingly
coarse layers reaching approximately 215 m near the bottom. The first
eight layers occupy the upper ~180 m. This is not the uniform 10 m top
mixed-layer spacing found in some climate model configurations.

**Next evidence needed:** Re-run the exact failed setup with the new error
diagnostic (or run the 300 s preset), retain the failure log if it recurs.
A `local_dt_limit` much less than 300 s indicates the next numerical
substep would also be unsafe *for the failing stage state*; it does not
prove that using that smaller dt from the initial state would stabilize the
whole integration. Check initial lateral pressure imbalance, rapid
velocity growth, minimum layer thickness, and numerical energy, before
considering any timestep or scheme change as a fix.

No complete local coupled integration was executed in this review: the
user's PyAM21 optical tables and macOS environment are not available
in the reviewing runtime.

## Opt-in numerical-energy isolation (debug branch)

For anomalous numerical-energy growth, use `--energy-trace` on a **fresh**
integration, keeping the tested physical configuration and mesh unchanged.
The trace evaluates the same global energy functional as `diagnostics.csv`
at each separately committed sub-operator boundary:

- each atmospheric dynamics substep;
- the ocean/ice/snow dynamics step (including its explicitly reported
  external energy);
- GM–Redi and eddy dissipation, either combined or individually per substep;
- atmosphere–surface/ice exchange (including TOA and upstream PyAM21 residual);
- post-exchange ice phase equilibration;
- ALE remapping (including its existing energy-change diagnostic).

The JSONL trace includes stage `ΔH`, identified external and upstream energy,
stage and interval unaccounted residual, component (atmosphere/ocean/ice-snow)
energy increments, and maximum wind/ocean/ice speeds with ocean edge and layer
indices. Records are flushed after every stage and an explicit `commit`
is written only after a coupled interval passes the existing acceptance checks.
An incomplete interval therefore retains its earlier trace events.

Example on Mac CPU (using the 3,000-Lloyd generator file):

```bash
python -u -I workflows/coupled/campaign.py \
  --config workflows/coupled/aquaplanet_full_depth_4096.yaml \
  --source ../PyAM21 \
  --radiation-data ../sea_esf \
  --mesh ../mesh_4096_exact_3000.npy \
  --device cpu --dtype float64 --threads 4 --compile \
  --stop-after-days 2.25 --energy-trace \
  --output energy-trace-4096 --verbose
python -I validation/coupled_debug/summarize_energy_trace.py \
  energy-trace-4096/energy_stages.jsonl
```

For **deeper ocean/ice isolation**, add `--energy-trace-ocean-stages`.
This splits each `ocean_ice_snow_dynamics` step into snowfall input,
phase adjustment, first implicit ocean–ice drag, combined SSPRK3 ocean–ice–snow
dynamics, vertical TKE mixing, second implicit drag, second phase adjustment,
and final snowfall input. The callback observes the resulting states only and
never alters the operators. It identifies whether the error is generated by the
combined SSPRK3 stage or by drag/mixing/phase; it does not resolve individual
SSPRK3 right-hand-side evaluations. The extra full-Hamiltonian diagnostics
may substantially increase runtime.

Example:

```bash
python -u -I workflows/coupled/campaign.py \
  --config workflows/coupled/aquaplanet_full_depth_4096.yaml \
  --source ../PyAM21 --radiation-data ../sea_esf \
  --mesh ../mesh_4096_exact_3000.npy \
  --device cpu --dtype float64 --threads 4 --compile \
  --stop-after-days 2.25 \
  --energy-trace --energy-trace-ocean-stages \
  --output energy-trace-ocean-stages --verbose
python -I validation/coupled_debug/summarize_energy_trace.py \
  energy-trace-ocean-stages/energy_stages.jsonl
```

For finer GM–Redi isolation, add `--energy-trace-substeps`, which **requires**
`--energy-trace` and samples each of its 10 substeps. This option is
computationally expensive because it re-evaluates the full coupled Hamiltonian
and state extrema at each stage. Use it only after a coarse trace implicates
mesoscale operators. Normal runs (without flags) have no trace overhead.

This branch modifies Python source hashes and thus the restart fingerprint.
The uploaded pre-patch restart cannot be resumed with the new branch under the
existing strict restart identity: use a fresh run. A trace cannot reproduce the
previous trajectory merely by reading the checkpoint. The trace distinguishes
stages but does not yet decompose the internal SSPRK RHS, and a stage energy
change is not an analytic conservation proof. It also does not constitute a
fully coupled climate validation.

## Audited day-1 restart -> day-1.5 energy diagnostic

The day 1.5 NetCDF snapshot is not a complete restart. Use the day-1
`restarts/restart_00086400.pt` from the original 2.25-day traced run.

If the previous run used precisely the source hashes pinned in
`workflows/coupled/restart_compat.py`, `--resume-compatible-manifest`
supports the traced version and the subsequent PyAM21 endpoint fix in
sequence. A restart from the deep-trace version needs only the endpoint
bridge. The bridge independently reconstructs the prior
fingerprint from the supplied manifest, checks it against the checkpoint,
checks configuration/mesh/radiation/runtime/compilation, and refuses all
other source changes. The new output uses the current source fingerprint.

```bash
# In vfs-a, on debug/coupled-energy-stage-trace.
# Replace this with the actual folder containing the PRIOR run and its manifest.
OLD_RUN=/path/to/previous-2.25-day-run

# First advance only 0.5 day using the new code, without tracing overhead:
python -u -I workflows/coupled/campaign.py \
  --config workflows/coupled/aquaplanet_full_depth_4096.yaml \
  --source ../PyAM21 --radiation-data ../sea_esf \
  --mesh ../mesh_4096_exact_3000.npy \
  --device cpu --dtype float64 --threads 4 --compile \
  --resume "$OLD_RUN/restarts/restart_00086400.pt" \
  --resume-compatible-manifest "$OLD_RUN/manifest.json" \
  --stop-after-days 1.5 --output segment-day1-to-day1p5 --verbose

# Second segment begins exactly at day 1.5, with the deep-stage energy trace:
python -u -I workflows/coupled/campaign.py \
  --config workflows/coupled/aquaplanet_full_depth_4096.yaml \
  --source ../PyAM21 --radiation-data ../sea_esf \
  --mesh ../mesh_4096_exact_3000.npy \
  --device cpu --dtype float64 --threads 4 --compile \
  --resume segment-day1-to-day1p5/restarts/restart_00129600.pt \
  --stop-after-days 2.25 --energy-trace --energy-trace-ocean-stages \
  --output segment-day1p5-to-day2p25 --verbose

python -I validation/coupled_debug/summarize_energy_trace.py \
  segment-day1p5-to-day2p25/energy_stages.jsonl
```

The first output includes `compatible_resume_audit.json`, which records
the source transition and both fingerprints. For a pre-trace checkpoint, the
same record is also available as `instrumentation_resume_audit.json`. The bridge requires the
original mesh and configuration and an audited source transition. Checkpoints
written after the fix can use the standard `--resume` option alone. You cannot obtain a true 1.5-day restart
from an old 1.5-day NetCDF snapshot.

## Resume the 600-second ocean run after the PyAM21 endpoint failure

Use the failed segment's saved configuration to retain its component timesteps,
vertical grids, compilation settings, and enabled physics. From the repository
root on the endpoint-fix branch, run the following in the original environment.

```bash
OLD_RUN=run-4096-ocean600-3day
python -u -I workflows/coupled/campaign.py \
  --config "$OLD_RUN/configuration.json" \
  --source ../PyAM21 --radiation-data ../sea_esf \
  --mesh ../mesh_4096_exact_3000.npy \
  --resume "$OLD_RUN/last_good_restart.pt" \
  --resume-compatible-manifest "$OLD_RUN/manifest.json" \
  --stop-after-days 3 --output run-4096-ocean600-3day-resumed --verbose
```

For the checkpoint at second 231,600 the log should report a resume at day
2.680556. The next committed interval ends at second 232,800.
