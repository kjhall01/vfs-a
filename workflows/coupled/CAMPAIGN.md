# One-year coupled aquaplanet

The [coastal campaign guide](../../handoff/COASTAL_CAMPAIGN.md) covers island runs, prescribed coastlines and the paired A100 benchmark.

The campaign advances the actual VFS moist atmosphere with PyAM21 physics,
Roquet ocean dynamics/TKE mixing, GM–Redi, mesoscale energy feedback/decay,
funded backscatter with a budgeted biharmonic sink, adaptive thermal ALE,
and moving two-category ice/snow. Ice has a
prognostic thermal store; its temperature is no longer a prescribed boundary.
It starts from merged PR #88. This is an initial spin-up experiment: no
365-day stability or equilibrated-climate result is claimed.

The former 20-hour failure was traced to the VFS/PyAM21 momentum map: the
transpose of an unweighted diagnostic reconstruction amplified impulses onto
very short edges. A shared kinetic-weighted reconstruction fixes that defect
while retaining reciprocal impulses and the energy ledger. A fresh 256-cell
radiation-off full-ocean run completes two days at the requested timesteps.
This does not establish year-long or real-radiation stability. See the
[debugging evidence and remaining limits](../../verification/atmosphere/pyam21/campaign/momentum-map/README.md).

## Install and preflight

From the repository root, using the same Python environment for every command:

```bash
python -m pip install .
python -m pip install -r workflows/pyam21/requirements.txt
git clone https://github.com/evanwellmeyer/PyAM21.git /path/to/PyAM21
git -C /path/to/PyAM21 checkout e1bdfcc3b23c05adea80b35af880702bb85718eb

python -I workflows/coupled/campaign.py \
  --source /path/to/PyAM21 \
  --radiation-data /path/to/input_data/sea_esf \
  --output aquaplanet-preflight --preflight
```

The preflight constructs the radiation package, executes one complete coupled
interval, and repeats it from a serialized restart, requiring bitwise replay.
Inspect `status.json` for `preflight_passed`. Missing tables or any component
failure return a nonzero exit code. This does not qualify a full year.

Seasonal radiation requires the external GFDL SeaEsfRad tables. See the
[radiation setup guide](../pyam21/RADIATION.md) for data preparation and the
[device guide](GPU.md) for measured runs. Automated tests use synthetic
radiation to exercise scheduling, shared fluxes and restart handling.

## Run the year locally

```bash
python -I workflows/coupled/campaign.py \
  --source /path/to/PyAM21 \
  --radiation-data /path/to/input_data/sea_esf \
  --output coupled-aquaplanet-365d --verbose
```

The default [configuration](coupled_aquaplanet_1yr.yaml) is editable; select
another with `--config FILE.yaml`. Defaults are 256 cells, 24 atmospheric
levels, 12 stretched ocean layers and 1000 m depth with a 20 m top layer.
The compatible rotation mesh has ten Lloyd iterations. The default is CPU
float64 with one thread. Select CPU/CUDA/MPS with `--device`; choose precision
with `--dtype float64|float32`. MPS requires explicit `--dtype float32` and is
experimental. CPU/CUDA keep float64 by default. `--threads N` overrides host
threads. See the [GPU setup and precision qualification guide](GPU.md).
Physics/ocean steps are 1200 seconds, atmospheric
substeps 240 seconds, radiation hourly. Mixing and GM–Redi each use ten
substeps per ocean step (120 seconds). The earlier ocean sweep supports the
chosen ocean step under its tested conditions; it does not validate the entire
coupled configuration. Optional regional compilation is described below;
see the device guide for the tested GPU configurations.

All three components start with **zero velocity**, with analytic moist
atmospheric stratification, a meridional SST gradient, stable ocean temperature
profiles, uniform 34 g/kg salinity, TKE of 1e-4 J/kg, mesoscale eddy energy of
0.1 J/kg, and thin/thick polar ice at 265 K. The eddy initial condition is an
explicit unresolved-energy reservoir, distinct from the resolved velocities.
A local energy-conserving phase solve initializes ice/ocean liquidus equilibrium.
No observational initialization, spun-up jet/current, SST restoring or ongoing
prescribed wind/heat flux is imposed. Initial dynamical adjustment is expected;
the frequent first-two-week output is intended to expose it.

The run starts January 1, 2001 on PyAM21's **Julian calendar**, a non-leap year,
and ends January 1, 2002. This implements exactly 365 days without claiming that
upstream supports a no-leap calendar. Solar geometry uses actual model time,
latitude and longitude, a 365-day orbital period, 23.44-degree obliquity,
eccentricity 0.01671 and fixed solar constant 1361 W/m². Both the seasonal and
diurnal cycles operate; SW fluxes are renormalized between radiation calls.
The radiation manifest records these settings and hashes the supplied tables.

## Use your saved 16,384-cell mesh

The same `mesh_16384_exact.npy` used by the timestep campaign is supported:

```bash
cp workflows/coupled/coupled_aquaplanet_1yr.yaml aquaplanet-local.yaml

python -I workflows/coupled/campaign.py \
  --config aquaplanet-local.yaml \
  --mesh /path/to/mesh_16384_exact.npy \
  --source /path/to/PyAM21 \
  --radiation-data /path/to/input_data/sea_esf \
  --output aquaplanet-mesh-preflight --preflight --verbose
```

`--mesh` accepts an Earth generator NetCDF or a NumPy `(N,3)` floating-point array of unit-sphere Cartesian
cell generators. It **infers `cells=N` and forces `lloyd_iterations=0`**, overriding
those two YAML fields. The existing mesh constructor normalizes generator vectors
and rebuilds the compatible topology/operators; there is no further relaxation.
The effective settings are saved in `configuration.json`; the generator mesh identity (the original file SHA256 for a converted NumPy mesh)
is saved in `manifest.json` and bound to restart identity. Moving or losslessly converting the mesh preserves that identity. Supply the same `--mesh` on every resume.

The mesh specifies the horizontal grid. Edit `aquaplanet-local.yaml` to choose
vertical resolution, depth, timesteps and thread count. The initial defaults are:

```yaml
atmosphere_levels: 24
ocean_levels: 12
ocean_depth_m: 1000.0
top_layer_m: 20.0
physics_dt: 1200
atmosphere_dt: 240
ocean_dt: 1200
radiation_dt: 3600
mixing_substeps: 10
eddy_substeps: 10
threads: 1
```

After the preflight passes, remove `--preflight` and choose a fresh output folder:

```bash
python -I workflows/coupled/campaign.py \
  --config aquaplanet-local.yaml \
  --mesh /path/to/mesh_16384_exact.npy \
  --source /path/to/PyAM21 \
  --radiation-data /path/to/input_data/sea_esf \
  --output aquaplanet-16384-365d --verbose
```

This requests the configured 365 days. Append `--stop-after-days 2` for a two-day
qualification attempt before committing to the year.
Continue a successfully completed segment with its saved restart and a new output folder. The
smaller radiation-off checks do not establish seasonal stability on this mesh.

## Optional compilation

Append `--compile` to either the preflight or campaign command. The default is
eager execution (`compile: false` in YAML); `--no-compile` explicitly overrides
`compile: true`. Neither flag changes the timesteps, thread count or physics.

```bash
python -I workflows/coupled/campaign.py \
  --config aquaplanet-local.yaml \
  --mesh /path/to/mesh_16384_exact.npy \
  --source /path/to/PyAM21 \
  --radiation-data /path/to/input_data/sea_esf \
  --output aquaplanet-compiled-preflight --preflight --compile --verbose
```

After this passes, remove `--preflight`, select a new output directory, and add
`--stop-after-days 2` for the initial segment. Retain `--compile` on its resume.
Switching between eager and compiled modes requires a fresh run: the compilation
policy/backend, effective configuration and source hashes are restart identity.
Compiled replay is required to be bitwise within the same supported environment;
eager/compiled trajectories can differ through floating-point reassociation.

The switch uses Inductor for nine repeated tensor kernels: four shared mesh
gradient/divergence/interpolation operations used by atmosphere, ocean and ice,
the atmospheric mixture-energy gradients, and four Roquet ocean thermodynamic
kernels. Each region requires full-graph capture. Public state checks,
positivity limiting, nonlinear solvers, PyAM21 column control flow and radiation
scheduling/optics retain their existing eager execution. GM–Redi, mixing,
backscatter, ALE and ice remain enabled as configured and use the shared compiled
operations where applicable. No package classes are patched globally.

Inductor needs a working native compiler. Setup, initial diagnostics and first
intervals can take minutes while kernels compile; a new shape may trigger another
compilation later. `run.log` and `manifest.json` record the selected regions and
backend. A compiler error fails visibly through the existing failure/restart
path, without silently retrying eagerly. Time several intervals after warmup
before assessing speed; output and fresh radiation calls still cost time.

For an eager/compiled comparison on your actual mesh and optical tables:

```bash
python -I validation/coupled_compile/run.py \
  --config aquaplanet-local.yaml \
  --mesh /path/to/mesh_16384_exact.npy \
  --source /path/to/PyAM21 \
  --radiation-data /path/to/input_data/sea_esf \
  --steps 8 --warmup 2 --output compile-comparison
```

This records setup separately, alternates execution order, compares final state
and whole-system budgets, and requires bitwise compiled restart replay. The
reported median excludes the first two intervals; inspect individual timings
because fresh radiation occurs only on its configured cadence. It times advance
calls, excluding NetCDF output and diagnostic inventories. Omitting
`--radiation-data` explicitly selects a radiation-off benchmark; the production
campaign still requires tables unless its smoke flag is explicitly selected.
See [CPU measurements and qualification limits](../../verification/atmosphere/pyam21/campaign/compilation/README.md).

## Logging and verbosity

Every initialized run/preflight writes `OUTPUT/run.log`, even without
`--verbose`. It contains UTC timestamps, initialization/mesh identity, progress
about every 30 wall-clock seconds (after a completed step), snapshot/restart
writes, scheduled budget summaries, completion, and full exception tracebacks.
The file is flushed on every log record. `--verbose` also prints these messages
to the terminal; without it, the terminal retains the final JSON status/error.
Numerical diagnostics are available in `diagnostics.nc` and `diagnostics.csv`, and run status in `status.json`.

Startup lists the enabled ocean processes, coefficients and substeps.
`diagnostics.csv` records actual GM–Redi/decay/ALE call counts, the number of
changed ALE columns, limited Redi fluxes, reservoir transfers, explicit-step
numbers, and local energy residuals. An enabled flag alone is not the evidence
that the process ran.

`radiation_residual` is the running maximum absolute column heating/flux
discrepancy in W/m², not a global mean. The signed cumulative area-mean
`radiation_energy_residual_J_m2` isolates its contribution to the upstream
energy residual; `upstream_minus_radiation_residual_J_m2` is the remainder.
Divide these accumulated quantities by elapsed model seconds for mean W/m².
Neither is an additional energy loss to add to `upstream_residual_J_m2`.
The [radiation-grid transfer](../pyam21/RADIATION.md#radiation-and-model-layer-masses)
conserves native radiative heating on the current host layers, including cached
calls. Real-data validation of this correction remains pending.

```bash
tail -f aquaplanet-16384-365d/run.log
```

## Outputs and restarting

| Output | Schedule | Contents |
| --- | --- | --- |
| `snapshots/state_*.nc` | Initial state, every 6 h through day 14 inclusive, daily on days 15–365 | 408 instantaneous native-mesh snapshots for a complete year; atmosphere T/p/water/wind/cloud, ocean CT/SA/layers/wind/TKE/mesoscale energy, ice area/volume/T/wind/cold content and snow |
| `diagnostics.nc`, `diagnostics.csv` | Initial state, every 6 h, segment endpoint | Combined full energy minus TOA, separately retained upstream AM2 imbalance, numerical residual, water/salt, component extrema and peak local exchange residuals |
| `restarts/restart_*.nc` | Initial state, every 5 days, segment endpoint | Atmosphere/cloud/physics/radiation caches and clock, all ocean/ice/snow tensors, cold content, accumulated budget ledgers |
| `manifest.json` | Initialization | Actual code, configuration, dependency and radiation-data identity |
| `run.log` | Startup, progress, output events and termination | Timestamped progress and failure tracebacks; always enabled |
| `status.json` | Diagnostic output and termination | Completion/failure status, last committed time and metrics |
| `last_good_restart.nc` | Caught error or interruption | Last complete coupled interval, never a partially applied exchange |

Day 14 is written once. Absent ice categories have diagnostic T=273.15 K; mask
these with `ice_area`. Snapshots and complete restarts are published by atomic replacement.
Restarts store named arrays, scalars, and a description of the component state.
The campaign also reads legacy `.pt` checkpoints with `weights_only=True`. The resume check verifies the
mesh, physics, runtime, configuration, source and radiation data. Binary output
files stay in your specified run directory. See the [Earth I/O guide](EARTH_SHELF.md#existing-inputs-and-netcdf-output) for the legacy transition and xarray examples.

The momentum-map correction changes the restart identity (standalone adapter
schema 4). Start a fresh run after pulling it; do not relabel old checkpoints.

For an initial shorter segment, append `--stop-after-days 2`. To continue, keep
the same configuration and give a **new** output directory:

```bash
python -I workflows/coupled/campaign.py \
  --source /path/to/PyAM21 \
  --radiation-data /path/to/input_data/sea_esf \
  --resume coupled-aquaplanet-365d/restarts/restart_00432000.nc \
  --output coupled-aquaplanet-continuation --verbose
```

That example resumes the day-5 restart. Output schedules remain anchored to the
original January 1 start. New segments include an initial diagnostic and, when
on the prescribed schedule, a boundary snapshot; deduplicate shared times when
combining segments. Existing output directories are never overwritten. The last
good checkpoint is best-effort on caught errors/SIGINT; an OS kill can only
recover a previously completed scheduled restart.

An explicit plumbing-only check is available without external radiation data:

```bash
python -I workflows/coupled/campaign.py --source /path/to/PyAM21 \
  --output radiation-off-preflight --preflight --smoke-no-radiation
```

This is visibly labelled radiation-OFF and cannot run more than one model day.
It is not the requested seasonal experiment.

## Ocean process configuration and order

The supplied YAML enables the existing research ocean process suite. Parameters
match the earlier integrated trade-wind experiment, including its 1e13 m4/s
rotational biharmonic sink; these are explicit experimental coefficients, not
calibrated climate values.

```yaml
gm_redi: true
gm_length_m: 10000.0
gm_efficiency: 0.1
gm_kinetic_fraction: 0.5
redi_diffusivity_m2_s: 100.0
redi_positive_transport: true
eddy_initial_energy_J_kg: 0.1
eddy_substeps: 10
eddy_dissipation: true
eddy_decay_days: 30.0
backscatter: true
backscatter_viscosity_m2_s: 10000.0
backscatter_days: 1.0
backscatter_fraction: 0.25
biharmonic_viscosity_m4_s: 10000000000000.0
ale_remap: true
```

Each physics interval advances atmospheric dynamics, then the ocean/ice
substeps. Every ocean substep includes existing phase/drag/mechanics/TKE mixing
and backscatter inside the SSPRK3 stages, followed by the configured GM–Redi
substeps and exact exponential eddy decay to heat. GM diffusivity is
`efficiency * length * sqrt(2 * kinetic_fraction * eddy_energy)`; it starts
near 316 m2/s with this reservoir. There is no continuous energy injection or
diffusivity floor. GM–Redi exchanges actual finite Hamiltonian work with the
reservoir; backscatter transfers funded energy into resolved flow, and the
biharmonic sink returns lost KE to the reservoir.

Shared atmospheric exchange and the final ice phase solve then run. ALE follows
**once per physics interval**, using the existing conservative PLM remap and
column-energy/entropy thermal projection. Its target is the initial stretched
layer **fractions of current column depth**: it preserves freshwater/free-surface
changes and does not restore the initial water mass or flatten the layers to
equal thickness. Passive tracers, TKE and eddy energy are remapped conservatively;
ice/snow/cold-content tensors are retained. Remap energy residuals remain in
the numerical budget, never counted as external forcing.

The optional positive Redi path limits whole local stencil-row fluxes separately
for salinity/passives/TKE/eddy energy, using available donor inventory. Signed CT
retains its original diffusion. The same limited flux enters the finite-energy
ledger. This prevents the rotated stencil from withdrawing TKE from an empty
layer without clipping state or adding a floor. The old standalone GM–Redi
default remains unchanged. Positivity is not a full maximum principle, and
inadmissible thermodynamics or unfunded energy conversions still reject.

To reproduce the previous reduced campaign, set `gm_redi`, `eddy_dissipation`,
`backscatter`, and `ale_remap` to `false`, and `eddy_initial_energy_J_kg: 0.0`.
`backscatter: false` also disables its biharmonic sink; use viscosity/fraction
zero with backscatter enabled for a sink-only case. Old YAML files inherit the
new process defaults, but retain any timesteps explicitly written in those files.
Update those entries or copy the supplied YAML to use the new timestep choices.
**Start a fresh run after this update:** existing checkpoints reject changed
source/configuration, including these process switches.

## Prognostic ice temperature and energy

For category k, the new tensor is `E_k = cp_i rho_i V_k (T_k - 273.15)` in J/m²
of horizontal cell area. `cp_i=2100 J/(kg K)` applies to the whole frozen
column, including its labelled snow subset. The combined Hamiltonian is the
existing ocean–ice–snow energy plus `sum(A E_k)`. Cold content is nonpositive;
absent categories carry zero. This is a **single thermal node** per category,
used as surface temperature, not a resolved vertical temperature/brine model.

Surface exchange uses that prognostic T for turbulent fluxes and emission.
The common radiation column is partitioned conservatively: shortwave by tile
area times absorptivity, longwave with unit emissivity and a common downwelling
flux inferred from the same cached net. Rain drains to liquid; intercepted snow
supplies its sensible enthalpy. Vapor exchange includes sublimation/deposition
latent heat through the existing frozen-mass energy datum.

Conduction through ice and porous snow is backward Euler:

```
C = cp_i rho_i V
K = area / (h_ice/k_ice + h_snow/k_snow)
E_trial = (E_old + Q_surface + dt K T_base_C) / (1 + dt K/C)
E_new = min(E_trial, 0)
```

The zero-C limit is the melting complementarity constraint, not discarded heat.
The reciprocal receiver transfers conduction and all surplus to ocean/phase
through its local first-law solve, subtracting the change in ice storage.
No energy is counted twice or removed by a temperature floor. Melting includes
the cold content's warming cost in the existing scalar phase-energy root; its
analytic derivative is independently tested. New ice joins the zero-C node
reference and is then cooled by conduction. Snow melts first as before.

Cold content uses the same donor advection and thin-to-thick ridging transfers
as the frozen carrier, inside the same SSPRK3 stages. It is conserved under
horizontal redistribution. Ocean/ice drag still deposits lost KE into TKE.
The phase/momentum/thermal boundary remains first-order split. Upstream AM2
energy defects and finite-step dynamics errors remain visible.

Limitations include the existing idealized liquidus, compression-only ice
mechanics, no brine, no resolved ice/snow thermal profile, and uncalibrated
mesoscale closures. Radiation climatologies retain the previous
local rectangular sampling approximation. Full-year stability and physically
credible climate statistics require the actual local experiment.

## ALE entropy rejection diagnostics

The thermal projection checks column energy and nondecreasing model entropy.
It now evaluates both iteration and final acceptance using temperature
reconstructed from the stored `h*T`. Earlier code could falsely converge at an
entropy boundary and then reject that same candidate in the final check.
The tolerances are unchanged; unsuccessful solves report the worst column,
residual relative to tolerance, and iteration count. Retain `run.log`,
`configuration.json`, the mesh and `last_good_restart.nc` for diagnosis.
See the [cold-column regression and checkpoint replay](../../verification/ocean/ale-cold-columns/README.md).
An additional near-tolerance stall was found in the user's later run. Once an
actual stored candidate violates entropy, the projection now retains a zero
entropy-change activation threshold for that column through the remaining
iterations. This prevents an ineffective sub-ulp correction from repeating
until the iteration cap. Final energy/entropy tolerances and the cap are unchanged.
See the [active-constraint regression and replay](../../verification/ocean/ale-active-set/README.md).
For a single-interval diagnostic using the checkpoint's original radiation cache:

```bash
python -I validation/coupled_debug/replay_ale.py \
  --source /path/to/PyAM21 --config /path/to/configuration.json \
  --mesh /path/to/mesh_16384_exact.npy \
  --restart /path/to/last_good_restart.nc --output ale-diagnostic
```

This explicitly reconstructs state for diagnosis across source versions. It
writes the pre-ALE ocean input and a result report, not a production restart.
It fails if the interval needs a fresh optics call; cached fields are never
extended past their scheduled refresh. `--compile` is available for comparison.
Start in a new output directory after updating source; the normal restart
fingerprint checks intentionally reject checkpoints from the previous code.
