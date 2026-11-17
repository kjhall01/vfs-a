# One-year coupled aquaplanet

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

**Outstanding external dependency:** the real GFDL SeaEsfRad tables are not
included. See [radiation setup](../pyam21/RADIATION.md). The advertised download
still returns HTTP 503 here as of 2026-11-15, so real-radiation execution remains
unverified. Synthetic radiation tests verify scheduling, shared fluxes and
restart handling only. The campaign never substitutes synthetic radiation or
silently disables it.

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
The compatible rotation mesh has ten Lloyd iterations. Everything runs on CPU
in float64 with one thread. Physics/ocean steps are 1200 seconds, atmospheric
substeps 240 seconds, radiation hourly. Mixing and GM–Redi each use ten
substeps per ocean step (120 seconds). The earlier ocean sweep supports the
chosen ocean step under its tested conditions; it does not validate the entire
coupled configuration. There is no compiled/GPU qualification.

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

`--mesh` accepts a NumPy `(N,3)` floating-point array of unit-sphere Cartesian
cell generators. It **infers `cells=N` and forces `lloyd_iterations=0`**, overriding
those two YAML fields. The existing mesh constructor normalizes generator vectors
and rebuilds the compatible topology/operators; there is no further relaxation.
The effective settings are saved in `configuration.json`; the input-file SHA256
is saved in `manifest.json` and bound to restart identity. Moving the unchanged
file to another path is allowed. Supply the same `--mesh` on every resume.

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

This requests the configured 365 days. Append `--stop-after-days 1` for a one-day
qualification attempt; the current 256-cell check fails as described above.
Continue a successfully completed segment with its saved restart and a new output folder. The
16,384-cell coupled timestep and runtime are not qualified by the smaller tests.

## Logging and verbosity

Every initialized run/preflight writes `OUTPUT/run.log`, even without
`--verbose`. It contains UTC timestamps, initialization/mesh identity, progress
about every 30 wall-clock seconds (after a completed step), snapshot/restart
writes, scheduled budget summaries, completion, and full exception tracebacks.
The file is flushed on every log record. `--verbose` also prints these messages
to the terminal; without it, the terminal retains the final JSON status/error.
Numerical diagnostics remain in `diagnostics.csv` and machine-readable status
in `status.json`.

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
| `diagnostics.csv` | Initial state, every 6 h, segment endpoint | Combined full energy minus TOA, separately retained upstream AM2 imbalance, numerical residual, water/salt, component extrema and peak local exchange residuals |
| `restarts/restart_*.pt` | Initial state, every 5 days, segment endpoint | Atmosphere/cloud/physics/radiation caches and clock, all ocean/ice/snow tensors, cold content, accumulated budget ledgers |
| `manifest.json` | Initialization | Actual code, configuration, dependency and radiation-data identity |
| `run.log` | Startup, progress, output events and termination | Timestamped progress and failure tracebacks; always enabled |
| `status.json` | Diagnostic output and termination | Completion/failure status, last committed time and metrics |
| `last_good_restart.pt` | Caught error or interruption | Last complete coupled interval, never a partially applied exchange |

Day 14 is written once. Absent ice categories have diagnostic T=273.15 K; mask
these with `ice_area`. NetCDF writes are atomic. A restart uses tensor/primitive
payloads with `weights_only=True`, not pickled model objects. It rejects changed
mesh, physics, runtime, configuration, source or radiation data. Binary output
files stay in your specified run directory.

The momentum-map correction changes the restart identity (standalone adapter
schema 4). Start a fresh run after pulling it; do not relabel old checkpoints.

For an initial shorter segment, append `--stop-after-days 2`. To continue, keep
the same configuration and give a **new** output directory:

```bash
python -I workflows/coupled/campaign.py \
  --source /path/to/PyAM21 \
  --radiation-data /path/to/input_data/sea_esf \
  --resume coupled-aquaplanet-365d/restarts/restart_00432000.pt \
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
