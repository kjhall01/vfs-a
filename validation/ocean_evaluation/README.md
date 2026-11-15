# Integrated ocean evaluation

This driver evaluates the checkpoint's Roquet CT/SA research ocean without
modifying its numerical methods. It loads research modules explicitly but uses
an installed `vfs` package. Install the current checkout (`python -m pip install .`)
first. No optional physics is silently omitted or changed after a rejection.

## Cases

- `rest`: 800–1200 m smoothly varying bathymetry, uniform reference CT/SA,
  initially flat free surface and zero velocity. No forcing or closures.
- `channel`: a closed spherical wet band (15–65° N), 1000 m depth, radius
  1000 km, salinity stratification and a meridional salinity front with a small
  zonal perturbation. Initial shear uses continuum thermal wind with zero flow
  at the bottom. Free surface is initially flat: this is only approximate
  balance, so adjustment is expected. It is not a validated Eady benchmark.
- `forced`: global Earth-radius ocean, flat 1000 m bottom, vertically stratified
  CT/SA and initially zero velocity. Smooth prescribed zonal wind stress,
  area-mean-zero surface heat forcing (including shortwave penetration), bottom
  drag and Roquet vertical TKE mixing. No freshwater, sea ice or ALE remapping.

All have eight stretched layers by default. `channel` and `forced` compare:

| Variant | Mesoscale transport | Backscatter |
|---|---|---|
| resolved | Off | Off |
| gmredi | Energy-dependent GM + prescribed Redi | Off |
| backscatter | Same GM–Redi | Local instantaneous momentum/reservoir pair |

For comparability, all non-rest variants begin with e=0.1 J/kg and include
30-day eddy decay to heat. This decay also occurs in the resolved baseline.
Default GM mixing length is 10 km, efficiency .1, kinetic fraction .5; Redi
is 100 m²/s. Backscatter magnitude is 1e4 m²/s with fraction .25 and 1-day
power timescale. These are explicit experimental parameters, not calibration.

## Run locally

Start with a brief smoke test before a long high-resolution job:

```sh
python -I validation/ocean_evaluation/run.py --case channel \
  --cells 128 --layers 8 --days .1 --dt 120 --output ocean-smoke

python -I validation/ocean_evaluation/run.py --case channel \
  --cells 1024 --layers 16 --days 2 --dt 60 --output ocean-channel-1024

python -I validation/ocean_evaluation/run.py --case forced \
  --cells 1024 --layers 16 --days 2 --dt 120 --output ocean-forced-1024

python validation/ocean_evaluation/analyze.py ocean-channel-1024
```

These higher-resolution commands are starting configurations, not qualified
maximum timesteps. `--compile` compiles selected thermodynamic kernels only;
startup cost can be substantial. `--threads N` controls PyTorch CPU threads.
A supplied `--mesh generators.npy` must have `(cells,3)` unit-sphere generators;
no extra Lloyd relaxation occurs. Without it the mesh gets two deterministic
Lloyd iterations. Radius changes geometry, not the unit-sphere generators.

For 16384×40 you can use `--cells 16384 --layers 40 --mesh ../mesh_16384_exact.npy`.
Plan an hours-long run rather than assuming the small screening run's speed.
The per-variant default wall limit is 7200 s; increase `--max-wall-seconds`
explicitly for longer jobs. Use `--variants backscatter` to run one variant,
although matching ablations are needed to interpret its behavior.

Use a fresh output directory for each run. The driver refuses to overwrite an
existing variant summary. It does **not** resume an interrupted integration.
A rejected step stops that variant and records the last fully accepted state;
the remaining requested variants still run. There is no automatic dt reduction
or parameter weakening that might hide a failure. Exit status is nonzero if
any variant rejects or times out.

## Outputs and interpretation

Each variant saves `summary.json`, streaming `series.csv`, `initial.npz`,
`last_accepted.npz` and `mesh.npy`. The root contains `provenance.json` (revision,
source hashes, package import, environment) and `campaign.json`. The analyzer
writes `metrics.json` and case PNGs. These are normal data files; NPZ has no
pickled Python objects. For review, send the small JSON/CSV files first and the
endpoint NPZ files if spatial fields are needed. A console completion is not
qualification of accuracy, equilibrium or timestep safety.

Energy error means `(H(t)-H(0)-external wind/heat work)/initial total mass`.
H includes resolved, TKE, mesoscale and thermal/PE terms. Mass, salt and passive
inventory errors are relative to their initial values. Mesoscale exchanges and
dissipation are internal, not external forcing. Their local roundoff residuals
are tracked separately. Backscatter is an instantaneous closure: finite RK
energy error remains part of the reported H error, with no hidden repair.

The reconstructed non-zonal KE proxy subtracts mass-weighted means within 12
latitude bands at each layer. It is **not** the native Hamiltonian KE, a clean
mesoscale EKE diagnostic, or proof of baroclinic instability. The potential N²
proxy uses the Roquet density polynomial evaluated at z=0; it is not the full
in-situ buoyancy frequency. Cell count includes land; inspect `wet_cells` for
the channel's active horizontal resolution.

The low-resolution runs screen conservation, integrated compatibility,
positivity, drift and sensitivity. They cannot establish mesoscale spectra,
eddy saturation, realistic overturning or climate skill. Fixed layer count,
coarse coastal geometry and initially imperfect channel balance matter.
