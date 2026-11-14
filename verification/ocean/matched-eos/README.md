# Matched-buoyancy linear/JMD95 comparison

This compares the explicit linear configuration with the temperature-coordinate
JMD95 configuration. It measures model differences, not error against observations.
No runtime model, default, fixture or tolerance is changed.

## Matching and references

Both models start with identical geometry, layer thickness, velocity, salinity,
passive content and TKE. At each initial layer midpoint, linear temperature is
chosen algebraically so that

    rho_linear(T_linear,S) - rho0
      = rho_JMD95(T_JMD95,S,p(z)) - rho_JMD95(T0,S0,p(z)).

This matches the density anomaly driving the Hamiltonian buoyancy force, with
maximum mismatch about 2.28e-13 kg/m³. Matching raw in-situ density would inject
pressure compression into the linear stratification and is deliberately avoided.
Midpoint matching does not match the EOS between midpoints, pressure derivatives,
thermal/salinity response, caloric energy, or all Hamiltonian force derivatives.
The initial temperatures differ; their final RMS difference alone is not an
accuracy score. Pressure is the same fixed-datum p(z) used by the existing energy.

The JMD95 reference uses compiled thermodynamics; linear thermodynamics is eager.
All physics/energy acceptance checks remain active. CPU float64, one thread per
process, no-grad trajectories. Some campaigns ran concurrently; recorded wall
seconds are not a throughput comparison. Meshes are Fibonacci Voronoi with ten
Lloyd iterations and compatible rotation, not the recovered 16384-cell mesh.

### Internal wave

Two layers (500/2500 m) on a sphere of radius 100 km. JMD95 initial T=17/4°C,
S=34/35.5; a 0.1-m degree-one thickness perturbation projects onto the internal
mode. Rotation, forcing, mixing and ALE are disabled. Both models run for one
complete JMD95 continuum-reference period (~32.75 h), with external-wave CFL .3.
Frequency is fitted from unwrapped modal phase using thickness amplitude and its
actual RHS derivative. The independent continuum reference uses JMD95 density
anomalies at the surface/interface, rather than a fit to a numerical trajectory.
The linear run is compared against that *same* reference, not its own optimum fit.

### Rotating flow

Four layers, 3000-m water column, Earth-radius sphere, rotation 7.292e-5 s^-1,
solid-body zonal flow with 5-m/s equatorial speed. JMD95 temperature/salinity range
17→4°C and 34→35.5. Internal interfaces are flat; free-surface slope balances
continuum centrifugal/Coriolis acceleration using the upper-layer surface
buoyancy. No adjustment to an exact discrete equilibrium is made. Run: 24 hours,
dt600 s, no mixing/forcing/ALE. Common mesh/operator imbalance and any continuum
initialization approximation affect both models. Between-model differences must
be compared with that common drift; a small difference does not prove exact balance.

### Forced column physics

32 cells, 100 m depth, 16/32 layers, 24 hours, dt600 s with eight mixing substeps.
A smooth thermocline separates ~15°C surface and ~10°C bottom water, with active
salinity 35-0.1*(T-10). Initial TKE is 1e-4 J/kg. Prescribed eastward wind stress
0.05 Pa, quadratic bottom drag .0025, shortwave 200 W/m² and nonpenetrative heat
-250 W/m². Boundary stress, heating and mixing advance; horizontal dynamics and
ALE are deliberately excluded to isolate column response. Density-profile RMS,
TKE profiles and velocity differences are retained.

Mixed-layer depth here is a **coarse diagnostic**: the first layer midpoint whose
buoyancy-driving density anomaly exceeds its surface value by 0.03 kg/m³.
The raw trajectory diagnostic has no interpolation and is not a standard potential-density MLD.
The summary also linearly interpolates the first threshold crossing of the
area-unweighted mean final density-anomaly profile; that is a distinct diagnostic,
not the mean of individually interpolated column depths.
Consequently equal reported depths can hide differences within one grid interval;
the full density profiles and vertical refinement matter more than equality alone.

## Measured results

All twelve model trajectories (six paired cases) completed. The largest energy
residual after prescribed inputs was 1.63e-6 J/kg in the rotating-flow case;
wave residuals were <=4.66e-9 J/kg and column residuals <=2.38e-9 J/kg.
No positivity/funding checks were relaxed. These are absolute specific-energy
residuals, not percentages of the small perturbation energy.

| Case | JMD95 | Linear | Interpretation |
|---|---:|---:|---|
| Wave, 128 cells: frequency / continuum reference | 0.994034 | 1.016760 | Linear frequency 2.286% above JMD95 |
| Wave, 256 cells: frequency / continuum reference | 0.996930 | 1.019739 | Linear frequency 2.288% above JMD95 |
| Rotating flow, 128 cells: one-day velocity drift RMS, m/s | 0.100102 | 0.100102 | Between-model RMS difference 4.81e-5 m/s |
| Rotating flow, 512 cells: one-day velocity drift RMS, m/s | 0.024960 | 0.024953 | Between-model RMS difference 4.47e-5 m/s |
| Column, 16 layers: interpolated mean-profile MLD, m | 15.700 | 15.718 | Final density-anomaly RMS difference 0.00293 kg/m³ |
| Column, 32 layers: interpolated mean-profile MLD, m | 12.423 | 12.607 | Final density-anomaly RMS difference 0.00870 kg/m³ |

The persistent ~2.29% wave-frequency difference is distinguishable from mesh
error: refining the mesh moves the JMD95 frequency toward its continuum reference
but hardly changes the relative EOS difference. This is a two-layer, one-period
comparison; it does not isolate pressure dependence from caloric/coordinate effects.
The jet's EOS difference is much smaller than common numerical drift. No unstable
baroclinic jet or eddy-growth experiment was run.

The column models are close *at each chosen resolution*, but changing layer count
moves MLD by ~3 m, much more than switching EOS. The density-profile discrepancy
also grows upon vertical refinement. Thus this campaign does not establish a
converged entrainment prediction or an EOS-independent mixing closure.
`analyze_matched.py` reproduces summary.json from the six raw case files.

## Physical scope

A matched initial density field does not guarantee the same response to heating.
`eos_response.py` directly compares the density partials at S=35 using the same
Boussinesq rho0=1025 normalization. At zero pressure the fixed linear coefficient
produces a thermal density response 68% stronger at 5°C, 11% weaker at 15°C, and
36% weaker at 25°C than implemented JMD95. At 0°C it is about 3.62 times as large.
This is an EOS derivative comparison, not a prediction that circulation errors
will have those percentages. The matched warm-column experiment cannot qualify
cold-water formation, thermobaricity, cabbeling, a global overturning circulation,
or long climate integrations. Constant-cp calorics also differ between models.

Linear EOS configurations are established scientific options:

- [MITgcm baroclinic-gyre tutorial](https://mitgcm.readthedocs.io/en/latest/examples/baroclinic_gyre/baroclinic_gyre.html)
  explicitly uses LINEAR for an idealized wind-driven ocean.
- [Oceananigans buoyancy documentation](https://clima.github.io/OceananigansDocumentation/stable/models/buoyancy_and_equation_of_state)
  supports linear EOS in hydrostatic and nonhydrostatic models, alongside nonlinear options.
- [MOM6 EOS documentation](https://mom6.readthedocs.io/en/main/api/generated/pages/Equation_of_State.html)
  documents both linear and nonlinear equations of state.
- [MITgcm global pressure-coordinate example](https://mitgcm.readthedocs.io/en/latest/examples/global_oce_in_p/global_oce_in_p.html)
  instead selects JMD95P. Availability of LINEAR is not endorsement for every application.

## Reproduce

With VFS installed, from the repository:

```sh
python validation/temperature_coordinate/matched_eos.py wave --cells 128 --compile --output wave128.json
python validation/temperature_coordinate/matched_eos.py wave --cells 256 --compile --output wave256.json
python validation/temperature_coordinate/matched_eos.py jet --cells 128 --layers 4 --compile --output jet128.json
python validation/temperature_coordinate/matched_eos.py jet --cells 512 --layers 4 --compile --output jet512.json
python validation/temperature_coordinate/matched_eos.py mixing --cells 32 --layers 16 --compile --output mixing16.json
python validation/temperature_coordinate/matched_eos.py mixing --cells 32 --layers 32 --compile --output mixing32.json
python validation/temperature_coordinate/eos_response.py --output response.json
```

Installed-wheel outside-checkout checks: **17 passed** (four matching/reference
checks plus all 13 linear-configuration checks). Parent full-suite results remain
in the linear-configuration report; the full suite was not rerun for this
harness-only change. Raw JSON includes campaign source hash, parameters, profiles,
sampled time series, budgets and explicit failures. No large state arrays or
binary wheel are uploaded. Every published file must remain <=128 KiB.
