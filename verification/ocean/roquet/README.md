# Roquet freezing configuration: evidence and limits

The [explicit configuration](../../../validation/temperature_coordinate/ROQUET.md)
uses published Roquet Freezing density coefficients, CT/SA state conventions,
analytic Hamiltonian derivatives, exact quadratic mixing work and algebraic
heating. JMD95 and linear defaults are unchanged. The caloric/entropy model is
explicitly idealized; this is not a complete TEOS-10 thermodynamic replacement.

## Verification

Installed-wheel outside-checkout portable suite: **1153 passed, one CUDA skip**,
one existing warning. Eighteen focused tests check:

- Published coefficients and positive-depth/negative-height sign convention,
  using independent decimal evaluation; density partials versus autograd.
- Full Hamiltonian derivatives versus autograd on open/coastal geometries,
  and actual limited spatial RHS work closure.
- Exact layer integration versus an independent 12-point Gauss integral.
- Finite mixing energy with opposing upper/lower fluxes, including their
  quadratic cross term, versus both the analytic change and actual H difference.
- Heating and cooling using the stable quadratic root.
- Cold stable/unstable and 0–30°C contrast profiles, each at zero/tiny/finite TKE.
- A forced step plus ALE with nonlinear EOS/inversion functions disabled and
  finite backward gradients through the complete operation.
- Explicit GSW initialization conversion and rejection of ambiguous T/S keywords
  and differentiable inputs at the offline conversion boundary.

No full-step energy exactness or differentiability across every acceptance branch
is claimed. Local funding retries, temperature bounds, positivity and model-entropy
checks remain in place. ALE still has iterative allocation and feasibility limits.

## Density response versus independent GSW

`response.json` compares density derivatives at SA=35 g/kg, CT=-1.5,0,5,15,25,30°C
and pressure 0/1000/3000/5000 dbar. Roquet uses the paper's approximate geopotential
depth D=p metres, i.e. VFS height z=-p. Density derivatives are compared in kg/m³/K
and kg/m³/(g/kg), not with inconsistent in-situ/reference density normalizations.
The linear comparison uses its constant coefficient times rho0=1025; its original
T68 coordinate is treated as a per-degree response control, not a CT conversion.

| Surface CT | Linear thermal-response discrepancy | Roquet discrepancy |
|---|---:|---:|
| -1.5°C | +487% | +2.1% |
| 0°C | +263% | -9.0% |
| 5°C | +68% | -14.6% |
| 15°C | -10.8% | -8.8% |
| 25°C | -35.6% | -1.1% |
| 30°C | -42.6% | +2.9% |

The maximum absolute Roquet thermal-response discrepancy on this small grid is
22.1%. This diagnostic checks local derivatives, not climate accuracy; some
combinations are far from common ocean water masses. No coefficients were refit.

## Matched-buoyancy wave and column experiments

The wave repeats the parent campaign's full internal-mode period, same h/u and
JMD95 buoyancy anomaly at each initial midpoint. SP/T68 are converted explicitly
to SA/CT using GSW at a single declared reference-water location
(lon=-30°, lat=45°), then CT is adjusted with the analytic quadratic root to match
buoyancy. This second adjustment is a deliberately different thermodynamic state;
its magnitude and density mismatch are retained. The comparison is against the
same JMD95 continuum reference and prior JMD95 trajectories, not a fitted Roquet
reference. Zero rotation; no forcing, mixing or ALE. Compatible Voronoi mesh,
ten Lloyd iterations, radius 100 km, layers 500/2500 m, 0.1-m degree-one wave.

The column compares JMD95 and Roquet using matched initial buoyancy, 32 cells,
32 layers, 100-m depth, a 24-hour dt600 run with eight mixing substeps. Same
wind stress, bottom drag, shortwave and cooling as the prior matched campaign;
horizontal dynamics and ALE are disabled to isolate column response. Interpolated
MLD is the first 0.03 kg/m³ crossing of the mean *density-anomaly* profile, not a
standard potential-density MLD. Its vertical convergence remains unqualified.

## Recorded trajectories

| Mesh | JMD95 frequency / continuum reference | Roquet frequency / same reference | Roquet vs JMD95 |
|---|---:|---:|---:|
| 128 cells | 0.994034 | 0.994774 | +0.07438% |
| 256 cells | 0.996930 | 0.997671 | +0.07441% |

The corresponding constant-coefficient linear difference in the parent campaign
was +2.29%. The initial density anomalies match to <1e-15 kg/m³, but temperatures
and salinity conventions differ. This is one complete two-layer mode period,
not a general wave-growth or climate qualification. Final Roquet energy residuals
are -3.34e-9 and -4.64e-9 J/kg; salt content drift is about 1–2e-13 relative.

At 32 layers, the warm matched-column run gives interpolated mean-profile MLD
12.245 m (Roquet) versus 12.423 m (JMD95), density-anomaly RMS difference
0.00229 kg/m³ and velocity RMS difference 0.000178 m/s. Roquet energy minus
inputs is -1.92e-10 J/kg. This comparison does not resolve the parent campaign's
vertical-resolution sensitivity and does not equate model entropy to TEOS entropy.

Native CT/SA full-step runs on 32 cells ×16 layers all complete 24 hours,
dt600, eight mixing substeps, prescribed forcing and ALE every step:

| Initial CT profile | Energy minus inputs, J/kg | Salt content relative change |
|---|---:|---:|
| Warm, 10–15°C | -2.15e-8 | -8.03e-15 |
| Cold, -1.5–0.5°C | -2.01e-8 | -8.64e-15 |
| Strong contrast, 0–30°C | -2.36e-8 | -8.52e-15 |

Final TKE remains nonnegative, with maximum local budget/tolerance ratio below
4e-7. These are synthetic startup/budget cases; the prescribed stress pattern
and small mesh are not a realistic ocean circulation. No sea-ice physics is
present, even if a temperature lies below a local physical freezing point.

### Retained diagnostic limitation

The initial wave setup used geographic GSW salinity corrections over the whole
idealized sphere. Density was matched at midpoints, but the resulting geographic
T/S compensation introduced an unintended background adjustment elsewhere in the
column. The modal equilibrium offset differed between meshes, biasing the raw
unwrapped-phase frequency fit (ratios 1.0004 and 0.9502). Those preliminary ratios
are **not wave-speed estimates**. `geographic-wave*.json` retains these runs;
`geographic-initialization.patch`, applied in reverse to the current campaign,
reconstructs their original source hash. No model/tolerance was changed to fix
this: controlled runs use one reference-water conversion location. The general
real-data GSW conversion helper still requires actual coordinates from the caller.

## Complete-step throughput

Dedicated sequential CPU float64 timing, 1024 cells ×64 layers, dt60 s, four
mixing substeps, full forcing and ALE each step. Two warmups per model, three
measured samples each, with model order rotated between repeats. No other test
or campaign jobs intentionally overlap this measurement. Same geometry/forcing;
Roquet fields are converted offline at one reference-water location. Different
EOS/calorics mean this is not an equal-accuracy or trajectory-equivalence claim.

| Configuration | Median s/step | Samples, s |
|---|---:|---|
| Compiled JMD95 | 11.3795 | 11.3795, 11.5183, 11.2618 |
| Eager linear | 2.4099 | 2.2566, 2.7173, 2.4099 |
| Eager Roquet | 2.7194 | 2.7508, 2.6635, 2.7194 |

Roquet is 4.18× faster than compiled JMD95 in this sample and costs 12.8% more
than linear. This includes the constant-cp caloric simplification; it is not the
isolated cost of replacing the density polynomial. No Mac/GPU/global scaling or
additional compiler speedup is inferred. Run `roquet_timing.py` to reproduce.

## Reproduce and provenance

Commands and equations are in ROQUET.md. Additional diagnostics:

```sh
python validation/temperature_coordinate/roquet_response.py --output response.json
python validation/temperature_coordinate/roquet_column.py --output column32.json
```

CPU float64, PyTorch 2.15.0+cpu, GSW 3.6.23 for offline conversion/reference only.
Production wheel SHA256: 77e7066f7a56699f13e164b27052b4095be2ef453308b706b8d8bb6431d1c6a1
(the unchanged parent wheel). Research modules/tests were copied outside checkout.
Some verification jobs overlapped; only dedicated sequential runs qualify as
throughput samples. No Julia execution comparison, GPU, float32, sea-ice, steric
sea-level, AMOC or climate qualification is claimed. Files larger than 128 KiB
are omitted whole; no binary wheel or full state arrays are uploaded.
