# Rotating wave and prescribed trade-wind aquaplanet — 2026-11-14

Seven integrations completed: 11,520 steps and 23 aggregate model days. The
model is the budgeted-biharmonic checkpoint `8b760f5`; only evaluation code was
changed. These runs assess short global adjustment and whether the paired
closure distorts a broad wave. They do not qualify equilibrium circulation,
mesoscale statistics, or spatial convergence.

## Prescribed trade winds

Earth radius, 256 cells, 12 stretched layers, flat 1000 m bottom, five days,
dt=300 s. Initial CT=4+16 exp(z/300), SA=35-.2 exp(z/200), velocity zero.
Wind stress is purely zonal before projection onto native edge normals:

    tau_east = .05 [-exp(-(latitude/20 degrees)^2)
                    + .5 exp(-((abs(latitude)-45 degrees)/15 degrees)^2)] Pa.

This gives easterly tropical trades and weaker midlatitude westerlies. Vertical
TKE mixing, bottom drag, energy-dependent GM, Redi and reservoir decay are
active. There is no heating, freshwater, restoring, land or sea ice. The
mesoscale reservoir starts at .1 J/kg and turbulent TKE at 1e-4 J/kg.

| Five-day endpoint | GM–Redi | + backscatter | + backscatter + sink |
|---|---:|---:|---:|
| Peak native edge speed (m/s) | .37313 | .37424 | .37424 |
| Mean surface eastward current within 20° (m/s) | -.07383 | -.07417 | -.07417 |
| Surface elevation RMS (m) | .029047 | .029053 | .029053 |
| H-minus-wind-work error (J/kg) | -5.496e-9 | -5.511e-9 | -5.501e-9 |
| KE in highest-energy ~1% of edges | 10.12% | 10.09% | 10.09% |

The tropical surface current is westward. Upper-100 m meridional transport
shows outflow away from the equatorial region in the latitude profiles. The
profiles retain spatial roughness and inertial adjustment; they are instantaneous
coarse-grid zonal means, not an equilibrated Ekman/overturning solution.

With the sink, total wind work is 4.6467e-4 J/kg, so the energy residual is
about 1.2e-5 of the supplied work. The final minimum potential-N² proxy stays
positive (~3.36e-6 s^-2), and the mesoscale reservoir stays positive. Backscatter
supplies 3.3784e-6 J/kg and the sink returns 4.4410e-8 J/kg to the reservoir.
The sink is weak for this smooth, coarse Earth-radius case: the marked
improvement in the smaller-radius channel does not imply it must substantially
change this flow. No obvious new short-edge concentration appears here.

See `trade-wind-profiles.png` and `trade-wind-fields.png`. Cell winds in these
plots are reconstructed diagnostics; native edge KE supplies the concentration
metric. The 100 m transport uses partial overlaps with model layers, not a
fixed number of top layers.

## Unforced rotating surface wave

Earth radius, 256 cells, eight layers, flat 1000 m bottom, uniform CT=10 and
SA=35. A smooth bump `0.1*exp((x-1)/.15)` m, minus its area mean, perturbs total
column thickness; the bump is divided among layers in proportion to their
rest thicknesses. It is centered at 0° longitude/latitude. The sampled initial
maximum is .08784 m, with compensating negative elevation elsewhere. Initial
velocity is zero. Rotation remains enabled. No external forcing or vertical
mixing is applied. All variants share reservoir initialization and decay.

Three variants run for two days at dt=120 s; GM–Redi also runs at dt=60 s.
The finer-dt result is a temporal reference, not analytic truth.

| Two-day endpoint | Elevation RMS (m) | Relative L2 height difference from dt=60 GM–Redi | Full-H error (J/kg) |
|---|---:|---:|---:|
| GM–Redi, dt=60 | .011416897 | 0 | -6.489e-9 |
| GM–Redi, dt=120 | .011416896 | 1.917e-7 | -3.245e-9 |
| + backscatter, dt=120 | .011418565 | 2.485e-4 | -3.237e-9 |
| + backscatter + sink, dt=120 | .011418562 | 2.479e-4 | -3.237e-9 |

The paired closure changes the final elevation field by about **0.025%** relative
to the finer-step baseline; RMS difference is 2.83 micrometers and normalized
pattern inner product is .99999998. Halving the baseline timestep changes its
height field by only 2.19 nanometers RMS. At this amplitude/mesh/coefficient,
there is little broad-wave distortion. This does not establish dispersion
accuracy against an exact solution or protection of every resolved wave scale.

Energy drift roughly doubles with twice as many steps at these tiny absolute
errors, consistent with accumulated roundoff rather than a clean temporal
truncation-error regime. This is an interpretation, not a proven isolation of
every error source. Decreasing surface-height RMS alone is not dissipation:
energy also exchanges with resolved motion and the pattern propagates/adjusts.

`wave-evolution.png` compares histories. `wave.gif` animates actual saved cell
samples with a fixed color range and no spatial interpolation.

## Numerical and diagnostic qualification

All seven runs completed with final volume/salt/passive-inventory errors below
1.78e-13. Saved snapshots, initial/accepted states, configurations and source
hashes are retained. These are CPU float64, uncompiled runs. Concurrent campaign
wall times are not performance benchmarks. No model numerical method was tuned
or changed during this evaluation. The checkpoint's prior 1227-pass/2-skip suite
was not rerun for these driver/analysis-only additions.

A new elevation diagnostic initially broadcast `(cell,1)` eta against `(cell,)`
areas, inflating RMS. Energy, inventory and native-speed diagnostics and the
integrations were unaffected. The analyzer now recomputes correctly shaped,
area-weighted RMS and initial-pattern projections from saved snapshots.
**Use `wave_trade_metrics.json` and the plots for elevation diagnostics; the
original raw CSV/summary elevation diagnostics are retained unchanged.** The
first finer-step run lacked intermediate snapshots and was repeated with corrected
diagnostics and snapshots. Its endpoints are bitwise identical across all 11
saved arrays, recorded in `diagnostic-correction.json`. That pilot is archived
under `diagnostic-pilot/` and excluded from the seven-run comparison.

## Reproduce or increase resolution

The driver supports `--case wave` and `--case tradewinds`, `--snapshots`, existing
mesh files, CPU thread selection and optional thermodynamic compilation. See
`validation/ocean_evaluation/README.md` for the full output/configuration guide.

```sh
python -I validation/ocean_evaluation/run.py --case tradewinds \
  --cells 1024 --layers 16 --days 5 --dt 120 --wind .05 \
  --biharmonic-viscosity 1e13 --variants gmredi backscatter \
  --log-every 360 --snapshots --max-wall-seconds 86400 \
  --output trades-1024

python -I validation/ocean_evaluation/run.py --case wave \
  --cells 1024 --layers 16 --days 2 --dt 120 \
  --biharmonic-viscosity 1e13 --variants gmredi backscatter \
  --log-every 60 --snapshots --output waves-1024
```

The larger-grid timesteps and viscosity are starting values, not qualified
limits or calibrated parameters. A no-sink backscatter control uses a fresh
output with `--variants backscatter --biharmonic-viscosity 0`. For the wave
half-step control use `--variants gmredi --dt 60`. Put campaign folders below
a common directory and run `waves_and_trades.py ROOT` to compare them.
The script bundle contains the evaluation helpers and current backscatter
implementation for extraction into an updated VFS checkpoint checkout.
