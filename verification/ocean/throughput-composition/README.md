# Temperature-native complete-step qualification

The optional research composition advances temperature content through dynamics,
prescribed surface forcing, TKE mixing and adaptive thermal ALE. The supported
entropy model and defaults remain unchanged. See
[API and equations](../../../validation/temperature_coordinate/COMPOSITION.md).

## Complete-step timings

CPU float64, one thread, compiled thermodynamic kernels in every variant;
60-second timestep, four mixing substeps, ALE every step. Timed calls include
reports and their checks. Two initial calls warm each configuration. Separate
calls count inversions and check trajectories; instrumentation is outside timing.

| Configuration | 128 cells × 16 layers, median of 3 | 1024 cells × 64 layers, single sample |
|---|---:|---:|
| Entropy baseline | 1.140 s | 20.241 s |
| Faster surface solve, entropy storage | 0.923 s | 56.955 s |
| Temperature storage, enthalpy path | 0.544 s | 11.693 s |
| Temperature storage, three-point T/S path | 0.448 s | 18.618 s |
| Experimental one-point T/S path | 0.407 s | 7.610 s |

The small-case three-point composition is 2.55× faster than this already-compiled
baseline. The large single sample is only 1.09× faster and is not evidence of a
stable gain: the surface-only case has a severe timing spike and the T/S path
is slower than the enthalpy path. Raw samples are retained. Do not multiply these
ratios by earlier gains or treat them as Mac/global SYPD forecasts.

At 1024×64 the three-point native composition makes nine enthalpy inversions
(including ALE), versus 36 in the entropy baseline and 33 in the temperature
enthalpy-path composition. It makes zero entropy inversions. This is a reliable
work-count reduction, not a guarantee of proportional wall-time savings.
Temperature RMS difference from the baseline after one step is 1.15e-14 °C;
energy change minus external input is -4.81e-12 J/kg for both. Such tiny
one-step differences do not establish long-run equivalence.

The one-point T/S result is exploratory. A budget-triggered three-point retry
exists, but an accepted budget does not certify layerwise allocation accuracy.
Fixed three-point work remains the comparison configuration. No quadrature
order or tolerance in the Hamiltonian was changed.

## Alternating large-case follow-up

A fresh process alternates the same compiled baseline and three-point candidate,
reversing order on the middle repeat. Each has two warmup calls and three timed
samples; no tests or other campaign runs concurrently. Complete-step medians
including ALE are **21.728 s → 9.709 s (2.24×)**. Baseline samples are
[21.728, 18.914, 28.293]; candidate samples are
[9.709, 9.979, 9.303]. This supports a gain on
this CPU case, but does not erase the variability in the first sequence or
establish performance on a Mac, 16,384 cells, a long run, or a different dt.
Both measurements use the same compiled thermodynamics; this is not a
compilation-versus-eager speedup. See `paired-1024x64-ale.json`.

## Duration and physical checks

- A 32×8, one-hour composed run (60 steps) includes all prescribed forcing,
  four mixing substeps and ALE every step. All three configurations complete;
  salinity, tracers and TKE remain positive. The three-point candidate's final
  energy change minus input is -1.51e-10 J/kg, identical to the entropy result
  at reported precision. Temperature RMS difference is 7.86e-14 °C.
- A 128-cell, one-layer, one-day solid-body/Williamson jet comparison completes in both
  coordinates. Their final velocity RMS difference is 8.96e-14 m/s. However,
  each departs from the analytic initial steady flow by 0.981 m/s RMS and
  10.35 m RMS thickness. Energy drift is -8.06e-5 J/kg. Coordinate agreement
  does not establish good balance on this coarse mesh.
- A 32-cell, three-layer resting ocean over varying bottom depth remains at
  rest for one day: maximum final speed 2.17e-13 m/s and reported energy drift
  zero. The native/entropy velocity RMS difference is 8.42e-16 m/s.
- A 32-cell, two-layer internal wave completes one continuum period
  (117898.9 s, 1113 steps). Frequency/reference is 0.976811 in both coordinates,
  a 2.32% low bias; final envelope/initial amplitude is 0.999381. Temperature
  coordinate energy drift is -1.58e-9 J/kg; endpoint coordinate differences are
  6.92e-14 °C RMS and 3.36e-13 m/s RMS. This is a coarse continuum comparison,
  not an exact discrete eigenmode or fine-grid convergence result.
- Thirteen focused tests cover surface equivalence, stable/unstable columns
  with zero/near-zero/positive TKE, removed inverse calls, deep 32-layer mixing,
  forced complete-step energy and inventories, ALE equivalence and identity,
  coastal remap/domain rejection, explicit midpoint retry, and a small composed
  gradient smoke test. They pass against the installed wheel outside checkout.

Balance/wave cases use unforced dynamics; they do not test the forced mixed
column composition. The first complete-step timings finish before regression jobs start; the
alternating follow-up timing runs after regression finishes. No timing job
intentionally overlaps tests or another campaign.

## Limits

Mixing remains restricted to fully wet columns. The ALE translation duplicates
the supported algorithm for comparison and is not a new public package API.
Finite time integration and nonlinear coordinate changes retain truncation
error. A small backward smoke test does not qualify every limiter branch.
No GPU, Mac, fine-mesh balance, coupled atmosphere, or long climate-run result
is claimed. Individual artifacts larger than 128 KiB are omitted.

Installed-wheel, outside-checkout portable regression: **1115 passed, one CUDA
skip**, one existing warning. Thirteen new composition checks pass.
