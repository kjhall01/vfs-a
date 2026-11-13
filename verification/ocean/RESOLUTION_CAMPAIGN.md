# Higher-resolution semi-implicit ocean mixing

This campaign tests the research semi-implicit closure with JMD95 at 32 and
64 layers in a 100 m column, and native spherical meshes with 128, 512 and
1,024 cells. The supported model and its defaults are unchanged. Runs use CPU/float64,
Torch 2.15.0+cpu and the installed unchanged wheel at parent commit
`c260d53ff83414c40b655d968c8cfea30a555472`.

## Reproduction and interpretation

Install the wheel, copy validation/tests outside the checkout, then run:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python validation/ocean_column/resolution_campaign.py --output results --workers 3
python validation/ocean_column/resolution_mesh.py --cells 128 --layers 32 --methods coupled16 coupled1 semi4 semi8 --output results/mesh128-32.json
python validation/ocean_column/resolution_mesh.py --cells 128 --layers 64 --methods coupled16 coupled1 semi4 semi8 --output results/mesh128-64.json
# Repeat for 512 and 1024 cells, both layer counts, with coupled1 semi4 semi8.
python validation/ocean_column/analyze_resolution.py --input results --output evidence
python -I -m pytest tests --fixture-mode=portable -q
```

The column matrix includes 24-hour wind forcing at both layer counts, one-hour
outer-timestep/substep/safety controls, a coupled dt75 numerical reference, and
64-layer twelve-hour cooling and rain/heating scenarios. Unless overridden,
outer dt is 300 s and mixing takes four substeps. The `reduction` metadata is
the research limiter safety factor; inherited `backtrack_safety` describes the
unused production-closure argument and does not configure this candidate.

Mesh tests advance 300 s of vertical mixing, with native shared-edge momentum
and its cell energy allocation. They do not step horizontal transport, pressure,
rotation or ALE. Layer interfaces and initial profiles are fixed functions of
physical depth, including unequal neighboring layer thicknesses. Meshes use
Fibonacci points and zero Lloyd iterations, not the recovered atmosphere SCVT
mesh. This tests batch cost and native vertical coupling, not jet balance.

The 128-cell reference uses sixteen coupled substeps. Larger meshes have no
fine coupled reference; their results establish budget and cost behavior only.
Neither numerical reference is an analytical solution or proof of asymptotic
convergence. Vertical comparisons conservatively overlap layer intervals, with
32-layer values compared against averaged 64-layer values. Initial profiles
are sampled at layer centers, so initial discretization error also contributes.

Timing uses one Torch thread. Mesh setup is excluded; CPU and wall times are
reported separately. Jobs overlap, so timings are not isolated throughput
benchmarks. Peak RSS includes mesh setup and all preceding methods in that
process. These 100 m tests do not qualify full-depth oceans, coasts/dry cells,
GPU performance, backward differentiation, or complete global time stepping.

Evidence files retain failure states and completed durations. Column summaries
contain initial/final profiles and hourly scalar diagnostics. Full oversized raw
profile files are omitted, with their sizes and hashes in `resolution/manifest.json`.
No raw file is split to bypass the 128 KiB upload cap.

## Regression

Installed unchanged wheel, outside checkout: **1,031 passed, one CUDA skip**
(269.36 s), with the existing tensor-to-scalar warning. A new geometry regression
checks that splitting 32 layers into 64 preserves each coarse-layer thickness
and the native horizontal areas. Corrected-Gibbs and dry-helper portable fixture
comparisons pass. No supported runtime, fixture, tolerance or default changes
are included.

## Vertical refinement and numerical controls

An additional one-hour 128-layer column uses outer dt75 with four mixing
substeps. At that same timestep/substep configuration, conservative vertical
comparison gives:

| Layer comparison | TKE relative L2 difference | Velocity relative L2 difference | Temperature RMS difference (C) |
|---|---:|---:|---:|
| 32 against 64 | 0.38968 | 0.06336 | 0.007078 |
| 64 against 128 | 0.17194 | 0.02516 | 0.001788 |

Differences shrink with refinement, especially for temperature, but a 17.2%
TKE difference does not establish a converged turbulent profile. At outer dt300,
the 32/64-layer TKE difference is 35.85%. The mixed-layer temperature threshold
has the same discrete depth in the dt75 comparisons; that coarse diagnostic
alone misses the substantial TKE differences.

One-hour differences against the same-layer coupled dt75 numerical control:

| Layers | Outer dt / mixing substeps | TKE relative L2 | Velocity relative L2 | Temperature RMS (C) |
|---|---|---:|---:|---:|
| 32 | 300 / 4 | 0.02242 | 0.02301 | 0.0002911 |
| 32 | 300 / 8 | 0.01714 | 0.02517 | 0.0002538 |
| 32 | 150 / 4 | 0.01286 | 0.00939 | 0.0002406 |
| 32 | 75 / 4 | 0.01021 | 0.00261 | 0.0002253 |
| 64 | 300 / 4 | 0.06780 | 0.04235 | 0.0001916 |
| 64 | 300 / 8 | 0.07622 | 0.04736 | 0.0002010 |
| 64 | 150 / 4 | 0.02994 | 0.01965 | 0.0001895 |
| 64 | 75 / 4 | 0.02056 | 0.00794 | 0.0001856 |

Refining the outer timestep improves TKE/velocity agreement in these controls.
Increasing mixing substeps alone is not monotonic: the 64-layer dt300/eight-
substep result differs more from the reference than the four-substep result.
Outer forcing/rotation splitting and the reference's own dt75 error remain.
The equal-outer-dt75 semi-implicit run uses four internal mixing steps whereas
the coupled control uses one; these differences are not pure solver errors.

Changing scalar funding safety from .9 to .5 at outer dt300/four substeps changes
TKE by 0.0598% at 32 layers and 0.0753% at 64 layers; temperature RMS differences
are 4.85e-5 C and 1.74e-4 C. Small bulk safety sensitivity can coexist with strong
local coefficient reduction. The minimum coefficient fraction is an extreme
statistic, not the fraction of total heat/salt transport suppressed. The current
reports do not measure a flux-weighted suppression fraction.

A heterogeneous 1,024-cell/64-layer control additionally reduces initial TKE by
1e-6 in cells with z-coordinate greater than .95 (a small northern cap), using
four semi-implicit substeps. Its timing and retry count probe a weak-turbulence
patch's batch effect; its physical initial state differs from the uniform case.
Reproduce with `--cells 1024 --layers 64 --patch --methods semi4`.

The first 512-cell/64-layer attempt was administratively interrupted before any
method completed, to run the remaining batches concurrently. Its interrupt log
is retained separately; the reported case starts afresh from the same initial
state. This interruption is not a numerical rejection.

For the 512/1,024-cell 64-layer cases, coupled and semi-implicit timings use
separate processes with identical setup. Their evidence files have `-coupled`
and `-semi` suffixes. Run the mesh script once with `--methods coupled1` and
once with `--methods semi4 semi8`; the table reports the maximum process RSS.
No fine numerical reference is supplied for those larger meshes.

## Longer forced columns

| Case | Completed hours | Trials | Whole-column deferrals | Energy residual (J/kg) | Minimum coefficient fraction |
|---|---:|---:|---:|---:|---:|
| L32-wind24 | 24 | 3455 | 0 | -5.440e-10 | 3.796e-22 |
| L64-wind24 | 24 | 3455 | 0 | -5.741e-10 | 4.945e-57 |
| L64-cooling12 | 12 | 1727 | 0 | -5.870e-12 | 7.862e-55 |
| L64-rain_heating12 | 12 | 1727 | 0 | -5.899e-11 | 6.758e-55 |

All seventeen column cases complete. The full energy residual includes prescribed boundary work, heating and water-energy input. All scenarios include wind stress; cooling adds -100 W/m², and rain/heating adds rain for six hours plus continuing radiative forcing. Salt and passive inventories remain conservative. Their maximum relative residuals over the matrix are 2.603e-15 and 3.549e-16, respectively.

At 24 hours, the 32/64-layer wind endpoints differ by 19.76% in TKE relative L2, 2.94% in velocity relative L2 and 0.01681 C temperature RMS. Both temperature-threshold mixed-layer depths are 15.625 m. There is no fine-time-step 24-hour reference. Extremely small minimum coefficient fractions occur without whole-column deferral; these are locally almost disabled scalar interfaces, not unmodified mixing.

## Native-mesh cost

| Cells | Layers | Coupled 1 CPU s | Semi 4 CPU s | Semi 8 CPU s | Peak RSS (MiB) |
|---:|---:|---:|---:|---:|---:|
| 128 | 32 | 29.31 | 5.76 | 12.77 | 287.5 |
| 128 | 64 | 62.43 | 14.07 | 27.66 | 310.9 |
| 512 | 32 | 88.43 | 24.08 | 44.91 | 362.7 |
| 512 | 64 | 393.44 | 86.30 | 115.47 | 447.7 |
| 1024 | 32 | 247.80 | 50.85 | 104.97 | 451.4 |
| 1024 | 64 | 731.21 | 133.06 | 211.20 | 635.2 |

All twenty-one mesh method cases complete. Maximum absolute total-energy residual is 1.027e-11 J/kg. TKE and passive amounts remain nonnegative, and no whole-column scalar deferral occurs. Minimum TKE can be extremely small; no floor is added.

Differences against sixteen coupled substeps on the 128-cell mesh:

| Layers | Method | TKE relative L2 | Velocity relative L2 | Temperature RMS (C) |
|---:|---|---:|---:|---:|
| 32 | coupled1 | 0.31823 | 0.000292 | 5.490e-04 |
| 32 | semi4 | 0.08391 | 0.000162 | 1.564e-04 |
| 32 | semi8 | 0.02890 | 0.000051 | 8.143e-05 |
| 64 | coupled1 | 0.31959 | 0.000482 | 3.579e-04 |
| 64 | semi4 | 0.08343 | 0.000191 | 1.719e-04 |
| 64 | semi8 | 0.02887 | 0.000063 | 9.470e-05 |

The weak-TKE cap control completes in 147.41 CPU seconds with 12 trials, 0 whole-column deferrals and minimum scalar coefficient fraction 2.054e-03. This is one heterogeneous-state stress test, not a bound on arbitrary-column batch cost.

## Scope and next qualification

This campaign supports the semi-implicit candidate's budget robustness and
relative cost advantage at larger batches. It does not justify treating the
TKE profile as vertically converged or the current implementation as a fast
global closure. Single-thread CPU costs remain substantial even for 1,024
columns, and simultaneous jobs make these timings unsuitable for a definitive
hardware-throughput claim. No GPU or compiled-kernel benchmark is included.

The next focused checks are:

1. Extend the 128-layer column to the longer forcing cases and refine the outer
   timestep as well as mixing substeps. Compare depth-resolved TKE, temperature,
   salinity and diffusivity, not only a temperature-threshold mixed-layer depth.
2. Record where limiting occurs and its flux-weighted reduction of heat/salt
   transport. Extremely small minimum fractions alone cannot quantify the
   impact on bulk entrainment or distinguish negligible deep fluxes from an
   important suppressed interface.
3. Profile the thermodynamic work/inversions and banded solves separately on an
   isolated large batch, then optimize the measured bottleneck while retaining
   independent full-energy checks and comparison against the coupled control.

Keep this as a research candidate with runnable qualification harnesses. Do not
change supported defaults or infer full global-dycore accuracy from vertical-
mixing batches. Evidence is retained in `resolution/`; four oversized raw
profile files are omitted rather than uploaded or split.
