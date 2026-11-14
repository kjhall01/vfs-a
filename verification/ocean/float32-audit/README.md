# Ocean float32 and device audit

The temperature-native research composition is tested in float32 and float64
with identical q4 Hamiltonian, fluxes, 60 s timestep and four mixing substeps.
No guards, tolerances, default dtypes or runtime numerical expressions change.
This is an audit, not qualification of float32 as a supported ocean mode.

## Device ownership

- `DryAtmosphere` and `MoistAtmosphere` are `torch.nn.Module` wrappers. Geometry
  and dynamics are registered; states remain separate and are explicitly moved
  with the model's state-conversion methods. Their docs recommend moving the
  model before initializing states.
- `VoronoiMesh`, `OceanColumn` and `OceanDomain` are modules with buffers.
- `Ocean`, `StratifiedOcean`, `TKEOcean` and research `TemperatureSystem` are plain
  Python objects. There is no complete public ocean `.to(device, dtype)` method.
  `OceanState.to(...)` moves state tensors but retains borrowed geometry.
- Prepared transport indices/degrees are ordinary cached tensors. Changing
  geometry/device/dtype after construction can invalidate them and is explicitly
  rejected. Compiled functions also belong to the configured model.

A thin ocean module owner would be useful for API consistency, buffer movement
and serialization. It must register shared geometry once, keep prognostic state
external, and rebuild derived caches/compiled wrappers after migration. Merely
inheriting `nn.Module` does not solve ownership or move unregistered caches.
This audit does not implement that public API change.

The harness copies mesh and column, casts them BEFORE constructing a fresh
StratifiedOcean, then binds explicitly converted state and prescribed fluxes to
that geometry. Connectivity remains integer. Thermodynamic coefficients are
Python scalars/lists evaluated with the input tensor dtype. Device synchronization
is included for a future GPU run, but recorded results are CPU-only.

## Measurement and independent conservation audit

Run from the repo with its VFS wheel installed:

```sh
python validation/temperature_coordinate/precision_campaign.py --compile \
  --cells 128 --layers 16 --steps 10 --output precision128.json
python validation/temperature_coordinate/precision_campaign.py --compile \
  --cells 1024 --layers 64 --steps 3 --output precision1024.json
```

One CPU thread, no autograd, two warmup calls and three fixed-input timing repeats.
Each precision uses the same compiled thermodynamic kernel definitions. The
float64 conservation oracle is eager and outside timing. Timed calls retain
reports and float64 reductions of their external-work arrays. Separate evolving
trajectories test conservation; no other timing/test job overlaps these runs.
The dtype ordering is sequential, so small timing differences are provisional.

The oracle promotes stored conservative contents and the candidate's rounded
geometry to float64 BEFORE evaluating the Hamiltonian or multiplying/reducing
inventories. Thus it audits the represented state, rather than only promoting an
already rounded energy total. Each dtype is measured against its own initial
state; initial float32 quantization is not counted as evolution drift.

Mass is Boussinesq reference mass (rho0 times volume); its relative error also
applies to volume. Energy includes resolved KE, the JMD95 Hamiltonian, caloric energy and TKE.
External heat/water-energy/stress report arrays are re-reduced in float64; their
per-element calculations are still in candidate precision, so this is not a
fully independent forcing-work formula. Prescribed net freshwater input is
computed independently in float64. Salt and passive inventories have no sources
in these cases. Reports include native-precision energy differences for comparison.

`forcing` means dynamics plus prescribed boundary operators, with mixing
explicitly disabled as a control. It is not a successful full-ocean run.
`ale` repeatedly alternates uniform and linearly varying target layer thicknesses
in its conservation trajectory; timing always remaps the same initial state to
a uniform target. Positivity is reported separately from conservation.

## Interpretation

The local mixing-budget tolerances are fixed physical tolerances originally
qualified in float64. Other energy checks scale with dtype epsilon. A rejected
float32 local solve can have a very small absolute error; a passing total-energy
check can nevertheless permit much larger error than its float64 counterpart.
Neither outcome alone establishes physical suitability for climate integration.

No float32 guard is relaxed to obtain a timing. Rejected cases are retained and
receive no successful whole-model speedup claim. No Mac/MPS/CUDA, long climate,
sharp-front, or full backward qualification is implied. Any future mixed-precision
candidate should preserve small energy transfers and freshwater increments,
with physical-unit drift criteria rather than solely dtype-scaled acceptance.

The small 128×16 ten-step forcing control has float32 relative mass error
1.19e-7 after subtracting prescribed freshwater. The intended net freshwater
addition itself is only 2.93e-8 of initial column mass (5e-6 kg/m²/s for 600 s,
100 m at rho0=1025 kg/m³). Thus a small-looking relative drift exceeds this weak
physical input by about fourfold. That comparison is specific to this case,
not a forecast of long-term mass drift.

## Recorded throughput

Seconds per call, median of three repeats. These are component/control timings,
not a successful float32 full-ocean result.

| Mesh | Workload | float64 | float32 | Ratio |
|---|---|---:|---:|---:|
| 128×16 | dynamics | 0.0925 | 0.0710 | 1.30× |
| 128×16 | ale | 0.1023 | 0.0583 | 1.75× |
| 128×16 | forcing | 0.1666 | 0.1036 | 1.61× |
| 128×16 | mixing | 0.2411 | rejected | — |
| 128×16 | full_ale | 0.6671 | rejected | — |
| 1024×64 | dynamics | 3.1101 | 1.7780 | 1.75× |
| 1024×64 | ale | 2.0178 | 0.9142 | 2.21× |
| 1024×64 | forcing | 3.9161 | 2.1102 | 1.86× |
| 1024×64 | mixing | 7.9551 | rejected | — |
| 1024×64 | full_ale | 12.5699 | rejected | — |

All pure float32 mixing/full cases reject with `lagged local budget failed`.
At 1024×64, the initial standalone mixing maximum local residual is 3.81e-11
J/kg, with maximum per-layer tolerance 2.23e-13 J/kg and worst normalized ratio
190.3. This residual is small physically, but the unchanged guard correctly
rejects it under the currently specified tolerances. Relaxing the guard was not
part of this audit and would not address the independently observed drift below.

## Two-hour conservation control

128×16, 120 steps of 60 s; mixing is absent in these completed controls.
Energy errors are changes minus external work/heat, evaluated in float64.
ALE repeatedly changes the target grid. Positive TKE/salt/passives remain
positive in the completed runs; positivity does not imply conservation.

| Workload | dtype | Energy error, J/kg | Relative mass error | Relative salt error | Relative tracer error |
|---|---|---:|---:|---:|---:|
| dynamics | float64 | -3.132e-10 | -7.991e-15 | -8.046e-15 | -6.678e-15 |
| dynamics | float32 | 1.596e-02 | 8.337e-07 | 4.473e-07 | 2.208e-07 |
| ale | float64 | -5.134e-12 | 1.567e-16 | -4.885e-15 | -4.517e-15 |
| ale | float32 | 7.730e-03 | -1.567e-16 | -4.266e-07 | -4.028e-09 |
| forcing | float64 | -3.033e-10 | -8.198e-15 | -8.046e-15 | -7.071e-15 |
| forcing | float32 | 1.033e-01 | 1.128e-06 | 6.002e-07 | 3.232e-07 |

Native float32 total-energy subtraction is not a reliable reference: in the
128×16 ten-remap test it reports +0.00276 J/kg while the promoted-state oracle
finds -0.00367 J/kg. The same state is being assessed at different arithmetic
precision. Promoting only the already-computed total cannot recover those bits.

These runs do not establish whether drift remains bounded over decades, nor
whether a mixed-precision scheme can meet climate accuracy requirements.
Preserving small conservative increments and thermal/TKE transfers is the next
numerical issue, rather than merely making failed checks more permissive.

The whole two-hour per-step JSON exceeds the upload cap and is omitted, not
split. `two-hour-summary.json` preserves endpoint and maximum absolute errors;
its provenance records the omitted raw file's byte count and hash.

In the forced two-hour float32 control, the 0.1033 J/kg excess is equivalent to
a mean 1.47 W/m² energy-budget error for this 100 m column. This is an average
over this short controlled run, not a measured climate drift rate.

## Regression scope

Sixteen installed-wheel checks pass outside the checkout: three new harness
checks (ownership/dtypes, promote-before-arithmetic inventory audit, and stale
cache rejection after a dtype move) plus thirteen existing composition checks.
No production package source changed. The parent full-suite result was 1115
passed and one CUDA skip; this audit does not claim another full-suite run.
