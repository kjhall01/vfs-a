# Ocean timestep campaign

Runs the **Roquet research ocean**, CPU float64, explicit SSPRK3, compatible
rotation. It does not establish a timestep for the atmosphere or JMD95 model.
Defaults: Earth radius, 4,000 m depth, 40 exponentially stretched layers, one
CPU thread. This is a fully wet, flat-bottom idealized ocean. Model guards and
numerical tolerances are unchanged.

## Start with a failure screen

From the repository, with VFS installed in the active Python environment:

```sh
python -I validation/timestep/run.py --cells 16384 --layers 40 \
  --hours 6 --dt 120 240 360 480 600 900 1200 \
  --screen --output timestep-16384-screen
```

For the previously relaxed mesh, append `--mesh /path/to/mesh_16384_exact.npy`.
The file must contain (16384,3) Cartesian generator coordinates. Its contents
are hashed; VFS rebuilds compatible geometry. Default zero additional Lloyd
iterations preserves those generators. Without a file the mesh is unrelaxed
Fibonacci, explicitly recorded; do not transfer its limit to another mesh.
No mesh download, package dependency on external datasets, or large upload.

The screen reports failures and budgets; **it never qualifies a timestep**.
Each trial runs in its own process, saves progress after every accepted step,
and has a configurable wall timeout (default 12 hours). Failed trials do not
stop the remaining trials. `--resume` reuses only completed trials with matching
configuration, code/runtime hashes and local endpoint files. Interrupted trials
restart from initialization, not a partially evolved state.

## Compare with a timestep-refined reference

After screening, restrict the candidate list to plausible values:

```sh
python -I validation/timestep/run.py --cells 16384 --layers 40 \
  --hours 6 --dt 120 240 360 480 600 --reference-dt 60 \
  --output timestep-16384-qualified --resume
```

This runs 60 s and 30 s references for **each case**, then compares candidates
against the 30 s endpoint. Reference differences must fit one-quarter of the
stated error budgets; a failed/unconverged reference cannot qualify anything.
The final step is shortened to reach the exact same physical endpoint. There
is no automatic binary search or assumption that acceptance is monotone in dt.
`summary.json` reports the largest **tested** timestep meeting all criteria
across all requested cases. `null` means none qualified, not zero stability.
If the largest candidate passes, the upper boundary remains unbracketed.

Default illustrative research budgets (all CLI-configurable):

| Quantity | Criterion |
| --- | --- |
| Edge velocity endpoint RMS | 0.001 m/s |
| Conservative Temperature endpoint RMS | 0.01 degrees C |
| Absolute Salinity endpoint RMS | 0.001 g/kg |
| Layer thickness endpoint RMS | 0.1 m |
| Specific TKE endpoint RMS | 1e-5 m2/s2 |
| Maximum energy minus prescribed input | 1e-3 J/kg initial ocean mass |
| Maximum mass/salt/passive inventory error | 1e-10 relative |

These are explicit experimental error budgets, not established climate accuracy
standards. RMS uses reference layer mass, or reference edge volume for velocity.
Comparisons are in corresponding layer indices, not interpolated fixed depths;
geometry differences are also reported separately. The reference controls
**temporal** error in this discrete model, not continuum error. No fixtures or
production guards are relaxed. Inspect raw errors, not just the boolean label.

## Cases and cadence

- `wave`: stable stratification and a localized 10 m free-surface pulse; no
  forcing, mixing or ALE. Tests explicit fast-wave dynamics in a deep ocean.
- `front`: unbalanced sharp temperature front with 0.5 m/s surface zonal flow;
  stresses active transport and pressure adjustment. No mixing or ALE.
- `forced`: pulse plus zonal shear, 0.1 N/m2 zonal stress, 200 W/m2 shortwave,
  -100 W/m2 nonpenetrating heat, bottom drag, prognostic TKE/mixing, ALE every step.
- `rest`: optional stratified rest control, not a useful stability test alone.

Salinity and passive inventories are closed. The forced case adds prescribed
energy but no freshwater mass. It is not qualification of freshwater exchanges,
coasts, topography, convective extremes, or real-world circulation.

Mixing substeps are `ceil(dt / mixing_dt)`, with default `mixing_dt=60 s`.
This holds the maximum physics substep approximately fixed while increasing
macro-dt; it does not falsely credit a larger timestep for under-resolving
mixing. Actual substeps can be inferred from each trial's dt and mixing_dt.
ALE targets the original stretched fractions of the evolving column depth,
not equal-thickness layers. ALE frequency changes with macro-dt and its error
is deliberately part of the complete-step comparison.

The reported wave Courant number is `(sqrt(g*max_depth)+max_speed)*dt/min_dc`.
It is an indicator, **not** the exact eigenvalue stability bound on this mesh.
A smooth pulse does not excite every grid-scale mode. Completion of six hours
is not evidence for multiyear stability.

## Extend the result before production

1. Screen at 6 hours, then qualify the surviving candidates with refined refs.
2. Repeat candidates near the limit at 24–72 hours. Prefer a timestep below the
   observed boundary; repeat at 24/40/64 layers and the actual target mesh.
3. Add the intended coast/bathymetry, sharper vertical structure and forcing
   extremes before using the result for a global configuration.

Multiple resolutions: `--cells 1024 4096 16384` (generated meshes only).
Choose `--cases wave` to isolate dynamics or `--cases forced` for full physics.
A small installation check is:

```sh
python -I validation/timestep/run.py --cells 32 --layers 4 \
  --hours .016666666666666666 --dt 30 60 --reference-dt 30 \
  --cases rest wave front forced --output timestep-smoke
```

Timing includes all per-step budget diagnostics and ALE, excludes setup, and
is not a replacement for the performance harness. No autograd tape is retained
while stepping. Endpoint `.pt` files contain only tensors and are loaded with
`weights_only=True`; they can be large and stay local. Do not commit or upload
them. Preserve compact JSON summaries and logs; omit any file over 128 KiB.

## Compilation and live progress

Append `--compile --verbose` to any command. `--compile` enables per-model
`torch.compile(backend='inductor', fullgraph=True, dynamic=True)` for Roquet
layer energy/derivatives, polynomial coefficients, parcel potential, and pure
caloric values. Public domain/positivity checks, transport, mixing control,
ALE acceptance and the Python timestep loop remain eager. This is selected
kernel compilation, not whole-model compilation. A compiler error is reported
as a failed worker; there is no silent eager fallback. A working CPU Inductor
compiler toolchain is required. No accelerator qualification is implied.

`--verbose` streams initialization and every accepted step to the terminal and
retains the same messages in each trial's `.log`. Lines include simulated time,
completed/total steps, elapsed run wall time, mean seconds per step, estimated
remaining wall time, and energy minus prescribed inputs. Timeout handling is
retained even while a worker is silent. Without `--verbose`, only the existing
per-trial start/result lines appear.

Compilation is lazy. Initial energy and the first dynamics/physics/ALE calls
may compile different graphs. Initial energy compilation is included in setup
wall time; later compilation is included in run wall time. The first ETA can
therefore overestimate subsequent costs. Neither reported SYPD nor mean step
time excludes startup compilation. Use the performance harness for warmed
throughput comparisons. Both flags are recorded in configuration/resume matching.
