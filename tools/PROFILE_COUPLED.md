# Fully coupled VFS performance profile (CPU FP64 / CUDA FP64)

## Consolidated coupled performance baseline

The integrated code uses the original campaign defaults. The 4,096-cell timing
configuration selects the multirate schedule explicitly. The command in
[TEMPERATURE_VALIDATION_PROFILE.md](TEMPERATURE_VALIDATION_PROFILE.md) reproduces
its 800 s atmosphere, 1,200 s SSPRK(4,3) ocean and 2,400 s exchange settings.
Use a fresh start for this source revision. The audited historical day-15
checkpoint bridge applies only to the source pair in PR 98.

| PRs | What landed | Evidence and scope |
| --- | --- | --- |
| [97](https://github.com/kjhall01/vfs-a/pull/97) | Batched passive column mixing, fixed Redi stencil, donor transport shortcut | Isolated CPU kernel timings and old-path comparisons. |
| [98](https://github.com/kjhall01/vfs-a/pull/98) | Shared vapor/sensible flux roundoff accounting and historical restart verifier | One reported 1,200 s replay of the day-15 checkpoint on its audited branch. |
| [99](https://github.com/kjhall01/vfs-a/pull/99), [100](https://github.com/kjhall01/vfs-a/pull/100) | Optional SSPRK(4,3) ocean and explicit closure/exchange schedules | Shorter dynamics wall time at equal simulated time and small coupled pilot. Long-run fidelity of the longer steps is open. |
| [101](https://github.com/kjhall01/vfs-a/pull/101), [103](https://github.com/kjhall01/vfs-a/pull/103), [105](https://github.com/kjhall01/vfs-a/pull/105) | Coupled timings, ocean RHS timing and optional line profiling | Measurement tools. Runs with actual SeaEsfRad include optics and cache intervals. |
| [102](https://github.com/kjhall01/vfs-a/pull/102), [104](https://github.com/kjhall01/vfs-a/pull/104) | Same-stage carrier reuse for atmospheric cloud transport and Gibbs face reuse for the ocean limiter | Forward parity on the tested stages. These PRs previously merged into experiment branches. |
| [106](https://github.com/kjhall01/vfs-a/pull/106) | Removed provisional parent flux/work that the composed ocean update replaces | Mac throughput went from 10.146 to 10.265 simulated days/hour. Experimental incidence batching stays disabled. |
| [107](https://github.com/kjhall01/vfs-a/pull/107) | Removed two duplicate temperature-coordinate validations per ocean RHS | Mac call count fell by 384 across 24 intervals. Elapsed time increased from 233.795 to 247.428 s in that comparison. |

The bounded atmospheric Euler stage now evaluates the mixture gradients and
carrier primitives once, evaluates chemical potentials at cell interfaces, and
constructs the water endpoint from the limited species amounts. The diagnostic
RHS continues to return water tendencies. The profiler reports centered momentum
and mixture gradients separately under the bounded atmospheric stage.

On a Linux CPU with PyTorch 2.15.0+cpu, float64 and four threads, the median
complete 800 s atmosphere-plus-cloud advance went from 1.611 to 0.931 s
(1.73×). The comparison used 4,096 Fibonacci cells, 32 layers, compatible
rotation, the campaign's regional compilation and 10 measured advances per
revision in A/B/B/A order after two warmups per process. Wind, pressure,
entropy, water and cloud fraction matched bitwise after seven consecutive
advances. This measurement covers atmospheric dynamics alone.


The profiler is **non-invasive**: it runs the real
`workflows.coupled.campaign.advance` with the same source, mesh,
SeaEsfRad, PyAM21, prognostic atmosphere, moving ocean/ice/snow, GM–Redi,
TKE mixing, backscatter, and ALE that the multirate production driver uses.

It is not a synthetic ocean or atmospheric RHS benchmark. Its instrumentation
is in `tools/`, which the coupled numerical-source fingerprint excludes.
It does **not** change any physics timestep, parameterization, budget check,
checkpoint schema, compiled regional kernel, or state update. Start with a
freshly initialized state or a **strictly matching** checkpoint.

## 1. Run on the M4 Pro, using actual configured SeaEsfRad

From the VFS repo root, with `vfs-joint` (or other env containing the pinned
PyAM21 and its actual SeaEsfRad tables) activated:

```bash
python -u -I tools/profile_coupled.py \
  --config workflows/coupled/aquaplanet_multirate_4096_40.yaml \
  --mesh "$HOME/Desktop/mesh_4096_exact_3000.npy" \
  --source "$HOME/Desktop/PyAM21" \
  --radiation-data "$HOME/Desktop/sea_esf" \
  --set physics_dt=2400 \
  --set atmosphere_dt=800 \
  --set ocean_dt=1200 \
  --set ocean_dynamics_integrator=ssprk43 \
  --set mixing_dt=1200 \
  --set mesoscale_dt=1200 \
  --set radiation_dt=4800 \
  --set ale_dt=4800 \
  --warmup-intervals 4 \
  --measure-intervals 24 \
  --sample-io \
  --output "$HOME/Desktop/vfs-profile-4096-ssprk43-01"
```

**Do not edit the reference YAML to try parameters.** Use `--set` overrides:
the profiler validates the exact constructed `CampaignConfig`, reports it
in `profile.json`, and includes it in the checkpoint fingerprint. The above
command does 4 warmup + 24 timed 40-minute intervals (18 h 40 min total
simulated time); the 16 measured model hours include eight 2-hour radiation
and ALE events. The start/stop model times in `profile.json` distinguish
the warmup from the timed section.

To profile your currently successful SSPRK3 configuration, omit both
`--set ocean_dt=1200` and `--set ocean_dynamics_integrator=ssprk43`,
and set `--set atmosphere_dt=600`; the pilot YAML already uses 600 seconds
and SSPRK3. To compare fairly, keep every other value, mesh and number of
timed/warmup intervals identical. `--threads 1`, `--threads 4` and
`--threads 8` enable a CPU parallel-scaling study.

The profiler defaults to the YAML's `compile: true`. Compilation happens
lazily. The four untimed warmup intervals should eliminate the major
first-use costs, but can miss rare shapes or new control-flow branches.
Inspect the per-interval CSV for any remaining compilation spikes. For a
controlled eager comparison add `--no-compile`, keeping other settings
the same. Do **not** directly compare the cold-start construction time with
the steady-state throughput.

## 2. Output files

- `summary.txt` — human-readable, sorted, *exclusive* per-operator wall
  time, percentages of timed coupled work, call counts and median inclusive
  call durations.
- `profile.json` — machine-readable nested inclusive/exclusive timing
  aggregates; exact config, runtime, PyTorch/host, peak resident memory,
  source fingerprint and modeled throughput.
- `intervals.jsonl` — one detailed record per timed physical interval:
  model time, wall time, all stage counts/times, full original ocean report.
- `intervals.csv` — simple per-interval cost and optics, GM/Redi,
  mixing-substep, ALE call counts. It highlights expensive radiation events.
- `sample_diagnostics.json`, `sample_snapshot.nc`,
  `sample_restart.pt` — **only when `--sample-io`**; individual diagnostics,
  NetCDF and portable checkpoint costs are measured after the primary timed
  simulation and excluded from its throughput.
- `failure.json` / `driver_failure.json` — if a stage is rejected, report
  the traceback, exact failed interval and partial per-stage timing; no
  solver exception is caught and repaired.

`sample_restart.pt` is a standard VFS checkpoint using the profiler's
exact config/source identity, but it is not intended to substitute for
the original campaign output. Avoid using these performance artifacts to
resume runs with a modified configuration.

### Meaning of exclusive time

A section may contain many child sections. Its *inclusive* duration includes
those children; its *exclusive* duration subtracts all children. Summing
inclusive durations double-counts. Sorting by exclusive time identifies
where work actually goes. Uninstrumented operations (mesh checks, diagnostics,
Python orchestration and synchronization) remain in the parent
`coupled_interval` exclusive cost.

The instrument includes:

- atmospheric bounded dynamics and individual bounded Euler updates;
- ocean/ice/snow outer integrator, ocean–ice tendencies, implicit drag,
  ice phase and snowfall;
- TKE vertical mixing, including each internal mixing substep;
- GM–Redi substeps, EKE decay and energy-funded backscatter;
- full PyAM21 column call, down/couple/up and the vertical diffusion sweeps;
- atmospheric vertical turbulence, RAS convection, stratiform cloud
  microphysics and surface tiled bulk fluxes;
- actual SeaEsfRad optical transfer versus cached radiative-driver calls;
- reciprocal ocean surface settlement, post-physics phase and ALE remap.

These are wall-time *regions*, not an exhaustive flop count. Tensor allocator
cost, gather/scatter, thread scheduling, Inductor runtime, and external native
kernels are included in their corresponding parent region.

## 3. Optional deeper traces

One extra ATen/PyTorch operator trace (CPU or CUDA only) can be requested:

```bash
# Add to the normal profile invocation:
--torch-trace-intervals 1
```

This runs an **additional** representative coupled interval after the
standard measurement and writes `torch_trace.json`, plus sorted
`torch_ops_cpu.txt` (and `torch_ops_cuda.txt` on CUDA).
The trace is intentionally excluded from throughput; it adds enormous
overhead and may consume gigabytes of storage. View
`torch_trace.json` in Perfetto/Chrome trace analysis.

For CUDA, the default timings synchronize once per complete coupled
interval (truthful end-to-end throughput). The nested stage timers normally
measure CPU dispatch/blocking, **not** device execution time. For accurate
GPU stage elapsed time, run separately with
`--synchronize-stages`; it synchronizes at every region boundary and
may substantially reduce throughput. Never compare its headline SYPD to
the default profile as if they had equivalent overhead.

Apple MPS requires FP32; VFS's stringent FP64 budget tests should be run
on CPU or FP64-capable CUDA hardware. Do not enable fallback for missing MPS
operations.

## 4. Representative state and safe restarts

The default starts from the analytic climate initialization; early transients
may have significantly different closure costs than an evolved state.
For accurate steady-state profiling, run from a **provenance-identical**
checkpoint:

```bash
# Add to the standard command:
--resume "/path/to/compatible/restarts/restart_00086400.pt"
```

The checkpoint identity requires the **same** source tree, exact YAML/overrides,
mesh, runtime, radiation tables and compilation policy. The profiler fails
closed if it does not match. Do not attempt an unaudited resume from your
previous 6-h / 18-h failed multirate runs if their source/cadences differ.

## 5. Interpreting results

Prioritize the largest exclusive contributors and inspect nested calls:

- If `atmosphere_bounded_euler` dominates, profile mesh gather/scatter,
  slow limiter paths, full-H EOS gradients and per-stage GPU sync.
- If `GM_Redi_internal_substep` dominates, compare limiting passes,
  vertical stencils and actual eddy substep time before changing the
  number of closure evaluations.
- If `ocean_mixing_internal_substep` dominates, count nonlinear
  iteration/reduction cost and factorization over 40 levels.
- If `radiation_optical_transfer` dominates, examine the 4800-s
  radiation cadence and optically expensive spectral kernels.
- If `RAS_convection` or `stratiform_cloud_microphysics` dominates,
  profile column loops versus tensor kernels.
- If `coupled_interval` exclusive remains large, inspect orchestration,
  compiled-graph re-entry, state checks, Python tensor-to-bool sync,
  and optionally the ATen operator trace.

**Known physics caveat:** A short profile passing does not qualify climate
stability, a full day of convection, or long-run numerical energy drift. It
records actual throughput over the chosen state/window, not a claimed
production 365-day SYPD.
