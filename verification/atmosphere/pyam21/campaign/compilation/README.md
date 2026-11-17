# Regional coupled CPU compilation — 2026-11-17

The campaign now accepts `--compile` / `--no-compile`, overriding YAML
`compile: false`. This is an opt-in execution policy, with unchanged model
parameters, enabled processes, numerical expressions and acceptance checks.
The production backend is native Inductor. Compilation is lazy, per instance,
and requires full-graph capture of each selected region; errors are not hidden
by an eager retry. The manifest and restart fingerprint identify the policy,
backend and targets. Changing policy requires a fresh campaign start.

## Choice of regions

A two-interval eager profile at 512 cells, 24 atmosphere / 12 ocean levels,
four threads and 1200/240/1200 s ocean/atmosphere/physics steps measured 6.632 s:

| Stage | Cumulative time | Fraction |
| --- | ---: | ---: |
| Atmospheric dynamics | 4.082 s | 61.5% |
| Ocean/ice dynamics and mixing | 1.209 s | 18.2% |
| Shared surface/column exchange | 0.610 s | 9.2% |
| Mesoscale update | 0.464 s | 7.0% |

These are profiled, radiation-off timings, not an optical-radiation profile.
The atmosphere mixture gradients accounted for 0.792 s inside dynamics.
The native PyAM21 column call accounted for 0.409 s inside exchange. The
[profile](profile.txt) records nested costs, which must not be added twice.

Compile four shared mesh tensor methods (`_grad_values`, `_div_values`,
`_cell_to_edge_values`, `_edge_to_cell_values`), atmospheric `mixture_gradients`,
and the existing four Roquet thermodynamic regions (`stratification`,
`parcel_potential`, `coefficients`, caloric `_values`). The shared mesh is the
same instance used by atmosphere, ocean and ice. GM–Redi, backscatter, mixing,
phase exchange and ALE continue to execute their existing processes; these
regions accelerate tensor work that they call without compiling their entire
control flow.

State validation, bounded transport decisions, nonlinear convergence,
PyAM21 convection/turbulence control, SeaEsfRad optics/scheduling, output and
transactional restart handling remain eager. This keeps data-dependent branches
and scientific errors inspectable. No upstream sources or package classes are
globally patched. The campaign CLI runs without gradient tracking to avoid
separate grad-mode specializations between initial diagnostics and stepping;
Python build/advance APIs retain their existing gradient behavior. Full coupled
backward execution has not been qualified.

This follows PyTorch's documented
[regional compilation guidance](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.where_to_apply_compile.html)
and [full-graph capture behavior](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html).

## Correctness checks

Linux x86-64, CPU float64, Python 3.12, PyTorch 2.15.0+cpu; installed VFS wheel
outside the checkout. PyAM21 is pinned to
`e1bdfcc3b23c05adea80b35af880702bb85718eb`.

- [Focused checks](focused-tests.txt): **74 passed, one optional GSW skip**.
  This run preceded addition of the separately executed native test below.
  Tests cover CLI/YAML precedence, per-instance isolation, idempotency, manifest
  and restart identity, compiler failure visibility, mesh adjoint identities,
  all coupled stages, fresh/cached radiation and bitwise restart replay.
- [Native Inductor test](native-radiation-test.txt): **one passed**, actual C++
  code generation. Three radiating intervals use analytic native-grid optics
  with the actual radiation scheduler, conservative heating map and PyAM21
  column physics (fresh/cache/fresh). Eager/compiled state, independently
  accumulated energy/water/salt budgets and bitwise compiled replay pass.
  The ordinary tests use full-graph Dynamo capture with the eager backend;
  this separate native test is now a required installed-wheel CI step.
- Frozen portable comparisons: [142 corrected-Gibbs tensors](baseline-portable.json)
  and [93 dry-helper tensors](dry-helpers-portable.json) pass.

Real GFDL optical tables are absent here. Analytic optics validate integration
and accounting, not SeaEsfRad fluxes. GPU, Mac performance, full-coupling
backward execution and seasonal/year-long stability remain unqualified.

## Timing and finite-step differences

Measurements below use the user's 24/12-level configuration, four threads,
1200/240/1200 s ocean/atmosphere/physics steps, ten mixing and eddy substeps,
and all configured ocean/ice processes. Radiation is explicitly OFF. Setup
and first two intervals are excluded from the steady median. Diagnostics and
file output are outside timed advance calls. Alternating execution order
reduces ordering bias, but these short shared-host samples are indicative,
not a prediction for the user's Mac.

| Mesh | Eager steady median | Compiled steady median | Throughput ratio |
| --- | ---: | ---: | ---: |
| [256 cells, zero Lloyd iterations](comparison-256.json) | 3.277 s | 2.851 s | 1.149× |
| [User's saved 16,384 cells](comparison-16384.json) | 82.753 s | 65.454 s | 1.264× |

The 256-cell run overlapped the larger run's warmup and is particularly sensitive
to contention. The 16,384-cell steady samples ran after the other test/benchmark
processes completed. Only two intervals per mode enter its steady median; the
individual eager times are 87.331 / 78.175 s, compiled 65.929 / 64.979 s. This
is a 20.9% reduction in measured interval wall time, not a universal speedup.

On 16,384 cells, setup/initial diagnostics took 16.816 s eager / 26.328 s
compiled; the first interval took 69.007 / 109.443 s. The experiment completed
four intervals (4800 model seconds) and a bitwise compiled replay of the last
interval. Each interval executed ten GM–Redi substeps, eddy decay and ALE in all
16384 columns, alongside the coupled dynamics/column/ice path. The mesh SHA256
is `3437509cb1520bb4e27aea7f3e9b989c64c19673d0f49523f0043ab81a9b0d0b`.
Final numerical energy residuals are -0.215103151954 J/m² eager and
-0.215103151346 J/m² compiled, differing by 6.08e-10 J/m². Recorded water and
salt drift are zero in both. Maximum atmospheric wind difference is 1.80e-8
m/s; the largest scaled field difference is 2.75e-8 in convection memory.
This short check does not qualify the user's earlier day-one failure or a
fresh-table seasonal integration.

The 256-cell experiment completed five intervals (6000 model seconds), with
three timed intervals after warmup. Numerical energy residuals are
-0.241809846859 J/m² eager and -0.241809846524 J/m² compiled; water residual is
zero and relative salt drift is -2.14e-16 in both. Compiled restart replay is
bitwise. Largest final max-error / max(1, max-field-magnitude) is 8.64e-8 in
atmospheric water content; maximum wind difference is 2.59e-7 m/s. Compiled and
eager trajectories are not expected to remain bitwise equal: reduction and
fusion roundoff can enter state-dependent physics decisions. The independent
budget comparisons, not bitwise eager equality, assess the native implementation.

A separate [512-cell pilot](pilot-512-eager-failure.json), using the default
10-Lloyd mesh instead of the user's unrelaxed configuration, failed in the
**eager** fifth interval with `outside linear reference domain` during ocean
mixing. Both modes completed four intervals. This is retained as a failed
physical pilot, not counted as a completed performance or long-run test. No
bounds, timesteps or acceptance tolerances were changed to bypass it.

## Reproduce

See [campaign usage](../../../../../workflows/coupled/CAMPAIGN.md#optional-compilation).
The reusable driver is `validation/coupled_compile/run.py`. From an installed
wheel check tree containing workflows, validation, tests and tools:

```bash
VFS_TEST_INDUCTOR=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -I -m pytest \
  tests/test_coupled_compilation.py::test_native_inductor_radiating_intervals_and_restart -q

python -I validation/coupled_compile/run.py \
  --source /path/to/PyAM21 --config /path/to/configuration.json \
  --mesh /path/to/mesh_16384_exact.npy \
  --steps 4 --warmup 2 --output compile-comparison-16384
```

The second command explicitly benchmarks without radiation. Supply
`--radiation-data /path/to/sea_esf` to benchmark actual optics, preferably with
more intervals covering multiple radiation refreshes. `comparison.json`
records kernel policy, configuration, mesh hash, source hashes, all interval
timings, final field differences, independent budget residuals and replay
status. Its `passed` status denotes completion and bitwise replay, not a
physical climate or speedup acceptance threshold. Initial setup/first-call
measurements may use an existing Inductor disk cache and are not guaranteed
cold-cache compilation costs.
