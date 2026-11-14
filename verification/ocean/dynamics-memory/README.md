# Ocean dynamics tensor-lifetime optimization

Reference: local commit `a62c199` (energy-dependent GM and decay to heat), with
unchanged installed production VFS code at the start of this pass.

## Profile and changes

The Linux line profiler samples process RSS at Python boundaries in horizontal
transport, the parent admissibility ledger, the local mechanical-work ledger,
and the temperature-coordinate integrator. It records the sizes of tensor
storage reachable from local names. Those sizes share storage and must NOT be
summed as independent allocations; the sampler can miss within-kernel peaks.
Fresh-process `ru_maxrss` measurements capture the process high-water mark.

The reference profile places the peak in the energy-constrained transport
calculation. Donor, original, target and final ledgers coexist with full edge
arrays. Five-field edge arrays are approximately 79 MB at 16384×40, and the
doubled extrema gather is approximately 157 MB. These buffers dominate the
avoidable overlap, rather than the small permanent eddy-energy scalar.

This pass changes ownership/lifetimes only:

- Release the doubled extrema gather and consumed left/right edge values once
  local bounds have been built; release donor/centered values after computing
  the same trace-change expression.
- Release consumed limiter budgets in the parent and local ledgers.
- Drop the shared stage after parent and target evaluation, and retain only
  original report fields actually needed by the final correction.
- Release superseded transport ledgers once the final ledger owns the returned
  fields. Reuse its identical connectivity and carrier references.
- Release the temporary flux report before endpoint validation in the research
  temperature integrator, and drop RK state references after their last use.

No limiter, Hamiltonian expression, numerical reduction, state mutation,
clipping, threshold, dtype or default is changed. All validation calls remain.
The trace-change calculation is moved earlier but uses exactly the same inputs
and arithmetic. Returned reports still contain every diagnostic and retain
backward graphs. `del` removes Python references, not tensors needed by autograd;
this is not a claim of smaller long-backpropagation memory.

Unlike the preceding research-only eddy changes, this pass touches production
ocean transport modules. Candidate verification therefore uses a newly built
wheel installed outside the checkout, not source imports or an editable install.
Atmospheric production code is unchanged.

## Equivalence

`dynamics_performance.py --mode check` captures three CT/SA fronts, each with
and without coasts, nonzero variable velocity, full tendency/work diagnostics,
three successive resolved steps with the eddy reservoir, both donor and FCT
transport ledgers, and derivatives of a one-step objective with respect to all
stored ocean fields. Both snapshots use the same runtime and separate installed
wheel executions. Every captured tensor is compared with exact `torch.equal`,
not a relaxed numerical tolerance. Snapshot tensors are reproducible temporary
artifacts, not replacements for frozen scientific fixtures.

The comparison contains **486 tensors**, all bitwise identical, including
one-step backward gradients in every open/coastal front case. No tolerances
were introduced or widened. `equivalence.json` records the result.

The rebuilt installed-wheel portable suite passes **1203 tests, 2 skipped**
in 202.79 s outside the checkout. Corrected-Gibbs and dry-helper comparisons
pass all 142 and 93 tensors. Original fixture bytes and tolerances are unchanged.
The wheel hashes, test log and fixture summaries are retained with this report.

## Performance method and reproduction

At 16384 cells × 40 layers, CPU float64, one thread and no compilation, time one
resolved step with the eddy reservoir and one passive tracer. The setup matches
the previous wave timing: generated mesh, 4000 m stretched column and dt=120 s.
Warm once, then take three sequential timings on the same initialized state.
Before and after run in fresh processes, without concurrent tests or profiling.
The timed operation includes dynamics and reservoir transport but no GM–Redi or
dissipation call. Whole-process RSS includes mesh, Python/PyTorch and allocator
memory; it is not a sum of live tensor sizes.

```sh
# Run each command using the corresponding installed before/after wheel and
# research module revision. Profile separately from time measurements.
python -I validation/temperature_coordinate/dynamics_performance.py \
  --mode check --output states.pt
python -I validation/temperature_coordinate/dynamics_performance.py \
  --mode profile --output profile.json
python -I validation/temperature_coordinate/dynamics_performance.py \
  --mode time --output timing.json
```

GPU/MPS, compiled memory/runtime, float32 and long backward rollouts are not
qualified by these eager CPU measurements.

## Measured result

| Unprofiled dynamics benchmark | Before | After |
| --- | ---: | ---: |
| Peak process RSS | 3.298 GB | 2.669 GB |
| Median step time | 18.442 s | 17.966 s |
| Individual times | 17.396, 18.461, 18.442 s | 17.966, 18.672, 17.269 s |

Peak memory decreases **19.1%**, or about **629 MB**, without changing the
captured numerical results. Median runtime decreases 2.6%, but samples overlap:
this establishes a memory reduction on the measured setup, not a reliable
runtime speedup. Profiling was disabled during these timing/RSS measurements.
The separate instrumented profiles are qualitative allocation-lifetime evidence;
profiling itself perturbs retention and process RSS. The benchmark executes
resolved dynamics with reservoir transport, not the subsequent GM–Redi/decay
stages or a full long simulation. Source hashes and raw results are retained
in `before-timing.json` and `after-timing.json`.
