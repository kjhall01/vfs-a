# Shared thermodynamic work within ocean stages

One transient `_StageEvaluation` holds the physical reconstruction, Gibbs face
quantities, dynamic/full energy gradients and donor transport ledger for one
unchanged RHS input. The base transport and sharp-front correction reuse these
values. No workspace is stored on a state/model or carried to a new SSPRK stage.

The physical equations, summation expressions, JMD95 quadratures, inversion
iterations/residual checks, limiters, boundary operators and mixing choices are
unchanged. Public `EntropyEnergy.gradients` still independently reconstructs its
input. Endpoint validation remains independent. The private coordinate-transform
helper preserves the original gradient expressions and their tensor graphs.

## Reproduce

Use the candidate installed package (an editable install with dependencies already
managed by Conda is also sufficient). Export the exact pre-change modules:

```bash
mkdir -p /tmp/vfs-stage-parent
git archive d08f03be0a649a7ede1982cc297a859a8cfc1c7a | tar -x -C /tmp/vfs-stage-parent
python -I validation/performance/stage_reuse.py --baseline /tmp/vfs-stage-parent --cells 128 --layers 16 --output stage128.json
python -I validation/performance/stage_reuse.py --baseline /tmp/vfs-stage-parent --cells 1024 --layers 64 --cases ocean --steps 1 --repeats 3 --profile --output stage1024.json
python -I validation/performance/stage_reuse.py --baseline /tmp/vfs-stage-parent --cells 128 --layers 64 --cases full full_ale --dt 600 --substeps 8 --steps 1 --repeats 2 --output stage600.json
```

The harness loads the parent's two transport modules under separate names and
switches only the transport implementation, using identical fixed initial states
and installed dependencies. The remaining implementation is shared, including the
unchanged public energy-gradient expressions extracted into a helper. Parent
source hashes are recorded. Every state and tensor report field is compared
bitwise before timing. Each method is warmed, repeat order alternates, and endpoint
diagnostics and call-count/cProfile instrumentation are outside timed intervals.
Full cases use the opt-in fast research mixing closure. These are single-thread,
CPU float64 forward tests. No other test job overlaps the benchmark processes.

## Mechanism and limits

At 128×16 over two ocean steps, temperature reconstructions fall from 50 to 26,
SurfaceGibbs evaluations from 792 to 408, and JMD95 gradients from 18 to 6.
This counts the complete dynamics/TKE-transport call, not just the RHS.

This pass removes duplicate evaluation without reducing required checks. It does
not add warm inversions to dynamics, compile kernels, change the timestep, change
mixing/remap frequency or qualify GPU/long-rollout behavior. Full-step improvements
are smaller because boundary, mixing and ALE work are unchanged. Wall timings
vary; raw interleaved samples are retained instead of asserting a universal gain.

## Atmosphere control

Between main after #56 (`8e8147b`) and the benchmark parent (`d08f03b`), the only
changed package files are `vfs/ocean/stratified/equations.py` and `tke.py`. The
atmosphere and shared geometry/stepping sources are identical. This optimization
also changes only ocean sources. Matched dry/moist controls therefore measure
execution variability of identical atmospheric code; they do not establish
whether older atmospheric configurations or different dependency builds regressed.

## Measured throughput

| Workload | Grid | dt / mixing substeps | Parent wall s/step | Candidate wall s/step | Speedup |
|---|---|---|---:|---:|---:|
| Dynamics + horizontal TKE transport | 128×16 | 60 s / none | 0.895 | 0.470 | 1.90× |
| Full forced ocean | 128×16 | 60 s / 4 | 3.255 | 2.794 | 1.17× |
| Full forced ocean + ALE | 128×16 | 60 s / 4 | 3.461 | 2.965 | 1.17× |
| Dynamics + horizontal TKE transport | 1024×64 | 60 s / none | 27.221 | 14.199 | 1.92× |
| Full forced ocean | 128×64 | 600 s / 8 | 12.104 | 9.818 | 1.23× |

The large dynamics call compares 64 state/report tensors exactly, reconstructs
physical temperature 13 rather than 25 times, and computes three rather than
nine JMD95 gradients. Its candidate cProfile call spends 8.86 s cumulatively in
temperature inversion; this is nested within validation and SurfaceGibbs work,
so these cumulative categories must NOT be summed. Warm/residual-controlled
inversion and reuse across validated operator boundaries are future candidates.
No full-ocean speedup at 1024×64 or on the user's Mac is inferred from this table.

At 128×64, dt600 with eight mixing substeps and ALE, median cost is
13.830 → 12.507 s/step (1.11×). All 262 compared tensors across the two dt600
cases are bitwise equal. These are one-step timing repeats, not long trajectories.

Identical-code atmosphere controls at 1024×24 give dry medians 0.173/0.157 s
and moist 0.895/0.875 s for parent/candidate timing labels. They compare 3/4
state tensors bitwise across three evolving steps. Since no atmospheric source
changed, the apparent 10%/2% differences measure timing variability, not an
atmospheric optimization. Small full-ocean gains should likewise be treated as
provisional; the large dynamics samples (26.26–28.09 versus 13.74–14.75 s) are
well separated. This is not a matched historical JW-A performance campaign.

## Regression validation

The installed candidate wheel matches the two edited runtime files byte for byte.
The complete portable suite, run outside the checkout, passes **1061 tests**, with
one CUDA skip and one existing warning (246.69 s). This includes frozen exact
sharp-front, near-fresh, upper-salinity, tilted-layer and coastal step/remap
comparisons. Five new checks cover stage/public-gradient equality, stage freshness,
retained backward derivatives and invalid salt/passive content rejection.
Corrected-Gibbs (142 tensor comparisons) and dry-helper (93) portable checks pass.
Fixtures and tolerances were not modified. No full-model backward qualification
is inferred from the focused gradient check. Final CLI case-name validation adds
no numerical change. Raw suite log and compact benchmark/profile results are in
`stage-reuse/`; every uploaded file is below 128 KiB.
