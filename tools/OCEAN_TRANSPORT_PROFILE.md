# Ocean transport deep profiler — real fully coupled VFS

This is a **read-only timing extension** of the real 4096-cell, 32-level atmospheric / 40-level ocean PyAM21–Roquet–ice–snow campaign. It works on the merged PR #104 optimization baseline in `perf/ocean-ice-rhs-breakdown-20261123`, without modifying any numerical source, energy ledger, scheme, or timestep. It does not disable or bypass model checks.

## Run on Mac CPU float64

```bash
# Run from the repository root.
python -m pip install -e .

python -u -I tools/profile_ocean_rhs.py \
  --config workflows/coupled/aquaplanet_multirate_4096_40.yaml \
  --mesh "input-data/mesh_4096_exact_3000.npy" \
  --source "../PyAM21" \
  --radiation-data "input-data/sea_esf" \
  --set physics_dt=2400 \
  --set atmosphere_dt=800 \
  --set ocean_dt=1200 \
  --set ocean_dynamics_integrator=ssprk43 \
  --set mixing_dt=1200 \
  --set mesoscale_dt=1200 \
  --set radiation_dt=4800 \
  --set ale_dt=4800 \
  --threads 4 \
  --warmup-intervals 4 \
  --measure-intervals 24 \
  --ocean-line-intervals 1 \
  --output "runs/vfs-ocean-deep-profile-01"
```

Use a fresh/empty output directory. Switch to an existing local branch with
`git switch tools/ocean-transport-deep-profile-20261123` and `git pull --ff-only`
if `--track` says the branch exists. Preserve any uncommitted files rather
than force-checkout. Real radiation data and the exact saved mesh are required
for comparable performance measurements.

**What is measured:** Four warmup physics intervals are excluded, followed
by 24 timed full 2400-s coupled intervals. Afterward, the model runs **one
additional ocean-source-line-instrumented coupled interval**. This additional
pass deliberately carries large tracing overhead. It is *not* included in
headline SYPD, and it is *not* a comparable timing benchmark by itself.

The existing `summary.txt` and `profile.json` gain nested stage timings for:

- `rhs_parent_admissibility_limiter`: Gibbs parent conservative FCT limiter;
- `rhs_Gibbs_face_traces`: parent Gibbs face reconstruction;
- `rhs_Gibbs_pressure_rotation`: compatible pressure/Coriolis momentum;
- `rhs_entropy_domain_validation`: ocean entropy/physical-domain checks;
- `rhs_parent_flux_divergence`: parent transport graph reductions;
- `rhs_local_flux_divergence`: local work correction graph reductions;
- `rhs_donor_transport_advance` if a persistent prepared transport object
  exists on the real core.

The line pass writes:

- `ocean_transport_lines.txt`: 90 highest cumulative-cost source statements
  in the two transport evaluators and their parent/local limiter functions;
- `ocean_transport_lines.json`: full source line, location, executions and
  measured wall-time attribution.

**Line timings are inclusive**: if one source line invokes a child Python
function, its elapsed time includes that child. Some statements span multiple
physical lines. The per-line data is for locating work (graph divergence,
work-cap arithmetic, heating source, correction flux, tensor allocations),
not for summing to the exclusive stage timing. The line pass uses Python
`sys.settrace` and incurs significant instrumentation overhead. Interpret
rankings together with normal exclusive stage timings.

With `--ocean-line-intervals 0` (default), no line tracing is performed and
the extra output files are omitted. For an A/B experiment keep the base
profiler flags, mesh, radiation, dtype, threads and configurations identical.

The numerical-source fingerprint covers `vfs/`, `workflows/`, and
`validation/`, but not `tools/`. This profiler changes only `tools/`,
tests, and CI. It can run on an exact configuration+source+runtime-matched
checkpoint via `--resume`, but will reject a checkpoint with a different
campaign fingerprint. Do not use it to resume incompatible branches.

## Expected findings and next stage

The existing profile attributes 21–22 s exclusive to the composed
`rhs_conservative_transport_total`, 16–17 s to its parent Gibbs evaluator,
and ~6.5 s to local target fields across 192 RHS calls.

This pass should distinguish whether the next optimization should address:
1. conservative graph divergence / `index_add` reductions,
2. parent Gibbs FCT positivity and budget arithmetic,
3. full-H work/heat source assembly,
4. physical entropy-to-temperature inversion and Gibbs evaluation, or
5. unnecessary per-stage intermediates.

Do not remove positivity checks or bypass work budgets to improve timing.
Any numerical optimization should first prove same-state CPU float64 forward
and budget parity and verify differentiability.
