# Ocean temperature-coordinate validation: same-stage deduplication

## Observed fully coupled reference

The 4096-cell/40-ocean-level/32-atmosphere-level Mac CPU FP64 profile,
four threads, compiled Roquet kernels, 24 2400-s intervals:

- Total: 233.795 s, 10.265 simulated days per hour
- `rhs_temperature_entropy_fields`: 2376 calls, 8.486 s exclusive
- `rhs_caloric_reference_evaluate`: 7704 calls, 7.513 s exclusive
- `rhs_temperature_ocean_check`: 2184 calls, 1.249 s exclusive
- `rhs_ridge_validation`: 1632 calls, 0.404 s exclusive

These costs are **exclusive**; they are not all direct temperature conversion,
and reduction of one nested check can move time between regions.

## Proof of redundant checks

On each *unchanged* Runge–Kutta ocean–ice–snow state:

1. `SnowOceanIce.tendency` previously checked snow plus delegated to a
   complete `OceanIce.check` — `RoquetOcean.check`, caloric evaluation,
   and ice admissibility.
2. `OceanIce.tendency` immediately performed its own full check.
3. The nested `CompressionRidging.tendency` performed the same complete
   check yet again before its sea-ice constitutive work.

The ocean/ice state is passed without mutation between these checks.
Consecutive SSPRK states have *different tensors* and must each be checked.

This experimental branch checks the snow store in step (1), performs the
complete coupled ocean/ice check in step (2), and lets the same-state nested
ridging RHS in step (3) reuse that validation. Public standalone snow,
ocean–ice and ridging checks are unchanged.

There is **no model-wide cache**, tensor-identity lookup, weakening of
positive salt/tracer admissibility, frozen-water restrictions, EOS range checks
or post-update step checks. The order in which error messages appear if both
snow and ocean states are invalid may differ; both remain rejected.

## Validation

`tests/test_ocean_rhs_profile.py` compares the entire coupled atmosphere,
ocean, ice, snow, and source report bitwise with the legacy triple-validation
call path, for both SSPRK3 and SSPRK43. It counts
`TemperatureOcean.fields` calls, requiring exactly two fewer full field
evaluations per ocean–ice RHS stage. A separate test confirms that standalone
`CompressionRidging.tendency` still performs its complete check.

CI: `Ocean temperature validation deduplication`; full installed-wheel suite
runs in parallel. Do not merge before CI verifies the final branch head.

## Run an A/B full coupled CPU benchmark

Use the same actual model configuration, same compiled flags and mesh, and
alternate baseline vs this branch to control CPU frequency/thermal variation.
The source-line pass is optional; for performance focus on nested method
timing, using `--ocean-line-intervals 0` to minimize perturbation.

```bash
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
  --output "runs/vfs-temperature-validation-01"
```

Compare full runtime, `rhs_temperature_entropy_fields`,
`rhs_caloric_reference_evaluate`, `rhs_temperature_ocean_check` and
`rhs_ridge_validation` at matched call counts and config. Don't ascribe
unrelated ocean mixing or radiation fluctuations to this change.

## Further research

More aggressive same-state elimination in phase, drag and boundary/exchange
routines requires explicit validation ownership across their state transitions.
Avoid persistent caches and do not skip checks across Runge–Kutta stages,
phase changes, freshwater injection or ALE remaps.
