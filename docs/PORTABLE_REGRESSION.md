# Portable fixture regression

Portable mode implements the reviewed budgets in the
[911aada investigation](CI_911AADA_INVESTIGATION.md). It checks CPU float64
forward regression across the measured environments, not scientific accuracy,
GPU behavior, long integrations or full backward rollouts. Numerical package
code and both fixtures are unchanged. Exact comparison is still the default
in pytest and both tools; select `--fixture-mode=portable` or `--mode portable`
explicitly. Exact failures remain failures, with no automatic fallback.

## Mapping and budgets

[`fixture_quantities.json`](../tools/fixture_quantities.json) explicitly maps
all 372 leaves (335 tensors) to a quantity and component count. Unknown paths,
containers, categories or fixtures fail; there is no name-based fallback.
`components` identifies the last axis: three water species, four ledger
species including dry air, or two tracers. Each component gets its own scale.
Zero means one field scale. Review mapping changes alongside any new fixture.

For each field/component, require `max(abs(actual-reference)) <= tau*S`,
where `S = max(abs(reference))`. This is a field-scaled absolute bound, not
pointwise relative error. There is no dimensionless floor: zero reference
scale requires exact zero. Distinct fields, species and units never share a
scale. `eps` is float64 machine epsilon; `128 eps = 2.842170943040401e-14`.

| Category | tau | Scope / rationale |
| --- | ---: | --- |
| state | 5e-13 | Ordinary states/geopotential; observed maximum 1.291e-13 |
| dry_helper | 5e-13 | Dry helpers/norms/forces; observed 7.311e-14 |
| tendency | 2e-12 | Ordinary tendencies, including cancellation; observed 4.272e-13 |
| gradient | 2e-13 | Analytic gradients by field/species; observed 1.683e-14 |
| inventory | 128 eps | Mass/species/tracer reductions |
| ledger | 128 eps | Amounts, integrated fluxes and beta; observed 1.766e-15 |
| energy | 2e-13 | Extensive totals/components; observed 1.233e-14 |
| jw_thermal | 1e-11 | JW initial/final mtheta only; differenced initialization |
| jw_tendency | 2e-10 | JW momentum/thermal RHS only; propagated initialization error |
| limiter_force | 1e-9 | Small force correction only, not full momentum tendency |
| limiter_heat | 2e-11 | Heat/entropy correction only; near-cancelling Gibbs work |
| cfl | 128 eps | Floating CFL metadata |
| energy_rate | 128 eps | Scale is reference sum of absolute gradient-times-RHS work |
| rotation_work | 128 eps | Scale is reference work_abs_scale in each formulation's units |
| rotation_relative | 128 eps | Absolute bound on an already normalized residual (S=1) |

The investigation records observed maxima, absolute allowances and justification
for the narrowly scoped JW/limiter budgets. These are provisional regression
budgets supported by the sampled CPUs/backends, not universal error bounds.
Integer/None metadata uses the separate `exact` category.

## Independent checks

The policy rejects mismatched structures, shapes and dtypes, non-CPU tensors,
and any nonfinite float, including matching NaNs. It also requires:

- Strictly positive pressure, layer mass, dry fraction and vapor; nonnegative
  condensates, ledger amounts and released heat. No floor or tolerance permits
  negative water. Existing kernel roundoff-policy tests remain separate.
- Beta in [0,1], exactly the fixture's active mask, limited-face count and stage
  counts. Euler water must equal recovery from the amount ledger **exactly**.
- Recomputed mass and each species conserved independently within 128 eps of
  that species' initial inventory. Absent species have zero allowance. Check
  reported inventories, ledger species totals and each ledger carrier amount.
- Dry mass conservation, positive dry thermodynamic fields and per-tracer
  area-weighted tendency cancellation, scaled by absolute tracer work.
- Actual instantaneous energy closure within 128 eps of actual absolute work,
  in addition to reference-scaled regression. Captured gradients/RHS are checked
  first, and their contraction must equal the reported rate exactly. Dry
  rotation has independent closure and exact normalized-work identity checks.

Finite-step energy drift is reported separately; SSPRK3 is not asserted to
conserve energy exactly. The existing 99 helper AST checks remain in pytest.
Adversarial tests perturb every mapped tensor and each comparison category,
species scaling, structures, nonfinite values and each invariant. Perturbations
below numerical budgets also demonstrate independent mask, ledger, positivity,
heat and work-identity rejection.

## CI and reproduction

Follow [CONTRIBUTING.md](../CONTRIBUTING.md) to install a wheel and copy tests,
tools and examples outside the checkout. Required CI runs the complete portable
pytest suite (including both fixture tests), both portable standalone tools and
the MoistAtmosphere example. Separate exact steps are informational and preserve
nonzero exit codes, failure JSON/JUnit and explicit step outcomes in the summary.
The job cannot pass a failed portable check. Branch-protection settings are not
changed by this workflow.

Reports include per-category maximum budget usage and invariant metrics.
`tools/record_environment.py` records CPU, backend build/dispatch details,
thread controls, dependencies and installed import path. Artifacts retain both
modes separately. For differences across **all** tensors, run
`tools/audit_fixture_differences.py --report audit.json`; the diagnostic
collector's completion alone is not a regression pass. See
[current validation](PORTABLE_REGRESSION_RESULTS.md) for actual runs.
