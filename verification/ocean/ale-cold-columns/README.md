# Stored-temperature ALE acceptance

The 16384-cell coupled campaign stopped after second 67200 with
`adaptive thermal projection decreases column entropy`. The temperature-state
remapper used its temporary solver temperature in its iteration entropy check,
but reconstructed temperature from the prognostic `h*T` for its final check.
Multiplication/division roundoff can put these on opposite sides of the same
entropy tolerance.

The iteration now evaluates entropy and thermal gradients at the actual stored
candidate temperature. A failed candidate remains in the solve. Neither the
256-epsilon energy tolerance nor the 512-epsilon entropy tolerance changes;
the 24-iteration cap, remap, projection objective and timestep settings remain.
Failures now identify the worst column, residual/tolerance and iteration count.
This changes the research temperature remapper only, not packaged VFS kernels.

## Reproducible counterexample

The deterministic 16-column input in
`tests/ocean/test_temperature_ale_roundoff.py` is weakly stratified cold water,
with small layer displacement and finite salinity contrast. It does not use
the user's checkpoint. On CPU float64:

| Quantity | Previous implementation | Corrected implementation |
|---|---:|---:|
| Iterations | 3, then final rejection | 4, accepted |
| Column 11 temporary entropy change / tolerance | -0.999716608 | — |
| Column 11 stored entropy change / tolerance | -1.000013766 | — |
| Minimum accepted entropy change / tolerance | — | -0.998899425 |
| Maximum absolute energy residual / tolerance | 0.002990128 | 0.003197348 |

Thus the old loop had declared convergence for a state its own final check
rejected. The corrected loop performs the extra iteration. The unchanged
nonlinear constraints are checked on returned state tensors; salt, depth,
passive tracer, TKE and EKE column contents remain conservative. A deliberately
stalled projection still rejects with detailed diagnostics. Tests cover one
and four threads. See [before](before.json) and [after](after.json).

With an installed wheel, copy `tests`, `tools`, `validation`, `workflows` and
this evidence directory outside the checkout (without `vfs/`), then run:

```bash
python -I -m pytest tests/ocean/test_temperature_ale_roundoff.py \
  tests/ocean/test_roquet_configuration.py tests/ocean/test_linear_configuration.py \
  tests/ocean/test_temperature_coordinate.py tests/ocean/test_temperature_composition.py \
  tests/ocean/test_adaptive_thermal_ale.py tests/test_coupled_campaign.py \
  --fixture-mode portable -q
python -I verification/ocean/ale-cold-columns/compare.py --output after.json
```

The focused suite passes **99 tests, with one optional GSW skip**. The frozen
portable comparisons pass **142 corrected-Gibbs / 93 dry-helper tensors**; see
[baseline](baseline.json) and [dry](dry.json). To obtain the old failure, pass
`--research-root` pointing at the workflow/research tree from `210b92f`.

## User checkpoint diagnostic replay

The user supplied the last committed checkpoint and configuration. The saved
generator mesh was recovered and its SHA256 matched the reported mesh. An
explicit diagnostic reconstruction advanced **67200 → 68400 seconds** on Linux
using the original radiation driver and original native-grid heating. The
cached radiation alarms were both 1201 s, so this 1200 s interval required **no
fresh optical-table call**; an attempted package call was guarded to fail.
This is not a production restart migration or a new real-table radiation run.

Both old and corrected remappers complete this particular replay on Linux.
The old pre-remap candidate was captured; the corrected replay's pre-remap
ocean/TKE/EKE tensors match it bitwise. The corrected full interval executes
10 GM–Redi substeps, eddy decay, ocean/ice/atmosphere physics and one ALE step
changing all 16384 columns. Maximum ALE column energy residual is
**1.31845624e-5 J/m²**. See [replay metrics](checkpoint-replay.json).

The precise Mac rejection is therefore not reproduced bitwise here. The
deterministic counterexample proves the acceptance inconsistency; the saved
state replay confirms the change works on the user's mesh/configuration for
one interval. GPU behavior, a fresh-table seasonal rerun, longer integration,
and the full year remain untested. No smaller timestep is inferred necessary
from this acceptance bug. Keep the original checkpoint, but use a fresh output
directory after updating: source fingerprints intentionally reject old runs.
