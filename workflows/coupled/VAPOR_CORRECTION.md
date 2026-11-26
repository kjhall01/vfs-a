# Positive vapor at the PyAM21 boundary

A nonpositive physics endpoint first draws on liquid and ice in its layer.
When that is insufficient, the adapter draws vapor from the nearest funded
layers in the same column, weighting transfers by layer pressure thickness.
Donors retain positive vapor. A finite column with insufficient available
water still rejects. Temperature adjustments account for the AM2 latent
energy of the transfer. The native endpoint solve uses the original upstream
energy target. Precipitation, surface exchange and radiation fluxes are retained.

The sparse report records donors, recipients, actual layer transfers and column
water residual. The existing campaign counters measure recipient events and
the global area-weighted vapor credited. Positive columns require no repair.

For the 1,200 s day-12 checkpoint from the pre-cleanup Earth campaign, use the
original configuration, mesh, geography, radiation tables, PyAM2.1 and runtime.
Supply a new output directory with `--resume /path/to/last_good_restart.nc`
and `--resume-compatible-manifest /path/to/manifest.json`. The verifier pins
the source inventories from both sides of PR #122 and the target of this change.

## Native atmospheric interfaces

The column physics returns layer means. A positive layer endpoint can still
leave a drier neighboring Charney–Phillips interface below the bounded
transport vapor floor after mapping. The adapter checks the mapped vapor
content against the new interface pressure mass, using the active transport
operator's floor (normally `1e-14` kg/kg).

An affected interface draws liquid, then ice, at the same node. If that is
insufficient, the nearest same-column interfaces can donate vapor above
their own floor. Donor debits use representable amounts. The original native
layer energy targets still apply to the new mixture. A column with insufficient
available water raises an error with its cell and interface. The normal path
returns the mapped water unchanged.

Both full-level PyAM21 vapor repairs and native interface repairs emit
`WARNING` records. A campaign writes them to `run.log` and also prints them
with `--verbose`. Native warnings include the model day, context, recipient
count, first location, minimum incoming vapor fraction and transport floor.
`interface_vapor_correction_events` counts recipient interfaces and
`interface_vapor_correction_kg_m2` accumulates global area-mean vapor
credited to them. The older `vapor_correction_*` fields continue to describe
the upstream physics-layer repairs. A parcel can participate at both stages.

To resume the day-28 pilot directly from its original checkpoint, keep the
source of PyAM21, configuration, runtime, mesh, geography and radiation inputs
from that run. Add these flags to the same campaign command, using a new
segment directory.

```sh
--resume OLD_RUN/last_good_restart.nc \
--resume-compatible-manifest OLD_RUN/manifest.json \
--output NEW_RUN
```

The source bridge recognizes the coupled limiters at `f525256`, the Redi
revision at `a894d0e` and main at `59b11db`. After unpacking, the campaign
checks the native interfaces before the first dynamics step. A restart repair
uses zero native energy change in each layer and enters the warning log and
cumulative ledger at the checkpoint time.

## Historical exact-zero-only transitions

For a checkpoint from the merged ocean performance stack, update VFS on the campaign machine.

```bash
git fetch origin
git switch main
git pull --ff-only origin main
python -m pip install -e .
```

Use your original `workflows/coupled/campaign.py` command with the same config,
mesh, PyAM21 checkout, optical tables, precision, threads and compilation setting.
Replace its output directory with a new one and add:

```bash
--resume /path/to/previous-run/last_good_restart.pt \
--resume-compatible-manifest /path/to/previous-run/manifest.json \
--output /path/to/vapor-fix-continuation
```

The restart bridge verifies the source inventory at both ends of each
transition. Choose the row matching the checkout that wrote your checkpoint.

| Checkpoint source | Checkout for continuation |
| --- | --- |
| Ocean performance stack at `5329122` (#113) or `3b090ed` (same sources on main) | Main at the integration of #109 and #113 |
| Earlier main at `dcda5cd` | Original vapor fix at `e9a5702` (use `git switch --detach e9a5702` before installing) |

Each pair changes only the vapor correction and its reporting or verification.
Crossing between the two ocean revisions fails the source check. Configuration,
runtime, mesh and radiation identities must also match. A successful transition
writes `compatible_resume_audit.json` with `numerical_change: funded_zero_vapor_v1`.

New campaign diagnostics accumulate:

| Column | Meaning |
| --- | --- |
| `vapor_correction_events` | Number of corrected cell/layer endpoints; repeated corrections count again |
| `vapor_correction_kg_m2` | Cumulative vapor transferred from condensate, averaged over the global area |
| `vapor_correction_proxy_residual_J_m2` | Peak absolute AM2 cooling roundoff in an affected layer; not additional boundary energy |

Existing checkpoints start these new counters at zero at the transition.
Subsequent ordinary checkpoints retain them. The usual native energy residual
continues to measure the energy-transfer solve.

## Verified scope

Linux CPU, Python 3.12, PyTorch `2.14.1+cpu`: 116 focused tests pass against an
installed VFS wheel outside the source checkout. These exercise liquid, ice
and mixed condensate transfers, positive-vapor no-op, unfunded and invalid
endpoints, native energy accounting, preservation of upstream energy defects,
and ordinary and audited campaign restarts. The 142-tensor corrected-Gibbs
and 93-tensor dry-helper portable fixture checks pass without changing their
fixtures or tolerances. CPU float32 has isolated phase-correction tests.

With VFS and the pinned PyAM21 installed, copy `tests`, `tools`, `workflows`
and `validation` outside the checkout and run:

```bash
python -I -m pytest tests/test_pyam21_workflow.py tests/test_endpoint_diagnostic.py \
  tests/test_coupled_restart_compat.py tests/test_coupled_campaign.py \
  tests/test_coupled_energy_trace.py --fixture-mode=portable -q
python -I tools/check_baseline.py --reference-version corrected-gibbs-v1 --mode portable
python -I tools/check_dry_helpers.py --mode portable
```

Private checkpoints and their derived measurements are excluded from this
repository. No long-run accuracy, microphysics timestep convergence, fresh
optical calls, GPU/MPS execution or backward behavior at the correction
boundary is qualified by the synthetic regression suite.
