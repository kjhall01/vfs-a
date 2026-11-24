# Positive vapor at the PyAM21 boundary

The conservative VFS adapter now handles an exact-zero vapor endpoint by
evaporating/sublimating a tiny amount of local cloud condensate. No PyAM21
checkout or package change is required. The nominal vapor target is
`1e-12 kg/kg`, declared in `workflows/pyam21/vapor.py`. Positive vapor endpoints
are unchanged, and negative or nonfinite outputs still reject. The operation
requires enough condensate in the same layer and preserves total water to
floating-point roundoff.

The correction cools the column in AM2's thermal/latent convention. VFS then
uses its existing native energy solve with the **original upstream energy
target**, so the phase transfer does not add external energy or erase an AM2
energy defect. Both endpoint and tendency reports are updated. The sparse
`PhysicsReport.vapor_correction` ledger retains the original affected rows.
See [the energy derivation](../pyam21/ENERGY.md#exact-zero-vapor-at-the-physics-boundary).

## Continue the failed run

Fetch the correction branch and reinstall VFS from the repository:

```bash
git fetch origin
git switch fix/pyam21-positive-vapor
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

The checked restart bridge recognizes the exact numerical source inventory of
main `dcda5cdd1832242557881e7d7438c820464db4fa`
and this correction's pinned target inventory. The old inventory is reproduced
from that public Git revision. All configuration, runtime,
mesh and radiation identity checks remain active. The transition writes
`compatible_resume_audit.json` with `numerical_change: funded_zero_vapor_v1`.
It does not convert arbitrary old branches or change checkpoint fields.

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
