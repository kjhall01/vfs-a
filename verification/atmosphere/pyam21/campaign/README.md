# Prognostic ice and coupled campaign verification

Parent: merged PR #88, `4f3e924e7b5f301f2f3a36233ba9ae85a47eabda`.
All changes are in the external workflows, tests and documentation. Installed
VFS numerical kernels and upstream PyAM21 sources are unchanged.

The [run guide](../../../../workflows/coupled/CAMPAIGN.md) states the numerical
model, initialization, output schedule, restart contract and limitations.

## Completed forward experiments

CPU float64, PyTorch 2.15.0+cpu, one thread, real AM2 physics, all component
dynamics and Roquet TKE mixing. **Radiation was explicitly off.** These are
short budget/implementation checks, not evidence of seasonal radiative climate.

| Experiment | Steps | Model duration | Numerical combined energy residual, J/m² | Water residual, kg/m² | Relative salt change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 cells / 24 air levels / 3 ocean layers, 200 m | 360 | 6 h | +0.00126809 | +8.73e-10 | +4.55e-15 |
| Default 256 / 24 / 12, 1000 m | 60 | 1 h | -0.0000420185 | +2.33e-10 | +2.14e-16 |

Ice starts at 265 K. At six hours, temperatures on present ice range from
265.192 to 266.499 K; at one hour in the default grid, 265.033 to 265.490 K.
Snow accumulates through actual atmospheric precipitation. The peak surface
local first-law residual is 3.16e-6 J/m² in the small run and 8.53e-6 J/m² in
the default run. The combined energy residual subtracts the separately tracked
upstream AM2 defect, and retains finite-step dynamics/splitting error.

The default configuration also passes a full-step preflight with bitwise
restart replay. Snapshot/restart binary files are intentionally local run
products; this directory retains configuration, manifest, CSV and JSON evidence.
The `smoke6h-*`, `default1h-*`, and `default-preflight-*` files are the exact
run outputs. `formatting-equivalence.json` proves identical Python ASTs before
and after the initial formatting pass at local checkpoint `d532f1c`; run manifests
identify the preformat bytes. The subsequent mesh/logging pass changes campaign
input/output orchestration; it does not rerun or relabel these archived experiments.

Reproduce these original reduced-ocean cases at commit `351dfca`. The later
ocean-process follow-up enables GM–Redi/backscatter/ALE by default and has its
own [evidence](ocean-processes/README.md); these historical numbers do not apply
to that composition. Select the saved configuration JSON (also
valid YAML) with `--config`, and adding `--stop-after-days .25
--smoke-no-radiation` to the documented campaign invocation. The default
one-hour case uses `--stop-after-days 0.041666666666666664`. Use an explicit
pinned PyAM21 checkout as described in the run guide.

## Independent checks

The new checks cover the combined energy/water budget with and without ice and
synthetic radiation; exact thermal-capacity/conduction balance; snow insulation;
melting complementarity and complete melt; phase-energy analytic derivative
against autograd; carrier-consistent cold-content transport/ridging; full native
dynamics; actual astronomical summer/winter declination; fixed solar constant
application to the upstream package; calendar/output boundaries; cached-radiation
restart replay; NetCDF field/clock roundtrip; resume; and atomic last-good
checkpointing after an injected failure. Synthetic package tests are explicitly
labelled and do not validate real radiative transfer.

Final focused suite: **70 passed, 1 skipped** (the real-data opt-in).
Frozen portable comparisons pass for all 142 corrected-Gibbs tensors and 93 dry
helper tensors. The installed-wheel suite is run from a copied external test
and workflow tree with isolated Python imports. **Full suite: 1400 passed, 3 skipped.** Results are retained
in `portable-suite.log`; fixture reports are `baseline.json` and `dry.json`.

## Remaining qualification

The advertised GFDL radiation data URL returned HTTP 503 on 2026-11-15.
No real SeaEsfRad run was possible here. Real-data preflight, a radiative short
run, and the requested full-year campaign remain to be executed locally once
the tables are available. There is no claim of year-long stability, a balanced
climate, GPU support, compiled timing, or long-trajectory differentiation.

## Saved mesh and persistent logging follow-up

`--mesh` loads the earlier `(N,3)` NumPy generator format, infers the horizontal
cell count, disables additional Lloyd relaxation and records/hashes the exact
input bytes. Tests cover geometry/order preservation, shared component mesh
ownership, unchanged-file relocation, changed-file restart identity, malformed
input rejection, and the complete mesh CLI preflight. Logging tests exercise
quiet/verbose terminal behavior, persistent startup/completion records, and a
full traceback plus last-good checkpoint after an injected failure.

The actual recovered `mesh_16384_exact.npy` was parsed and validated as
16,384×3 float64 unit vectors; SHA256
`3437509cb1520bb4e27aea7f3e9b989c64c19673d0f49523f0043ab81a9b0d0b`.
No mesh binary is committed. No 16,384-cell coupled/radiative run is claimed.
Focused follow-up: **78 passed, 1 skipped** (the real-data opt-in); result
retained in `mesh-logging-tests.log`.
