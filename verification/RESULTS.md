# Initial infrastructure results — 30 October 2026

Frozen runtime: `8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648` (merged PR #12).
No `vfs/` sources, numerical fixtures or comparison tolerances changed.

Local validation used a wheel built from `git archive` of that exact commit,
installed outside the checkout. Wheel SHA256:
`7194b139d0ef52dd9759718974c981a3be99ea235efce279a58ce595c7ee32b5`.
The new provenance check first rejected the previously installed wheel, which
still contained pre-merge source text. Rebuilding the exact baseline resolved
the mismatch; the check was not weakened.

Environment: Apple M3, macOS 14.6.1 arm64, Python 3.12.13, Torch 2.15.0,
CPU float64, one Torch thread, Accelerate backend. Initial sandbox runs retained
the denied CPU query in their manifests; an authorized host run recorded the
CPU brand and reproduced the metrics exactly. Full numerical library/build
information is in each manifest.

* Installed-wheel portable suite: **539 passed, 1 CUDA skip**; the pre-existing
  bounded donor-subcycle tensor-to-scalar warning remains visible.
* Both standalone portable comparisons passed (142 moist/control and 93 dry
  tensors). These are migration checks, not scientific acceptance criteria.
* All six documented explicit, ARK4 and Held–Suarez example commands completed.
* Three new infrastructure tests cover repeatability/nonmutation, nonfinite
  measurement failure, and missing/changed/extra runtime source rejection.
* Exact fixture mode was not rerun in this infrastructure increment. Previous
  platform-specific outcomes remain in the cleanup report; portable success is
  not an exact pass.

`C1-GEO-001`, Fibonacci grids, zero Lloyd iterations, radius 6371220 m,
seed 20261030, produced:

| Cells | Relative area error | Normalized grad/div adjoint residual | sin(latitude) gradient RMS error (m⁻¹) |
|---:|---:|---:|---:|
| 16 | 2.220446e-16 | 2.020205e-17 | 3.403769e-9 |
| 64 | 2.220446e-16 | 0 | 8.251462e-10 |
| 256 | 2.220446e-16 | 7.197518e-18 | 2.158184e-10 |

Constant gradients were exactly zero in these runs. Signed integrated divergence
and normalized residuals, minima and maximum errors are also saved. These are
preliminary measurements for a single manufactured scalar; they do not establish
general operator accuracy or qualify a dycore. No timestep is advanced by this
experiment. The local run directory is `/tmp/vfs-c1-host-environment` (ephemeral);
reproduction commands are in the [entry document](README.md).

CI now runs this measurement against the installed baseline and uploads its
manifest, metrics, mesh/input tensors and executed driver sources in `cpu-checks`.
Hosted execution has not yet been claimed here. C2–C9 remain planned campaign
work with existing tests mapped as starting evidence in the tracker. No numerical
defect was diagnosed or fixed by this increment.
