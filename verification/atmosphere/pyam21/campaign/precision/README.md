# Explicit precision and MPS execution preparation

The campaign and both comparison drivers accept CPU/CUDA/MPS plus float32 or
float64. CPU/float64 remains the default. MPS requires explicit float32;
unsupported combinations and missing hardware fail without fallback. Geometry
preprocessing stays CPU/float64, then model buffers, physics maps and states
are initialized at the requested device/precision. The setup helper restores
the caller's default dtype. Integer/bool state survives explicit floating-point
tree conversion, including gradients.

## Numerical policy

Initial FP32 trials exposed two fixed double-precision checks: the lagged TKE
budget's 1e-9 relative threshold and reciprocal drag's 1e-12 Newton residual.
The campaign configures the former as `max(1e-9,32*eps)`; the research drag
default is `max(1e-12,32*eps)`. Explicit drag tolerances remain literal. FP64
thresholds and stored expressions are unchanged. The FP32 allowance is
3.814697265625e-6; this is an arithmetic roundoff policy, not a physical
accuracy target. An injected 1% solve defect still fails the mixing ledger.
Independent reciprocal momentum and KE-to-TKE tests exercise the FP32 drag.
Other guards, all physical processes and the fixed timestep schedule remain.

## Executed CPU evidence

Tests run outside the source checkout against the installed VFS wheel on
PyTorch 2.15.0+cpu. Native CPU Inductor is exercised in float32 and float64.
The focused suite includes all-component analytic-radiation fresh/cache/fresh,
dtype/device ownership, real disk checkpoint replay, CLI configuration/logging,
NetCDF, integer/bool/gradient preservation and failure-path tests. GPU tests
are defined separately and skip on this host. Final focused suite: **156 passed,
6 GPU skips**; see `focused-tests.log`.
Frozen portable comparisons retain **142 corrected-Gibbs / 93 dry tensors**.

The reproducible [`capture_cpu.py`](../cuda/capture_cpu.py) now also accepts
`--dtype float32`. The paired capture uses 32 cells, 24 atmosphere and 12 ocean
levels, physics/ocean 1200 s, atmosphere 240 s, radiation 3600 s, ten mixing/eddy
substeps and every configured process. Four intervals cover fresh/cache/cache/
fresh analytic optics through the actual radiation scheduler.

CPU float64 remains **bitwise identical** to merged base `969400e` across the
initial state and all four intervals: 240 tensors and 127 scalar/metadata
leaves. Float32 completes, but is **not scientifically equivalent**:

| Measurement after 4800 s | CPU float64 | CPU float32 |
| --- | ---: | ---: |
| Native numerical energy residual (J/m²) | -0.128371 | 2781.843303 |
| Native water change (kg/m²) | -1.16e-10 | -0.0625 |
| Native relative salt change | 0 | 1.1489e-7 |

These inventories are reduced in each mode's native precision and include
diagnostic rounding. Direct state differences, evaluated after promotion to
CPU float64, include maximum ocean temperature difference **5.3635e-5 K**,
maximum specific eddy-energy difference **0.0753064 J/kg**, and eddy-content
maximum absolute difference divided by reference maximum **0.1671293**.
The last two metrics must not be confused: layer thickness enters eddy content.
See every tensor's difference and input settings in `cpu-precision.json`.

## Limits

MPS/CUDA hardware and real optical tables are unavailable here. This is executed
CPU precision evidence plus GPU wiring/test definitions, not an executed GPU
qualification or a speedup claim. Float32 is experimental; the eddy-energy
disagreement makes the short completion insufficient evidence for a long
campaign. Use the [GPU guide](../../../../../workflows/coupled/GPU.md) and
real-table comparison with `--reference-dtype float64` on the intended hardware
and mesh. No old-checkpoint device/precision migration is provided.
