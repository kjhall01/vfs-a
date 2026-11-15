# Stationary Hamiltonian ice columns — 2026-11-14

Implemented on `feature/hamiltonian-ice-columns`, based on `ec5ccbb`.
No production ocean/atmosphere kernels or defaults change. Research modules
are explicit and the API rejects nonzero ocean velocities.

## What was recovered

The inspected archive is `VFS-main.zip`, dated 2026-10-11, SHA256
`ced96143a44bf6eaac6ae86bff9ae9d80d06d45e5d4acf321befd1d0c20f1d3c`.
Its `vfs/ice/{state,thermo,transport,evp,albedo}.py` contains useful scaffolding.
The present implementation rebuilds the thermodynamic piece against the current
Roquet full Hamiltonian; no old EVP code is copied into the active model.

The old transfer ledger alone did not establish conservation of the actual
combined functional. Its fresh-ice and brine bookkeeping ideas are retained;
latent and moving-boundary gravitational energy now enter the common functional.
Snow, fraction transport and EVP are deferred. The archive's global internal-force
correction cannot substitute for a discrete stress-work proof.

## Evidence

Fifteen focused tests cover direct independent full-H subtraction, identity of
its ocean terms with the existing model, local water/salt conservation, unchanged
passive/TKE/eddy contents, freezing, partial/complete melting, coastal masks,
zero-flux equilibrium, unforced supercooling, an energy-reversible heat cycle,
analytic phase derivative versus independent autograd, finite-difference
backward sensitivity, conduction resistance, and explicit rejection of unsupported
motion/exhausted phase brackets. Forward execution is tested with autograd disabled.

The seasonal driver uses 32 independent three-layer columns (20/60/120 m),
fresh zero-layer ice, prescribed sinusoidal air temperature and 10 W/m²/K air
conductance. It includes neither ocean dynamics nor horizontal ice transport.
All runs use CPU float64, one thread, PyTorch 2.15.0+cpu.

| Quantity | 1-day step | Half-day step |
|---|---:|---:|
| Steps / duration | 365 / 1 year | 730 / 1 year |
| Maximum ice thickness | 1.09806 m | 1.09397 m |
| Final maximum thickness | 1.00861 m | 1.00498 m |
| Largest single-step energy residual | 8.55e-7 J/m² | 1.03e-6 J/m² |
| Largest accumulated energy residual | 3.85e-5 J/m² | 8.09e-5 J/m² |
| Residual / initial maximum absolute column energy | 2.53e-14 | 5.31e-14 |
| Largest water discrepancy | 5.82e-11 kg/m² | 8.73e-11 kg/m² |
| Median thermodynamic update | ~0.026 s | ~0.027 s |

Salt contents are unchanged by construction and checked exactly in the focused
cases. Both seasonal integrations melt all ice in summer and regrow it in winter.
The ~4 mm timestep difference illustrates forcing/phase timing error despite
energy closure. More steps accumulate more floating-point subtraction error;
energy residual is not a trajectory-accuracy estimate. These timings describe
small stationary columns, not full coupled-model overhead or GPU throughput.

CSV trajectories and JSON summaries with source hashes are retained in
`seasonal-daily/` and `seasonal-halfday/`. The public runner reproduces both;
use the [component guide](../../../validation/sea_ice/README.md).

## Qualification boundary

This is a closed thermodynamic foundation, not a complete energy-closed sea-ice
model. The common functional already exposes the required ice loading derivative,
but moving-ocean loading, phase momentum exchange, reciprocal drag, ice transport
and dissipative stress work still need implementation. No EVP dynamics, ice KE,
snow enthalpy, partial concentration, TEOS chemical equilibrium or entropy proof
is claimed. Backward evidence is confined to a smooth phase branch. No GPU or
climate skill qualification was performed.

## Regression checks

The installed wheel outside the checkout passed **1239 tests, 2 skipped** under
the established portable fixture policy. That run included the first 12 ice
tests. Three additional edge-case tests and the final ice bracket change were
then checked with the complete **15-test** ice suite in the same external
harness; all passed. No packaged source changed, so the existing wheel was used;
research modules were explicitly copied into the harness. Logs are retained as
`wheel-tests.log` and `ice-tests.log`. Frozen comparisons pass for 142
corrected-Gibbs tensors and 93 dry helpers (`baseline.json`, `dry-helpers.json`).

The final bracket change permits warming within roundoff of the upper salinity
limit while rejecting additional freezing. It only bounds the root-search
interval; it does not clip any prognostic state. Both seasonal runs were repeated
against that final source, with source hashes recorded in their summaries.
