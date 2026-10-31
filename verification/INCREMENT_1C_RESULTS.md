# Increment 1c — small-run results

All 22 development runs completed on macOS Apple M3, Python 3.12.13,
PyTorch 2.15.0 / Accelerate, CPU float64, one Torch thread. Concurrent local
processes make timings descriptive, not performance benchmarks. Full metrics,
reference errors and artifact hashes: [machine-readable evidence](results/increment_1c_small.json).
Methods and runnable commands: [increment 1c](INCREMENT_1C.md).

## Smooth full-period return

Liquid fraction area-weighted L2 error (kg/kg), **both CP interfaces**.
Vapor and ice norms, weighted inventories, extrema and intermediate fields are
recorded separately; ice shares the initial liquid profile at one tenth its scale.

| Flow | Cells | dt=T/120 | dt=T/240 | Paired dt-halving L2 difference |
|---|---:|---:|---:|---:|
| rotation_equatorial | 16 | 0.0001459103 | 0.000145897 | 4.181165e-08 |
| rotation_equatorial | 64 | 0.0001361719 | 0.0001361533 | 1.305089e-07 |
| rotation_equatorial | 256 | 9.627981e-05 | 9.632043e-05 | 3.916184e-07 |
| rotation_tilted | 16 | 0.0001466407 | 0.0001466347 | 2.49238e-08 |
| rotation_tilted | 64 | 0.0001126094 | 0.0001126041 | 1.438776e-07 |
| rotation_tilted | 256 | 9.401506e-05 | 9.404742e-05 | 3.141081e-07 |
| NL2010_case4 | 16 | 0.0001979546 | 0.0001979528 | 2.788789e-07 |
| NL2010_case4 | 64 | 0.0002148495 | 0.0002148426 | 4.981088e-07 |
| NL2010_case4 | 256 | 0.000209251 | 0.0002093432 | 1.923215e-06 |

Rotation return errors decrease overall with resolution, but this is not an
asymptotic order study. Deformational errors do **not** decrease monotonically;
the data do not establish convergence. Paired timestep differences are smaller
than return errors, separating visible timestep sensitivity from the dominant
coarse spatial/mixing effects. One halving does not establish SSPRK3 temporal order.

At dt=T/240, liquid variance retention from 16 to 256 cells rises from 0.220
to 0.714 (equatorial) and 0.334 to 0.768 (tilted). Case 4 retains only
0.153, 0.098, 0.122 across 16/64/256 cells. Corresponding peak losses and
centroid angles are in the JSON. Irreversible numerical mixing prevents exact
return; boundedness alone is not an accuracy or shape-preservation guarantee.

## Conservation, positivity, limiter and reference evidence

- Maximum absolute relative carrier change across all final states: 1.91e-14;
  individual dry/vapor/liquid/ice inventory relative changes: at most 1.93e-14.
- Every recorded Euler water tensor exactly matches its authoritative ledger.
  All carrier/dry/vapor remained positive; condensates remained nonnegative.
  Empty-condensate Case 4 retained exact zeros and exercised 49,133 limited
  face evaluations. No vapor floor was inserted.
- Limiting occurred even for smooth profiles. Actual maximum carrier-outflow
  fraction was 0.713; maximum donor substeps was 1. Constant controls used the
  safe-high-order shortcut (count 0). Per-stage counts and beta minima remain
  in the artifacts; these are actual production diagnostics.
- Directly sampled flow is discretely divergent. Peak Case 4 divergence across
  time was 1.115e-6, 6.365e-7, 2.215e-7 s^-1 at 16/64/256 cells. Carrier was
  evolved, not reset. Measured vertical graph flux was zero in these one-layer cases.
- Smooth initial/final order-12 versus order-8 quadrature differences were at
  most 6.65e-12 kg/kg. At maximum deformation the worst difference was 1.08e-6:
  midpoint reference resolution is less secure and no midpoint convergence
  claim is made. DOP853 tolerance refinement changed averages by at most
  3.57e-14. These refinement differences are estimates, not certified bounds.
- Compact-cloud reference quadrature differences reached 1.75e-8 initially/
  finally and 3.12e-7 at maximum deformation. This case exercises limiting,
  not a smooth-profile convergence claim.
- Aggregate measured runtime was about 130 s; artifacts before manifests total
  about 64.3 MB. No large campaign was run. The local directories are
  `/tmp/vfs-c5-{smooth,constant,empty}-final`; rerun commands create new protected
  directories. Compact committed evidence includes hashes for every raw artifact.

## Constant-composition negative result — separate issue, no core fix

The 16-cell constant controls have maximum stage composition deviations of
1.001e-6 (equatorial), 7.896e-7 (tilted), and 5.897e-6 (Case 4) kg/kg.
Final liquid L2 errors, including both interfaces, are respectively 3.660e-7,
2.447e-7 and 1.637e-6. All three inventories remain conserved and no limiter
activates. Constant composition is **not preserved** in these full-period probes.

This is not explained by the SSPRK3 extraction: exact atmospheric pre/post
comparisons passed. The frozen `gibbs_face` mixes logarithmic gas-density means
with arithmetic condensate-density means. With unequal density and identical
endpoint composition, the returned face fractions differ from that composition.
Discrete divergence evolves bottom pressure/density, exposing this distinction.
Top-interface errors alone misleadingly appear near roundoff; a regression test
now ensures that bottom-interface errors cannot be omitted from norms.

Minimal reproduction using the unchanged installed runtime:

```python
import torch
from vfs.atmosphere.gibbs_means import gibbs_face
from vfs.atmosphere.mixture_thermodynamics import MixtureThermodynamics
torch.set_default_dtype(torch.float64)
q = torch.tensor([[.9889, .01, .001, .0001]])
T = torch.tensor([280.])
_, face, _ = gibbs_face(MixtureThermodynamics(), T, torch.tensor([1e5]), q,
                       T, torch.tensor([9e4]), q)
print(face - q)
```

Dry/vapor/liquid/ice differences are approximately
`[-1.0060946e-6, -1.0173876e-8, 9.2388042e-7, 9.2388042e-8]`.
For constant q and equal T, gas weights are `q_g*logmean(p_i,p_j)` and
condensate weights `q_c*(p_i+p_j)/2`, normalized by their sum. An independent
formula test reproduces the mismatch. No pressure, scalar, mean or limiter
was changed to conceal it.

**Separate proposal:** investigate whether a common density mean for species
can preserve constant composition while retaining the Gibbs identity and its
energy pairing. Any candidate requires a new derivation and separate reviewed
numerical fix, with the existing ledger/energy/gradient regressions. Simply
replacing the face trace by constant q or projecting fluxes in this harness
would change the production method under test and is not proposed here.

## Validation and limits

Local installed-wheel suite: 563 passed, one CUDA skip (including 12 C5 tests).
Tests cover analytic return by independent forward trajectory integration,
time-dependent velocity, stage times/state handling, small-dt constant transport,
Gaussian quadrature, histogram versus node-by-node face audit, corrupted ledgers/
fluxes/limiter counts, and sensitivity to bottom-interface errors. The small-dt
test is not a claim of full-period constant preservation.

Both portable comparison tools and all six examples passed for the same wheel
during extraction validation; the dependent PR changes no runtime source.
Same-environment atmospheric equivalence remains the #16 result: 298 exact
tensors, scalar/ARK4 solver records and four exact full-step gradient tensors.
Required hosted results for this dependent PR are recorded in its description,
separately from these local measurements.

This does not qualify coupled energy closure, temporal order, 3-D transport,
IMEX backward, GPU execution, or atmospheric rollout accuracy. Terrain-rest
construction remains the next queued increment; the constant-composition
finding needs separate review before treating C5 as qualified.
