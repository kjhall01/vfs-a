# Proposed paired Gibbs face revision — review evidence

This is a separately authorized numerical fix following [#18](https://github.com/kjhall01/vfs-a/pull/18),
not a silent revision of the verification baseline. Historical baseline
`8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648`, fixtures, tolerances and original
[C5 evidence](INCREMENT_1C_RESULTS.md) remain intact. Proposed numerical commit:
`a7facc1bcdb88f7dc9c10f4fb6059f29b0a9bac7`. The sole changed runtime file is
`vfs/atmosphere/gibbs_means.py`; manifests record that revision separately from
baseline and the approved SSPRK3 interface extraction. Source checking rejects
this change unless explicitly selected with `--numerical-revision`.

## Formula and pairing audit

The [derivation](CONSTANT_COMPOSITION_DERIVATION.md) establishes the CP carrier
and species condition, including horizontal half-layer interpolation and vertical
interface fluxes. For incidence B, integrated carrier flux G, carrier M and
species amount Q, M'=M+BG and Q'=Q+BJ preserve a uniform fraction q when
J=qG. If the pressure-derived target mass differs by r, water has an additional
−qr error; physical dry differs from ledger dry by r. The probes retain this
independent check rather than hiding it by resetting carrier mass.

For condensates only, replace the arithmetic partial-density mean by
`d_c* = logmean(rho_i, rho_j) * (q_ci + q_cj)/2` (kg/m³). Gas partial densities
retain their logarithmic means. The existing sums then recompute **all three**
matched traces: composition (kg/kg), entropy (J/kg/K), specific volume (m³/kg).
No species-only patch, floor, clipping or constant-composition detection occurs.
At constant composition every species weight is q times the same total-density
logarithmic mean, so normalization preserves q algebraically. Condensate gas
constants are zero; their density weights are free in the Gibbs log-density term.
The entire recomputed tuple preserves the Gibbs identity for unequal T and p.
This is an algebraic statement; the floating-point results below measure its
implementation, including cancellation in the unchanged gas logarithmic means.

The candidate is symmetric and consistent, but is **not an internal mean of
condensate partial density**. The investigation's anticorrelated example exceeds
the endpoint interval by 0.05114493007861455 kg/m³. The audit of
`bounded_transport.py` and `BoundedGibbs` found no use of that interval property:

- The donor reference uses node carrier/species amounts and carrier outflow.
  The common-beta limiter restricts actual amount losses and requires admissible
  endpoint amounts and the species-flux/carrier sum pairing, not internal face
  partial densities. A changed candidate can change beta, active faces and mixing;
  it does not establish local-extremum preservation.
- `_gibbs_corrections` pairs actual entropy/composition trace changes with energy
  covector jumps and the negative carrier transpose. Limiter work uses the actual
  corrections; its sign split allocates mechanical work or nonnegative heat.
  Neither argument requires internal partial-density means. `face_alpha` is
  recomputed consistently even where returned only as a diagnostic.

No corresponding limiter, entropy-transport, energy-allocation or integrator
formula change was identified as necessary. Positive vapor remains required.
Exactly zero and one-sided condensates are supported without a floor.

## Small face, energy, stage and derivative evidence

[Machine-readable checks](results/gibbs_fix_checks.json) contain all 116 face
records, actual Euler stage budgets, derivative probes, C3 component matrices,
C4 inventories, environments and manifests. Gibbs residual is
`sum(q* delta_mu) + s* delta_T - alpha* delta_p` in J/kg. Its cancellation
normalization is the sum of absolute individual chemical, entropy and pressure
work terms; a zero denominator is reported as null.

On local Apple M3 CPU/float64, maximum absolute residual was **1.83445e-10 J/kg**,
maximum cancellation-normalized residual **1.50370e-14**. Linux AMD EPYC 9V45
CPU/float64 gave the same absolute maximum and **1.50408e-14** normalized.
All tuple components match the offline candidate. Deliberately perturbing the
entropy trace makes the independent 60-digit Decimal identity check respond.
Equal states, unequal T/p, varying composition, zero/one-sided condensates and
nearly equal states are included. Constant-face fraction error is at most
1.79544e-15, at the unchanged logmean series/exact branch boundary. The new face
regression bound includes the reported logarithm-subtraction conditioning;
no historical tolerance was altered.

Twelve actual 16-cell/3-layer bounded Euler/SSPRK3 cases cover both flow signs,
constant/zero/one-sided clouds and nearly equal thermodynamic states. Constant
composition stage/final drift is at most **2.22045e-16**. All authoritative water
ledgers match exactly; carrier, dry and vapor remain positive and condensates
nonnegative. One-sided cases actively limit (40–117 faces per Euler stage).
Observed stages and the public SSPRK3 result agree exactly.

Fresh old/new C2–C4 runs use the same 16-cell, 3-layer states, dt=1 s and two
directions. **All C2 derivative-sweep records are exactly equal.** C3 signed
instantaneous total work and cancellation-normalized work are:

| State | Old W | Proposed W | Old normalized | Proposed normalized |
|---|---:|---:|---:|---:|
| Smooth | 15.8516 | 21.3984 | 6.58547e-15 | 8.88990e-15 |
| Sharp | 36.1641 | 51.8672 | 1.54632e-14 | 2.21776e-14 |
| Reversed | -41.6094 | -57.5234 | -1.78751e-14 | -2.47117e-14 |
| Dry control | -157.875 | -157.875 | -7.00729e-14 | -7.00729e-14 |

Moist component cancellation scales are about 2.3–2.4e15 W. Sharp allocated heat
changes from 4.09589409e12 to 4.09626084e12 W; reversed from 1.56070732e12 to
1.56075006e12 W. These are intended changes, not equality regressions.
New heat-work pairing residuals are 0 and 0.00048828125 W respectively.
C4 records every actual stage, independent incidence sums, positivity, exact
ledger and final convex-combination budgets. Instantaneous cancellation does
not imply finite-step energy conservation.

The same-environment capture compares **all 648 leaves**, including 298 tensors.
47 leaves change, confined to bounded moist RHS, Euler/SSPRK3 and their ledgers.
Dry and centered moist tendencies, energy derivatives and RHS VJPs, plus dry
and moist ARK4 states, stages and solver diagnostics, are exactly unchanged.
Existing smooth and active-limiter full-step directional derivative tests pass
for wind, pressure, entropy and water. At zero condensate, a one-sided liquid
perturbation through a complete bounded step gives AD 6.72473940e-5; FD absolute
errors are 2.67e-15, 8.04e-14, 3.02e-13, 2.52e-12 as h decreases from 1e-2 to
1e-5. Limiter masks remain unchanged. This is branch-local evidence, not proof
of differentiability across limiter switches or qualification of IMEX backward,
GPU, reduced precision or atmospheric rollouts.

## Same 22 C5 experiments

[Full paired results](results/gibbs_fix_c5.json) retain every old/new metric,
artifact hash, limiter count, inventory, stage extremum, runtime and field
comparison. `tools/compare_gibbs_transport.py` checks all artifact hashes,
identical initial tensors/geometry/configuration, stage times and budgets,
same Python/Torch/platform and exactly the one allowed runtime source change.
The reference quadrature and both CP interfaces are unchanged. Old results
are never overwritten. These are prescribed-flow transport experiments, not
coupled dynamical energy-closure experiments.

Maximum constant-composition stage drift (kg/kg), 16 cells/120 steps:

| Flow | Old | Proposed |
|---|---:|---:|
| Equatorial rotation | 1.00052e-6 | 5.37764e-16 |
| Tilted rotation | 7.89564e-7 | 3.26128e-16 |
| NL2010 Case 4 | 5.89698e-6 | 5.46438e-16 |

No limiter activation occurs in these constant cases. The prescribed discrete
carrier still changes: no pressure or mass reset conceals divergence. Full
stage divergence and carrier-outflow measurements remain in the paired data.

Smooth final liquid area-weighted L2 error (kg/kg), 240 steps, both CP interfaces:

| Flow/cells | Old | Proposed | Limited face evaluations old → proposed |
|---|---:|---:|---:|
| Equatorial/16 | 1.45897e-4 | 1.45541e-4 | 14828 → 14804 |
| Equatorial/64 | 1.36153e-4 | 1.36113e-4 | 67725 → 67699 |
| Equatorial/256 | 9.63204e-5 | 9.63140e-5 | 213044 → 212998 |
| Tilted/16 | 1.46635e-4 | 1.46263e-4 | 17236 → 17267 |
| Tilted/64 | 1.12604e-4 | 1.12626e-4 | 69243 → 69222 |
| Tilted/256 | 9.40474e-5 | 9.40515e-5 | 221452 → 221508 |
| Case 4/16 | 1.97953e-4 | 1.98024e-4 | 10857 → 10810 |
| Case 4/64 | 2.14843e-4 | 2.14904e-4 | 43689 → 43608 |
| Case 4/256 | 2.09343e-4 | 2.09343e-4 | 167148 → 167230 |
| Case 4/64, empty clouds | 1.14144e-4 | 1.14181e-4 | 49133 → 49138 |

**The nonmonotonic deformational convergence remains unresolved.** Changes are
small compared with total error and are not uniform improvements. All 120-step
counterparts are recorded to distinguish timestep sensitivity from this spatial
behavior. For Case 4, proposed 120 → 240-step errors are 1.98023401e-4 →
1.98024401e-4 (16 cells), 2.14905421e-4 → 2.14903912e-4 (64), and
2.09250538e-4 → 2.09342907e-4 (256). Timestep halving does not restore monotonic
spatial convergence. These small differences are not evidence of temporal order. Maximum final relative inventory changes across the proposed cases:
carrier 1.90958e-14, individual species 1.92069e-14. All recorded Euler ledgers
are exact; dry/vapor stay positive, clouds nonnegative. No scientific acceptance
threshold or larger campaign is introduced.

## Reproduction and validation status

Build/install the wheel and run from outside the source checkout as in
[CONTRIBUTING](../CONTRIBUTING.md). With REPO pointing to this checkout, choose a
fresh OUT directory and explicitly select the proposed numerical revision:

```sh
REV=a7facc1bcdb88f7dc9c10f4fb6059f29b0a9bac7
python -I "$REPO/tools/check_gibbs_fix.py" --repo "$REPO" --output "$OUT/faces" --numerical-revision "$REV"
python -I "$REPO/tools/run_energy_verification.py" --repo "$REPO" --output "$OUT/energy" --numerical-revision "$REV"
python -I "$REPO/tools/run_transport_verification.py" --repo "$REPO" --output "$OUT/smooth" --numerical-revision "$REV"
python -I "$REPO/tools/run_transport_verification.py" --repo "$REPO" --output "$OUT/constant" --cells 16 --steps 120 --profile constant --numerical-revision "$REV"
python -I "$REPO/tools/run_transport_verification.py" --repo "$REPO" --output "$OUT/empty" --cells 64 --steps 240 --flows NL2010_case4 --profile empty_condensate --numerical-revision "$REV"
python -I "$REPO/tools/compare_gibbs_transport.py" --old "$OLD/smooth" "$OLD/constant" "$OLD/empty" --new "$OUT/smooth" "$OUT/constant" "$OUT/empty" --output "$OUT/paired.json"
```

For historical runs use a separate checkout/wheel without the numerical revision
and omit that option; keep their output paths distinct. Local source manifests
record the dirty driver development state and actual archived driver hashes;
those hashes, not the checkout label alone, identify the executed driver.

Local installed wheel: Python 3.12.13, Torch 2.15.0, Apple M3/Accelerate CPU,
float64: **570 passed, 2 failed, 1 CUDA skip**. Both failures are preserved old
moist-reference comparisons. Standalone moist portable and exact comparisons
fail; dry portable passes all 93 comparisons. All six documented examples run.

[Hosted run 41171844985](https://github.com/kjhall01/vfs-a/actions/runs/41171844985),
code commit `2ccdfe8`, confirms Linux Python 3.12.15/Torch 2.15.0+cpu, AMD EPYC
9V45 (backend details in the JSON): **570 passed, 2 failed, 1 CUDA skip**, same
failed tests. Hosted dry portable, six examples, Held–Suarez archive comparison,
C1, C2–C4, small C5 and proposed-face/stage probes complete. Informational exact
moist fails and dry passes, separately recorded; a successful collection step is
not an exact-comparison pass. No cross-backend bitwise claim is made.

The required job is **red**, not ready to merge. The portable moist fixture first
fails at cloud RHS wind (local error 5.83144e-15, historical bound 1.31791e-15).
Both pytest's historical fixture test and its CLI exit-status test retain failure.
Comparator corruption tests now use an in-memory capture as their mutation
reference, with historical path/shape coverage checked; they test comparison
categories without conflating them with old/new algorithm equality. No stored
reference was regenerated and no tolerances loosened.

**Reference proposal:** review these manifests, independent checks and intended
old/new differences before approving any new numerical baseline or reference
fixture. This evidence is not a replacement golden fixture. Keep the historical
fixture gates and their failures visible until a separately reviewed reference
policy is agreed. The candidate resolves the demonstrated composition defect,
but is not grounds to claim C5 convergence or other paper claims. Leave this PR
unmerged; no baseline promotion, release or tag is made here.
