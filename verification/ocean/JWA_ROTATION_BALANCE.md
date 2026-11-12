# JW-A initial balance: rotation geometry comparison

The recovered 16,384-cell atmosphere mesh shows a 23.6% reduction in initial
mass-weighted RMS momentum residual when the rotation uses consistent
corner association and primal/dual intersections. The residual is reduced, not
eliminated. Energy-work closure remains at roundoff. This is a t=0 diagnostic;
no alternative model is integrated and no production operator is changed.

## Current source and configuration

GitHub main was checked at `9c756fdb6cb2b3fd8a24679b8cbc4743fa1584ce`, containing
the integrated atmosphere/ocean API and documentation update. Its runtime
rotation is the one diagnosed here. The equiangular-grid PR does not replace
that rotation, and the rotation/PV audit is diagnostic-only.

The run uses the recovered `mesh_16384_exact.npy` generator points, with no
additional relaxation. The accompanying archived mesh diagnostics record 3000
Lloyd iterations; their convergence flag is false because the full requested
stopping criteria were not all met. The frozen generator hash is recorded in
[the results](jwa-rotation-balance-results.json).

Configuration:

- Current `DryHydrostaticConservativeDynamics`, CPU float64.
- 32 levels, fixed pressure lid 5000 Pa, hybrid sigma power 2.
- Mesh radius 6,371,000 m and the current default dynamics constants.
- `constant_ps_surface_phi`, `direct_u_gl2_adjoint` column reconstruction.
- Unperturbed public JW initializer, `theta_source='phi_derivative'`,
  `wind_source='U_jw'`, point-sampled theta, no mass-flux projection or fitting.
- Surface pressure 100,000 Pa; maximum edge-normal wind 34.9966145 m/s;
  theta range 226.527088–527.349567 K.

This tests the current public initializer. It is not claimed to reproduce every
initialization detail of the archived 20-day run. In particular, its maximum
edge-normal wind differs slightly from that archived summary. The analytic JW
helper and runtime also use slightly different nominal radius/gravity constants;
these are held fixed across every comparison, not adjusted to improve balance.

## What changes in the comparison

Pressure, thermal fields, velocities, kinetic energy, continuity, scalar
transport and vertical momentum terms are identical in all variants. Only the
rotation stencil J changes. The same absolute vertex vorticity, edge averaging
and fixed-vorticity energy pairing are applied to each J:

\[
A=-\tfrac12[\operatorname{diag}(\zeta_e+f_e)J
 +J\operatorname{diag}(\zeta_e+f_e)],\qquad
\dot u_{rot}=\tfrac12[A-M^{-1}A^TM]u,
\]

where M is the actual diagonal kinetic metric, edge area times layer pressure
thickness divided by gravity. A sparse reassembly of the stored stencil matches
the production rotation; an independent dense small-matrix test checks the
pairing. Mesh buffers are not replaced.

The horizontal gradient-wind residual is pressure gradient + rotation + kinetic
Bernoulli gradient. Full momentum additionally includes vertical exchange and
metric terms. RMS weights are M; edge-area/equal-level RMS and per-level
measurements are also retained in the JSON.

## Results

All accelerations below are in m/s².

| Rotation | Horizontal residual RMS | Full momentum RMS | Maximum full residual |
| --- | ---: | ---: | ---: |
| Current production | 4.40326e−5 | 4.40312e−5 | 5.26931e−4 |
| Index association only | 3.21786e−5 | 3.21774e−5 | 3.87246e−4 |
| Index + true intersections | 3.36513e−5 | 3.36493e−5 | 4.23538e−4 |

The consistent geometric comparison improves full RMS by **23.58%** and maximum
residual by **19.62%**. Additional metric-skew projection of that J leaves the
reported balance norms unchanged at the shown precision. The raw production
port and its stored skew projection likewise give the same norms.

The index-only comparison has a lower residual for this one state, but retains
the geometric compatibility defect established by the PV audit. It is not a
preferred correction based on this scalar score; errors can cancel in one
initial state.

The unchanged pressure-gradient RMS is 1.30087e−3 m/s², kinetic Bernoulli RMS
8.59227e−5 m/s² and vertical-term RMS 6.92007e−8 m/s². The maximum production
residual occurs near latitude 42.24°, level index 14 (zero-based), at approximately
24,529 Pa. The correction does not remove errors from pressure reconstruction,
kinetic Bernoulli reconstruction, initialization sampling or vertical resolution.
No partition of the remaining residual among those causes is established here.

## Energy interpretation

The existing closed semidiscrete invariant is atmospheric energy **plus the
fixed-pressure lid reservoir**. The geometry/PV inconsistency does not negate
that demonstrated energy budget. With each tested J, the same metric-skew
pairing still gives zero rotation work independently of curl compatibility.

| Quantity | Production | Index + intersections |
| --- | ---: | ---: |
| Signed rotation work / sum of absolute edge-layer work | −1.25e−18 | −3.12e−19 |
| Closed energy rate / sum of absolute momentum and scalar work | −4.08e−16 | −4.15e−16 |

The unnormalized closed rates are about −18 W globally; these are cancellation
residuals of much larger terms, not measured physical energy dissipation. The
Hamiltonian functional and its matched pressure/transport derivatives remain
unchanged. Energy closure alone never established complete Poisson/Jacobi
structure, exact PV conservation or accurate balance.

These results support a separately reviewed geometric correction. They do not
yet qualify long-run behavior or the variable-thickness PV-flux alternative,
which is **not** included in this JW-A comparison.

## Reproduction

From an installed-wheel directory containing tools and tests:

```bash
python -I tools/audit_jwa_rotation_balance.py --mesh /path/to/mesh_16384_exact.npy --output jwa-rotation-balance-results.json
python -I -m pytest tests/test_jwa_rotation_audit.py tests/ocean/test_rotation_pv_audit.py -q
```

The mesh array is omitted because it exceeds the 128 KiB upload cap. The result
artifact records its hash, both diagnostic driver hashes, state hashes, exact
initialization options, all component norms and per-level residuals. Relevant
runtime/initializer source hashes are in
[JW-A provenance](jwa-rotation-balance-provenance.json). The underlying
[rotation/PV audit](ROTATION_PV_AUDIT.md) documents the geometric alternatives.
