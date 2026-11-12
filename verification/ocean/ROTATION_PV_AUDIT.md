# Rotation, circulation and potential-vorticity compatibility

The current horizontal rotation fails the curl–divergence compatibility test.
This is a discrete inconsistency, not a failure of `curl(grad(q))` or energy
work cancellation. It is present on Fibonacci, refined-icosahedral and the
recovered 16,384-cell atmosphere meshes. No production operator is changed by
this audit.

## Definitions and independent checks

Let D be cell divergence, C vertex curl and J the stored +k× rotation (physical
Coriolis acceleration is −f J u). Let K_vc be the spherical area shared by a
Voronoi cell and its Delaunay dual triangle. The geometric cell-to-dual transfer
is P_vc = K_vc/A_v. It preserves constants and area integrals. The tested identity
is

\[
CJ=PD.
\]

This is the discrete circulation statement relating the curl of rotated flux
to its divergence transferred to dual cells. The comparison uses flux density,
not integrated edge transport. D, C and J are assembled as sparse matrices;
separate tests compare D and C with public runtime methods. A second path-sum
assembly reproduces the unprojected rotation reconstructed from the stored
unprojected tangential stencil and its handedness.

K is computed independently with oriented spherical triangles. Primal/dual
intersection points are normalized sums of adjacent generator positions.
The independent areas partition both primal and dual cells. The production
corner construction instead uses midpoints of **Voronoi edges**, which generally
are not those intersection points. Its corner areas partition the primal cells,
but their sums at vertices generally differ from the dual triangle areas used
by curl. Normalizing those production corners by A_v consequently fails to
preserve constant thickness.

## Operator results

The dimensionless defect is ||CJ−PD||_F / ||PD||_F. It describes all edge-basis
inputs, not a fractional forecast error. The last column is an independent
matrix-only comparison using matching intersections and vertex associations.

| Mesh | Stored rotation defect | Compatible geometric comparison |
| --- | ---: | ---: |
| Fibonacci 64, no relaxation | 0.3371 | 2.02e−15 |
| Fibonacci 64, 20 Lloyd steps | 0.2053 | 2.19e−15 |
| Fibonacci 256, 20 Lloyd steps | 0.2872 | 9.71e−15 |
| Fibonacci 1024, 20 Lloyd steps | 0.2644 | 5.00e−14 |
| Fibonacci 4096, 20 Lloyd steps | 0.4772 | 3.18e−13 |
| Refined icosahedron, 162 cells | 0.1196 | 4.55e−15 |
| Refined icosahedron, 642 cells | 0.09392 | 2.88e−14 |
| Recovered atmosphere, 16,384 cells | 0.09803 | 2.46e−12 |

Resolution and relaxation are not interchangeable in this table; the Fibonacci
sequence does not demonstrate a convergence rate. The recovered mesh is loaded
without further relaxation. Its source array hash is recorded in the artifact.
The fine-mesh geometric residuals reflect finite-precision spherical geometry;
they are not claimed to be bitwise zero.

Two separate geometric issues are isolated:

1. **Corner association.** `verticesOnCell[i,k]` is the exit vertex of edge slot k.
   The current path sum starts at source+1 and includes target. Its effective
   transfer uses the *next* corner fraction at each vertex: C J_raw equals
   P_shift D to roundoff. Assigning the preceding fraction to each traversed
   slot restores compatibility with the stored corner transfer.
2. **Corner partition.** That index change alone does not make the stored
   corner areas the true dual intersections. On the recovered mesh, the
   remaining operator defect is 0.04544; the stored transfer's maximum error
   on a constant field is 0.22835. Using true intersections as well reduces
   the defect to 2.46e−12.

The raw port is already nearly metric-skew on these meshes. The geometry-level
skew projection changes the operator only at numerical-roundoff scale and does
not materially change the observed compatibility defect. This audit therefore
does not support blaming that projection for the primary geometric discrepancy.
The geometric comparison is also metric-skew to geometry roundoff, showing that
these two constraints need not conflict in this setting.

## Uniform-PV tendency test

A global positive constant PV cannot be initialized on the physical rotating
sphere with positive thickness: integrated relative vorticity and integrated
planetary vorticity both vanish. This audit instead prescribes a **synthetic f**
on the same closed topology, solely to test the algebra. It is not an Earth
benchmark or a new supported Coriolis configuration.

Use H=1000 m, U=1 m/s, q0=1e−7 m⁻¹ s⁻¹ and scalar potential
U R (x+0.3 y z). Set u=G(potential), h=H(1+a z), h_v=P h and
f_v=q0 h_v−C u. This f is held fixed when evaluating the instantaneous tendency.
Thus initial (C u+f)/h_v is exactly q0 to roundoff. The production
`ConstantDensityDynamics.rotation_acceleration` is called with this temporary
prescribed f; the mesh buffer is restored afterward.

Continuity supplies hdot=−D[I(h)u]. Uniform PV requires

\[
C\dot u=q_0 P\dot h.
\]

The full Bernoulli force has zero curl to roundoff and cannot repair a rotation
defect. Report the dual-area-weighted RMS of the discrepancy divided by the RMS
of q0 P hdot. This is an *instantaneous tendency discrepancy*, not accumulated
PV drift or a model error percentage.

| Recovered 16,384-cell mesh | Uniform depth | 20% smooth thickness variation |
| --- | ---: | ---: |
| Production rotation | 0.02683 | 0.02716 |
| Same absolute-vorticity pairing, compatible geometric J | 2.93e−11 | 0.002304 |
| Constant-PV flux form with compatible J | 3.40e−11 | 3.39e−11 |

The last two rows are diagnostic algebraic alternatives, not runtime corrections.
The constant-PV flux comparison uses −q0 J[I(h)u], which both preserves uniform
PV under CJ=PD and does zero kinetic work in the actual thickness metric when J
is metric-skew. This does **not** establish the general varying-PV discretization,
potential-enstrophy conservation, Jacobi identity or nonlinear stability.

The additional variable-thickness residual shows that geometry alone is not the
whole issue: the current method averages *absolute vorticity* onto edges and
antisymmetrizes in the layer metric. That is not generally equivalent to a
compatible PV-times-mass-flux expression. Production rotation work still cancels
at roundoff. Both observations can hold simultaneously.

The fine-mesh instantaneous relative PV-rate RMS is about 5.2e−9 s⁻¹ for the
specified U and H. It must not be extrapolated linearly into a long-run error.
Smaller mesh results and all alternative pairings are retained in the JSON files.

## Reproduction and scope

Base source: main `9c756fdb6cb2b3fd8a24679b8cbc4743fa1584ce`. CPU float64,
PyTorch version and driver/input hashes are recorded in
[standard results](rotation-pv-results.json) and
[frozen-mesh results](rotation-pv-frozen-results.json). Relevant installed-source
hashes are recorded in [provenance](rotation-pv-provenance.json).

From an installed-wheel check directory containing tools and tests:

```bash
python -I tools/audit_rotation_pv.py --output rotation-pv-results.json
python -I tools/audit_rotation_pv.py --generator-npy /path/to/mesh_16384_exact.npy --output rotation-pv-frozen-results.json
python -I -m pytest tests/ocean/test_rotation_pv_audit.py -q
```

The frozen mesh array exceeds the artifact upload cap and is not copied into
this PR. The standard seven-case audit is self-contained; reproducing the eighth
requires the hash-matching recovered array. No large matrices are uploaded.

The checks diagnose a problem before any timestep is taken. They do not measure
its effect on existing jets, waves or long integrations. Atmospheric cores share
the geometric buffers, but their complete PV budgets are not audited here.
No production code, fixtures or tolerances change. A correction would require
separate review of both the shared geometry and the variable-thickness PV-flux
form, with atmospheric/ocean regression and balance/wave measurements. The
present work establishes a concrete inconsistency and a diagnostic route to
removing it; it does not qualify a replacement model.

The curl/divergence requirement follows the compatible C-grid formulation in
[Thuburn et al. (2009)](https://www2.mmm.ucar.edu/people/skamarock/Papers/cv_43.pdf)
and the energy/PV distinction in
[Ringler et al. (2010)](https://www2.mmm.ucar.edu/people/skamarock/Ringler_et_al_JCP_2009.pdf).

Validation: installed-wheel suite 831 passed, one CUDA skip; corrected-Gibbs and
dry-helper portable comparisons pass. Four focused audit tests pass, including
buffer restoration and an intentionally misindexed comparison that fails the
compatibility oracle. Runtime source hashes match the base source listed above.
