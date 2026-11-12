# Compatible rotation implementation and checks

`rotation_scheme="compatible"` installs the corrected geometry and carrier/PV
formulation in the dry/moist atmosphere and constant-density/JMD95 ocean paths.
The default historical formulation remains available as `"legacy"`. See
[the public guide](../../docs/COMPATIBLE_ROTATION.md) for selection and equations.
The new geometry does not replace the independent tangential KE reconstruction.

The implemented correction consists of:

- Oriented kite areas at actual primal/dual intersections.
- Exit-vertex-consistent corner association in the rotation path sum.
- Cached area-conservative cell-to-dual carrier transfer.
- Symmetrically averaged PV applied to carrier flux, rather than absolute
  vorticity applied to velocity with a thickness-dependent skew projection.

The Hamiltonian functional, pressure gradients, scalar transport, vertical
exchange, remapping and timestep formulas are unchanged. Rotation is still
skew in the corresponding energy-work pairing. Tests check general nonuniform
PV work as well as the special constant-PV vorticity/continuity identity.

## Recovered 16,384-cell mesh

Both runtime schemes use identical hash-checked generator points and identical
initial u, ps and mtheta arrays, 32 levels, 5000 Pa lid, sigma power 2, unperturbed
public JW initialization and no mass-flux projection. The configuration is the
same as the [preceding t=0 diagnostic](JWA_ROTATION_BALANCE.md).

| Measurement | Historical runtime | Compatible runtime |
| --- | ---: | ---: |
| CJ−PD relative Frobenius defect | 0.0980295 | 1.95e−12 |
| Uniform-PV tendency relative defect, 20% thickness variation | 0.0271562 | 2.93e−11 |
| t=0 momentum mass-weighted RMS (m/s²) | 4.40312e−5 | 3.36493e−5 |
| t=0 maximum momentum residual (m/s²) | 5.26931e−4 | 4.23538e−4 |
| t=600 s momentum mass-weighted RMS (m/s²) | 2.75192e−5 | 2.35916e−5 |
| t=600 s wind-change mass-weighted RMS (m/s) | 0.0149482 | 0.0119338 |
| t=600 s maximum surface-pressure change (Pa) | 67.3221 | 49.6062 |
| t=600 s signed relative mass change | 5.88e−20 | −2.44e−18 |

The run uses ten SSPRK3 steps of 60 s. Initial RMS acceleration falls 23.6%; RMS
wind change after ten minutes falls about 20.2%. Neither is a forecast-error
estimate or a long-run qualification. No timestep or damping is tuned.

Compatible rotation work has relative cancellation residuals 2.17e−19 initially
and −3.34e−19 at 600 s. Total atmosphere-plus-lid instantaneous work residuals are
−4.15e−16 and −6.34e−17 relative to absolute constituent work. These are
semidiscrete work checks, not claims of exact finite-step energy conservation.

The uniform-PV test uses synthetic prescribed f, as explained in the
[operator audit](ROTATION_PV_AUDIT.md); a globally positive constant PV is not a
physical Earth initialization. This implementation uses the actual production
constant-density ocean method in that test. In the JSON, the inherited audit
field `production_pairing_relative_error` measures disagreement with the **old
absolute-vorticity pairing**; it is not an error oracle for the corrected PV-flux
method. The constant-PV flux oracle and work identities are the relevant checks.

## Regression and component checks

The installed wheel was tested outside the source checkout:

- Full portable suite: **840 passed, one CUDA skip**, before the final three
  added component tests. The final correction test file separately passes all
  **10 tests**, including those additions and moist centered IMEX.
- Corrected-Gibbs atmospheric fixture comparison and dry-helper portable
  comparison pass without regenerating fixtures or changing tolerances.
- Compatible geometry tests cover 16/64/256-cell meshes, independent sparse
  curl/divergence/dual-transfer oracles and weighted skew symmetry.
- A nonuniform-PV test checks zero work and velocity/thickness reverse-AD
  derivatives against finite differences.
- Mesh save/load preserves the scheme and serialized geometry exactly;
  reconstructed rotation agrees within floating-point roundoff. NumPy and Torch
  evaluation of the derived Coriolis factor can differ by an ulp.
- Dry IMEX, moist bounded stepping, moist centered IMEX and moist energy-work
  closure execute on corrected meshes.
- Active stratified-ocean stepping and adaptive remapping execute with and
  without coast masks; volume, salt and passive inventory checks pass and
  blocked edge velocities remain zero.
- The standalone compatible-rotation example executes.

Historical default outputs remain reproducible. No old fixtures or tolerances
change. CPU float64 is the measured scope; no GPU/compile/long-backward claim.
Near-coast uniform-PV conservation is not established. Full atmospheric PV,
enstrophy, Jacobi structure and sustained wave/climate accuracy remain separate
questions. The correction is explicit rather than silently changing existing
runs that construct a mesh without a scheme argument.

## Reproduction

After installing a wheel and copying tools/tests/examples outside the checkout:

```bash
python -I -m pytest tests/test_compatible_rotation.py -q
python -I examples/compatible_rotation.py
python -I tools/qualify_rotation_correction.py --mesh /path/to/mesh_16384_exact.npy --output rotation-correction-results.json
```

[Results](rotation-correction-results.json) record driver, mesh and initial-state
hashes; [provenance](rotation-correction-provenance.json) records the tested runtime
source hashes. The frozen input array exceeds 128 KiB and is not uploaded.
The implementation branch starts from the audit PR's head
`2e91f23b1f22e18de5f1f013c20fb46c10c9bc65` and preserves that evidence.
