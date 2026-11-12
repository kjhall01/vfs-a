# Horizontal bottom drag over bathymetry

`OceanBoundaryStress` permits varying stationary column depth. It applies the
same horizontal quadratic stress to each local bottom layer, with the same
native momentum metric, interpolated layer thickness and KE-to-TKE allocation
as the flat-bed implementation. The default flat-depth example reproduces every
original reported result exactly. Optional domain and column depths must agree
in wet cells. No slope-normal projection, slope-area multiplier or unresolved
topographic form drag is introduced. Original #53 evidence remains unchanged.

Installed-wheel portable suite outside checkout: **950 passed, one CUDA skip**.
Corrected-Gibbs (142 comparisons) and dry-helper (93 comparisons) portable
regressions passed with unchanged fixtures and tolerances.

## Boundary regression

Twenty-four boundary tests include nine new checks: along/across-slope currents
with and without coasts, exact no-op drag at rest over a slope and seamount,
inconsistent geometry rejection without input mutation, bitwise recovery of
the flat-bed formula, and a combined varying-depth dynamics/TKE step. They
check native momentum and independent global/cell kinetic-plus-TKE budgets,
nonnegative TKE, unchanged passive contents, and untouched dry cells.

## Refinement

`validation/ocean_boundary/bathymetry.py` uses 32/128/512/2048-cell spherical
meshes with two Lloyd iterations, depth D=100+40*z metres (unit-sphere z), four
layers with fractions [0.1,0.2,0.3,0.4], Cd=0.0025 and a 100-second drag substep.
Along-slope flow is 0.3*(zhat cross r); across-slope flow is
0.3*(zhat-z*r), in m/s. Both have an analytic horizontal vector speed.

The independent reference uses that speed in the drag formula, retaining the
same discrete edge thickness. Area-weighted relative errors of the velocity
**increment** therefore isolate reconstructed-speed error, not the existing
thickness interpolation or an entire boundary-layer solution.

| Cells | Along-slope error | Across-slope error |
|---:|---:|---:|
| 32 | 0.6191% | 1.3507% |
| 128 | 0.1907% | 0.7098% |
| 512 | 0.0666% | 0.3293% |
| 2048 | 0.0212% | 0.1632% |

Boundary energy residual divided by drag loss has magnitude below 1e-17 in
these runs. Independent rest probes at uniform potential temperatures 10 and
12 degrees C, SP=35, show exactly zero boundary velocity/TKE change. Existing
Hamiltonian pressure RMS is 1.68e-20 to 1.76e-19 m/s²; these are uniform-composition
probes, not a stratified topographic balance qualification. Results are in
[refinement.json](bathymetric-drag/refinement.json).

## Combined example

`python examples/ocean_boundary_stress.py --bathymetry` runs 32 cells, 16 layers,
five 10-second steps and depth 61.36–138.64 m, with resolved dynamics, wind,
drag and per-layer TKE mixing/dissipation. Explicit initial k=0.001 m²/s².
Wind supplies 1.68808e14 J and drag transfers 3.81802e13 J into TKE. The boundary
energy residual is -0.1484 J; full-step energy minus wind work is 5.53e-12 J/kg.
Minimum final k=0.0006135 m²/s². See [results](bathymetric-drag/spherical.json).

This qualifies short CPU float64 behavior and smooth-depth drag refinement.
It does not establish realistic bottom Ekman layers, abrupt-step/coastal speed
reconstruction accuracy, stratified pressure-gradient accuracy, GPU or long-run
stability. Separate surface TKE input and fully slope-aware traction remain absent.
