# Ocean radiation qualification

Optional `OceanSurfaceHeating` applies prescribed signed non-solar surface heat
and two-band penetrating shortwave to the existing JMD95/entropy Hamiltonian.
Current thicknesses define optical interfaces. The bottom policy explicitly
exports transmitted radiation or deposits it in the bottom water layer. There
is no atmospheric radiation calculation, bulk air–sea exchange or freshwater
operator. See [API/equations](../../docs/OCEAN_RADIATION.md).

## Analytic optics and energy

`validation/ocean_boundary/radiation.py` uses a 40 m column, 200 W/m² incoming
shortwave, -50 W/m² surface heat, a=0.6, lengths 0.5/20 m and 3600 s forcing.
For 1, 4, 16 and 64 layers, analytic interface-flux differences agree with the
implemented layer heating to relative errors <=5.06e-16. With bottom export,
water receives 139.17317734 W/m² and 10.82682266 W/m² exits the bottom. With
bottom absorption, water receives 150 W/m². These integrated values are
independent of vertical partition to roundoff.

Independent total Hamiltonian increments minus prescribed heat have magnitude
<=2.11e-11 J/kg; maximum individual-layer discrepancies are <=3.55e-11 J/kg.
These checks establish conservative optical deposition and thermal inversion,
not empirical correctness of the illustrative optical coefficients. See
[results](radiation/optics.json).

## Combined evolution

`examples/ocean_radiation.py` combines resolved dynamics, heating and per-layer
TKE mixing/dissipation on 32 cells, 16 layers and varying depth, for five
60-second steps. Net water heating is 2.12262e19 J; bottom export is
1.72668e18 J. Full-step energy minus external heat is -1.32e-11 J/kg, including
dynamics and mixing errors. The heating-only residual is 4.75e-12 J/kg. Final
minimum k is 0.0001411 m²/s². Initial k=0.001 m²/s² is explicit. See
[results](radiation/spherical.json).

## Regression and extraction

Installed-wheel portable suite outside checkout: **967 passed, one CUDA skip**.
Corrected-Gibbs (142 comparisons) and dry-helper (93 comparisons) portable
regressions passed with unchanged fixtures and tolerances.

Seventeen new tests cover analytic absorption, both bottom policies, signed
surface cooling/heating, resolution-independent column absorption, varying
depth, dry masking, exact zero forcing, cancellation, zero-forcing and smooth
shortwave derivatives, invalid optics/fluxes, rejection without mutation and
TKE composition. The 24 existing TKE tests also pass.

The shared nonnegative TKE heating path reproduces its pinned parent exactly
across 28 tensor comparisons, including zero/tiny/finite heat increments and
a complete mixing step with every report field. Reproduce with
`validation/ocean_boundary/compare_heating.py --repository /path/to/vfs-a`
against an installed candidate wheel. See [comparison](radiation/extraction.json).

## Limits

CPU float64, short forward trajectories and selected smooth derivatives only.
No GPU/float32/long-run qualification, calibrated optical water types, scattering,
chlorophyll, sky/surface radiation calculation or seabed thermal reservoir.
Atmospheric latent/sensible flux calculations and freshwater exchange remain
absent. Supplying a heat flux alone does not implement evaporation mass loss.
Thermal-domain failures reject rather than clip. Independent energy tolerances
scale with the thermal background and cannot establish relative accuracy of
arbitrarily tiny increments. First-order process splitting remains unchanged.
