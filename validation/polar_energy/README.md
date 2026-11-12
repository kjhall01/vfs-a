# Polar accuracy and energy/tracer mapping

Optional diagnostics for both cell-centered and pole-inclusive equiangular grids.
No runtime operators, interpolation defaults or model energy definitions change.

```bash
python -I validation/polar_energy/polar.py --output /tmp/analytic
python -I validation/polar_energy/mapping.py --output /tmp/mapping.json
python -I validation/polar_energy/data_poles.py snapshot.nc --output /tmp/poles.json
```

The first two commands are offline. The third takes a single 2D wind snapshot
with `latitude`, `longitude`, scalar `time` and `level` coordinates, and
`u_component_of_wind`, `v_component_of_wind` in m/s. It assumes both exact poles
are present and uses the WB2 radius of 6,367,500 m. It does not modify the input.
To reproduce the recorded input, explicitly open the HTTPS Zarr store documented
in [the WB2 campaign](../equiangular_wb2/README.md), select 2020-01-15T12:00:00,
250 hPa, load only the two wind variables and save that selection as NetCDF.
Cloud reading is optional and not a package dependency. Input data are omitted
from Git; all committed evidence files are smaller than 128 KiB.

## References and polar results

`polar.py` uses spherical harmonics x, z and xz and their tangential gradients
and rotations on a unit sphere. Their divergence/curl are known analytically.
Closed-form area integrals give true primal-cell and dual-cell averages;
independent Gauss–Legendre quadrature checks those integrals in offline tests.
Exact normal-flux line means and exact tangential-circulation line means are
separate audit inputs. A single edge sample is not generally both line means.

There are three different references:

- Analytic point values for ordinary finite differences.
- Analytic control-volume means for native VFS divergence and curl.
- Spherical finite differences of ERA5 winds: an independent discretization,
  not truth. WB2 archived derived fields are another discretization, described
  in the preceding campaign; neither is an analytic reference for real weather.

[Recorded results](../../verification/equiangular/polar-energy/) cover 16, 32,
64, 128, 256 and 720 latitude intervals, both layouts, three modes and two
velocity constructions. Native operators supplied exact line means recover
analytic area means to floating-point geometry error (maximum about 7e-11 at the finest
polar cells). Ordinary collocated-wind adapters retain polar truncation error.
For tilted solid-body rotation, pole-inclusive last-ring cell-curl RMS error is
0.0129332, 0.00325277, 0.00115694 at 64, 256, 720 intervals: approximately first
order as the ring approaches the pole. These unit-sphere numbers scale as U/R
for a physical velocity scale U and radius R. They are not ERA5 errors in s^-1.
The campaign also records error introduced solely by dual-to-cell averaging.
Exact integral closure does not imply pointwise polar accuracy.

In the recorded ERA5 snapshot, each exact-pole row has constant local u/v values
about -0.00043/-0.00124 m/s. Adjacent-ring mean Cartesian wind speeds are
10.315 m/s north and 5.134 m/s south. This is a strong discontinuity relative to
that smooth extrapolation. Replacing only the pole rows with each adjacent
ring's Cartesian mean, projected into the pole's longitude-dependent local
basis, changes the combined ±89.75° VFS-minus-spherical-FD RMS as follows:

| Diagnostic | Original | Pole sensitivity variant |
| --- | ---: | ---: |
| Divergence (s^-1) | 5.22233e-5 | 2.18081e-6 |
| Cell curl (s^-1) | 7.84028e-5 | 3.93096e-6 |

This is one snapshot and a sensitivity intervention, not verified replacement
winds or a production correction. It identifies strong input-pole sensitivity;
it does not establish which discretization is closer to the real atmosphere.

## Analogous Hamiltonian

The same physical column energy can be integrated on either horizontal grid:

H = sum_c A_c E_column,c + 1/2 sum_e W_e I(m)_e u_e^2,

where W_e is face length times center distance. For a constant-density ocean,
m = rho0 h; for a pressure-coordinate atmosphere, m = dp/g. `face_kinetic`
implements this positive face-metric quadratic on the equiangular grid, including
collapsed polar faces. Its velocity derivative is checked independently with
finite differences; differentiating this diagnostic does not build a dycore.

The source calculation calls the actual `ConstantDensityEnergy.terms` on a
Voronoi ocean state. The target uses the analogous kinetic functional and the
same free-surface potential rho0 g eta²/2. This qualifies a mechanical-energy
comparison only. Active JMD95 dynamic enthalpy/reference thermal energy and
atmospheric internal/gravitational energy need matched column states, vertical
coordinates, thermodynamic variables and boundary terms before comparing full
Hamiltonians. A hydrostatic atmosphere's fixed-pressure lid reservoir must not
be silently dropped. No full active-ocean or atmosphere mapping is qualified.

A matching energy functional is reasonable for diagnostics. Energy-closing
dynamics would additionally need compatible adjoints, interpolation derivatives,
rotation and pressure-work operators; this experiment does not provide those.

## Mapping ledger

The existing `MeshToLatLonKNN(k=8)` is explicitly nonconservative. This experiment
maps thickness and tracer content hq separately, then forms q'=R(hq)/R(h).
Positive interpolation weights preserve concentration bounds and constant q,
but do not imply global inventory conservation. It maps mass-weighted Cartesian
reconstructed velocity, then projects into target local components and faces.
The reconstruction itself is not an energy isometry.

The ledger records source edge kinetic energy, reconstructed source-cell kinetic
energy, mapped energy-density integrals, target collocated/face kinetic energy,
free-surface energy, mass and tracer inventories. It separates

- transfer defect: integral of mapped source energy density minus source H;
- reconstruction defect: target H of mapped fields minus mapped energy integral.

Their sum is the total energy change. The second term includes smoothing,
projection and differences in discrete energy representation; it is not purely
an unresolved-turbulence estimate and is not deposited into heat.

The synthetic smooth-flow/sharp-tracer experiment uses 64/256/1024 Voronoi cells
(three Lloyd iterations), two target resolutions (16×32, 32×64, plus corresponding
pole-inclusive layouts). These are mapping probes, not production mesh studies.

| Source cells | Target energy change | Tracer inventory change |
| ---: | ---: | ---: |
| 64 | -10.51% to -10.33% | -0.163% to -0.096% |
| 256 | -3.96% to -3.82% | -0.326% to -0.018% |
| 1024 | -1.16% to -1.09% | -0.438% to +0.290% |

Mass changes range from -1.97e-6 to +1.42e-5 relative; constant tracer error is
below 3e-16 and concentration bounds hold to roundoff. Resolution does not
monotonically remove inventory error for a discontinuous tracer with this map.
These numerical ranges are case-specific, not general error bounds.

A conservative overlap mapper should transfer mass and tracer inventories using
cell-intersection areas, then recover concentrations. Energy density can be
transferred as a separate conserved diagnostic. Recomputed nonlinear energy
need not agree with that transferred energy, even for a conservative mapper;
the discrepancy must remain explicit. Implementing spherical overlap weights,
full thermodynamic mappings and their gradients remains future work.

## Validation limits

Tests run against an installed wheel outside the checkout, CPU float64. Offline
checks cover independent analytic integrals, exact face budgets, observed polar
convergence, the quadratic energy derivative and mapping bookkeeping. They do
not establish GPU behavior, remapper differentiability with respect to mesh
geometry, active thermodynamic energy conservation, or coupled dynamics.
