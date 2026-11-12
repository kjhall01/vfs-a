# Equiangular dataset diagnostics: qualification

The coordinate-aware grid supports global regular half-step centers and global
pole-sampled coordinates, in either latitude order, with eastward cyclic
longitude. It retains input tensor ordering. The existing size-based constructor
remains available. The dycores and historical fixture tolerances are unchanged.

## Analytic vector fields

`tools/qualify_equiangular.py` uses a unit-radius sphere and three scalar spherical
harmonics: sin(latitude), cos(latitude) cos(longitude), and
sin(latitude) cos(latitude) cos(longitude). These have degrees 1, 1 and 2.
For each, let g = grad(q) and r = (g_north, -g_east). The measured field is
F = g + r, with exact divergence −l(l+1)q and exact radial curl +l(l+1)q.
The driver also measures divergence of r and curl of g separately, whose exact
answers are zero. Neither interpolation of analytic gradients nor the collocated
adapter is assumed to inherit the native discrete curl(grad) identity.

Stress is the same vector field scaled by 0.1, a linearity/units check rather
than an independent physical forcing experiment. Measurements cover 16/32/64/128
latitude intervals with twice as many longitude intervals. A pole-sampled layout
has one additional latitude sample. All coordinate arrays descend in latitude.

The table gives absolute dimensionless errors for the tilted degree-one mixed
field. Polar-band maximum includes all samples with |latitude| >= 75 degrees,
not just the repeated pole itself.

| Layout | Samples | Divergence RMS | Curl RMS | Polar-band curl maximum |
| --- | ---: | ---: | ---: | ---: |
| center | 16 × 32 | 0.0108587 | 0.0211032 | 0.128889 |
| center | 32 × 64 | 0.00290136 | 0.00602573 | 0.066727 |
| center | 64 × 128 | 0.000767956 | 0.00166804 | 0.0336514 |
| center | 128 × 256 | 0.000201998 | 0.000453372 | 0.0168618 |
| pole | 17 × 32 | 0.00929831 | 0.0202214 | 0.0966045 |
| pole | 33 × 64 | 0.00253323 | 0.00583901 | 0.0519449 |
| pole | 65 × 128 | 0.000681304 | 0.00162632 | 0.0264344 |
| pole | 129 × 256 | 0.00018151 | 0.000443815 | 0.0132752 |
RMS errors decrease on both layouts. Near-pole maximum errors for tilted fields
are approximately first order; an exact zero at a sampled pole is not evidence
that the neighboring ring is accurate. The references are point values while
polar diagnostic values are cap averages. Do not interpret the small area-
weighted error as uniformly second-order pointwise accuracy or as real-data
validation. No higher-order reconstruction or polar filter is applied.

[Full measurements](../verification/equiangular/coordinates-results.json) include
all modes, component leakage, interior/polar maxima and global integrals, plus
runtime/driver hashes. Reproduce with an installed wheel outside the checkout:

```bash
python -I tools/qualify_equiangular.py --output coordinates-results.json
python -I examples/equiangular_dataset.py
python -I -m pytest tests/test_equiangular.py tests/test_equiangular_coordinates.py -q
```

## Structural and dataset checks

Tests cover centered/pole layouts and both latitude orders; periodic seams and
longitude rolls; area totals; grad/div adjointness; global divergence and curl
integrals; native curl(grad) circulation cancellation; conservative dual-to-cell
averaging; analytic solid-body rotation on polar sectors; radius scaling;
field-value finite-difference/autograd agreement; xarray dimension/coordinate
alignment; copied read-only/reversed NumPy coordinates; and invalid coordinates
or inconsistent bounds. A synthetic 721 x 1440 coordinate layout checks area and
constant-field Laplacian at ERA5's common 0.25-degree sampling. It is not a test
on downloaded ERA5 meteorology.

CPU float64 is the measured scope. GPU/compile performance, masked coastlines,
regional boundaries, arbitrary cell bounds, ellipsoidal metrics and derivative
accuracy on noisy observational products remain outside this qualification.
The existing portable fixtures are retained, without changed tolerances.

The optional [WeatherBench2/local-data campaign](../validation/equiangular_wb2/README.md)
adds real-data comparisons, regional budgets, coordinate invariance, and
common-scale diagnostics. It does not add package or ordinary-test network access.
