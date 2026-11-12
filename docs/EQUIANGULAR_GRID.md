# Equiangular spherical grid

`vfs.geometry.EquiangularGrid(nlat, nlon, radius=6.371e6)` provides global
latitude–longitude geometry, area integrals and matched finite-volume operators.
It reuses `LatLonGrid` coordinates and areas and can be passed to its remapping
helpers. It is not a replacement mesh for the atmospheric or ocean dynamical cores.

```python
import torch
from vfs.geometry import EquiangularGrid

grid = EquiangularGrid(90, 180)  # CPU float64; cell centers exclude the poles
lat, lon = map(torch.deg2rad, grid.meshgrid())
q = lat.cos() * lon.cos()
east, north = grid.grad(q)
laplacian = grid.div(east, north)  # Also grid.laplacian(q)
area_mean = grid.mean(q)
```

## Placement and units

All methods accept tensors with arbitrary leading axes and horizontal axes last.
Fields must match the grid's dtype/device; move the grid explicitly with `.to()`.

| Quantity | Trailing shape | Convention |
| --- | --- | --- |
| Cell scalar | `(nlat, nlon)` | South to north, longitude increasing from 0° |
| East face component | `(nlat, nlon)` | Index j is the face between cells j and j+1, periodic |
| North face component | `(nlat+1, nlon)` | Interior index i lies between rows i−1 and i; endpoints at poles |
| Native curl | `(nlat+1, nlon)` | Interior latitude boundary i, longitude boundary j+1; repeated cap values at poles |

`grad(q)` returns normal derivatives in q-units/m. `div(east, north)` accepts
normal flux densities, multiplies by face lengths and divides net transport by
cell area. Do not pass already integrated face transports. `laplacian` is a
scalar operator; applying it separately to east/north vector components is not
the spherical vector Laplacian.

`integral(q)` sums q times exact spherical cell area over the horizontal axes;
`mean(q)` divides by total area. Column mass in kg/m² yields kilograms; column
energy in J/m² yields joules. Three-dimensional density needs vertical integration
first. These methods do not infer thermodynamics, masks or layer weights.

## Conservative identities

Shared face transports cancel globally. With cell-area inner product and face
weights equal to face length times center distance, the operators satisfy

\[
\langle q, D F\rangle_A=-\langle Gq,F\rangle_W,
\qquad
\langle q,DGq\rangle_A=-\langle Gq,Gq\rangle_W.
\]

`face_inner(east_a, north_a, east_b, north_b)` evaluates the weighted face product.
These identities hold to floating-point roundoff and make the scalar Laplacian
negative semidefinite. They do not alone establish conservation of a time-stepping
model or kinetic energy in an interpolated dataset.

## Collocated winds and poles

`grid.wind_to_faces(u, v)` averages neighboring cell-centered eastward/northward
components onto faces. Then `grid.div(*grid.wind_to_faces(u, v))` gives a
conservative diagnostic divergence. Interpolation is explicit; it is not
energy-preserving reconstruction or conservative transfer between grids.

Polar boundary faces have exactly zero length. Gradient and wind adapters return
zero placeholders there; finite user-supplied polar values also have zero flux.
There are no scalar samples at the poles and no division by cos(±90°).
Near-pole cells still have short east–west distances, affecting conditioning,
accuracy and any explicit diffusion timestep. No polar filtering is applied.

The first implementation supports the full sphere only, without land masks or
regional boundaries. Pole-centered datasets (including repeated longitudes at
a pole) require an explicit conversion to this cell-centered layout. Missing
values are not filled or skipped. `LatLonGrid` remains available for existing
regridding layouts.

Run `python -I examples/equiangular.py` for integral, adjoint and refinement
measurements, including polar errors. Tests cover CPU float64 algebra, spherical
harmonics and field-value gradients; they do not qualify GPU execution or complete
model rollouts. Geometry is prescribed rather than differentiable.

## Measured refinement

For q = cos(latitude) cos(longitude), the exact scalar Laplacian is −2q/R².
The example reports error multiplied by R², making these values dimensionless:

| Grid | Area-weighted RMS error | Maximum error in polar rows |
| --- | ---: | ---: |
| 16 × 32 | 0.00652093 | 0.0475606 |
| 32 × 64 | 0.00178220 | 0.0243518 |
| 64 × 128 | 0.000479829 | 0.0122478 |
| 128 × 256 | 0.000127905 | 0.00613292 |

This case approaches second-order area-weighted convergence, but polar-row
maximum error decreases only at approximately first order. Conservative identities
are not a guarantee of uniformly second-order pointwise accuracy near the poles.

## Curl and wind-stress curl

`curl(east, north)` computes outward circulation per dual-cell area from the
same face layout as `grad`. Its result has shape `(..., nlat+1, nlon)`.
Interior entry `(i,j)` is at latitude boundary i and longitude boundary j+1,
modulo 360°. Both polar rows repeat a **single cap-average value**: there are
not nlon independent polar points. `corner_area` divides each cap's area equally
among its repeated entries; use `corner_integral` for its area integral.

For a scalar q, `curl(*grad(q))` vanishes to roundoff because endpoint
differences telescope around closed loops. The full-sphere curl integral also
vanishes to roundoff. These are native discrete identities; interpolating an
independently evaluated analytical gradient does not promise exact cancellation.

For collocated stress arrays with shape `(..., nlat, nlon)`:

```python
# Continuing the grid/coordinates from the example above:
tau_east = 0.1 * lat.cos()  # N/m²
tau_north = torch.zeros_like(tau_east)
stress_curl = grid.curl_from_collocated(tau_east, tau_north)  # N/m³, cells
corner_curl = grid.curl_from_collocated(tau_east, tau_north, location="corner")
```

The helper interpolates onto faces before taking curl. Its default cell result
uses `corners_to_cells`, an area-overlap average of piecewise-constant dual-cell
values. This preserves the area integral and smooths the field; it is not an
exact pointwise curl. Wind input in m/s instead gives relative vorticity in s⁻¹.
Wind-stress curl is not Ekman pumping: the latter additionally involves density
and Coriolis, including their spatial variation where applicable.

The native solid-body rotation test matches dual-area-averaged analytic curl,
including both polar caps, to roundoff. Tilted rotation and collocated-wind
calculations have separate refinement checks. The pole caps use the circulation
around the first/last center latitude, never a singular polar derivative.

## Curl and dynamical rotation

The existing Voronoi mesh also computes circulation on dual cells. Direct tests
check `curl(grad(q))`, global circulation cancellation and the curl adjoint on
16, 64 and 256-cell meshes, including relaxed and unrelaxed configurations.
These checks do not require accurate TRiSK tangential reconstruction.

Atmospheric and ocean rotation use that vertex curl to construct absolute
vorticity. Their energy-work pairing and PV compatibility are distinct: a
weighted skew rotation has zero kinetic-energy work, but that alone does not
establish the discrete vorticity/continuity relationship needed for consistent
PV transport and geostrophic balance. The existing
[rotation investigation](../verification/ocean/ROTATING_BALANCE.md) records those
unresolved consistency limits. No rotation operator is changed here. Equiangular
curl is a diagnostic operator and has no TRiSK rotation dependency.
