# Equiangular spherical grid

`vfs.geometry.EquiangularGrid(nlat, nlon, radius=6.371e6)` provides global
latitude–longitude geometry, area integrals and matched finite-volume operators.
It implements the `LatLonGrid` coordinate interface. It is not a replacement
mesh for the atmospheric or ocean dynamical cores. Dataset-layout support here
does not extend the supported layouts of the separate remapping helpers.

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
| Cell scalar | `(nlat, nlon)` | Input latitude order; eastward periodic longitude order |
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
The default constructor has no scalar samples at the poles. Neither layout
divides by cos(±90°).
Near-pole cells still have short east–west distances, affecting conditioning,
accuracy and any explicit diffusion timestep. No polar filtering is applied.

The grid supports the full sphere only, without land masks or regional
boundaries. Both centered and pole-sampled datasets are supported as described
below. Missing values are not filled or skipped. `LatLonGrid` remains available
for existing regridding layouts.

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
For the default centered layout, interior entry `(i,j)` is at latitude boundary i and longitude boundary j+1,
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
PV transport and geostrophic balance. The [compatible rotation option](COMPATIBLE_ROTATION.md) corrects the identified
Voronoi geometry and uniform-PV flux inconsistencies; wider PV and long-run
qualification remains separate. Equiangular curl is a diagnostic operator and
has no TRiSK rotation dependency.


## Dataset coordinates: centered and ERA5-style layouts

```python
grid = EquiangularGrid.from_coordinates(
    ds.latitude.values, ds.longitude.values, radius=6.371e6,
)
# Coordinates stay in the supplied order. For ordinary ERA5 files this is
# north-to-south latitude and 0, 0.25, ..., 359.75 degree longitude.
u = torch.as_tensor(ds.u10.transpose(..., "latitude", "longitude").values.copy(),
                    dtype=torch.float64)
v = torch.as_tensor(ds.v10.transpose(..., "latitude", "longitude").values.copy(),
                    dtype=torch.float64)
divergence = grid.div_from_collocated(u, v)
vorticity = grid.curl_from_collocated(u, v)
```

Both outputs retain the input horizontal shape and ordering. `grad` still
returns face fields; its components are always geographically eastward and
northward, including for descending latitude. The boundary at index i separates
input rows i−1 and i. `lat_bounds_deg` follows latitude order; `lon_bounds_deg`
is an unwrapped eastward sequence. No interpolation to another latitude grid,
latitude flip, seam roll or component sign change is required from the caller.

Accepted layouts:

- Uniform half-step latitude centers excluding both poles (e.g. −89.5 to 89.5).
- Uniform point latitudes including both poles (e.g. 90 to −90 in 0.25° steps).
- Either latitude order, any longitude offset, −180/180 or 0/360 conventions,
  and an eastward cyclic sequence whose seam occurs inside the array.
- Optional `lat_bounds` and `lon_bounds`, each with shape `(n+1,)` or `(n,2)`.
  These must match the contiguous midpoint control volumes in axis order;
  latitude endpoints are clipped to the poles. They are validated, not used to
  introduce arbitrary off-center/nonuniform cells. Wrapped longitude bounds
  are accepted modulo 360°.

Coordinates and bounds are degrees; validation tolerance is 1e−6 degrees.
Regional subsets, Gaussian/nonuniform latitude, descending longitude and a
repeated 0/360 endpoint are rejected. Remove a redundant periodic endpoint
explicitly from both coordinates and fields before constructing the grid.
Geometry is copied/detached to CPU float64. Field differentiation is supported;
coordinate differentiation is not. Use `.to()` to match tensor dtype/device.

[ECMWF describes regular-grid ERA5 values as point samples](https://confluence.ecmwf.int/pages/viewpage.action?pageId=226500583).
The diagnostic control volumes are midpoint quadrature regions around those
samples; their exact area weights do not turn point samples into exact cell
averages or recover conservation lost in an upstream interpolation. The radius
is explicit and defaults to 6,371,000 m; set it for the desired spherical
convention. No ellipsoidal geometry is inferred from metadata.

## Pole-sampled control volumes

A pole-sampled scalar row is one cap, stored as repeated longitude entries,
with one equal fraction of cap area per entry. Its outer boundary is halfway
between the pole and the first nonpolar row. `integral` uses these fractional
weights; `grad` first averages the duplicate polar scalar samples. `div`,
`laplacian` and cell-located `curl_from_collocated` return one cap-average value
repeated across the polar row. This is an explicit topology convention, not a
missing-data fill or a pointwise derivative at every polar longitude.

East faces inside that cap are collapsed and carry no flux. The northward
component on the cap boundary retains its longitude dependence. **Do not
average polar u/v components across longitude:** even a smooth physical vector
has longitude-dependent components in the local east/north basis at a pole.
The adapter averages each local component along its coordinate direction.

The dual cells adjacent to a sampled pole are triangular sectors. The two
endpoint rows of the native corner array have zero area and zero placeholders;
they are not estimates of polar vorticity. Use `curl_from_collocated(...,
location="cell")` (the default) for the cap value. `corner_integral` ignores
those zero-area placeholders. On the original centered layout, the endpoint
dual rows instead represent nonzero-area caps, as described above.

## Collocated diagnostic accuracy

`wind_to_faces` uses arithmetic component interpolation at coordinate
midpoints. `div_from_collocated` applies the matched divergence; default
`curl_from_collocated` additionally uses conservative dual-to-cell averaging.
These operations smooth the input. They preserve the discrete global flux and
circulation budgets, not kinetic energy or pointwise extrema.

`tools/qualify_equiangular.py` measures analytic degree-one axial/tilted and
degree-two vector fields on both latitude layouts at four resolutions. It
reports area-weighted RMS, interior maximum, endpoint maximum, and maximum
above 75° latitude for divergence, curl and linearly scaled stress curl.
Results and interpretation are in [the qualification report](EQUIANGULAR_VALIDATION.md).
The references are analytic point values: cap-averaged diagnostic outputs also
include a representation error relative to a polar point reference.

Run `python -I examples/equiangular_dataset.py` for a synthetic ERA5-style case,
or pass `--file data.nc --u u10 --v v10`. This explicit xarray boundary preserves
coordinate labels and checks component alignment. It loads selected arrays;
subset time/levels first for large files. No cloud access or data download is
performed. For stress, provide stress component names; the curl has stress-units
per metre. It is not Ekman pumping. Coastlines and NaNs still require an explicit
application-specific treatment; zero-filling land is not a general solution.
