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
