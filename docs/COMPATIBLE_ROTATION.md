# Compatible horizontal rotation

Create a Voronoi mesh with `rotation_scheme="compatible"` to use matched
curl/divergence geometry and PV-times-carrier-flux rotation:

```python
from vfs import VoronoiMesh

mesh = VoronoiMesh(n_cells=1024, lloyd_iters=20,
                   rotation_scheme="compatible")
```

The same mesh can be supplied to `DryAtmosphere`, `MoistAtmosphere`, or the
constant-density/JMD95 ocean dynamics, including `StratifiedOcean`. No atmosphere
code is imported by the ocean. `examples/compatible_rotation.py` initializes and
advances a small dry JW case using the installed API.

The default `rotation_scheme="legacy"` retains the historical numerical path.
Select the scheme when constructing geometry; changing the string afterward
would not rebuild its buffers and is unsupported. Existing fixtures and their
tolerances remain unchanged. Native mesh save/load retains the scheme; a module
state dictionary requires reconstructing matching mesh configuration first.
The first implementation is exposed on the Voronoi constructor; cached legacy
meshes must be regenerated from their generators to use compatible geometry.

## Geometry and equations

Corner areas use actual primal/dual intersections, with the vertex association
matching the oriented edge path. The cell-to-vertex area transfer P preserves
constant fields and integrated carrier mass. For rotation J, curl C and
divergence D, the geometric relation is CJ=PD to floating-point geometry error.
The unrelated tangential velocity/kinetic reconstruction is unchanged.

For carrier h, edge interpolation I and outward relative vorticity C u:

\[
F=I(h)u,\qquad q=(Cu+f)/P(h),\qquad
Q_{ij}=\tfrac12(q_i+q_j)J_{ij},\qquad \dot u_{rot}=-QF.
\]

Vertex q is averaged to edges before the symmetric edge-pair average. The
atmosphere uses layer pressure thickness in Pa as carrier; the ocean uses layer
thickness in metres. Momentum acceleration remains in m/s². With edge-area
matrix W, J is W-skew and so is Q. Thus

\[
\langle u,\dot u_{rot}\rangle_{W I(h)}=-F^T W QF=0.
\]

This is the same kinetic-energy metric as the existing energy functional, up
to the fixed gravity/reference-density factors. The Hamiltonian, pressure work,
scalar transport, vertical exchange, remapping and time integrators do not change.
For constant q, the geometric identity gives C udot = q P hdot under horizontal
continuity. Using absolute vorticity times velocity with a thickness-dependent
skew projection does not, in general, have that property.

## Scope

The corrected scheme requires nonnegative signed primal/dual kite areas and
finite positive dual carrier thickness; invalid geometry/states reject explicitly.
It does not clip corner weights or carrier mass. The geometric transfer and
rotation are differentiable with respect to state tensors; geometry is prescribed.

Validation covers CPU float64, metric-skew work, horizontal constant-PV
preservation, derivatives, dry/moist IMEX, moist explicit stepping/work, active-ocean
steps/remapping, and coast masking. Coastal tests establish closed normal flux
and energy masking, not a coastal PV theorem: dual masses near land include the
finite dummy cell values retained by the existing domain convention.

Horizontal constant-PV preservation is not a proof of full three-dimensional
atmospheric PV conservation, general potential-enstrophy conservation, a Poisson
Jacobi identity, or long-run stability. Baroclinic and vertical terms remain part
of those broader budgets. Generic legacy `Ocean` reference equations have their
own energy/transport formulation and are not qualified by these checks.
GPU, compiled execution and long backward trajectories remain unqualified.

See [correction measurements](../verification/ocean/ROTATION_CORRECTION.md) and
[the preceding diagnostic](../verification/ocean/ROTATION_PV_AUDIT.md).
