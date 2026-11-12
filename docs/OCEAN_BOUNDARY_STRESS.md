# Ocean boundary stress

`OceanBoundaryStress` adds prescribed surface stress and optional stationary
horizontal quadratic bottom drag over varying bathymetry to `TKEOcean`.
Stress is an edge-normal tensor
`[nEdges]` in Pa, with the state's dtype/device. It is a force on the ocean;
a wind speed or an atmospheric bulk drag law is not implied.

```python
from vfs.ocean.stratified import TKEOcean, TKEClosure, OceanBoundaryStress

model = TKEOcean(ocean, closure=TKEClosure(mixing_length=2.),
    boundary=OceanBoundaryStress(bottom_drag_coefficient=0.0025))
state = model.state(resolved_state, tke=k)
state, report = model.step_with_report(state, dt, surface_stress=tau_edge)
print(report.boundary.surface_work)  # external work, J
print(report.boundary.bottom_ke_loss.sum())  # internal KE -> TKE, J
```

`boundary=None` is the unchanged default. Supplying stress without a boundary
object raises. `OceanBoundaryStress()` enables surface forcing with zero bottom
drag; omitting stress means zero surface stress. Inactive coast edges receive
neither stress nor drag. No mass, salt or tracer flux is introduced.
`boundary_with_report` runs boundary physics alone. The full step order is
resolved dynamics, optional ALE, boundary stress, then TKE mixing/dissipation.
This splitting is first order. See the [runnable example](../examples/ocean_boundary_stress.py); pass
`--bathymetry` for a smooth varying-depth case.

## Momentum and energy

With edge area A=dc*dv, interpolated thickness H=I(h), and native kinetic
metric M=rho0*A*H, the top-layer update is

\[
u_t^*=u_t+\Delta t\,\tau_e/(\rho_0 H_t),\qquad
W_e=\Delta t A_e\tau_e (u_t+u_t^*)/2.
\]

W is exact finite-step work on the discrete resolved kinetic energy. It may
be negative when stress opposes the current. It is an external contribution
to the ocean energy budget, not heat or an extra TKE source.

The bottom-layer update follows the wind update, also for a single layer:

\[
u_b'=\frac{u_b^*}{1+\Delta t C_D s_b/H_b},\qquad
D_e=\tfrac12 M_b[(u_b^*)^2-(u_b')^2]\ge0.
\]

Speed s_b is frozen at entry to this substep. It combines the native edge-normal
velocity with a tangential component reconstructed from cell Cartesian winds.
The reconstruction estimates full horizontal speed; it is not an energy
isometry. Energy accounting always uses the original native kinetic metric.
The update cannot reverse the velocity through drag alone and remains
first-order for general coupled flow.

Every D_e is deposited in bottom-layer TKE. A cell receives fraction
w_ce*h_cb/I(h_b)_e, where w_ce is the existing thickness interpolation weight.
This matches the Hamiltonian's cell kinetic-energy allocation even for unequal
neighbour thicknesses. The boundary substep therefore satisfies

\[
\Delta(H_{\rm ocean}+E_{\rm TKE})=\sum_e W_e
\]

to floating-point roundoff. All finite-step drag loss, including numerical
damping, is retained in ocean TKE. There is no seabed heat reservoir. Subsequent
TKE dissipation heats water through the existing thermodynamic energy solve;
drag loss is not added to heat a second time.

`BoundaryStressReport` contains surface/bottom native edge impulses (N s),
edge/summed surface work, edge bottom KE loss, cell-layer TKE gain, vertically
integrated edge momentum residuals, and cell/global boundary energy residuals.
Native edge impulses are diagnostic covectors, not Cartesian total momentum.
Full-step dynamics drift and mixing inversion error remain separate from the
boundary ledger. Boundary work closes against kinetic/TKE scales without
subtracting the large thermal background.

## Scope

- Bathymetry may vary; column and optional domain depths must agree in wet
  cells. Drag acts on the local lowest layer using its interpolated thickness.
  Stress and work are per horizontal area. There is no seabed-normal projection
  or sloping-area multiplier; this is a horizontal hydrostatic drag closure.
  Explicit slope-aware traction and unresolved topographic form drag are deferred.
  The lowest layer's current is not forced to zero: drag represents the unresolved
  velocity gradient to the stationary seabed.
- Cd is a prescribed nonnegative constant, not a roughness or stability model.
  The example's 0.0025 is a starting choice, not a VFS calibration.
- No separate surface TKE flux, wave breaking, Langmuir production, atmospheric
  coupling or horizontal EKE closure. Wind stress acts on resolved currents.
  Exactly zero TKE still needs a source/seed for the existing shear closure to
  activate; bottom drag on a nonzero current can supply bottom TKE. The example
  explicitly initializes a small TKE seed.
- No claim of realistic Ekman-layer depth or entrainment with the prescribed
  mixing length. Small CPU float64 runs and smooth derivatives are checked;
  GPU, long-rollout, float32 and coastal reconstruction accuracy are unqualified.

[Qualification](../verification/ocean/BOUNDARY_STRESS.md) distinguishes independent
constant-viscosity column references from spherical evolving-TKE smoke tests.

[Bathymetry qualification](../verification/ocean/BATHYMETRIC_DRAG.md) records
along/across-slope reconstruction refinement, rest and coast checks.
