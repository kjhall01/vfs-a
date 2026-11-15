# AM2–VFS energy transfer

The first adapter applied AM2 temperature tendencies directly to native CP
temperatures. Even with conservative species amounts and a work-adjoint wind
map, that does not transfer the same energy: AM2 uses constant `CP_AIR` and latent
heats, VFS uses mixture enthalpy, mass sources change the hybrid mass metric, and
CP entropy interpolation changes full-level enthalpy. The initial 0.4–0.6 W/m²
six-hour comparison also omitted the sensible enthalpy and kinetic energy carried
by precipitation and evaporation.

The upstream contracts used here are the pinned
[AM2 staged driver](https://github.com/evanwellmeyer/PyAM21/blob/e1bdfcc3b23c05adea80b35af880702bb85718eb/src/pyam21/physics/am2.py),
[moist-process endpoints](https://github.com/evanwellmeyer/PyAM21/blob/e1bdfcc3b23c05adea80b35af880702bb85718eb/src/pyam21/physics/moist.py),
and [vertical-diffusion energy return](https://github.com/evanwellmeyer/PyAM21/blob/e1bdfcc3b23c05adea80b35af880702bb85718eb/src/pyam21/physics/vert_diff.py).

The conservative mode instead treats **energy as the transferred quantity**.
The package thermodynamics and upstream physics remain unchanged. This is an
explicit alternative host update, not a claim of identical temperature evolution
to the original AM2 SCM.

## Independent boundary ledger

For a stationary prescribed ocean, the atmospheric energy input over one interval is

\[
B = \Delta t\,[Q_s+E h_v(T_E)-P_l h_l(T_l)-P_i h_i(T_i)
 +(E-P_l-P_i)\Phi_s
 +\tfrac12 E|\mathbf v_E|^2-\tfrac12(P_l+P_i)|\mathbf v_P|^2
 +F_{\rm net,top}-F_{\rm net,bottom}].
\]

Vapor flux E is signed into the atmosphere; precipitation is positive out.
The host species enthalpies are `h_v = Lv_ref + cp_v*(T-T_ref)`,
`h_l = cp_l*(T-T_ref)`, and `h_i = -Lf_ref + cp_i*(T-T_ref)`.
The stress-work term is zero because the imposed surface is stationary. AM2's
configured dissipative heating returns atmospheric drag losses to heat; disabling
that option produces an upstream energy deficit which remains visible.

`WaterBoundary` declares the three temperatures and two Cartesian tangential
velocities. Defaults use SST and zero velocity for incoming vapor, and the old
lowest air temperature/velocity for precipitation and departing vapor. These
temperatures are boundary assumptions, not quantities diagnosed by AM2. Different
rain/snow temperatures and water velocities can be passed explicitly. The boundary
object is an input to each call; replay must supply the same overrides.

Water momentum is `dt*(E*v_E - (P_l+P_i)*v_P)`, placed at the bottom layer and
mapped by the existing cell-vector transpose. Native edge momentum obeys
`M1*u1 = M0*u0 + I_physics + I_water`. The altered mass metric therefore does not
silently add momentum by making new water co-moving throughout the column.
Global Cartesian angular momentum is not asserted by this generalized work map.

## Preserve the upstream energy residual

With old layer masses `m = dp/g`, the actual AM2 outputs define

\[
\Delta A_k=m_k\{c_{p,A}\Delta T_k+L_{v,A}\Delta q_{v,k}
 -L_{f,A}\Delta q_{i,k}
 +\mathbf v_k\cdot\Delta\mathbf v_k+\tfrac12|\Delta\mathbf v_k|^2\}.
\]

Its independently declared boundary energy is
`B_AM2 = dt*(Q_s + Lv_AM2*E + Lf_AM2*P_i + net absorbed radiation)`.
The upstream residual is `r_AM2 = sum(ΔA_k) - B_AM2`. Radiation uses the net
top/bottom fluxes; a separate diagnostic compares those with integrated heating.

The native layer-energy target is
`ΔH_target,k = ΔA_k + bottom_k*(B - B_AM2)`.
Thus, algebraically,

\[
\sum_k\Delta H_{\rm target,k}=B+r_{\rm AM2}.
\]

The residual is **not subtracted**. Deliberately injected undeclared heating and
actual upstream diffusion with dissipative heating disabled remain energy defects
after transfer. Conversely, closed phase changes or energy-returning drag produce
zero net native full-H change when AM2's own ledger closes. Tests cover both.

For temperature and tracers, conservative mode uses the actual endpoint returned
by PyAM21's moist stage, checked against the tendency reconstruction at a
128-epsilon scale. Available total water bounds the comparison scale for intermediate
phase exchanges. This avoids a subtraction roundoff negative when the upstream
state has exhausted a condensate exactly. Source amounts are `dp*(q1-q0)`.
Remaining amounts are mapped with normalized nonnegative donor weights; signed
upstream endpoint roundoff is never clipped. The original temperature/tendency
mapping remains unchanged in legacy mode. The mass-source placement still follows
the declared hybrid-column convention, not a resolved vertical mass-source remap.

## Solve the native enthalpy constraints

VFS's hydrostatic full Hamiltonian, including its fixed-pressure lid, has layer
ownership `H_k = dp_k/g * (mean_Gauss(h) + KE_native,k + phi_s)`.
After water, pressure and momentum have been updated, their contributions are
known. The solver determines CP entropy so each layer attains its prescribed
energy increment. It does not modify the water or momentum result.

There are L layer constraints and L+1 CP unknowns. The initial guess is the previous
temperature-increment map. Each Newton step chooses the minimum Euclidean-norm CP
log-temperature increment. At fixed pressure/composition, CP entropy changes by
`delta S_i = mass_i * cp_i * delta log(T_i)`. Native interpolation gives the
analytic Jacobian with only two nonzero entries per layer:

\[
J_{k,k}=\frac{\Delta p_k}{g}c_{p,k}\langle(1-x)T_g\rangle,
\qquad
J_{k,k+1}=\frac{\Delta p_k}{g}c_{p,k+1}\langle xT_g\rangle.
\]

The row-scaled least-norm solve is `delta = -A.T @ solve(A @ A.T, residual/scale)`.
An explicit bounded step size limits each trial log-temperature increment to 0.25;
it is a nonlinear-solver safeguard, not clipping of prognostic fields. The solve
rejects nonfinite states or failure to converge within 12 iterations. Residuals
use stable finite enthalpy differences and a 128-epsilon layer thermal-energy
scale. No forward autograd or cross-column/global correction is used.

The underdetermined vertical allocation is a stated numerical choice. It has not
been established as entropy-stable, optimal for cloud physics, or accurate on long
climate trajectories. Selected smooth derivative checks include a zero-valued
source, for which two Newton maps retain the constraint's implicit derivative.

## Scope and next interface

CPU float64 tests and short radiation-free integrations qualify the transfer and
its budgets. They do not establish full radiation or climate skill. The legacy
mode is retained for reproducibility, and restart fingerprints distinguish modes.
Saturation, convection and microphysical rates still use AM2's constants and
parameterizations. Conserving the host energy does not establish that those
closures are in thermodynamic equilibrium with the host mixture or retune them.

Interactive ocean/ice coupling still needs a common mass/enthalpy datum with the
ocean's `cp*CT` and ice references, identical exchanged water states on both sides,
reciprocal stress and dissipation allocation for moving surfaces, and one shared
time-integrated flux record. This pass prepares an explicit atmospheric boundary;
it does not claim that simply wiring two existing ledgers together conserves the
combined system. Real SeaEsfRad and its input-data manifest remain untested here.


The standalone contract above is extended by the optional
[shared exchange workflow](../coupled/README.md): one integrated record carries
all tile fluxes to both components, and recipient midpoint stress work replaces
zero stationary-surface work. Both sides consume the same cached net radiation.
AM2's energy defect remains explicit. No package source is changed.
