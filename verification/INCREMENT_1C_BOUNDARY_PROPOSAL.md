# Increment 1c: prescribed-transport boundary proposal

**Status: proposal, not a completed C5 experiment. No runtime extraction has been
implemented.** The user's instruction requires presenting a separate proposal
before changing a runtime interface to isolate transport. That condition applies
here; neither a copied RK scheme nor a stage-mutating monkey patch is used.

## Source and dependency boundary

PRs [#13](https://github.com/kjhall01/vfs-a/pull/13) and
[#14](https://github.com/kjhall01/vfs-a/pull/14) are both merged into main.
Their merge commits are respectively `7a1c99e30aa139637bd0dc380af04b9d724bf6a3`
and `ba241cdcae1fb98f01231f632bacb16fbedb5e90` (confirmed through GitHub API).
This proposal starts from the latter; no dependency branch is needed.
The numerical baseline stays **8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648**.
The complete `vfs/` source tree still matches that baseline.

## Actual production trace

| Operation | Existing function | Prescribed-flow consequence |
|---|---|---|
| Scalar/state validation and interface thermodynamics | `MoistConservativeDynamics.transport_state`, `temperature_from_entropy` | Positive vapor and dry air remain necessary; do not reset entropy or prescribe temperature after initialization. |
| Pressure thickness and carrier continuity | `DryHydrostaticConservativeDynamics._scalar_transport_primitives` through the moist kernel | Computes edge mass flux `u_e*cell_to_edge(dp)`, its divergence, dps/dt, hybrid-coordinate mu and omega. No arbitrary supplied-flux argument at this boundary. |
| CP face graph | `bounded_transport.graph` | Horizontal layer flux is averaged to interfaces; vertical omega links follow it. A supplied state velocity reaches this code unchanged. |
| Candidate face reconstruction | `BoundedGibbs._gibbs_corrections` → `gibbs_face` | Temperature, pressure and composition determine Gibbs means. Returns Gibbs-minus-arithmetic traces; replacing these with invented centered candidates would change the method under test. |
| Actual high-order transfer and limiter | `BoundedGibbs.evaluate` → `limited_traces` → `bounded_transfer` | Builds full candidate species flux, complementary dry flux, actual continuity target, donor reference/subcycles and common all-species limiter. |
| Entropy/heat and mechanical work | remainder of `evaluate` | Retain the production entropy/heat update; momentum forces are computed, but their evolved wind must not feed the prescribed-flow experiment. No coupled energy claim follows. |
| Euler species installation | `euler_with_info` | Installs water from authoritative amounts/area. Never replace with state + dt*diagnosed water RHS. |
| Three Euler evaluations and convex combinations | `step_with_info` | Calls the bound Euler method at all three stages, without time or a stage-velocity interface. This is the missing boundary. |

Source: [bounded moist](../vfs/atmosphere/bounded_moist.py),
[transfer](../vfs/atmosphere/bounded_transport.py),
[Gibbs means](../vfs/atmosphere/gibbs_means.py),
[shared carrier primitives](../vfs/atmosphere/conservative.py).

The generic [dry SSPRK3](../vfs/core/integrators.py) advances an RHS and uses a
different arithmetic form for combinations. It cannot replace the bounded
Euler ledger. Replacing velocity only once per outer step is also insufficient
for a time-dependent flow. The 1b observer only records stages; turning it into
a state-mutating patch would hide a new stepping interface in a measurement tool.

## Recommended smallest extraction — separate prerequisite PR

Extract **only the existing bounded SSPRK3 sequence** into one ordinary function
in `bounded_moist.py`, provisionally:

```python
bounded_ssprk3_with_info(euler_with_info, state, time, dt)
# callback signature: euler_with_info(state, stage_time, dt) -> (state, info)
```

The three callback times are `time`, `time + dt`, `time + dt/2`; every call gets
positive full `dt`. Move the current statements and their convex-combination
arithmetic without rewriting them. Keep final physical validation in the
existing `BoundedGibbs.step_with_info` wrapper, which delegates with a time-ignoring
adapter to its unchanged Euler method. Existing public constructors, stepper
instances, return values and methods retain their behavior. No registry,
inheritance layer, new transport algorithm or alternative RK implementation.

The verification driver can then supply a plain Euler callback that:

1. Constructs a borrowed-geometry state view with the current prognostic
   ps/entropy/water and the analytical edge-normal velocity at `stage_time`.
2. Calls the **unchanged** `transport.euler_with_info(view, dt)`.
3. Retains the returned pressure, entropy, species and actual ledger. Returns
   the original Euler result for the existing combination arithmetic. Wind
   entries in those combinations are overwritten by the next analytical stage
   velocity before evaluation and never affect transport. Assign final output
   wind at `time + dt` for reporting.

This exercises the complete production carrier/reconstruction/limiter/Euler
path, including entropy heat. It bypasses **feedback from computed momentum
advancement**, not reconstruction. The current code may still calculate unused
momentum forces; removing that work would be an unrelated optimization.
No raw-flux injection or reconstruction extraction is needed for this first C5
boundary. No prognostic carrier, entropy or species is reset.

Acceptance for the extraction is same-environment bitwise comparison of coupled
states, RHS, all three ledgers, component work and representative gradients, plus
existing installed-wheel portable checks. New tests must verify stage times and
positive full-dt budgets, and detect a deliberately wrong second/third stage time.
Do not change formulas to improve a transport result.

**Provenance consequence:** this changes runtime source text, even if all numerical
expressions are unchanged. The present manifest check correctly rejects it.
Keep the original numerical baseline identifier and its original hashes. Before
using an extracted wheel, separately record the approved extraction commit,
explicit changed-file hashes and equivalence evidence. Do not silently repin the
baseline, disable source checking, or claim that a changed wheel is byte-identical.
The current proposal changes neither the runtime nor the verifier.

## Benchmark specification to implement after the boundary is approved

Read original sources:

* Williamson et al., *A standard test set for numerical approximations to the
  shallow water equations in spherical geometry*, original
  [ORNL/TM-11895 report, §3.1, equations 75–79](https://www.osti.gov/servlets/purl/5232139),
  subsequently JCP 102 (1992), 211–224,
  [DOI](https://doi.org/10.1016/S0021-9991(05)80016-6).
* Nair & Lauritzen (2010), *A class of deformational flow test cases for linear
  transport problems on the sphere*,
  [author manuscript, §3.1.2 and §3.2.3, equations 14–16 and 28–33](https://www.image.ucar.edu/staff/rnair/NairLauritzen_BM_JCP2010.pdf),
  [DOI](https://doi.org/10.1016/j.jcp.2010.08.014).

Use longitude lambda eastward and latitude theta northward (radians), unit
position r=(cos(theta)cos(lambda),cos(theta)sin(lambda),sin(theta)), physical
radius R=6371220 m. Velocities u/v are local east/north components in m/s.

Rotation: U=2*pi*R/T, T=12 days, u=U(cos(theta)cos(alpha)+sin(theta)cos(lambda)sin(alpha)),
v=-U*sin(lambda)sin(alpha). Equivalently v_cart=(2*pi*R/T)*a×r,
a=(-sin(alpha),0,cos(alpha)). Initial planned runs: alpha=0 (equatorial track)
and pi/4 (tilted). The latter is an explicitly chosen variant, not one of the
original report's four prescribed orientation values. Use the exact inverse
Rodrigues rotation of evaluation points for the reference at any time.

Select Nair–Lauritzen **Case 4**, including its zonal background, rather than
reversing dt. In original dimensionless units R*=1, T*=5, k=2, lambda'=lambda−2*pi*t*/5:

```
u* = 2*sin(lambda')**2*sin(2*theta)*cos(pi*t*/5) + (2*pi/5)*cos(theta)
v* = 2*sin(2*lambda')*cos(theta)*cos(pi*t*/5)
psi* = 2*sin(lambda')**2*cos(theta)**2*cos(pi*t*/5) - (2*pi/5)*sin(theta)
```

For the chosen physical period T=12 days, set t*=5*t/T and multiply velocity
by 5*R/T, streamfunction by 5*R²/T. The flow has an exact return at T; t=T/2
is maximum deformation, not zero total velocity. Rotation of coordinates plus
positive-time flow reversal must both be retained. Intermediate references need
independent characteristic integration with a tolerance-refinement audit.

Smooth profile: double Gaussian sum f=sum_i exp(-5*|r-r_i|²), centers at
(lambda,theta)=(5*pi/6,0),(7*pi/6,0). Adapting that profile to multispecies
composition is a VFS choice: propose q_v=.01+.001*f, q_l=.001*f,
q_i=.0001*f; q_d=1−sum(water). Vapor background is part of the IC, not a floor.
Add a compact cosine-bell condensate case with exactly empty regions and a
constant-composition control. These are not original moist benchmark ICs.

## Carrier, references and measurements still to implement

Use one full layer first, repeated interface composition, flat terrain and
positive lid. This minimizes vertical exchange; still record actual vertical
fluxes and verify they vanish to roundoff, never delete them by assumption.
Sample analytical velocity at the documented edge location and project onto
`edge_normal`. Measure discrete `mesh.div(u_e)` and the **actual** mass-flux
divergence separately. Continuum nondivergence is not a discrete guarantee.
Evolve ps and species together through every Euler stage and convex combination.
The constant-composition test is mandatory even when carrier changes.

Start without projection. A later vertex-streamfunction flux construction would
change the sampled velocity: document orientation, length weights and the
velocity difference, audit its incidence cancellation, and report it as a distinct
configuration. Do not introduce it silently to make constant carrier appear exact.

Initialize/reference **cell averages**, independently integrating the smooth
profile over spherical Voronoi polygon triangles. Refine quadrature at least
three levels and archive differences and sphere-area checks alongside transport
errors. Use inverse rotation at quadrature points, and initial averages at a full
return. At maximum deformation, use independently refined backward trajectories;
label trajectory and quadrature error separately. Never call an approximate
intermediate reference exact. Sharp support needs its own quadrature-error study.

For each species e=q−q_ref report L1=sum(w*|e|)/sum(w),
L2=sqrt(sum(w*e²)/sum(w)), Linf=max|e|, with w=area and separately carrier mass;
errors are kg/kg. Record initial/half-period/final arrays, inventories in kg,
stage bounds and remaining dry fraction, peak loss, meaningful centroid/phase
shifts, mass-weighted variance/mixing, constant-composition error, limiter masks,
actual node outgoing dt*carrier / min(endpoint carrier), donor subcycles,
wall time and artifact bytes. Preserve failures and negative results.

Reuse 1a/1b manifests, archives and failure/output protection, the exact water
ledger check and independent small-case node-by-node face audit. Do not apply
that all-face scan per node at large resolution. A scalable segmented audit must
first agree with the small oracle; optimization is not this increment's purpose.

Planned development matrix: 16/64/256 cells, positive T/120 and T/240 steps,
rotation at both orientations and Case 4; align half-period outputs exactly.
Check actual outflow before running and record any failed case without adjusting
bounds or tolerances. Compare time-halving at fixed grid and grid refinement at
fixed sufficiently small dt; two dt values are a diagnostic separation, not a
claimed asymptotic order. No 1024/4096/16384 campaign, terrain rest, new physics,
or coupled energy-conservation claim.

## Delivery gate

Approve the separate SSPRK3 extraction and its explicit source-provenance
exception before the prescribed-flow stepping driver. Until then, C5 transport
accuracy/return results, full stage-time tests, quadrature studies and the planned
resolution/time matrix remain **pending**, not passed. The single-Euler boundary
probe below establishes only how velocity reaches production transport.

### Reproducible single-Euler boundary probe (completed locally)

```sh
cd /tmp
python -I /path/to/vfs-a/tools/check_transport_boundary.py \
  --repo /path/to/vfs-a --output /tmp/vfs-carrier-entry
```

This calls the original coupled Euler operator **once**, with constant
composition and analytical t=0 wind. It does not isolate a full timestep or run
a return experiment. The shared manifest verifies the frozen installed source;
arrays, stage info and driver source are archived. No projection is applied.

Apple M3/macOS CPU float64, dt=1 s, one layer, T=12 days:

| Cells | Equatorial rotation max | Tilted rotation max | Case 4 max |
|---:|---:|---:|---:|
| 16 | 1.5294e-7 | 1.7219e-7 | 8.4417e-7 |
| 64 | 6.0736e-8 | 7.5823e-8 | 6.3646e-7 |
| 256 | 1.7538e-8 | 3.3918e-8 | 1.9997e-7 |

Values are maximum absolute sampled-velocity discrete divergence (s⁻¹).
Largest one-second ps change was 0.08020 Pa; ps was **not** reset. Constant
water-fraction errors were at most 3.47e-18 kg/kg and actual water ledger
installation was exact in all nine probes. No limiter faces were active, so this
probe provides no active-limiter or sharp-profile evidence. These data motivate
retaining carrier evolution; they are not transport convergence results.

Validation for this proposal/probe: 12 existing energy-observer/provenance tests
passed against the installed wheel outside the checkout; both standalone portable
fixture comparisons passed. The formatted committed probe reproduced the initial
probe's nine metric rows exactly. No fixtures, tolerances, numerical source or
source-verification policy changed. Full prescribed-flow analytical-return,
stage-time, quadrature and corruption tests remain pending with the gated driver.
