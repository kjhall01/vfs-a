# Active JMD95 ocean: energy-derived experimental runtime

This is a separately selected numerical formulation on top of draft #30
(`8efd7ff6f16d538e2438601d8d3b537f938ecb24`). It implements active potential
temperature and salinity, nonlinear pressure-dependent JMD95 density, and a
matched spatial energy budget. The archived ocean and constant-density model
remain unchanged, including defaults, fixtures and regression tolerances.

## Supported equations and ownership

```python
from vfs.ocean import Ocean
from vfs.ocean.jmd95 import JMD95Dynamics, checked_ssprk3

model = Ocean(mesh, column, dynamics=JMD95Dynamics(), stepper=checked_ssprk3)
```

See `examples/ocean_jmd95.py`. State remains the existing tensor-owned OceanState;
mesh and column belong to the model. `JMD95Energy` owns the energy definition and
its analytic derivatives. `JMD95Dynamics` pairs them with transport and reuses
the existing ocean kinetic-metric rotation. Neither new module imports atmosphere.
Shared spherical geometry and the existing SSPRK3 sequence remain shared.

| Symbol / stored field | Meaning | Units / location |
| --- | --- | --- |
| h | Positive material-layer thickness | m; cell/layer |
| u / u_e | Horizontal normal velocity | m/s; edge/layer |
| Theta / temp_h = h T | Potential-temperature content | m degC; cell/layer |
| Sigma / salt_h = h S | Practical-salinity content | m PSS-78; cell/layer |
| hq / tracer_h | Optional passive contents | concentration times m |
| z, eta | Height and surface displacement from fixed mean sea level | m |
| H_bottom | Fixed bathymetric depth | m, positive down |
| rho0 | Constant inertial reference density | kg/m3 |
| rho(T,S,p) | JMD95 in-situ density | kg/m3 |
| Psi | Parcel dynamic-enthalpy potential defined below | m2/s2 |
| A_c, A_e | Cell area and grad/div edge metric dc*dv | m2 |

T is the potential-temperature argument of the retained JMD95 polynomial
(the MITgcm implementation specifies IPTS-68 degrees Celsius), not in-situ or
Conservative Temperature. No temperature/salinity scale conversion is hidden.
Inviscid T and S are material scalars. Their volume inventories are conserved;
`temperature_content` is not heat in joules. There is no caloric EOS, entropy,
heat-release correction, nonlinear mixing parameterization, surface forcing,
vertical remapping, convection, open boundary or wetting/drying in this candidate.
Global wet geometry and static closed coast masks are supported by algebra tests.
No realistic coastline/bathymetry campaign is claimed.

## A pressure-dependent energy functional

Layers are ordered from top to bottom. Their physical boundaries are

```text
z_bottom,k = -H_bottom + sum_{j>k} h_j
z_top,k    = z_bottom,k + h_k
eta        = sum_k h_k - H_bottom.
```

Use the **fixed-datum** thermodynamic pressure
`p(z)=-rho0*g*z/10^4` in dbar and the reference column
`rho_ref(z)=JMD95(S0,T0,p(z))`, with T0=10, S0=35. Define this model's conventional
Boussinesq buoyancy and parcel potential as

```text
f(T,S,z) = g [rho(T,S,p(z)) - rho_ref(z)]/rho0 = -buoyancy
Psi(T,S,z) = integral_0^z f(T,S,x) dx.
```

In particular Psi_z=f for the continuum integral, so moving a material parcel
changes this energy by minus buoyancy work. The reference density removes the
resting reference column's compressibility profile. This is an explicit choice
of modeled buoyancy; it is not a claim of exact equivalence to every seawater
Boussinesq convention. The reference contribution is part of the functional.

The dynamic-enthalpy approach is motivated by
[Young (2010), DOI 10.1175/2009JPO4294.1](https://doi.org/10.1175/2009JPO4294.1)
and the thermodynamic consistency discussion in
[Tailleux, Boussinesq thermodynamic consistency](https://arxiv.org/abs/1011.5071).
Our conventional density-anomaly buoyancy, potential-temperature material
variable and discrete layered construction are specified here; they are not
asserted to be Young's complete thermodynamic system (which uses Conservative
Temperature and a specific-volume buoyancy). For adiabatic material T/S this
potential gives the explicitly modeled mechanical/dynamic-enthalpy invariant.
It does not establish diabatic total-seawater-energy conservation.

EOS pressure must be a fixed function of physical height for this derivation.
The archived solver uses depth below the instantaneous free surface instead.
The candidate therefore deliberately changes the pressure convention, not just
an EOS flag. Near a positive free surface p is slightly negative sea pressure:
the signed polynomial is used without a pressure floor. Physical EOS validity
under extreme extrapolation is not established by the finite/positive-density
checks. This candidate is for small free-surface excursions and oceanic T/S.

The Hamiltonian is

```text
H = rho0/2 sum_e,k A_e I(h)_e,k u_e,k^2
  + rho0*g/2 sum_c A_c eta_c^2
  + rho0 sum_c,k A_c integral_zbottom^ztop Psi(T_k,S_k,z) dz.
```

Replacing the last term with `g sum A h rho'(T,S,p(zmid)) zmid` would introduce
extra pressure-derivative forces. That simpler expression is not this nonlinear
EOS Hamiltonian. A pressure-independent affine anomaly reduces the continuum
last term to the linear-EOS expression in #30. Constant reference composition
has zero anomaly energy, but its content derivatives are still nonzero: setting
the anomaly energy value to zero alone does not make active scalars passive.

## Exact derivatives of the chosen quadrature

Both height integrals use fixed Gauss-Legendre quadrature (default order 4).
The quadrature *defines the discrete Hamiltonian*. Its moving nodes are included
in the derivatives, so energy cancellation does not rely on quadrature accuracy.
The independent 16/24-point checks separately measure integration accuracy.

For parcel nodes x_i=z xi_i and weights w_i,

```text
Psi_q   = z sum_i w_i f(T,S,x_i)
Psi_q,T = z sum_i w_i f_T(T,S,x_i)
Psi_q,S = z sum_i w_i f_S(T,S,x_i)
Psi_q,z = sum_i w_i [f(T,S,x_i) + x_i f_z(T,S,x_i)].
```

Analytic JMD95 partials include the dbar-to-bar factor. `jmd95_partials` groups
the existing polynomial in Horner form; its value and derivatives are tested
against the retained EOS and autograd. Keeping the new evaluator local avoids
changing the archived evaluator's arithmetic. This small coefficient duplication
is intentional and covered by equivalence tests. The MITgcm documented density
check at S=35.5, T=3, p=3000 dbar is 1041.83267 kg/m3, at its published precision:
[MITgcmutils source](https://github.com/MITgcm/MITgcm/blob/master/utils/python/MITgcmutils/MITgcmutils/density.py).

Let < > denote layer quadrature at z_bottom+h xi, and let E_b,c be the cell's
anomaly energy divided by rho0 A_c. At fixed stored contents Theta and Sigma,

```text
b_T,k = d E_b,c/d Theta_k = <Psi_T>_k
b_S,k = d E_b,c/d Sigma_k = <Psi_S>_k
b_h,k = d E_b,c/d h_k
      = <Psi>_k - T_k<Psi_T>_k - S_k<Psi_S>_k
        + h_k<xi Psi_z>_k + sum_{j<k} h_j<Psi_z>_j.
```

The final sum accounts for lifting every layer above k. Differentiating at
fixed concentrations would miss the two negative content terms. Production
uses these analytic expressions, not autograd-generated tendencies. Tensor
operations remain differentiable; independent tests use AD and finite differences.

## Paired operators and the zero-velocity issue

Let b_h, b_T, b_S now include the full H derivatives divided by rho0 A_c.
Use velocity-independent arithmetic face concentrations T_f, S_f and q_f:

```text
F           = I(h) u
h_dot       = -D F
Theta_dot   = -D(F T_f)
Sigma_dot   = -D(F S_f)
(hq)_dot    = -D(F q_f)
u_dot       = R_skew(u) - G b_h - T_f G b_T - S_f G b_S.
```

The discrete grad/div adjoint identity gives scalar work
`rho0 sum A_e F [G b_h + T_f G b_T + S_f G b_S]`.
Because `H_u=rho0 A_e F`, momentum cancels it. The unchanged ocean rotation is
skew in that same kinetic metric and does no work. This gives an energy-derived
skew evolution operator, with global closed-boundary volume and scalar inventories.
It does **not** prove a discrete Poisson bracket, Jacobi identity, PV or enstrophy
conservation. The inherited rotation/kinetic accuracy limitations in #28 remain.
SSPRK3 adds temporal truncation error; it is not an energy-preserving time map.

The atmospheric work supplied the design lesson: separate a consistent baseline
from dissipative transport corrections, and identify where their energy goes.
The new baseline uses the same centered traces in scalar fluxes and the matched
force. It does not import atmospheric Gibbs means or silently center only the
force while leaving donor scalar fluxes. Therefore no donor sign switch enters
this pressure force. Any future limiter requires a new, ocean-specific budget
for its mixing work; adding donor fluxes alone would break this argument.

Centered transport does not enforce scalar bounds. `checked_ssprk3` validates
all three full Euler endpoints **before** the SSP convex combinations, then the
final state. It rejects negative salinity, nonfinite tensors or invalid thickness
and density without clipping, retrying or shrinking dt. This is an explicit
acceptance policy, not a monotonicity, positivity or wave-CFL theorem. Negative
Celsius temperature is allowed. Use the checked stepper explicitly; the generic
Ocean facade does not automatically select it when a dynamics object changes.
Other integrators need their own stage acceptance and energy-error assessment.

## Evidence and reproduction

[Machine-readable measurements](jmd95_results/results.json) contain every
reported refinement, source SHA256s and the verification-driver hash. They contain
no detailed machine inventory or large binary fields. New uploaded files are
below 128 KiB. Geometry is regenerated from recorded deterministic mesh settings;
bitwise cross-platform mesh reproduction is not claimed.

```bash
python -m pip wheel --no-deps --no-build-isolation . -w /tmp/jmd95-wheels
python -m pip install --no-deps --force-reinstall /tmp/jmd95-wheels/*.whl
# Copy tests, tools and examples OUTSIDE the checkout, then run there:
python -I -m pytest tests --fixture-mode=portable -q
python -I examples/ocean_jmd95.py
python -I tools/verify_ocean_jmd95.py --output /new/nonexistent/jmd95-run
```

Local installed-wheel suite: **642 passed, one CUDA skip** (nine new tests).
Local measurements use an installed wheel, CPU float64, one thread,
PyTorch 2.15.0+cpu. The environment reuses existing scientific dependencies;
it is not an independent fresh dependency-resolution test.

| Check | Measurement |
| --- | --- |
| Maximum analytic/AD energy-gradient relative discrepancy | 6.11e-16 |
| Spatial energy rate / sum of absolute local work | -7.08e-18 |
| Order-4 anomaly energy vs independent order-24 integral | 1.23e-14 relative |
| Flat / variable-bottom homogeneous-layer rest RMS acceleration | 6.37e-21 / 1.11e-19 m/s2 |
| Pressure jump at u=+/-epsilon, epsilon=1e-3...1e-12 m/s | Exactly zero in these probes |
| Full RHS difference at the smallest +/- velocity | 1.17e-16 m/s2 RMS; scales with epsilon |
| SSPRK3 measured temporal orders | 2.971, 2.988 |

The hydrostatic pressure reference independently integrates the archived EOS
and its AD partials at physical heights. Horizontal T/S gradients are analytic
spherical-coordinate fields; the force is a physical layer mean. Flat, fixed
layer interfaces isolate active pressure accuracy from rotation and terrain.

| Cells | Pressure RMS error (m/s2) | Reference RMS (m/s2) |
| --- | --- | --- |
| 64 | 1.03954e-9 | 1.08358e-7 |
| 256 | 2.51128e-10 | 1.08376e-7 |
| 1024 | 6.48409e-11 | 1.08375e-7 |

The internal-wave check uses two homogeneous material layers (500/2500 m),
T=17/4 degC, S=34/35.5, a 100 km sphere, no rotation and a degree-1 spherical
mode. An independently evaluated hydrostatic pressure matrix gives the internal
speed 3.76838 m/s and period 117898.9 s. Runs cover **0.05 period** (5894.94 s),
with 0.1 m initial interface amplitude and dt<=30 s resolving the fast mode.

| Cells | Endpoint projection error / initial amplitude | Delta H / initial perturbation energy |
| --- | --- | --- |
| 32 | 0.00339898 | 1.99e-5 |
| 64 | 0.00169232 | 4.90e-6 |
| 128 | 0.000878905 | 7.84e-6 |

Volume, T and S inventory relative changes are at most 3.32e-16 in these runs.
The much smaller total-energy relative changes (5.08e-15 to 2.06e-14) reflect
an enormous constant background and must not replace the perturbation-energy
normalization. Subtracting large background energies and EOS cancellation limit
these tiny wave-energy differences. This is a short projection check; it does
not establish many-period phase accuracy, damping, nonlinear wave stability or
long-run conservation. No separate timestep extrapolation accompanies the spatial
wave table. The separate 400 s timestep study uses dt40/20/10 and a dt1.25 reference,
with a dt0.625 refinement discrepancy of 9.38e-15 in its nondimensional RMS norm.

The rest cases have horizontally flat upper interfaces and homogeneous layers,
with topography confined to the bottom layer. They do not qualify a smooth
stratified profile crossing sloping coordinates or arbitrary terrain balance.

Legacy same-environment ocean capture: **164 tensors exactly equal**. Corrected
Gibbs portable comparison: **142** tensors; dry helper portable comparison:
**93** tensors, both passed. All 97 pre-existing runtime Python files remain
byte-identical to #30. Only two ocean runtime modules are added. Historical
fixtures and exact/portable tolerance policies are untouched.

Frozen scientific drivers intentionally reject added runtime modules. CI runs
their pinned campaigns with the pinned e959b700 wheel after running the candidate's
full installed suite and exact legacy comparisons. It then restores the candidate
wheel for the new JMD95 example/probes. This preserves the old source-hash policy;
the frozen campaign is not presented as a JMD95 validation. New probes hash the
actual installed candidate source. Hosted CI status is reported in the PR.

## Remaining development

1. Measure many-period internal-wave phase/damping and rotating stratified flow;
   resolve the joint rotation/kinetic consistency limitation from #28.
2. Derive an ocean-specific bounded/mixing extension with an explicit energy
   destination and continuous zero-velocity behavior. No atmospheric heat term
   can be copied into this adiabatic potential-temperature budget.
3. Derive and verify vertical remapping, mixing, convection, forcing and boundary
   work before enabling those sources. Nonlinear EOS availability alone does
   not qualify cabbeling/mixing energetics.
4. Test smooth stratification over slopes, realistic coasts, larger meshes,
   accelerators, compilation and longer backward trajectories. Current gradient
   evidence is a selected CPU float64 one-step T-content direction; second
   salinity derivatives at S=0 are not qualified.
