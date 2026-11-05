# Constant-density ocean: separate numerical change

> Status reconciled against main `e959b7005832a5041e082b6e300838e15bfbaa9e`:
> #26 and #27 are merged and their heads are ancestors of main. Draft/unmerged
> language below describes the original publication, not current status.
> The reference solver remains default; the constant-density candidate remains
> explicit. Current investigation: [rotating-flow diagnosis](ROTATING_BALANCE.md).

This dependent change adds `ConstantDensityDynamics`, selected explicitly from
`vfs.ocean.conservative`. `Ocean` still defaults to the extracted reference.
The reference equations and their 1,416-array capture are not replaced.

## Equation scope and ownership

This is a layered, constant-density free-surface ocean with passive hT, hS and
hq contents. It has no active EOS, stratification energy, vertical remap, or
interlayer momentum exchange. Density is rho0 regardless of T/S. Recover fields
through `model.physical_fields(state)` to use the selected dynamics' density;
the retained `state.density()` helper is explicitly a legacy linear-EOS value.
The old regrid adapter's density request also uses that legacy helper.

`ocean.energy.ConstantDensityEnergy` owns H and its analytic derivatives.
`ocean.conservative.ConstantDensityDynamics` owns the paired equation assembly.
Geometry and the explicit RK algorithms remain shared and unchanged. No
atmospheric pressure-coordinate energy, lid reservoir, moist carrier, or limiter
is reused as ocean thermodynamics. The small Euclidean interpolation transpose
remains ocean-local until a shared contract has been established for both domains.

Let I be the actual cell-to-edge interpolation, D divergence, G gradient,
A_c cell area and A_e=dc*dv the compatible edge metric. For fixed bottom depth:

```text
eta_c = sum_k h_ck - H_bottom,c
M_ek = rho0 A_e I(h)_ek
H = 1/2 sum_ek M_ek u_ek^2 + rho0 g/2 sum_c A_c eta_c^2
H_u = M u
H_h = rho0/2 I^T(A_e u^2) + rho0 g A_c eta
B = H_h / (rho0 A_c)
F = I(h) u
h_dot = -D(F)
u_dot = -G(B) + M^-1 [M R - R^T M] u / 2
(h q)_dot = -D(F q_upwind)          [also for passive T and S]
```

I^T uses the same two side-area weights as I; it is not an unweighted average.
The thickness derivative is taken at fixed stored velocity and scalar contents.
It includes the changing layer-dependent KE metric. Grad/div adjointness then
cancels the entire Bernoulli/continuity work. R is the vorticity-weighted mesh
rotation stencil, with its vorticity held fixed when forming R^T. The transpose
scatters those exact coefficients. A second rotation evaluation at M*u would
incorrectly recompute vorticity and is not the transpose.

The skew covector has zero u-paired work. At a static coast, input/output
projection restricts it to active edges; potential energy is restricted to wet
cells. This gives a closed mechanical budget, but does not prove accurate coastal
vorticity, a physical viscous tangential condition, or a Poisson Jacobi identity.
No forward tendency is obtained by autograd. Autograd and finite differences are
used only for independent verification.

This intentionally changes KE, its Bernoulli derivative and rotation relative
to the archive. Thus old and new energies have different numerical values, even
on the same state. Closure of the new H is not conservation of the archived H.

## Use

```python
from vfs.ocean import Ocean, OceanColumn
from vfs.ocean.conservative import ConstantDensityDynamics

column = OceanColumn(3, total_depth=3000.)
model = Ocean(mesh, column, dynamics=ConstantDensityDynamics())
# h, temperature, salinity: [cell, layer]; u_e: [edge, layer]
state = model.state(u_e=u_e, h=h, temperature=temperature, salinity=salinity)
next_state = model.step(state, 30.)
budgets = model.budgets(next_state)
# budgets['constant_density_mechanical_energy'] is a connected tensor in J.
```

Run `examples/ocean_conservative.py` for a complete CPU/float64 example. This
explicit selection avoids silently changing the reference solver's behavior.
The snapshot format is unchanged. No coupled atmospheric model is restored.

## Reproduce the evidence

Build/install the wheel, copy tests/tools/examples outside the checkout, then:

```bash
python -I -m pytest tests --fixture-mode=portable -q
python -I tools/verify_ocean_energy.py --out new-ocean-energy-run
python -I tools/capture_ocean.py --inputs tests/fixtures/ocean/inputs.npz \
  --compare tests/fixtures/ocean/archive_capture.npz
python -I examples/ocean_conservative.py
```

The numerical driver refuses to overwrite a directory. It records source,
script, input and mesh hashes; actual wave/time-study mesh buffers; environment;
raw errors and work terms. The fixed work probes use the archive's saved
32-cell/three-layer mesh. The wave/time sweeps use the current generator and
store its actual buffers. Their wave numbers are therefore not a direct paired
comparison with the old audit's regenerated meshes.

`energy_results/results.json` contains the raw measurements. No subjective
scientific PASS thresholds are applied. Unit assertions concern algebraic
identities at reduction-scaled floating-point precision and finite-difference
agreement, not a tolerable climate error.

## Measured evidence and limits

Installed wheel, CPU float64, one thread: **575 tests passed, one CUDA skip**
(46.74 s). The 20 ocean tests include exact reference capture, ownership/output,
AD/analytic energy gradients, actual-RHS work, nonzero pressure exchange,
inventories and uniform tracers, flat/variable-bottom rest, and one-step
directional derivatives. All 457 existing structural executable definitions
remain unchanged; the 298 atmospheric tensors/solver records remain exact.
No atmospheric fixture or tolerance changes are included.

| Saved 32-cell state | Reference normalized dE/dt | Candidate normalized dH/dt |
|---|---:|---:|
| Global sheared/random flow | 1.50032e-3 | 0 |
| Closed-coast sheared/random flow | 4.84555e-3 | 3.81495e-17 |
| Free-surface-aligned flow | 3.42179e-2 | 9.27209e-17 |

Normalization is the sum of absolute local H_u*u_dot and H_h*h_dot terms.
Absolute candidate rates are 0, -0.00378418 W and 3.05176e-5 W, respectively,
on planetary-domain energies of order 1e16–1e18 J. These are floating-point
cancellation results for the stated configurations, not a universal zero-drift
claim. In the aligned state the pressure KE/PE mismatch changes from 6.60515%
to exactly zero in this evaluation. That state has 32 cells; the earlier audit's
3.69% result was a different 64-cell state.

Analytic-versus-AD gradient maximum relative errors are at most 1.28461e-16.
The nonzero-direction central-difference sweep reaches relative errors of
1.29e-12 (global), 2.71e-13 (coast) and 3.99e-16 (aligned), with cancellation
growth at smaller increments. Finite differences along the nearly energy-null
RHS resolve only roughly 1e-10–1e-6 of the local work scale; they do not establish
the 1e-17 cancellation observed through independent AD.

External spherical-harmonic wave projection errors, divided by initial 0.1 m
amplitude, at 0.2 period, H=4000 m, one layer, zero planetary rotation:

| Cells | Degree 1 | Degree 2 |
|---:|---:|---:|
| 32 | 0.02988555 | 0.07981940 |
| 64 | 0.01442045 | 0.04288668 |
| 128 | 0.00758788 | 0.02300681 |
| 256 | 0.00392074 | 0.01095773 |

Fitted orders against sqrt(4*pi/n) are 1.9434 and 1.8986. Timesteps are adjusted
to the same fraction of each analytic period and stay below 300 s. This is one
endpoint projection, mixing phase and damping; it is not a multi-period fit,
independent time extrapolation, internal-wave or full dispersion qualification.

On a fixed 32-cell sphere of radius 100 km, H=1000 m, a 0.1 m perturbation,
end time 400 s, SSPRK3 steps 40/20/10/5 s give nondimensional state RMS errors
1.15864e-10 / 1.45415e-11 / 1.82111e-12 / 2.27609e-13 against dt=0.625 s.
The norm concatenates h differences divided by H and u differences divided by
sqrt(gH). Pairwise orders are **2.9942, 2.9973, 3.0002**. Halving the reference
step again changes it by 2.88693e-15 in the same norm. This estimates temporal
error for the same spatial operator, not error against an analytic PDE solution.

Relative energy changes in that time study are -7.40535e-6, -9.26362e-7,
-1.15817e-7 and -1.44801e-8: SSPRK3 is not exactly energy conserving. The small
ten-step example changes volume by 1.67299e-16 relatively and H by -6.03333e-12.

Still unqualified: general centered-carrier positivity, active T/S energetics,
nonlinear EOS, stratified/topographic balance, geostrophy, nonlinear transport
convergence, PV/enstrophy, internal/Rossby/Kelvin modes, vertical remap, sources'
mechanical/thermal conversion, long runs, GPU/compile and full backward rollouts.
The original active-EOS/remap/process paths remain experimental reference code.
The next numerical change should derive stratification energy and active-content
forces together; simply reusing the archived baroclinic pressure would break
the closure established here.
