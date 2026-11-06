# Conservative vertical ALE remap

Implemented on main `5841cda8b4585418b18f0ba0486315870b1a59b2` after #31 merged.
This is an explicit, optional extension of the active JMD95 ocean. The user
authorized an inventory/momentum-conservative remap with measured, nonzero
energy changes. It does not add a heat correction or claim energy conservation.
All 99 pre-existing runtime Python files and all defaults/fixtures/tolerances
are unchanged. The new runtime module imports ocean code only.

## Public use and ownership

```python
from vfs.ocean import Ocean
from vfs.ocean.jmd95 import JMD95Dynamics, checked_ssprk3
from vfs.ocean.ale import ALEStepper, ReferenceGrid, VerticalRemapper

stepper = ALEStepper()  # ReferenceGrid + bounded linear VerticalRemapper
model = Ocean(mesh, column, dynamics=JMD95Dynamics(), stepper=stepper)
state = model.step(state, dt)

# Same update, with connected tensor diagnostics for the REMAP ONLY:
state, report = stepper.step_with_report(model, state, dt)
print(report.energy_change)  # kinetic, free_surface, dynamic_enthalpy, total (J)
```

The two calls above illustrate alternatives; calling both advances two steps.
No report, timestep counter, or autograd graph is retained by the stepper.
The ordinary `model.step` avoids computing the optional before/after energies.
`examples/ocean_ale.py` provides a complete executable example.

For a caller-owned, less frequent schedule, use the components explicitly:

```python
target = ReferenceGrid()
remapper = VerticalRemapper(method="plm")  # "pcm" is the constant reference
for step in range(number_of_steps):
    state = checked_ssprk3(model, state, dt)
    if (step + 1) % remap_interval == 0:
        state, report = remapper.remap_with_report(model, state, target(state))
```

`ReferenceGrid(fractions)` accepts a tensor of shape `(layers,)` or
`(cells,layers)`, positive and summing to one in each column. The default uses
`OceanColumn.rest_thickness` fractions. New thickness is the current column
total times these fractions. The physical bottom and free surface remain
fixed during remapping. On a flat resting bottom this is a simple stretched
reference (z-star-like) target. Over variable bathymetry, reference fractions
do not imply globally flat interfaces. No density-adaptive coordinate, slope
control, partial cells, wetting/drying or automatic minimum-thickness repair
is implemented. The selected target must exceed the dynamics' minimum thickness.
The number of layers is fixed by the existing OceanColumn/OceanState geometry.

## Conservative integration

For old layer interval I_i and destination J_j, with q=T, S or a passive tracer,

\[
 (hq)'_j = \sum_i \int_{J_j\cap I_i} q_i(z)\,dz.
\]

`remap_contents(old_h, content, new_h, method="plm")` exposes this operation.
Thickness has shape `(...,layers)`; content has the same shape or an additional
final field axis. This low-level function can change layer count independently
of OceanState. Input tensors must already agree on dtype/device.

PCM uses each original mean. PLM uses a linear reconstruction about the old
layer midpoint, with nonuniform-grid minmod slopes and an additional endpoint
bound. Boundary layers have zero slopes because no boundary concentration is
prescribed. Slopes integrate to zero over their source layer, preserving the
original mean. Endpoints remain within neighboring source means; destination
means consequently remain bounded, up to roundoff. Signed potential temperature
is allowed; nonnegative salinity stays nonnegative. JMD95 density is evaluated
from the remapped T/S and target pressure geometry, not independently remapped.
Separate T/S bounds do not establish static stability for a nonlinear EOS.

First and last intersections are found with batched `searchsorted`; prefix
sums integrate full intervening layers, and analytic linear integrals handle
partial layers. Memory scales with column count times layers times fields,
without a dense old-layer/new-layer overlap matrix. Thin single-layer
intersections use the supplied target width to avoid cancellation of large
depths. A common final interface eliminates roundoff-sized uncovered slivers;
larger discrepancies in column extent and indistinguishable interfaces are
rejected. No scalar clipping or inventory redistribution is applied afterward.

Each wet column preserves sum(h), sum(hT), sum(hS), and each passive inventory
to floating-point roundoff. Uniform concentrations are preserved to roundoff.
These are Boussinesq volume and material-scalar inventories: hT is not heat in
joules, and sum(rho(T,S,p)*h) is not asserted invariant.

## Momentum and energy

The velocity remap uses h_e=I(h) at mesh edges, with the SAME area-weighted
cell-to-edge interpolation used by the existing kinetic energy. Edge interfaces
are cumulative h_e, consistent with interpolating the cell interfaces because
I is linear with height-independent weights. Remap h_e*u, then divide by new
h_e. This preserves each active edge column's integrated normal momentum

\[
 \sum_k \rho_0 A_e h_{e,k} u_{e,k},\qquad A_e=d_c d_v.
\]

This does not prove global angular-momentum conservation of the dynamical core.
Finite land placeholders and inactive edge velocities are retained unchanged.
No coast projection or atmosphere import is hidden in the remapper.

For a report, evaluate the actual quadrature-defined JMD95 Hamiltonian before
and after remapping. Report changes in kinetic, free-surface, dynamic-enthalpy,
and total energy separately. Free-surface energy is unchanged except roundoff.
Neither dynamic-enthalpy change nor total change has a prescribed sign. The
report also contains per-column content residuals and edge momentum residuals;
global cancellation cannot conceal local conservation failures.

An exact two-layer test illustrates the energy distinction: thicknesses (1,1),
velocities (-1,+1) remap to thicknesses (0.5,1.5), velocities (-1,+1/3).
Momentum remains zero, while kinetic energy falls from 1 to 1/3 in normalized
units. No arbitrary velocity rescaling, compensating temperature increment,
or artificial energy reservoir is used. Diagnostic differences are not heating.

## Integration and differentiation scope

`ALEStepper` calls the unchanged checked SSPRK3, then remaps once after the
accepted full step. Full Euler-stage checks remain in force. A failed stage
never reaches the remapper. Horizontal centered transport is still not bounded;
vertical remapping does not repair earlier negative salinity/thickness.
The checked dynamics still require an unforced model. No convection or mixing
parameterization is added; numerical remap mixing remains measurable error.

The full ALE algorithm is not claimed third-order merely because its dynamics
step uses SSPRK3. Remap frequency, vertical resolution and reconstruction all
affect error; repeating a lossy remap more frequently need not improve accuracy.
Tensor expressions retain autograd through contents and geometry away from
overlap/limiter switches. Tests include selected finite-difference directional
checks, not smoothness at switches or a long-backward/GPU/compile qualification.
No forward tendency or remap is generated with autograd.

## Evidence and reproduction

See `ale_results/results.json` and `ale_results/validation.json`. Scientific
measurements report errors and convergence without acceptance thresholds.
Numerical invariant tests use explicit floating-point tolerances.

The analytic smooth-profile PLM L2 errors at 16/32/64/128 layers are
8.37e-4, 2.13e-4, 5.12e-5 and 1.21e-5 (orders 1.97, 2.06, 2.08).
After 50 grid round trips the corresponding errors are 6.15e-2, 8.34e-3,
2.04e-3 and 3.69e-4. Repeated remapping is measurably diffusive.

The full-period coarse sharp-interface wave results are **not** evidence of
small energy error. Here the energy normalization is the initial perturbation
energy (~7.31e9 J), rather than the much larger background Hamiltonian:

| Layers | Remap interval (steps) | Projection RMS error / initial amplitude | Fitted phase (rad) | Remap energy / initial perturbation energy |
| --- | ---: | ---: | ---: | ---: |
| 4 | None | 0.0627 | -0.0806 | 0 |
| 4 | 1 | 0.4418 | -0.5702 | +2.99e4 |
| 8 | 1 | 0.2677 | -0.3331 | +7.13e3 |
| 8 | 10 | 0.2670 | -0.3321 | +7.12e3 |

The positive change is dominated by dynamic enthalpy; kinetic energy decreases
under remapping in these cases. Refining the vertical grid helps, while reducing
remap frequency from every step to every ten steps has little effect here.
These are very small waves across an unresolved sharp T/S interface on a coarse
grid, so the ratios do not estimate a universal climate energy bias. They do
show that the current fixed-target remap cannot be treated as an energetically
negligible extension of the material-layer wave model. Across all runs,
volume/T/S inventory relative changes remain at or below 1.50e-13.

The continuous-profile rest probe also shows nonzero acceleration before and
after remapping. After remapping its RMS acceleration decreases from 5.02e-9
to 1.58e-10 m/s² over 8/16/32 layers. Energy changes have both signs across this
sequence. Better interface reconstruction/target placement and energy treatment
remain future work; the implementation is deliberately opt-in.

Build/install a wheel, then copy `tests`, `tools`, and `examples` to a directory
outside the checkout and run there with isolated Python:

```bash
python -I -m pytest tests --fixture-mode=portable -q
python -I examples/ocean_ale.py
python -I tools/verify_ocean_ale.py --output ale-results
```

Use `--quick` to omit the four complete-period wave runs. CI runs the full
installed-wheel test suite, the example, and the quick measurements. Historical
scientific manifest checks retain their pinned wheel, as on the parent PR.

The measurement driver covers analytic smooth-profile remaps, 50 grid round
trips, and a continuously stratified physical rest state sampled on neighboring
deformed columns. The rest probe measures spurious acceleration, not an exact
well-balanced property. The wave cases use a 100 km sphere with 32 cells, no
rotation, two water masses subdivided into 4 or 8 layers, a 0.1 m perturbation,
and the independently derived continuum two-layer period (about 32.75 hours).
The observable is column temperature-content projection, invariant under
remapping. Layer-thickness projection would be misleading on the reset grid.
Reported fitted amplitude/phase use the reference frequency and do not estimate
an independent dispersion relation from one period.

## References

- [MOM6 vertical Lagrangian method](https://mom6.readthedocs.io/en/main/api/generated/pages/ALE.html).
- [MOM6 vertical reconstruction](https://mom6.readthedocs.io/en/main/api/generated/pages/Vertical_Reconstruction.html).
- [Griffies, Adcroft and Hallberg (2020), vertical Lagrangian-remap primer](https://doi.org/10.1029/2019MS001954).

## Follow-up: avoid remapping healthy material layers

`DistortionLimitedGrid` is an optional target policy. It retains a column's
current grid while every thickness/reference-thickness ratio is within [0.5,2].
It resets the entire column to its reference target if any ratio leaves that
band. These configurable thresholds are numerical policy, not scientific
acceptance thresholds. It does not control interface slopes or prevent invalid
horizontal Euler stages. It must be applied before layers become invalid.

```python
from vfs.ocean.ale import ALEStepper, DistortionLimitedGrid
stepper = ALEStepper(target=DistortionLimitedGrid())
```

An identity target now returns the original state exactly, avoiding unnecessary
roundoff diffusion. A test exercises both retention and an actual thin-column
reset, including inventory conservation. Trigger switching is not a smooth
operation; no differentiability at that switch is claimed. Defaults remain
unchanged, and fixed-target tests/results above remain evidence of the problem.

Reproduce the complete-period comparison with:

```bash
python -I tools/verify_ocean_ale.py --output ale-adaptive --distortion-limited
```

The compact `ale_results/distortion_limited.json` records the four-layer result.
The small-amplitude wave remains inside the permitted thickness band, so this
case avoids remapping entirely and should recover the material-layer control.
This demonstrates the benefit of avoiding unnecessary remaps, **not** improved
accuracy or energy conservation when distorted layers actually require remapping.
Resolving required-remap errors across sharp T/S interfaces remains open.

Measured full-period result: normalized projection RMS error 0.0626679,
phase -0.0805977 rad (4.62 degrees in magnitude), and zero remap energy change.
This exactly matches the earlier no-remap control. Total dynamics/time-stepping
energy change remains 3.32e-5 of initial perturbation energy. The installed-wheel
suite after this extension reports 660 passed and one CUDA skip.
