# Entropy/salt ALE integration and vertical resolution

This research composition integrates the sharp-front entropy transport of draft
#38 (`9e9a92c7479983a9cf45a2f740445d5079da9f06`) with the remap machinery of
#32 (`0f6aa8f266e38d4d94b140b6a11186944ebf86ae`). The latter's ocean module,
example, tests, driver and historical reports are retained unchanged. Atmosphere,
rotation, existing runtime expressions, defaults, fixtures and tolerances are unchanged.

## What is conserved

`tools/ocean_entropy_ale.py` owns a small `EntropyRemapper` and
`EntropyALEStepper`. It composes the checked full horizontal SSPRK3 step with a
caller-chosen target grid, then uses the existing bounded overlap reconstruction:

- Cell columns conserve thickness, entropy content, salt content and passive contents.
- Edge columns remap momentum using the same interpolated thickness as kinetic energy.
- Unchanged columns, land placeholders and inactive velocities are retained exactly.
- Invalid geometry or thermodynamic inversion rejects the result; there is no clipping,
  heat compensation, hidden remap schedule or automatic timestep retry.

For every content C=h*q, the new content is the integral of the old reconstructed
q over the new layer. Consequently, sum(C_new)=sum(C_old) at roundoff. Each overlap
profile is locally bounded. This does not imply a temperature maximum principle:
T is obtained by nonlinear inversion of the remapped entropy and salinity.

Temperature content is deliberately **not** conserved by this version. Remapping
T and then computing entropy would implement a different thermodynamic scheme.
Neither variant is an energy-conserving physical mixing closure. The new report
exposes temperature-content change, each Hamiltonian component's energy change,
per-column content residuals, edge momentum residuals and actually changed columns.
The horizontal entropy-production ledger remains separate from the remap ledger.

For a complete step, the energy accounting is:

    total energy change = horizontal finite-step change + measured remap change.

Spatial Hamiltonian work closure in the horizontal solver does not remove either
finite-step drift or the remap defect. No energy residual is called physical heating.

## Which resolution question is being tested?

1. **Original full-period sharp-interface wave:** same two water masses, mesh,
   amplitude, timestep rule and target policy as #32, now subdivided more finely.
   Each water mass receives half the layers, so upper-layer thickness is 1000/N m
   and lower-layer thickness is 5000/N m. This is controlled vertical refinement,
   not a representative stretched ocean grid. Both remapped and unremapped controls
   are measured. The observable is column temperature content, invariant under the
   original remap.
2. **Integrated entropy trajectory:** same initialization, new sharp-front solver,
   matched energy and entropy remap. These are shorter **0.05-period** integrations,
   not complete wave qualification. The observable is column entropy content,
   invariant under the new remap. Do not compare its short-window fitted phase or
   amplitude directly with the old full-period values; a short fit is ill-conditioned.
3. **Single-remap smooth stratification:** the same continuous linear T/S profiles
   sampled on distorted columns at 8–128 layers, reset to reference columns. This
   measures reconstruction/thermodynamic conversion, not long-run dynamical accuracy.
   Flat boundary slopes can dominate the maximum error and reduce global RMS order.

A layer count is not an accuracy guarantee. For development, compare 32 and 64
layers, then refine further if the desired thermocline, surface layer or wave mode
is not resolved. Target-layer distribution and remap frequency matter as well as N.
No extrapolation of the four/eight-layer errors to a climate grid is justified.

## Reproduction

Build/install the wheel, then copy tools/tests/examples outside the checkout.
Use CPU float64, one torch thread and the parent reference dependencies.

```sh
python -I tools/probe_ocean_entropy_ale.py --legacy --layers 32 --interval 1 --output legacy-32-1.json
python -I tools/probe_ocean_entropy_ale.py --legacy --layers 32 --interval 0 --output legacy-32-0.json
python -I tools/probe_ocean_entropy_ale.py --layers 32 --periods .05 --interval 1 --output entropy-32-1.json
python -I tools/probe_ocean_entropy_ale.py --layers 32 --periods .05 --interval 0 --output entropy-32-0.json
python -I tools/probe_ocean_ale_smooth.py smooth-rest.json
python -I -m pytest tests --fixture-mode=portable -q
```

Repeat at the documented resolutions. CI executes short actual-remap integration
and the smooth sweep; full-period research runs are supplied as compact JSON.
No GPU, compile, long-backward, long-run climate or coupled atmosphere qualification
is implied. The inherited informational historical exact fixture discrepancy is
kept separate from required portable regression checks.

## Validation and scope

Installed-wheel portable suite outside checkout: **763 passed, one CUDA skip**.
All 164 legacy ocean integration tensors remain exactly unchanged; atmosphere
portable comparisons pass for 142 moist and 93 dry tensors. Nine new integration
tests cover native inventories, local reconstructed bounds, explicit energy
accounting, identity/land/inactive preservation, absent passives, invalid geometry,
SSP entropy accounting, actual wave remaps and a genuinely triggered thin-column reset.
Eighteen original ALE tests are retained. All new uploads are below 128 KiB.

## Full-period original-remap results

| Layers | No-remap RMS | Remapped RMS | Remapped phase magnitude (degrees) | Remap energy change (J/m²) |
|---|---:|---:|---:|---:|
| 16 | 0.062668 | 0.191072 | 13.222 | -294.001 |
| 32 | 0.062668 | 0.155110 | 10.484 | -650.641 |
| 64 | 0.062668 | 0.136792 | 9.117 | -823.282 |

Earlier #32 remapped RMS values were 0.442 at four layers and 0.268 at eight.
The new 64-layer RMS is about 2.18 times the no-remap value; refinement is helpful,
but does not eliminate the remap penalty. The unremapped controls barely change
with vertical subdivision, as expected for this two-water-mass setup. Remapped
phase improves, but energy change is non-monotonic across the complete sweep.
The approximately 0.058 J/m² initial perturbation energy is small; absolute remap
energy changes are therefore reported alongside the ratios in the raw JSON.

## Single-remap smooth stratification

| Layers | Temperature RMS error (°C) | Maximum error (°C) |
|---|---:|---:|
| 8 | 0.0702608 | 0.2439258 |
| 16 | 0.0253587 | 0.1243523 |
| 32 | 0.0090155 | 0.0624770 |
| 64 | 0.0031925 | 0.0312762 |
| 128 | 0.0011293 | 0.0156428 |

Here global RMS decreases at about order 1.5, while the boundary-dominated
maximum error decreases at about order one. This is distinct from the original
smooth scalar PLM test's approximately second-order convergence. Neither is a
full-wave convergence claim.

## Integrated entropy/ALE: 0.05 period, 197 steps

| Layers | No-remap RMS | Remapped RMS | Remap energy (J/m²) | Actual remap events |
|---|---:|---:|---:|---:|
| 8 | 0.0011150 | 0.0036325 | -63011.6 | 197 |
| 16 | 0.0011150 | 0.0024206 | -63326.6 | 197 |
| 32 | 0.0011150 | 0.0018244 | -63449.1 | 197 |
| 64 | 0.0011150 | 0.0015315 | -63440.7 | 197 |

Every remapped run resets 32 columns in each of 197 steps (6,304 column events).
Per-remap native inventory residuals remain near roundoff. The absolute entropy
budget residual in each JSON must be interpreted against its large background
inventory; focused tests check the scaled combined horizontal/remap budget.

The reference-thermal term dominates the integrated remap energy loss. Increasing
resolution from eight through 64 layers barely changes that loss, despite improving
the short-window wave observable. Thus this campaign establishes a functioning,
conservative composition, **not an energetically benign ALE formulation**. Nonlinear
thermal energy is not preserved by separately conserving entropy and salt. An
energy-aware thermodynamic remap or explicitly justified mixing treatment remains
necessary work. No unreported temperature adjustment has been applied.

The sharp-interface displacement is much smaller than even the refined layers.
As an interpretation, mixing across that unresolved interface can remain important
through this sweep; these data do not establish the eventual asymptotic convergence
rate. A smoother thermocline or different grid schedule is a different experiment,
not a reason to discard this negative result.

Compact evidence is in [entropy-ale-results](entropy-ale-results); original #32
results remain in [ale_results](ale_results). All runs used CPU float64.
