# Potential-enthalpy ALE experiment

This research pass follows draft #39 (`8358aece76ce1e4bb19ead890a6bbeb392a8550a`).
It retains the entropy-based dynamical state, JMD95 density EOS, matched Hamiltonian,
sharp-front transport, rotation and existing grid policy. Only the explicitly
selected remapper changes. No installed runtime file, default, fixture or tolerance
is modified. The old entropy remapper remains available for controlled comparisons.

## Thermodynamic choice

Write thickness as h, salinity as S, specific entropy as eta and potential enthalpy
as a(S,T), using the same surface reference as the existing energy functional.
The old remap preserves sum(h eta) and sum(h S). Since a is nonlinear in eta and
S, reconstructing T from these averages changes sum(h a). That thermal change was
the dominant measured energy loss in #39.

`tools/ocean_enthalpy_ale.py` instead performs the explicit sequence:

1. Compute a(S,T) from the original entropy state and existing reference.
2. Conservatively remap h a, h S and passive contents with the existing bounded
   PCM/PLM overlap integrator. Remap edge momentum exactly as in #39.
3. Recover T from a(S,T)=a_new at the remapped salinity, using the existing bracketed
   analytic Newton inversion; reconstruct h eta from that same reference.
4. Validate the complete thermodynamic state. Report every energy component,
   native inventory residual, potential-enthalpy residual and entropy change.

There is no post-step temperature offset or globally redistributed heat correction.
Potential enthalpy is the conserved thermal quantity during projection. Entropy is
then diagnosed and is not constrained to its old inventory. Ocean physical mixing
conserves energy and generally produces entropy; conserving entropy is appropriate
for reversible material motion but not necessarily for the averaging in a remap.

The potential-enthalpy conservation residual includes the complete conversion
back to entropy and inversion to T, not just the linear overlap calculation.
Identity remaps return the original state; unchanged columns and land/inactive
placeholders are retained exactly. Tensor expressions retain gradients away from
limiter, overlap and inversion branch changes; one directional derivative is tested.
This is not a global differentiability or long-backward qualification.

## What it does not solve

The full Hamiltonian contains kinetic and gravitational/dynamic-enthalpy terms
as well as the surface-reference thermal contribution. Preserving sum(h a) removes
one source of error but does **not** enforce total-H conservation. Mechanical energy
changes remain visible and may have either sign. No measured residual is hidden as
physical heating. Nor does energy conservation alone guarantee accurate waves or
correct mixing of a thermocline.

PCM averages h a and h S with common positive overlap weights. Higher-order PLM
reconstructs the fields independently. Its local scalar bounds alone are not a
proof of a discrete entropy inequality. The audited entropy changes, including
roundoff-sized negative values in unmixed/uniform cases, remain in the evidence.
The tested mixed states produce entropy; no universal entropy-stability claim follows.

## Controlled measurements

The new driver uses **column salinity content** for the wave projection: salt is
conserved by both remappers. Entropy was the projection in #39, but is no longer
invariant under potential-enthalpy remapping. Thus its old wave RMS cannot be
silently reused as the matching reference for this experiment.

At 8 and 32 layers, rerun no remap, entropy remap and enthalpy remap on identical
initial states, mesh and timestep. A 64-layer enthalpy run extends the thermal-energy
comparison with the prior campaign, without claiming a new same-driver 64-layer
control. All wave trajectories here cover **0.05 period**, not a complete period.
Every remapped wave performs 197 resets. Tests also cover repeated back-and-forth
remaps, coupled-step entropy accounting, land, missing tracers and extreme salinity.

The 32 one-remap audits combine four profiles, coast/no-coast, PCM/PLM and the two
remapping choices. Energy per square metre uses the whole mesh surface area;
coastal cases are not normalized by wet area alone. Raw column enthalpy residuals
have units of h*a and must be compared with column inventory scales.

## Next step toward total energy closure

A further correction must use the **same discrete Hamiltonian** and its actual
edge kinetic energy, not only a constant heat-capacity temperature offset. One
possible research design is to allocate edge energy changes to adjacent columns
with explicitly conservative weights, then solve a local thermal adjustment for
each column's full energy target. The thermal adjustment would replace strict
potential-enthalpy conservation by the corresponding mechanical-to-thermal transfer.

Such a solve must retain volume, salt and momentum, respect the caloric domain,
and check the combined entropy change. Its spatial allocation is part of the
numerical formulation and must be justified; a scalar global energy fix is not an
adequate replacement. This pass establishes the thermal remap baseline and exposes
the remaining mechanical defect. It does not implement or claim that further solve.

## Reproduction

Install the wheel and copy tools/tests/examples outside the checkout.

```sh
python -I tools/probe_ocean_enthalpy_ale.py --audit --output audits.json
python -I tools/probe_ocean_enthalpy_ale.py --layers 8 --scheme none --output wave-8-none.json
python -I tools/probe_ocean_enthalpy_ale.py --layers 8 --scheme entropy --output wave-8-entropy.json
python -I tools/probe_ocean_enthalpy_ale.py --layers 8 --scheme enthalpy --output wave-8-enthalpy.json
python -I -m pytest tests --fixture-mode=portable -q
```

Repeat at 32 layers; run the enthalpy variant also at 64 layers. CI runs the audits
and a shorter actual-remap wave smoke. Compact reports include source hashes.
All uploaded files are below 128 KiB. No GPU, long-run, coupled climate or full-period
wave qualification is claimed. Scientific errors are reported without accuracy gates.

## Measured results

All trajectories cover 0.05 period with 197 steps. Energy values below are
accumulated **remap** changes, separated from horizontal finite-step energy drift.

| Layers | Remapper | Salinity projection RMS | Remap total (J/m²) | Remap thermal (J/m²) |
|---|---|---:|---:|---:|
| 8 | none | 0.00111498 | 0.000000 | 0.000000 |
| 8 | entropy | 0.00517453 | -63011.563031 | -63084.479186 |
| 8 | enthalpy | 0.00566237 | 88.023548 | 0.000033 |
| 32 | none | 0.00111498 | 0.000000 | 0.000000 |
| 32 | entropy | 0.00236218 | -63449.071709 | -63303.638624 |
| 32 | enthalpy | 0.00282468 | -131.511334 | 0.000083 |
| 64 | enthalpy | 0.00238989 | -167.476346 | -0.000284 |

The prior 64-layer entropy remap had −63,440.7 J/m² accumulated total remap
change (same setup and duration; its old wave observable differs). The new method
reduces the magnitude of this energy defect by about 99.7–99.9% across these cases.
The sign is not constrained: the eight-layer enthalpy remap gains energy while the
32/64-layer runs lose it. Dynamic enthalpy dominates the residual. Thermal residuals
are consistent with floating-point conversion/accumulation error in this experiment.

The matched salinity projection errors are larger with enthalpy remapping
than with entropy remapping. This thermal conservation improvement must not be
presented as improved wave accuracy. The no-remap controls remain better in this
short window. Neither short-window RMS nor repeated-reset tests substitute for a
complete-period coupled wave campaign.

The 32 one-remap audit rows are preserved in `enthalpy-ale-results/audits.json`.
The thermal conservation and native-inventory checks pass in all tested cases.
Mixed fronts produce entropy; tiny negative entropy changes in uniform cases are
assessed relative to their large inventory scales in the focused tests.

## Validation

Installed-wheel portable suite outside checkout: **783 passed, one CUDA skip**.
Twenty focused tests include extreme salinity, land, absent passives, repeated
remaps, full-step entropy accounting and an independent finite-difference check
of one thermal derivative through the complete remap. All 164 legacy ocean
integration tensors are exactly unchanged. Atmosphere portable comparisons pass
for 142 moist and 93 dry tensors. Runtime, defaults, fixtures and tolerances are
unchanged; historical exact fixture discrepancies remain a separate category.
