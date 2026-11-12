# Forced-column qualification campaign

**Verdict: active mixed-layer evolution is not qualified.** All primary scalar-
mixing cases reject their first 150/300-second step because local buoyancy work
exceeds an adjacent TKE reservoir. Reducing the timestep permits a short advance
but does not produce sustained evolution. This is a controlled rejection, not
an accepted negative TKE state or a repaired energy budget.

This report uses the merged ocean at `8e8147bd31c341a452b8df2f0c019d39a7dd5b8b`.
No production equations, coefficients, guards, fixtures or tolerances change.
The [reproduction guide](../../validation/ocean_column/README.md) defines all
inputs, the isolated f-plane metric adapter, criteria and limitations.

## Experiment and references

A 100 m column begins at rest with a warm/fresh upper 20 m transitioning smoothly
to colder/saltier water. Potential temperature is 10–15 C; salinity 34.5–35.
Wind stress is 0.1 Pa and f=1e-4 s⁻¹. Requested duration is 48 h (2.75 inertial
periods). Initial k=1e-4 m²/s²; prescribed L=2 m and existing default closure
coefficients are fixed. The primary matrix combines 16/32/64 layers,
300/150-second steps, and wind-only, additional cooling, or additional
rain/heating forcing: 18 cases.

Six smaller-step wind probes and two separately labelled seed sensitivities
investigate the failure. Six wind-driven momentum/TKE controls disable scalar
mixing, retaining shear production, TKE diffusion and dissipative heating.
These controls cannot establish entrainment or density-driven mixing accuracy.

The two-component one-m² column adapter preserves the native rho*h momentum
metric and rho/2 sum h*(u²+v²) energy exactly. Rotation is an analytic f-plane
rotation split around the installed production physics. No spherical rotation,
horizontal pressure/transport, ALE or basin dynamics are exercised. Independent
checks verify metric/work, wind impulse and the analytic depth-integrated wind
response. This is an actual prognostic-TKE physics campaign, not a new dycore grid.

## Active scalar-mixing failures

| Layers | dt=300 s | dt=150 s | dt=30 s | dt=3 s |
|---:|---:|---:|---:|---:|
| 16 | 0 s | 0 s | 90 s | 108 s |
| 32 | 0 s | 0 s | 60 s | 93 s |
| 64 | 0 s | 0 s | 60 s | 93 s |

Entries are accepted simulation time before the first rejected attempt, for
wind-only forcing. Cooling and rain/heating also reject immediately at both
primary timesteps. Their mixed-layer responses therefore cannot be assessed.
All failures report `local buoyancy mixing exceeds available TKE in an adjacent
layer; no nonlocal borrowing`. A failed attempt is transactional: forcing and
mixing from that attempt are not committed to the accepted state or budget.

At 32 layers and dt=3 s, increasing initial k to 1e-3 and 1e-2 m²/s² extends
accepted time to 225 and 333 s, respectively, then triggers the same rejection.
These sensitivity runs do not replace the failed primary runs. Increasing the
seed alone does not establish robust behavior.

## Interpretation and next numerical task

The guard correctly detects an unaffordable local energy transfer. The campaign
shows that the current prescribed-diffusivity scalar exchange plus rejection
policy cannot sustain this ordinary forced stratified configuration. It does
not show that the energy guard should be loosened or that a long climate run
would be sound. Smaller timesteps delay failure toward local reservoir depletion;
there is no demonstrated successful timestep-converged active trajectory.

The next focused change should make each adjacent-interface scalar exchange
energy-affordable before committing it, reducing the local exchange when needed
while preserving its paired content transport and measured Hamiltonian work.
A coupled positive TKE/mixing solve is another option. Either requires separate
qualification of zero-reservoir behavior and ordering with TKE production and
transport; no remote reservoir borrowing, clipping or arbitrary heat injection
should be used to hide this failure.

Only after that change should the same fixed campaign be rerun and extended to
independent entrainment/mixed-layer references. These results do not justify
coefficient tuning, a claimed realistic Ekman layer, or moving directly to a
forced basin qualification.

## Completed momentum/TKE controls

All six controls reach 48 h. The analytic transport is
(-0.975609006, -0.974399605) m²/s. Relative transport error decreases from
3.7501e-5 at dt=300 s to 9.375e-6 at dt=150 s for every layer count, approximately
second-order for the split rotation/wind impulse. Final energy-minus-input
residual magnitudes are 1.23e-9–2.61e-9 J/kg; mass, salt and passive inventory
residuals are zero in these controls.

Integral agreement does not imply converged profiles. At 48 h, halving dt changes
the full two-component velocity profile by 4.81%, 7.14%, and 7.63% for 16, 32,
and 64 layers (Euclidean norm relative to the dt=150 s profile). At dt=150 s,
vertical refinement changes the profile by 8.53% (16→32) and 2.75% (32→64),
after pair-averaging fine velocities to the coarse layers, relative to the
averaged fine profile. Initial profiles are centre sampled, not common exact
layer averages. Surface work ranges from 761 to 2095 J/m² across controls.
Neither active mixing nor full profile convergence is established.

The generated figure exceeds 128 KiB and is omitted from this PR.
The reproduction script can regenerate it locally.

## Regression and evidence

Installed-wheel suite outside the checkout: **985 passed, one CUDA skip**
(CPU float64). Three harness checks independently cover native kinetic energy,
rotation and wind impulse, and short-run analytic integral transport. The final
harness-only rerun also passes all three. Production source is unchanged.

[Compact results](column-campaign/summary.json) contain every case, including
rejections; individual case files contain initial/final profiles and hourly
energy residuals. [Profile comparisons](column-campaign/profile-comparison.json)
quantify control sensitivity. [Omissions](column-campaign/omissions.json) lists
oversized raw profile files that were left out. Raw SHA256 values identify the
source records; wall times and JSON hashes are not reproducibility gates.
See [provenance](column-campaign/provenance.json) and
[test output](column-campaign/tests.log).
