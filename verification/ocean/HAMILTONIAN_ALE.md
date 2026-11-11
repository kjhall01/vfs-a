# Column-local Hamiltonian ALE projection

This research pass follows draft #40 (`450bf96d6fa7b03b383ed83866ab9c0f26823933`).
It composes the existing potential-enthalpy remap with a local nonlinear energy
projection. JMD95, the energy functional, rotation, horizontal transport, momentum
remapping and installed runtime are unchanged. This is an explicit research choice,
not a new default or a general-purpose production ALE scheme.

## Local energy ownership from the existing functional

The discrete Hamiltonian is the sum of edge kinetic energy, column free-surface
energy, quadrature-defined dynamic enthalpy and reference thermal energy. Its
kinetic energy is linear in cell thickness when edge velocities are held fixed:

    K = sum_c,k h[c,k] * dK/dh[c,k].

The derivative already uses the transpose of the actual cell-to-edge interpolation,
including its two edge-side weights and coast mask. Consequently the column share
K_c = sum_k h[c,k] dK/dh[c,k] sums to the original edge energy without an arbitrary
half-edge split. Each column also owns its existing thermal, dynamic-enthalpy and
free-surface terms. `column_terms` uses exactly the original quadrature and EOS.
Independent tests verify that these column terms sum to the original Hamiltonian
and that their temperature derivative matches differentiation of the original H.

An edge momentum remap can affect energy owned by either neighboring column.
Thus a thermally corrected column need not itself have changed thickness. The
ownership rule accounts for that adjacency; no energy is spread globally.

## The constrained solve

First obtain the conservative provisional state from `EnthalpyRemapper`. Then
hold its h, salt, passives and edge velocities fixed. For each wet column solve:

    H_c(h_new, S_new, u_new, T_provisional + alpha_c) = H_c(state_before_remap).

Alpha_c is a uniform temperature increment over that column. This is an explicit
**allocation policy**, not a unique consequence of Hamiltonian mechanics. The scalar
value is determined by the full nonlinear energy constraint, not by an arbitrary
offset or a constant-heat-capacity estimate. The corresponding energy derivative is

    dH_c/dalpha = rho0*A_c*sum_k h_k*(a_T + mean_quadrature(Psi_T)).

Here a is the same surface potential enthalpy and Psi is the same JMD95 parcel
potential as the dynamical Hamiltonian. Ten bracketed analytic Newton iterations
retain tensor graphs; forward tendencies/solves do not invoke autograd. Brackets
lie inside the existing caloric temperature domain. Nonpositive derivatives,
unbracketed targets and roots that miss the roundoff-scaled residual check reject.
There is no clipping, automatic retry, cross-column redistribution or hidden fallback.

After solving, recover entropy using the same thermodynamic reference, validate
the state, and recompute the energy residual through the complete conversion back
to the stored entropy coordinate. Component differences are formed before summing
to reduce cancellation. Closure is to floating-point precision, not bitwise exact.
Global and column-reduced residuals can differ at that precision because reduction
order differs against a large background thermal energy.

The target is the state **immediately before remapping**, after the checked full
SSPRK3 step. The projection does not erase or compensate the horizontal integrator's
finite-timestep energy error. The returned ledger records both provisional and final
energy changes, per-column residuals, temperature increments and entropy changes.

Potential enthalpy is no longer separately conserved after the correction: its
change balances the remap's mechanical-energy change. Volume, salt, passives and
momentum remain exactly those of the conservative provisional remap.

## Entropy feasibility is a separate constraint

The final column entropy must not decrease beyond a roundoff-scaled tolerance.
This check is applied to the whole remap, including its mixing contribution and
energy adjustment. It is not enough that the energy root exists.

All 16 original profile/coast/PCM-PLM audits pass both constraints. An additional
feasibility sweep exposes failures: at zero velocity, S=35 and potential temperature
10 + epsilon*[1,0,-1] °C, a strong prescribed grid reset with PCM is rejected for
epsilon = 1e-6, 1e-4 and 0.01 °C because this projection decreases column entropy.
The epsilon=1 °C case and the four corresponding reversed-gradient cases complete.
The failing inputs are not altered, relaxed or repaired.

These failures establish a limitation of **this allocation policy and prescribed
remap**, not a proof that every energy/entropy-compatible representation is impossible.
A richer vertical adjustment, a different target grid, retention of unresolved
energy, or a justified mixing closure would require its own derivation and tests.
Until then, a caller must handle rejection explicitly; this candidate should not
be presented as a universally available replacement for the existing remapper.

## Wave scope and reproducibility

The wave setup and salinity-column projection are identical to #40. The 8/32/64-layer
runs cover 0.05 period (197 steps), with actual remapping every step. Compare the same
observable and same duration. No full-period, climate, GPU, compile or long-backward
qualification follows from these short trajectories. A directional derivative
through the complete corrected remap is tested away from switching surfaces.

Install the wheel and copy tools/tests/examples outside the checkout:

```sh
python -I tools/probe_ocean_hamiltonian_ale.py --audit --output audits.json
python -I tools/probe_ocean_hamiltonian_feasibility.py feasibility.json
python -I tools/probe_ocean_hamiltonian_ale.py --layers 8 --output wave-8.json
python -I tools/probe_ocean_hamiltonian_ale.py --layers 32 --output wave-32.json
python -I tools/probe_ocean_hamiltonian_ale.py --layers 64 --output wave-64.json
python -I -m pytest tests --fixture-mode=portable -q
```

CI runs the audits, feasibility cases and a shorter actual-remap trajectory. Reports
retain failures and numerical error measurements without subjective accuracy gates.
All uploaded files are below 128 KiB; no full fields, logs or wheels are uploaded.

## Measured wave results

| Layers | #40 remap energy (J/m²) | Corrected remap energy (J/m²) | Corrected salinity RMS | Maximum temperature increment (°C) |
|---|---:|---:|---:|---:|
| 8 | 88.023548 | -0.000100132 | 0.005664004 | 3.473e-08 |
| 32 | -131.511334 | 0.000216952 | 0.002825002 | 2.364e-08 |
| 64 | -167.476346 | 0.000166886 | 0.002390018 | 2.611e-08 |

Every wave completed 197 actual remaps. Column residuals remain around 1e-15
relative to background energy; accumulated global remap residuals are around 1e-4
J/m². This is numerical energy closure, not exact arithmetic. The total trajectory
energy drift remains separate and includes the horizontal integrator's error.

The prior enthalpy-remap salinity RMS values were 0.005662365, 0.002824675 and
0.002389893. The new results are very similar; energy closure has not repaired
mixing or wave-propagation accuracy. The entropy-producing mixed-front cases
remain feasible; the separate weak-stratification failures remain rejected.

## Validation

Installed-wheel portable suite outside checkout: **799 passed, one CUDA skip**.
Sixteen focused tests cover original-H column decomposition, independent thermal
derivatives, per-column closure and entropy, unchanged provisional salt/momentum,
land and identity handling, absent passives, thermal-domain and entropy rejection,
combined-step accounting, forbidden forward autograd and a directional derivative
through the corrected remap. No installed runtime/default/fixture/tolerance changes.
All 164 legacy ocean tensors remain exactly unchanged. Both atmosphere portable
comparisons pass (142 moist and 93 dry tensors). Existing historical exact fixture
discrepancies remain informational and separate from required portable checks.
