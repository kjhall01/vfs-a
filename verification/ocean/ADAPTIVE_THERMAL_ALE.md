# Adaptive layerwise thermal ALE correction

This research pass follows draft #41 (6dec5c01ab3353f33fbb9bd39a85bf5c394dc748).
It replaces that candidate's uniform column temperature adjustment with an
adaptive layerwise allocation. The target grid and provisional conservative remap
are unchanged. No installed runtime, defaults, rotation, atmosphere, fixtures or
validation tolerances change.

## Formulation and implementation

After potential-enthalpy remapping, freeze thickness, salt, passive inventories
and edge velocities. Let T0 be the provisional temperature, d = T - T0, and
w_k = h_k / sum(h). In each column seek a small mass-weighted change,

    minimize 1/2 sum_k w_k d_k^2
    subject to H_c(T) = H_c(before remap), S_c(T) >= S_c(before remap).

H_c is precisely the previous Hamiltonian column ownership, including kinetic
energy assigned through the actual interpolation transpose. S_c is mass-weighted
reference entropy. Temperature remains inside the existing thermodynamic domain.
This objective is an explicit allocation choice, not uniquely specified by the
Hamiltonian. It gives each unit of layer mass the same temperature-change penalty.

The implementation performs sequential linearized weighted minimum-change solves.
For energy gradient g, entropy gradient s, residual r and current displacement d,
the energy-only step is

    step = -d + W^-1 g (g.d - r)/(g.W^-1.g).

If that predicts an entropy decrease beyond the unchanged roundoff allowance,
project s orthogonally to g in the W^-1 metric, then add a correction in that
direction to make the linearized entropy change zero. This avoids a cancellation-
prone two-by-two normal-equation determinant. A required entropy direction that
is numerically rank deficient causes explicit rejection. Independent dense KKT
solves check both active and inactive constraint cases.

Analytic derivatives use the same reference thermodynamics and JMD95 quadrature
as the full Hamiltonian. Forward solves never call autograd. Fraction-to-boundary
damping keeps attempted temperatures inside the EOS domain; no state clipping or
inventory floors are introduced. After at most 24 iterations, reconstruct entropy
and independently evaluate full nonlinear H and S. Failed constraints reject.
The unchanged tolerances are 256 epsilon times the absolute column energy scale
and 512 epsilon times the absolute column entropy inventory. Identity returns the
original object. Land entropy is untouched. Temperature increments in the new
report have shape (columns, layers), unlike the parent's (columns,) report.

This is a **feasibility-seeking sequential projection**, not a certified global
nonlinear minimum. It stops once nonlinear feasibility checks pass. Active-set
switches and iteration-count changes are piecewise differentiable; a finite-
difference/AD test checks one smooth branch, not all possible branches.

## Previously rejected weak-stratification cases

Use zero velocity, salinity 35, temperatures 10 + a*[1,0,-1] degrees C and the
same strong reset to thickness fractions [0.2,0.3,0.5]. All three weak stable
cases rejected by the uniform correction now complete with unchanged tolerances.
All 32 columns activate the entropy constraint and finish in three iterations.

| a (C) | Maximum correction (C) | Maximum relative H residual | Minimum relative entropy change |
| --- | --- | --- | --- |
| 1e-6 | 1.38e-8 | 4.19e-16 | -3.05e-16 |
| 1e-4 | 1.38e-6 | 5.16e-16 | -2.70e-16 |
| 0.01 | 1.27e-4 | 4.27e-16 | -3.42e-16 |

The tiny negative entropy residuals are retained: this is nondecrease within
roundoff, not strict positive entropy production. Absolute column entropy
residuals are in the raw JSON and should be interpreted with their large column
mass. The a=1 case and all four reversed-gradient cases also complete. All 16
inherited front/near-fresh/upper-salt/tilted, coast and PCM/PLM audits complete.

## Scope and limitations

The adaptive solve resolves a demonstrated allocation failure. It does not prove
that every target grid admits an energy-conserving, entropy-admissible correction.
Rank deficiency, EOS limits and nonlinear nonconvergence remain explicit failure
modes. There is no fallback that silently violates a constraint. Total energy
closure trades potential enthalpy against dynamic/kinetic changes; separate
potential-enthalpy conservation is not claimed. The correction addresses the
remap only, not time-integrator drift.

The method does not yet prefer mixed layers or protect thermoclines through
custom penalties. It does not adapt the target grid or introduce the atmospheric
hybrid coordinate. Long wave runs, sharp fronts over many cycles, GPU behavior,
and gradients through active-set transitions remain unqualified.

## Reproduction

Build/install the wheel and run copied tests/tools outside the source checkout.

```sh
python -I -m pytest tests --fixture-mode=portable -q
python -I tools/probe_ocean_adaptive_thermal.py --audit --output audits.json
python -I tools/probe_ocean_adaptive_thermal.py --layers 8 --periods .05 --output wave-8.json
python -I tools/probe_ocean_adaptive_thermal.py --layers 32 --periods .05 --output wave-32.json
python -I tools/probe_ocean_adaptive_thermal.py --layers 64 --periods .05 --output wave-64.json
```

The JSON records driver/implementation hashes, CPU float64 measurements and raw
residuals. Hosted CI additionally executes the audits and a .001-period smoke run.

## Matched short wave measurements

Each run covers .05 internal-wave period and 197 actual remaps, using the same
salinity-projection observable as #41. These are short comparisons, not long-run
wave qualification. The correction preserves closure without materially changing
the measured wave error.

| Layers | Uniform correction RMS | Adaptive correction RMS | Accumulated remap energy (J/m²) | Maximum relative column H residual |
| --- | --- | --- | --- | --- |
| 8 | 0.00566400410095 | 0.00566400545035 | -3.33772e-05 | 9.21e-16 |
| 32 | 0.00282500205568 | 0.00282500143345 | -3.33772e-05 | 5.3e-16 |
| 64 | 0.00239001801123 | 0.00239001727191 | -3.33772e-05 | 5.35e-16 |

## Validation

Installed-wheel portable suite: **816 passed, one CUDA skip**. The 17 new tests
cover independent constrained algebra, rank failure, all three recovered cases,
conservative fields/coasts, identity/missing passives, EOS-domain failure, no
forward AD, the composed entropy ledger and one smooth directional derivative.
All 164 legacy ocean integration tensors remain exactly equal. Corrected Gibbs
atmosphere (142 comparisons) and dry helpers (93) pass their existing portable
checks. CPU float64 only; no CUDA or long-rollout qualification is claimed.
