# Instantaneous, locally funded backscatter

Replaces the finite-step prototype at `34cc640`; no default dynamics changes.
See [equations and integration](../../../../validation/temperature_coordinate/BACKSCATTER.md).

The timescale now sets a local power allowance f E/tau. Nonnegative local
vertex coefficients remain inside the adjoint-curl operator. A positive
incidence-power bound limits each cell's debit without a global reduction.
Momentum and reservoir tendencies share the same Hamiltonian and are evaluated
at every resolved SSPRK3 stage. All finite-step energy corrections are removed.

Tests independently differentiate full H to check its contraction with both
closure tendencies. They cover open/coastal geometry, active/inactive funding,
empty cells, unchanged remote forcing, zero flow/energy/coefficients, backward
finite differences, all three integration stages, source timestep rejection,
and composition with GM–Redi and decay. The locality test checks distant
accelerations bitwise, not just that some remote flow remains nonzero.

Four 20-step channel experiments cover isolated closure RK integration and
coupled dynamics with weak/strong viscosity. Maximum local power residual is
**3.933e-23 W/kg**. Reservoir contents stay nonnegative. With the deliberately
very strong coefficient, the 10 s integration accumulates about **1.124e-6 J/kg**
of total energy error: this is time-discretization error, not exact finite-step
conservation. It is retained explicitly in `budgets.json`.

The smooth isolated-closure refinement over 400 s gives:

| dt (s) | KE + reservoir energy error (J/kg) |
|---:|---:|
| 100 | 2.7754e-12 |
| 50 | 3.9102e-13 |
| 25 | 5.1658e-14 |
| 12.5 | 6.4194e-15 |

The refinement diagnostic sums only the changing KE and reservoir terms of the
same full H, avoiding subtraction against unchanged thermal/PE datums. Error
ratios approach eight, consistent with SSPRK3 in this smooth regime. This is
not a convergence claim across limiter switches or for the full coupled model.

Reproduce with `backscatter_campaign.py --mode budgets|refinement|time --output FILE`.
CPU float64, short integrations only; no GPU, long-run spectral/physical skill,
or angular-momentum conservation claim. Prior finite-kick results remain in
the parent directory as historical evidence, not current acceptance criteria.

Installed-wheel portable suite outside the checkout: **1217 passed, 2 skipped**,
200.33 s. This includes 14 current backscatter cases, replacing the 15 old
finite-kick tests. Both frozen comparisons pass (142 corrected-Gibbs, 93 dry
helpers). All 486 default-path tensors remain bitwise identical, including
backward sensitivities. Wheel/research source hashes and full logs are retained.

At 16384×40, one-thread eager CPU float64, isolated tendency evaluation has
median **0.3811 s** (0.4151, 0.3798, 0.3811 s). Three evaluations per SSPRK3 step
suggest about **1.14 s** incremental cost, approximately 7.6% of the prior
15.063 s dynamics timing. This is an estimate, not a timed combined step.
Isolated-process peak RSS is 0.777 GB and is not additive full-model memory.
