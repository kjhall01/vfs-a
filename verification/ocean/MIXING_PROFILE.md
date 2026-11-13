# Ocean mixing cost and cheaper evaluation

This is a CPU/float64 fixed-input profile of the research semi-implicit mixing
candidate. Supported runtime, defaults, tolerances and fixtures are unchanged.
The initial conditions are the 100 m, unequal-layer native-mesh cases from the
resolution campaign. Both 128 and 1,024 cells use 64 layers and a 300 s interval.
Mixing uses four substeps. No other campaign jobs overlap these measurements.
One warmup is excluded, followed by two unprofiled samples and a separate
instrumented mixing update. Torch uses one thread.

The dynamics control uses the supported SSPRK3 ocean step with horizontal TKE
transport and no mixing, remapping or boundary forcing. It is a cost control,
not a global circulation accuracy test or a realistic full-depth ocean setup.
A separate control advances five consecutive 60 s dynamics steps over the same
300 s interval. Neither control qualifies a new physics-call schedule.

## Reproduction

Run outside the checkout with the installed wheel:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python validation/ocean_column/profile_mixing.py --cells 128 --layers 64 --output profile128.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python validation/ocean_column/profile_mixing.py --cells 1024 --layers 64 --output profile1024.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python validation/ocean_column/benchmark_cadence.py --cells 1024 --layers 64 --output cadence.json
```

Nested timers temporarily wrap selected functions without changing their inputs
or outputs. Exclusive categories do not overlap; inclusive categories include
children and must not be summed. The outermost-category table attributes a
wrapped operation and all its descendants to its entry point. `cProfile` records
the top fifty functions by self time. Profiled durations include instrumentation
overhead; use the unprofiled samples for absolute costs. Numerical output is
compared bitwise against the uninstrumented update.

## Measurements

| Cells × layers | Mixing CPU s | One dynamics step CPU s | Mixing/dynamics | Mixing share of two-block cost |
|---|---:|---:|---:|---:|
| 128 × 64 | 9.85 | 2.18 | 4.53 | 81.9% |
| 1024 × 64 | 123.63 | 22.25 | 5.56 | 84.7% |

For the 1,024-cell instrumented call:

| Exclusive category | Seconds | Fraction of profiled wall time | Calls |
|---|---:|---:|---:|
| surface_gibbs | 48.788 | 58.07% | 1371 |
| jmd95_parcel_potential | 32.612 | 38.82% | 186 |
| banded_factor | 0.042 | 0.05% | 19 |
| banded_solve | 0.045 | 0.05% | 19 |
| momentum_diffusion | 0.069 | 0.08% | 8 |

The two thermodynamic kernels account for 96.89% of the profiled time. At the outermost-operation level, buoyancy-path work accounts for 34.17%, dissipation heating 33.27%, and state validation 14.10%. These are alternative partitions of the same call, not extra costs to add to the exclusive table.

Both instrumented updates reproduce all six output state tensors bitwise. The large-mesh energy residual is -1.03e-11 J/kg; the four-substep call retains 11 trials and zero whole-column deferrals.

**Timing variability:** the 1,024-cell instrumented call takes 84.01 s, less than the unprofiled samples (120.83–127.07 s wall time). The cause is not established; cache, allocator or host effects may contribute. Therefore these are provisional costs, not a demonstrated profiler overhead or an isolated hardware-throughput standard. Category fractions use the instrumented call only; cost budgets use the unprofiled CPU samples. Optimizations must be checked with interleaved/repeated before-and-after controls.

| Target mixing fraction, same call frequency | Allowed mixing CPU s | Required reduction from measured mixing cost |
|---:|---:|---:|
| 10% | 2.47 | 50.00× |
| 20% | 5.56 | 22.22× |
| 30% | 9.54 | 12.96× |

These targets are demanding at equal call frequency. Eliminating the banded-solve cost cannot materially close this gap. The 20% budget requires a reduction in the repeated thermodynamic work, not a small loop optimization.

## Why there are so many thermodynamic calls

For four mixing substeps, the pilot makes 48 enthalpy-to-temperature inversions
and 29 entropy-to-temperature inversions. Each inversion uses two bracket
endpoint evaluations, twelve fixed iterations and a final residual evaluation:
1,155 of the 1,371 surface-reference evaluations come from these inversions.
They restart from a broad-domain bracket even when a nearby temperature is
already known. This count is for the measured state, not every possible retry
pattern.

The four dissipation-heating inversions call the parcel-energy routine 112 times:
per substep, one initial derivative, six iterations with four derivative
calculations each, and three final path-quadrature calculations. Heating needs
only the temperature derivative, but `parcel_potential` computes energy and all
three derivatives, including the reference-density contribution which has zero
temperature derivative. Layer averaging and the parcel integral use nested
four-point quadratures. Changing their order would change the defined discrete
Hamiltonian; avoiding unused outputs need not change it.

Other duplicated work includes midpoint conjugates computed in both the scalar
trial and path-work routine, construction of a default buoyancy-work expression
that the path-work implementation replaces, and repeated state validation that
reconstructs temperature. These are implementation opportunities, not evidence
that guards should be removed.

## A cost budget tied to the call schedule

Let C_m be the cost of a mixing call and C_d the cost of all dynamics steps
between mixing calls. If mixing may occupy a fraction f of their combined cost,

    C_m <= f/(1-f) * C_d.

The 10%, 20% and 30% scenarios in the evidence are planning choices, not
scientific acceptance thresholds or claims about other models. A provisional
20% budget means mixing may cost at most one quarter of the dynamics work in
that interval. More dynamics steps between physics calls increase that allowance,
but do not make the physics call itself cheaper. Remapping, forcing, I/O and
other components are excluded; a full-model budget also needs an absolute
throughput target. An inefficient dynamics implementation must not become a
reason to accept an inefficient closure.

The five-step, dt60 dynamics interval takes 109.64 CPU seconds (samples
111.75 and 107.53 s). Paired for cost accounting with one dt300/four-substep
mixing call, mixing would represent 53.0% of the two-block cost. A 20% target
allows **27.41 CPU seconds per mixing call**, a **4.51-fold reduction** from the
measured 123.63 s. A 30% target allows 46.99 s (2.63-fold reduction).

This is a more attainable provisional engineering milestone than the equal-
frequency 5.56 s target. It assumes a multi-rate driver: dynamics every 60 s,
mixing every 300 s with four internal mixing substeps. The costs were measured
separately; the combined schedule's splitting error, stability and long-run
budgets have not been qualified. There is no automatic cadence/default change.

## First candidate: the same energy, evaluated less often

Retain JMD95, the discrete Hamiltonian, native momentum/TKE energy transfers,
conservative scalar fluxes and the existing acceptance checks. Use a local
thermodynamic work buffer containing temperature, salinity and potential
enthalpy through a mixing call; convert the public entropy state at entry/exit
rather than repeatedly recovering the same fields. Cache geometry-dependent
quadrature pressures and reference-density quantities while layer geometry is
fixed. Caches must belong to that explicit call and cannot survive a state or
geometry change implicitly.

For temperature recovery, use a known nearby temperature as the Newton starting
point, retain the bracket as a safeguard, and stop only after satisfying the
existing physical-domain and numerical residual checks. For fixed salinity,
precompute the surface Gibbs polynomial's salinity-dependent coefficients and
evaluate just the caloric quantities required by the inversion. Compute the
full thermodynamic record when it is actually needed. No speedup or reduced
iteration count is assumed without measurement.

Give dissipation heating a temperature-derivative-only kernel and a
residual-controlled iteration. Keep its cancellation-safe heat-increment
integral and independent full-H check. Compute accepted-state energy/entropy
checks once per accepted substep, reusing fields already calculated at that exact
state. Preserve all genuinely distinct admissibility checks. Remove duplicated
midpoint/default-work evaluation only after showing that the retained work and
reports agree to their existing numerical criteria.

This changes evaluation strategy rather than the physical formulation. It is
the preferred first implementation because it addresses the measured repeated
work without first weakening the energy constraint or replacing JMD95.

## Second candidate: frozen-conjugate trials, nonlinear acceptance

If the first candidate misses the cost budget, move full thermodynamics out of
repeated funding trials. Let x contain layer salt and potential-enthalpy amounts
at fixed thickness, and let mu be the corresponding derivatives of the thermal
plus stratification Hamiltonian. Freeze mu during trial construction:

    Delta H_trial = sum_i mu_i dot Delta x_i.

For the stratification energy averaged over a layer, the frozen conjugates in
potential-enthalpy/salinity coordinates are a = Psi_theta / h0_theta and
b = Psi_S - a*h0_S. The conservative reference-enthalpy transfer cancels
analytically. This avoids making small TKE budgets pay subtraction error from
large background thermal energies.

Because Delta x is a conservative flux divergence, this work can be written as
interface fluxes times differences of mu. Use it to estimate affordable scalar
conductances with inexpensive banded solves. Preserve the native exact KE loss,
TKE production, conservative scalar flux form and positive transport structure.
Then evaluate the actual nonlinear Hamiltonian/path work for the proposed
accepted state, rather than on every cheap trial.

Frozen conjugates alone do NOT close the original nonlinear Hamiltonian. The
remainder R = Delta H_actual - Delta H_trial must be accounted for using the
same defined local energy-flux allocation. If positive R is to be paid from TKE,
that debit must be affordable in every affected layer. Otherwise reduce the
conservative exchange or take a smaller substep and retry. In the current
ordered split, use (M+d)*k_new = M*k_transport - B_actual and heat = d*k_new,
where M is layer mass and d the fixed dissipation coefficient times timestep
and mass. Replacing trial work with actual work must update both final TKE and
dissipated heat consistently, before thermal inversion; it is not a TKE-only
patch applied after heating. A global energy
check alone cannot establish local affordability. Negative R must also have
an explicit allocation; it cannot silently disappear. Evaluate this remainder
using cancellation-safe increment/path expressions, not naive subtraction of
large total Hamiltonians. Keep the independent full-H check as a check, not as
the numerical source of a tiny local work debit. Retain entropy, domain,
positivity and full-energy checks before commitment, with the current solver
available as a diagnostic fallback.

The weak-TKE interfaces seen in the resolution campaign are the hard case:
small absolute work errors can exceed their available TKE. This design is only
a proposal until fallback frequency, coefficient reduction, accuracy and cost
are measured on those cases. It does not justify clipping TKE, hiding work in
an untracked reservoir, or claiming that a frozen linearized energy equals the
nonlinear JMD95 energy.

## Decision sequence

First prototype the thermodynamic work buffer, warm-started residual-controlled
inversions and derivative-only heating against the current implementation.
Measure the same 1,024-by-64 case in isolation, plus weak-TKE and long-column
controls. Retain full budgets and track accepted fluxes and limiter activity.
Only then decide whether frozen-conjugate trials or a different physics cadence
are needed. Neither a faster banded solver alone nor an unmeasured GPU speedup
addresses the evidence presented here.

## Validation and evidence

Both instrumented cases match the six uninstrumented state tensors bitwise;
unprofiled repeat outputs also match exactly. Installed-wheel focused tests:
**8 passed** (8.61 s). Corrected-Gibbs and dry-helper portable comparisons pass.
No full-suite rerun is claimed for this profiling-only change; the parent
campaign retains its 1,031-test result. All benchmark inputs and outputs are
CPU/float64, with no GPU or backward qualification. Proposed numerical changes
have not been implemented or timed.

Compact profiles, unprofiled samples, call-schedule budget calculations and
regression evidence are in `mixing-profile/`. No binary profiler dump is uploaded.
Every uploaded file is below 128 KiB.
