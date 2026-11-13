# Linear-EOS column control

## Finding

Linear density alone does not cure the coupled solver's failures. Constant heat
capacity makes it much cheaper, but the original work-evaluation path still fails.
Using the **analytic affine mixing-work formula**, with the same linear physics,
solver settings and budget guards, completes five of six one-hour controls that
include timestep and vertical-resolution variations. The dt600 case still rejects,
and a requested 48-hour run rejects after 5 h 55 min. This is a useful diagnostic
control, not a replacement ocean or a completed long-run qualification.

This result shifts the next priority toward cancellation-safe mixing-work evaluation
for JMD95, alongside nonlinear solver robustness. It does not establish that a
linear EOS is physically adequate or that every JMD95 convergence failure has the
same cause. No supported package code, EOS default, fixture or tolerance changed.

## Three matched controls

All use the same initial T/S, native momentum metric, forcing, mixing length, TKE
closure and solver settings as the prior offline column campaign:

1. JMD95 density and the existing surface Gibbs caloric reference.
2. Linear density with the existing caloric reference (isolates density-law changes).
3. Linear density with constant cp=3990 J/kg/K (simplifies both density and calorics).

The linear law is

    rho = rho0 [1 - alpha (T-T0) + beta (SP-S0)],
    Psi = g z [-alpha (T-T0) + beta (SP-S0)].

alpha=0.0001908821582383638 K^-1 and beta=0.0007568878444130999 per SP match
JMD95 partial derivatives at T=12.5 C IPTS-68, SP=34.75 and p=0 dbar, divided by
rho0=1025 kg/m3. This matches local derivatives, not the entire nonlinear density
profile or pressure dependence. T0=10 C and S0=35 retain the existing datum.

The constant-cp control uses h0=cp*T and eta=cp*log((T+273.15)/273.15), independent
of salinity. This deliberately idealized caloric model uses the harness temperature
coordinate directly; it is not a GSW/TEOS replacement. Its entropy inversion still
uses the existing bracketed routine; enthalpy inversion is explicit. The adapter
rejects horizontal stepping and remains outside the installed VFS package.

## Analytic work control

For equal exchanged reference volume V between two fixed layers,

    W = g V (rho_lower-rho_upper) (z_upper-z_lower).

More generally, let f_H and f_S be time-integrated upward potential-enthalpy and
salinity content fluxes per area. With fixed geometry and constant cp,

    W_interface = rho0 A g (z_upper-z_lower)
                  [-alpha f_H/cp + beta f_S].

This is exact for the affine model, not a small-exchange approximation. Interface
work is assigned to adjacent layers with the same geometric fractions used by the
coupled prototype's energy-flux definition. Therefore the analytic expression is
algebraically equivalent to its finite-energy-difference work, apart from finite
precision thermodynamic reconstruction and subtraction errors.

`ExactLinearWorkClosure` changes only this research work evaluation. Full actual
Hamiltonian, entropy, positivity and nonlinear-residual acceptance checks remain.
The existing finite-work path remains the default; no negative-TKE clipping, hidden
substeps, changed tolerance or compensating heat source is introduced.

## Results

In the 15-case primary matrix, all six cautious/existing 16-layer dt300 cases
complete the hour. All nine accelerated coupled cases reject: for each EOS,
16 layers at dt300 and dt75, plus 32 layers at dt300. Finite-work evaluation thus
remains problematic even with fully affine thermodynamics.

One-hour dt300 controls (single-thread CPU, concurrent jobs; indicative timing):

| Thermodynamics | Accelerated coupled | Cautious coupled time | Mean cautious iterations/step |
|---|---|---:|---:|
| JMD95 + existing calorics | Rejects at 40 min | 58.4 s | 108.0 |
| Linear + existing calorics | Rejects at 35 min | 57.1 s | 104.8 |
| Linear + constant cp | Rejects at 40 min | 5.80 s | 100.8 |

A separate interleaved fixed-input benchmark (three timed repetitions after warmup,
16 layers, dt150) gives median closure times of 5.72, 6.19 and 0.617 seconds,
respectively. The methods took 122, 155 and 155 iterations. Much of the cost reduction
comes from the caloric evaluation/inversion, not fewer nonlinear iterations. These
are CPU column measurements, not GPU or full-dycore throughput predictions.

With **analytic work**, accelerated linear-cp runs complete one hour at dt300,
150, 75 and 18.75 seconds with 16 layers, plus dt300 with 32 layers. The 16-layer
dt300 case takes about 2.0 s in the concurrent matrix and averages 19 iterations
per step. Its final fields agree closely with the completed cautious finite-work
control: relative velocity difference 1.76e-9, relative TKE difference 1.41e-8,
temperature RMS difference 1.64e-10 C, and maximum absolute TKE difference
1.46e-13 m2/s2. Relative ratios in near-zero TKE layers can still be enormous;
this is not local entrainment qualification.

The dt600 analytic-work case rejects on its first step. The attempted 48-hour
dt300 run rejects at 21,300 seconds (5 h 55 min). Its accepted-prefix energy-minus-input
residual is -3.30e-10 J/kg; that does not make the failed rollout a success.

Same-method comparisons against the one-hour dt18.75 numerical endpoint:

| dt (s) | Velocity relative L2 (%) | TKE relative L2 (%) | Temperature RMS (C) |
|---:|---:|---:|---:|
| 300 | 0.9825 | 36.51 | 0.0002474 |
| 150 | 0.4894 | 17.45 | 0.0001264 |
| 75 | 0.2189 | 7.64 | 0.00005684 |

The reference itself is not proven converged, and no analytic trajectory is claimed.
The independent analytic benchmark is the exchange-energy identity, not the evolving
column. Accurate work evaluation improves feasibility, but does not remove temporal
error or justify a larger supported timestep. Final actual-H residuals remain small
(the fine one-hour case accumulates about -1.51e-9 J/kg); they are not asserted zero.

## Interpretation and next step

The same linear physics succeeds with analytic work where energy differences fail.
That directly implicates work evaluation/reconstruction sensitivity in the tested
case; simply blaming JMD95 nonlinearity or the existence of per-layer TKE is too
strong. Remaining dt600 and long-run failures show that this is not the only issue.

Keep JMD95 as the target. Next compare a cancellation-safe nonlinear work calculation
against the current difference calculation: integrate analytic thermodynamic energy
derivatives along the conservative heat/salt exchange, then check the full H
independently. The affine case supplies its exact limiting reference. Preserve the
local energy flux/allocation and distinguish numerical residuals from physical
mixing costs. Only after that comparison should a more elaborate nonlinear solver
or a change to the physical work allocation be selected.

## Reproduction and evidence

See [commands](../../validation/ocean_column/README.md#linear-eos-controls).
`linear-eos/primary` contains the primary cases and interleaved benchmark;
`linear-eos/exact` contains the analytic-work matrix and failed long attempt.
The [analysis](linear-eos/analysis.json) never compares failed prefixes against
completed endpoints. Source hashes accompany runs. Reversible zero-context patches
reconstruct the initial primary experiment sources from the final research sources;
apply with `patch -p0` in a scratch directory containing those files.

The tests independently check the equal-volume work formula and its exact linear
scaling, analytic Hamiltonian derivatives, both caloric controls' conservation and
positivity, and the interface-work allocation. No GSW dependency, external dataset,
GPU/compile qualification or end-to-end backward qualification is added. All
published artifacts satisfy the 128 KiB file cap.

Final installed-wheel portable suite: **1010 passed, one CUDA skip**. Five final
linear-control checks pass; the initial linear/coupled focused run passed eleven
checks. Corrected-Gibbs and dry-helper portable fixture comparisons pass.
