# Coupled ocean column prototype

## Status

The research-only `CoupledClosure` solves an end-of-step TKE consistency problem
with implicit momentum, scalar and TKE transport and dissipation. Accepted steps
retain positive TKE/passive contents, conservative scalar fluxes and checked
Hamiltonian budgets. It is **not a supported replacement** for `TKEClosure`.

The conservative, unaccelerated configuration completes the one-hour wind checks,
but is expensive. The accelerated configuration passes short tests yet fails the
one-hour matrix. The current prototype does not establish a faster solution at
matched accuracy. It does establish an executable coupled formulation and exposes
where a better nonlinear solver is needed. No runtime package, defaults, fixtures,
tolerances, atmosphere code or physical boundary sources are changed by this PR.

## Equations and ownership

The unknown is cell-layer end TKE k. At each nonlinear iterate, diffusivities are
computed from k, and native edge momentum plus scalar contents are solved using
batched tridiagonal backward-Euler systems. All iterates start from the same
original state; provisional steps do not advance time or accumulate energy.

For a column, the integrated TKE equation is

    M (k_new-k_old) = P - B - Q + divergence(F_k).

M is rho0 times cell area times layer thickness. P is the actual native resolved
kinetic-energy loss, including backward-Euler temporal loss. B is nonlinear
buoyancy/thermodynamic work. Q is dissipated TKE, deposited as heat using the
existing full-H thermal inversion. F_k is explicitly represented vertical TKE
energy transport. A layer can receive energy through F_k during the same update;
there is no arbitrary column-wide redistribution or nonlocal budget borrowing.

The final TKE controls all diffusivities and the implicit dissipation coefficient.
This is first-order endpoint coupling, **not** a second-order method. Thermal
conversion of dissipated energy is applied after the coupled transport solve;
its nonlinear heating inversion is independently checked. Scalar transport now
uses a full tridiagonal solve rather than colored adjacent-pair sweeps. These
changes, interface averaging and backward-Euler dissipation all affect the
numerical solution; comparison with the previous solver does not isolate time
coupling alone. In particular, the old isolated decay update is exact while this
prototype's isolated decay is backward Euler.

Shared C-grid edge momentum is computed from the current batched cell TKE and
charged back with the existing native kinetic metric. Unequal neighboring layer
thicknesses retain the known column-to-column kinetic-energy exchange term, whose
domain sum vanishes. No independent per-cell momentum substeps are introduced.
All linear solves are banded and batched; nonlinear iteration can still be costly.

## Interface diffusivity

The `resistance` option computes momentum/scalar conductance from two half-layer
resistances in series:

    G = 2 K_upper K_lower / (h_upper K_lower + h_lower K_upper).

Exact zero coefficients give zero conductance, with no floor or singular reciprocal.
Unlike arithmetic averaging, this prevents the more turbulent side from forcing
finite scalar mixing through a side with zero diffusivity. This is a spatial
flux discretization choice, not evidence of a calibrated turbulent closure.

For TKE's own transport, the nonlinear identity

    sqrt(k) gradient(k) = (2/3) gradient(k^(3/2))

provides a finite divided-difference conductance, including a positive coefficient
when one side has zero TKE and the other is turbulent. Thus energy can propagate
into a depleted layer before/simultaneously with renewed momentum/scalar mixing.
The exactly zero-TKE whole-column state still cannot bootstrap turbulence.

An explicit `arithmetic` option preserves the first prototype's interface choice.
Its failed attempts are retained. Observed iterations can drive one reservoir
almost to zero while continuing to demand positive work from it; this does not
constitute a mathematical proof that no other positive root exists.

## Work and energy residuals

Let Psi be the layer-averaged JMD95 dynamic enthalpy and h0 the transported
surface reference potential enthalpy. Conjugate gradients are evaluated at a
midpoint thermodynamic state:

    a = Psi_theta / h0_theta
    b = Psi_S - a h0_S.

An interface energy flux contains the conserved h0 transfer plus weighted a and b
times enthalpy and salt transfers. The remaining actual nonlinear Psi increment
is local buoyancy work. This defines a **particular local work allocation**, with
its nonlinear remainder retained; energy closure alone does not establish its
physical accuracy or uniqueness.

The reference h0 flux cancels algebraically in B. Making local TKE pay the floating
point enthalpy-to-entropy-to-temperature reconstruction error can swamp exhausted
reservoirs. That error is instead reported as `caloric_reconstruction_residual`.
It is not hidden: the final energy check evaluates the complete actual Hamiltonian,
using the same production acceptance tolerance as the existing closure.

Positive source/sink decomposition is used only for the inner Patankar iteration.
Final acceptance checks the **actual** TKE energy-equation residual, not merely
agreement between successive linearized iterates. A signed direct solve is used
when it produces an admissible answer. Otherwise the positive iterate is acceptable
only if it independently passes both nonlinear-consistency and energy-residual
checks. Negative solutions are never clipped and no energy is silently deposited
into heat to rescue a failed solve.

The nonlinear state criterion uses rtol=1e-6 and atol=1e-12 m2/s2. A separate local
budget criterion uses atol=1e-13 J/kg and rtol=1e-9 of the integrated local source/work
scale. These are explicit research solve tolerances, not TKE floors or relaxed
production energy/entropy guards. Nonconvergence raises before committing inputs.
The reported budget closes to floating-point reconstruction and nonlinear-solve
residuals, not an assertion of bitwise-zero energy error.

## Evidence

The final focused tests cover an independent dense reference for the banded solver,
stable/unstable mixing, unequal shared-edge geometry, local/global budgets, scalar
conservation, positive transport into an initially zero-TKE layer, exact preservation
of an entirely zero-TKE state, implicit decay against an independent scalar root,
and unchanged inputs after failed convergence.

All eight six-minute wind cases complete (16 layers; dt=180, 90, 45, 11.25 seconds;
existing and coupled methods). Errors below use **each method's own** 11.25-second
endpoint, not analytic truth or an established converged limit:

| Method | dt (s) | Velocity relative L2 (%) | TKE relative L2 (%) | Temperature RMS (deg C) | Observed time (s) |
|---|---:|---:|---:|---:|---:|
| Existing | 180 | 0.7162 | 8.7557 | 0.002704 | 1.12 |
| Coupled | 180 | 0.3946 | 22.0815 | 0.000277 | 4.22 |
| Existing | 90 | 0.2587 | 2.0955 | 0.000812 | 2.11 |
| Coupled | 90 | 0.1877 | 11.2885 | 0.000163 | 5.35 |
| Existing | 45 | 0.1176 | 0.8080 | 0.000204 | 4.21 |
| Coupled | 45 | 0.0813 | 5.1021 | 0.000079 | 5.84 |

There is no across-the-board accuracy gain. TKE temporal error is larger in the
prototype; velocities and temperature have smaller same-method temporal differences.
Different semidiscretizations mean these differences cannot rank absolute physical
accuracy. Times are concurrent one-thread CPU observations, not a controlled speed
claim or a prediction of GPU/global-model throughput.

With 80 iterations, relaxation 0.7 and safeguarded two-iterate acceleration, all
eight one-hour coupled cases reject after partial progress, whereas all five
existing-method wind controls complete. Wind rejection times range from roughly
22 to 50 minutes; the 32-layer dt300 case rejects at 30 minutes. Cooling and
rain/heating also reject. Their accepted prefixes retain tight budgets; they are
not completed qualifications. Loosening only the state atol to 1e-10 does not
rescue the dt300 case. Failed cases are never compared against completed endpoints.

Disabling acceleration and using relaxation 0.3 with a 240-iteration limit completes
all three one-hour wind checks: dt600 in 34.1 s (137.5 iterations/step), dt300 in
52.6 s, and dt150 in 105.0 s (87.7 iterations/step). Turning acceleration back on with that same
damping still rejects at 30 minutes. The cautious configuration is useful as a
reference, not a demonstrated performance solution. Adjacent-timestep results and
all failure ledgers are in [the analysis](coupled-column/analysis.json). Against
the cautious dt150 endpoint, dt300 differs by 0.502% in velocity and 17.6% in TKE;
dt600 differs by 1.37% and 52.8%, respectively. The dt150 endpoint itself is not a
converged reference. These differences provide no justification for increasing
the supported model timestep.

## Next decision

Keep the existing supported closure. The most useful next prototype is a robust
block Newton or nonlinear block relaxation method with analytic derivatives,
positive line search and a banded preconditioner. It should solve the coupled
momentum/content/TKE residuals directly rather than repeatedly solving nested
transport systems with slowly converging coefficients. Native edge/cell coupling
must remain explicit. No forward-autograd tendency construction is needed.

Before optimizing implementation, require the nonlinear solver to complete the
fixed one-hour matrix at strict residual tolerances. Then test timestep and
vertical refinement, isolate the interface/dissipation changes, and compare wall
time at matched velocity, TKE and tracer accuracy. Only after that should the
48-hour forcing campaign or production integration be considered.

This candidate adds no new surface TKE input, length-scale equation, horizontal
EKE closure, atmospheric coupling, GPU/compile qualification or long-run validation.

## Regression and retained evidence

Final installed-wheel portable suite: **1005 passed, one CUDA skip**. Seven new
focused tests passed outside the checkout. Corrected-Gibbs and dry-helper portable
comparisons pass. All accepted prefixes and completed runs in the final experiment
have absolute energy-minus-input residuals below 4e-11 J/kg. Failed runs remain
failed, regardless of their small residuals before rejection.

The scalar ledgers of earlier failed formulations and reversible source patches
are retained alongside the final cases. `campaign-source.patch` differs from the
final source only in failure diagnostics. Each uploaded artifact is below 128 KiB.
The old oversized campaign figure remains omitted. No forward runtime source was
changed; the research prototype and its tests are outside the installed package.
