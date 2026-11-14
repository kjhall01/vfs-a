# Cheaper discrete ocean energy and tensor-kernel fusion

`StratifiedOcean(..., quadrature_order=2)` selects two Gauss points in each of the
two integrals defining the dynamic-enthalpy Hamiltonian. The established default
is still four in each. This changes the discrete energy, not the JMD95 EOS. All
pressure/content derivatives, heating, ALE correction, and the research mixing
workspace use the same selected rule. This option is experimental scientific
configuration, not a claim of global-run qualification.

For a layer with midpoint z and thickness h, the existing construction is

    H_anomaly = rho0 * A * h * sum_i w_i Psi(T,S,z+h*(xi_i-1/2))
    Psi(T,S,z) = z * sum_j w_j g/rho0 * [rho(T,S,p(z*xi_j))-rho_ref(p(z*xi_j))]

Differentiating these finite sums, including their moving sample locations,
produces the implemented forces. Reducing the order preserves that derivative
identity for the new energy. It does not make the new energy equal to the old
one, and exact spatial work closure does not imply exact time-integrator energy
conservation. The EOS is evaluated at q² locations, but whole-model cost also
includes surface thermodynamics, nonlinear inversions, transport and mixing.
Temperature/salinity polynomial intermediates are already broadcast across
quadrature locations: not every EOS operation is repeated q² times. The expanded
pressure-dependent arithmetic and arrays are the main quadrature-dependent cost.

## Accuracy comparison

`validation/energy_quadrature/accuracy.py` evaluates the same physical states at
orders 2, 3, 4, 8 and 12. Order 12 is a numerical reference, not analytical truth;
order 8 is a convergence check. Cases span 100, 4000 and 6000 m, 8/32/64 layers,
unequal neighboring layer thicknesses, smooth profiles, sharp horizontal T/S
fronts and constant-reference rest. The 32-cell mesh is intentionally a cheap
operator test. Pressure errors refer to the JMD95 Hamiltonian pressure block,
not the complete entropy/front-limited timestep. This is not fine-mesh balance
or wave-growth qualification.

Across non-reference profiles:

| Order | EOS locations/layer | Maximum pressure-force relative L2 difference | Maximum anomaly-energy difference, J/kg |
|---|---:|---:|---:|
| 2 | 4 | 6.58e-6 | 1.77e-4 |
| 3 | 9 | 1.35e-9 | 4.38e-8 |
| 4 (default) | 16 | 1.11e-12 | 5.06e-12 |
| 8 | 64 | 7.29e-13 | 9.24e-14 |

Maximum absolute RMS acceleration difference for order 2 is 3.10e-11 m/s².
At 100 m its relative pressure difference is about 2e-12; at 4000 m it reaches
1.47e-6, and at 6000 m 6.58e-6. Increasing the layer count does not remove this
floor: the inner integral still spans from sea level to the parcel depth.
The shallow full-ocean timing profiles therefore provide little evidence about
deep-ocean approximation error. Reference-rest cases are retained separately.

Independent reverse-mode differentiation verifies analytic energy derivatives
and actual tendency work at orders 2/3/4, including coasts. No forward tendency
uses autograd. Reduced-order heating and ALE tests retain existing energy guards.

## Execution experiments

A pure `_values` kernel is separated from `SurfaceGibbs.evaluate`; its public
validation checks and default expressions are retained. The isolated performance
harness offers `--compile-thermo` for JMD95 density/partial and heating kernels,
and `--compile-gibbs` for surface Gibbs and the research caloric kernel. These
flags originally patched kernels inside the benchmark process. The final combined
path calls `ocean.compile_thermodynamics()` and owns compiled functions per model.
Single-kernel attribution switches remain research process-local patches. Guards,
inversion residuals and funding decisions stay in place. The method compiles
thermodynamic kernels, not the whole model, and compilation is lazy. Compilation may change roundoff and limiter diagnostics.

`kernels.py --compile` tests an isolated EOS-partials kernel at 1024×64. Warm
median seconds were 0.134 eager versus 0.00883 compiled at order 4; order-2 eager
was 0.0233. These are kernel timings, not model speedups. Compiled order-2 was
slower than compiled order-4 in this sample, demonstrating shape/compiler and
sampling effects. Compilation startup is separately reported (about 24 s for
the first order-4 call). No Mac/MPS/GPU speedup is inferred.

Full-step comparisons use the existing short performance profiles, CPU float64,
one thread, warm entropy inversions, and the research fast mixing closure. Their
independent order-8 energy diagnostics use the original eager kernels. Tensor
captures remain local; uploaded evidence is restricted to files below 128 KiB.

## Scope of this pass

Detailed report construction is not removed: many transport/work ledger fields
also determine the update or enforce numerical guards. Separating optional
report retention is a remaining memory optimization, not a demonstrated source
of the thermodynamic speedups here. Physics-call cadence and mixing funding
iterations are unchanged. No fitted replacement EOS or altered TKE energy
allocation is introduced. Lower quadrature and fusion are measured separately
so an EOS approximation is not mistaken for an implementation-only speedup.

## Full-step results and regression checks

1024 cells × 64 layers, dt=60 s, four mixing substeps, warm inversions,
CPU float64, one thread. Each large entry is one timed step after warmup;
compilation/startup is excluded. These are exploratory samples, not confidence
intervals. Full includes dynamics, research fast mixing and boundary operations;
it excludes ALE in this table.

| Configuration | Seconds/full step | Speedup vs eager q4 |
|---|---:|---:|
| Eager q4, parent performance path | 61.25 | 1.00× |
| Eager q2 | 40.69 | 1.51× |
| JMD-only compiled q4 experiment | 34.25 | 1.79× |
| Final per-model compiled q4 | 18.70 | 3.28× |
| Final per-model compiled q2 | 14.83 | 4.13× |

These gains are additional to the field-reuse parent (#72), not relative to the
original model. The eager baseline precedes the pure-kernel extraction, whose
default arithmetic is unchanged. Wheel hashes and phases are in
[provenance](energy-quadrature/provenance.json). Earlier global-patch prototypes
measured 16.01/12.41 s for q4/q2; these are retained, not substituted for the final
API results. Small samples are also retained, including a q2 ALE slowdown.

Final compiled q4 endpoint state differences reach 7.18e-18 relative L2 versus
eager q4; compiled q2 reaches 1.65e-15 on this shallow test. Reports are not
bitwise identical: some limiter coefficients differ by 1 despite tiny endpoint
differences. No threshold or guard was relaxed. Both final runs have independent
order-8 energy-minus-input residual -1.135e-11 J/kg; this is a short-run diagnostic,
not proof of deep-ocean or long-run equivalence. Mixing trial counts/cadence are
unchanged. Deep quadrature errors must be assessed using the accuracy campaign.

Final installed-wheel suite outside checkout: **1084 passed, one CUDA skip, one
existing warning**. Ten focused tests cover energy gradients/work, reduced-order
heating/remap, default equality, per-model compilation isolation and guards.
Corrected-Gibbs (142 tensor comparisons) and dry (93) portable fixtures pass.
Three final compiled CLI cases (ocean, full, full_ale) complete. Their timings and
the earlier fusion probe overlap tests and are excluded from speedup claims.
Native Inductor is exercised by the campaign; the unit test uses backend=eager.

Usage:

```python
ocean = StratifiedOcean(mesh, column, warm_start=True, quadrature_order=4)
ocean.compile_thermodynamics()  # lazy; warm up before timing
```

The performance runner accepts `--warm-start --compile-thermodynamics` and,
separately, `--energy-quadrature 2`. Start with compiled q4: it preserves the
existing discrete energy. q2/q3 remain opt-in approximation candidates. Defaults
remain eager q4. No Mac, accelerator, long-rollout or full-model backward
qualification is claimed. Compilation requires a working PyTorch compiler
backend and can incur substantial startup time.
