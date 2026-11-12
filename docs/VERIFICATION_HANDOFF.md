# Verification coverage

[Verification results](../verification/README.md) distinguish regression checks
from physical-accuracy measurements. Examples demonstrate API use; their short
runs do not qualify climate statistics or broad stability ranges.

## Atmospheric checks

| Area | Available checks | Limits |
| --- | --- | --- |
| Energy and operators | Conservative derivatives, analytic wave actions, pressure-gradient work | Semidiscrete closure does not establish finite-step conservation |
| Time integration | Explicit stage checks and small-case ARK4 refinement against DOP853 | Broad timestep/resolution stability and IMEX backward remain unqualified |
| Moist transport | Species-amount ledgers, sharp fronts, positivity and selected directional derivatives | Centered IMEX is not bounded; limiter switches are nonsmooth |
| Held–Suarez forcing | Source budgets, zero-source limits and pinned archive comparison | No equilibrated climate or forced-IMEX qualification |
| Initial balance | JW initialization and regression measurements | Analytic balance does not imply discrete balance |

JW initialization supports perturbed and unperturbed states. Select
`phi_full_mode="direct_u_gl2_adjoint"` explicitly for the conservative core.
Initialization choices, mesh quality and any mass-flux projection affect balance
residuals. Topographic rest requires a consistent surface-pressure field; uniform
surface pressure over terrain is not a demonstrated resting state.

There is no dedicated mountain-wave initializer, absorbing upper layer or open
lateral boundary. The hydrostatic equations require a compatible wave regime.

## Ocean checks

The [ocean verification index](../verification/ocean/README.md) covers JMD95 density,
caloric accuracy, pressure work, internal waves, balance, sharp-front transport
and vertical remapping. Results are specific to their recorded configuration.
Longer runs without the adaptive remapper do not qualify that remapper.

## Interpreting results

Record mesh, levels, timestep, dtype/device and solver settings alongside errors.
Separate instantaneous energy closure from timestep drift, forcing work and remap
changes. Atmospheric energy includes the fixed-pressure lid reservoir;
Held–Suarez drag does not return its kinetic loss as heat.

For bounded transport, use actual stage ledgers for carrier closure, species
amounts and limiter activity. Reconstructing these from the final RHS loses the
intermediate admissibility information.

CPU float64 is the baseline. GPU, reduced precision, long-rollout gradients and
coupled atmosphere–ocean evolution require separate qualification. See
[portable regression](PORTABLE_REGRESSION.md), [bounded gradient limits](BOUNDED_GRADIENT_LIMITATION.md)
and [contributing](../CONTRIBUTING.md) for checking instructions.

The optional [polar and energy/tracer mapping audit](../validation/polar_energy/README.md) distinguishes analytic point/area references, ERA5 pole-input sensitivity, and
nonconservative interpolation defects from nonlinear energy reconstruction.
Full active thermodynamic mappings remain unqualified.
