# Instantaneous Hamiltonian backscatter with local funding

This **opt-in research closure** replaces the earlier finite-kick prototype.
Its public evaluation is `EddyBackscatter.tendency(model, state)` with no dt.
There is no exponential withdrawal, quadratic finite-step energy debit, or
post-step temperature/energy repair. The old `apply` API is removed.

```python
from roquet import RoquetOcean
from backscatter import EddyBackscatter

# Illustrative explicit parameters, not calibrated recommendations.
closure = EddyBackscatter(viscosity=1e4, timescale=86400., fraction=.25)
model = RoquetOcean(mesh, column, backscatter=closure)
# state must be EddyState, e.g. model.state(..., eddy_energy=.1)
state, report = model.step_with_report(state, dt)
# report.backscatter contains the three SSPRK3 stage power reports.

# Inspect the instantaneous closure without advancing time:
du_dt, deddy_h_dt, power = closure.tendency(model, state)
```

Without `backscatter`, the original stepping path and report type are unchanged.
When enabled, momentum forcing and eddy-reservoir source are evaluated from
**the same stage state**, alongside resolved dynamics, at all three SSPRK3
Euler evaluations. Existing authoritative advected scalar amounts are retained;
the reservoir source is added to those amounts, and the momentum source to the
velocity update. TKE, CT, SA and other passives receive no backscatter source.
GM–Redi and eddy decay remain explicit separately composed stages.

## Instantaneous operator and energy pairing

C is the existing edge-to-vertex curl; C* is its adjoint under A_e=dc_e dv_e
and A_v=dual_area. h_e=I(h), h_v is the positive mean of the three incident
edge thicknesses, and nu>=0 is the negative-viscosity magnitude [m²/s].
With a local vertex coefficient alpha_v in [0,1],

    F = (1/h_e) C*(alpha_v nu h_v C u).

Coasts mask the input and output velocities. The mechanical metric in the full
Roquet Hamiltonian is M_e=rho0 A_e h_e. Its instantaneous kinetic power is

    P_e = (dH/du)_e F_e = M_e u_e F_e,
    sum P_e = rho0 sum_v A_v alpha_v nu h_v (C u)_v² >= 0.

`RoquetHamiltonian.kinetic_power` allocates edge power to adjacent cells with
the existing geometric interpolation transpose I^T, whose weights sum to one.
This allocation is an explicit local accounting convention, not a separately
modelled physical energy flux. The reservoir tendency is

    d(h e)_c/dt = -I^T(P)_c / (rho0 A_c).

The reservoir conjugate is rho0 A_c, so the full Hamiltonian contraction
vanishes locally under this allocation and globally, to roundoff. Local
negative kinetic power replenishes that cell's reservoir. No second EOS or
energy functional is introduced. Forward tendencies use analytic operators;
autodiff is used only for independent verification and backward sensitivities.

## Strictly local power limiter

Available power is Q_c = fraction * rho0 A_c (h e)_c / timescale [W].
It is a **power cap**, not a demand to spend the whole allowance.

Each edge forcing has contributions from its two endpoint vertices. Compute
those incidence powers before cancellation and accumulate their positive parts
into cell demands D_c using I^T. Set r_c=min(1,Q_c/D_c), or one for zero demand.
Each vertex takes the minimum r_c over the cells adjacent to its three edges.
The resulting alpha_v multiplies the viscosity **inside** C*.

Every positive contribution to a cell is therefore scaled by at most that
cell's r_c; negative contributions can only reduce its debit. Thus the signed
local power is <=Q_c. Nonnegative vertex coefficients preserve the global
positive-work identity. There is no layer-wide reduction, iterative/global
solve, or communication beyond the fixed local stencil. An empty cell can
suppress neighboring vertices but cannot veto distant forcing. Ignoring
cancellations when calculating demand makes this limiter conservative.

## Time integration and positivity are separate

Instantaneous closure does not imply exactly conserved finite RK steps.
SSPRK3 evolves both paired tendencies and has time-discretization error.
Its smooth-regime energy error converges at third order; limiter switches can
reduce the observed order. Large negative-viscosity rates may require shorter
steps for accuracy even when the reservoir remains nonnegative.

The integration wrapper enforces dt*fraction <= timescale, sufficient for the
isolated forward-Euler reservoir source. Advection and the source together can
still exhaust a cell; each combined Euler endpoint is checked and rejects with
"reduce dt" if this occurs. There is no clipping or hidden adaptive energy
repair. Constitutive power evaluation itself remains independent of dt.

## Scope and diagnostics

The report contains kinetic and reservoir powers, local power residuals,
local allowances and vertex coefficients. Units are W, not J. Stage reports
can be integrated with SSPRK3 weights (1/6,1/6,2/3) for accumulated exchange.

At uniform thickness the forcing is negative rotational viscosity. It omits
anti-diffusion of divergence; varying thickness/coasts do not imply exactly
divergence-free increments or angular-momentum conservation. Negative viscosity
can energize grid-scale noise. No new biharmonic sink, spectral calibration,
resolution-dependent coefficient or long-run stability claim is supplied.
The literature motivates the approach ([Jansen et al., 2019](https://doi.org/10.1029/2019MS001750));
this is not that paper's complete scale-aware closure.

See [current verification](../../verification/ocean/backscatter/instantaneous/README.md).
The parent report documents the superseded finite-step prototype.
