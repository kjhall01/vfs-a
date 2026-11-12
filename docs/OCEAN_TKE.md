# Prognostic ocean TKE

`TKEOcean` composes an existing `StratifiedOcean` with an optional one-equation
vertical turbulence closure. The original ocean state, dynamics, remapper and
defaults are unchanged. This is an experimental prescribed-mixing-length closure,
not a calibrated mixed-layer model or a GOTM implementation.

```python
from vfs.ocean.stratified import TKEOcean, TKEClosure

model = TKEOcean(ocean, closure=TKEClosure(mixing_length=2.0))
state = model.state(resolved_state, tke=k)  # k tensor, m²/s², same layout as h
state, report = model.step_with_report(state, dt)
budgets = model.budgets(state)             # total = resolved_total + tke
```

`TKEState.ocean` holds the existing entropy/salinity state; `tke_h` stores h*k on
cells and layers. `specific_tke()` returns k. Initialization is explicit, requires
matching shape/dtype/device and rejects negative or nonfinite k. Exactly zero k
is allowed, without a floor. Dry-cell k must be zero.

`mix_with_report(state, dt)` runs vertical physics alone. `step_with_report` runs
the existing checked dynamics, optional requested ALE, then vertical physics.
This is first-order operator splitting, even though the dynamics uses SSPRK3.
`closure=None` transports/remaps k without vertical production, mixing or decay.
`remap_with_report` returns separate resolved-ocean and TKE energy ledgers.
Run [the example](../examples/ocean_tke.py) with an installed wheel.

## Closure

For prescribed length L in metres and specific TKE k in m²/s²,

\[
K_m=C_\mu L\sqrt{k},\qquad K_h=K_m/\mathrm{Pr},\qquad
K_k=K_m/\sigma_k,\qquad \epsilon=C_\epsilon k^{3/2}/L.
\]

The default dimensionless coefficients are C_mu=0.1, C_epsilon=0.7, Pr=1 and
sigma_k=1. They are explicit starting values, not calibrated coefficients for
VFS. L is required; no second length-scale/dissipation equation is solved.
`c_mu=0` isolates decay, `c_epsilon=0` disables decay and `mix_scalars=False`
isolates shear/TKE exchange without active or passive scalar mixing.

Top and bottom fluxes are zero. At each step the diffusivities are frozen from
initial k, with arithmetic averaging to vertical interfaces. Closed implicit
vertical diffusion mixes native edge-normal momentum, potential enthalpy, salt,
passives and TKE. It does not reconstruct cell winds for the momentum solve.
This relies on the existing layer adjacency and cell-to-edge thickness mapping;
it is not an arbitrary sloping-layer stress-tensor discretization.

## Energy transfers

The combined Hamiltonian is

\[
H_{\rm combined}=H_{\rm ocean}+\sum_{c,k}\rho_0 A_c h_{ck}k_{ck}.
\]

The added energy is linear in stored h*k. Conservative horizontal content
transport therefore preserves its integral without a new pressure force. TKE is
packed as one extra passive content only inside the existing transport/remap
call, then unpacked; user tracer layout is retained. Numerical transport heating
and ALE thermal correction remain their existing resolved-energy processes and
are not relabelled as turbulence production.

**Shear.** The momentum mass matrix is the exact native kinetic metric
M_e=rho0 dc_e dv_e I(h)_e. The vertical solve satisfies

\[
M(u'-u)=-\Delta t D^T G D u',\qquad
K'-K=-\Delta t\,(Du')^T G(Du')-\tfrac12(u'-u)^T M(u'-u).
\]

Both nonnegative terms are deposited into TKE. The report separates physical
shear production from backward-Euler finite-step damping. Interface production
is shared between its adjacent layers; edge energy is then distributed with
weights w_ce*h_c/I(h)_e. This is the same cell decomposition as h*dK/dh in the
existing Hamiltonian, including unequal neighbouring thicknesses. Native
vertically integrated edge momentum is conserved to solver roundoff. For
horizontally unequal thicknesses those cell fractions vary between layers:
vertical edge work can redistribute cell-assigned K between neighbouring
columns. `kinetic_column_exchange` records that zero-global-sum redistribution.
The local energy residual subtracts this known exchange; the independent global
energy residual does not. A shared-edge model does not imply isolated energy
budgets for every cell column.

**Buoyancy mixing.** A closed diffusion solve conserves potential enthalpy and
salt contents, with temperature recovered from the caloric reference. The change
in full thermodynamic Hamiltonian (dynamic enthalpy plus reference thermal energy)
is debited from column TKE. Stable thermal stratification consumes TKE in the
tested cases; unstable stratification releases it. The debit includes numerical
caloric inversion residuals, rather than assuming h0 inversion is exact.

A positive column cost is drawn proportionally from layer TKE; a release is
allocated by layer mass. This is an explicit column allocation policy, **not a
local layerwise -K_h N² production law**. It permits different layers in a column
to share their energy budget. There is no inter-column funding. If the cost
exceeds available TKE, the step rejects without mutating its input; reduce dt.
This limitation matters before using the closure for entrainment or mixed-layer
science. No alternate allocation or timestep is silently substituted.

**Dissipation.** Following production and vertical TKE diffusion, the local decay
has the exact fixed-L solution

\[
k'=\frac{k}{\left(1+\Delta t C_\epsilon\sqrt{k}/(2L)\right)^2}.
\]

Its lost energy heats the same layer. The thermal solve uses the analytic
JMD95/caloric energy derivative at fixed thickness and salinity; it is not simply
a constant-cp temperature increment. A three-point temperature-path quadrature
avoids subtracting large reference energies while finding the increment. The
result is independently checked against actual Hamiltonian differences.
Thermal-domain failures and column entropy decreases beyond roundoff reject.

The report contains shear, implicit damping, column buoyancy work, heat,
kinetic/thermal/TKE-transport residuals, entropy changes, shared-edge column
exchanges and independent column/global energy residuals. Transfers are joules, entropy changes J/K. Energy checks permit
floating-point error scaled by the thermal background; they do not certify
relative accuracy for arbitrarily tiny TKE. Existing SSPRK3 energy drift is
reported separately and is not repaired by this closure.

## Boundaries of the implementation

- CPU float64, unbatched states and fixed closed coast masks are the tested scope.
- No surface wind/wave TKE injection, bottom drag, Langmuir turbulence, prescribed
  background diffusivity or prognostic mixing length. Zero TKE stays zero without
  a seed or a future source, even in an unstable column.
- No horizontal EKE/backscatter, isotropic turbulent-pressure term, or atmospheric
  physics coupling. Atmospheric TKE remains a separate task.
- A prescribed L and constant Pr do not establish realistic stability functions,
  mixed-layer depths, entrainment rates or internal-wave dissipation.
- Positivity follows the implicit diffusion/decay construction in exact arithmetic;
  numerical failures reject instead of clipping. Buoyancy work can restrict dt.
- Gradients are piecewise through switches, reference inversions and rejection
  boundaries. Selected short smooth checks do not establish long-run backward,
  GPU, float32 or compile behavior. The zero-k branch has zero chosen derivative.
- The vertical mixing solve and operator splitting are first order in time.
  Energy closure is not a statement of physical accuracy.

The standard one-equation structure is described in the
[GOTM scientific manual](https://gotm.net/manual/stable/pdf/a4.pdf).
The finite-step transfers and column buoyancy allocation above are specific to
this implementation. [Qualification results](../verification/ocean/TKE.md)
separate budget evidence from physical-validation gaps.
