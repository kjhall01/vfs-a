# Standalone Held–Suarez forcing

`vfs.atmosphere.held_suarez.HeldSuarezForcing` preserves the thermal relaxation
and Rayleigh drag formerly in PR #8's `HeldSuarezPhysics`. It owns prescribed
buffers only, not geometry, prognostic state or dynamics. There is no restored
registry, general process framework, legacy dycore or implicit source splitting.

Run `python examples/held_suarez.py` for a complete CPU float64 example using
`DryAtmosphere`, tensor initialization and a single forced SSPRK3 step. Its
essential composition is explicit:

```python
forcing = HeldSuarezForcing()
def rhs(stage):
    return model.tendency(stage) + forcing.tendency(stage)
next_state = ssprk3(rhs, state, dt=1.0)
```

Both terms are evaluated at every existing RK stage. Calling `model(state, dt)`
alone still advances the unforced model. This example does not configure forced
ARK4, demonstrate climate equilibration, or establish a safe timestep generally.

## Equations and budgets

Thermal forcing relaxes interface theta toward `Teq / (exner_interface / c_p)`.
It returns `dm * kT * (theta_eq - theta)` in Pa K/s at cell interfaces. The
latitude-dependent rate ramps from k_a=1/(40 days) aloft to k_s=1/(4 days) near
the equatorial surface, below sigma_b=0.7. The prescribed equilibrium uses the
existing 315 K equatorial value, 60 K meridional contrast, 10 K vertical contrast
and 200 K minimum. The existing sigma/pressure safeguards remain unchanged.

Rayleigh drag returns `-cell_to_edge(kv) * u_e` in m/s² on edge/full levels.
Its full-level rate ramps below sigma_b to k_f=1/day at the surface. Drag removes
kinetic energy; the original equations do **not** return that work as heat.
Pressure and supported container tracers receive zero source. The conservative
dry core itself remains unbatched and tracer-free.

`forcing.tendency_terms(state)` returns separate `thermal_relaxation` and
`rayleigh_drag` State-shaped tendencies. To keep their budgets distinct, contract
each with `model.dynamics.boundary_energy_rate_closed(state, term)` for watts;
report the conservative RHS contraction separately. These are instantaneous
source contributions including the fixed-lid energy boundary, not an exact
finite-step energy ledger. The example prints them alongside finite-step energy
and mass changes. Neither term is folded into the conservative energy-closure claim.

`HeldSuarezConstants` accepts the same constants. Optional `Teq` (K) and `kT`
(s⁻¹) broadcast to cell/interfaces; `kv` (s⁻¹) must carry cell/full axes for
edge averaging (not a scalar). Their original buffer conversion behavior is
preserved; it converts prescribed data to the state's dtype/device, not the
prognostic state. Graph connections remain intact. Move prescribed buffers with
`forcing.to(...)` when moving the model.

## Preservation evidence

The inherited method bodies have identical executable ASTs to `849ccb8`'s
HeldSuarezPhysics, apart from the class name and new documentation. A temporary
same-environment comparison executes that archived source against the installed
new wheel for 1, 3 and 12 levels, with standard and prescribed-rate configurations.
It compares rates, tendencies, all three forced SSPRK3 stage inputs, the resulting
state, energy and energy gradients with respect to wind, theta content and pressure.
All 132 compared tensors agree exactly (rtol=atol=0) on Apple M3 CPU,
float64, PyTorch 2.15.0/Accelerate. Fixtures and numerical operators are unchanged. Focused tests check the explicit
source equations, independent zero-source limits, component budgets, per-stage
composition, input preservation and connected prescribed-buffer gradients.

These are CPU float64 regression and representative gradient checks, not a new
Held–Suarez climate campaign, accelerator qualification or forced-IMEX validation.

Local installed-wheel checks outside the checkout: **527 passed, 1 CUDA skip**;
both portable comparison tools and all three examples passed. The existing
bounded-donor gradient warning remains unrelated to this forcing. Hosted check
outcomes are recorded separately in PR #10 after its updated head runs.
