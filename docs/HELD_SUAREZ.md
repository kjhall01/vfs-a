# Held–Suarez forcing

`vfs.atmosphere.held_suarez.HeldSuarezForcing` provides thermal relaxation
and Rayleigh drag. It owns prescribed buffers; the atmospheric model owns
geometry, dynamics and prognostic-state validation.

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

## Verification scope

Tests cover source equations, zero-source limits, component budgets, per-stage
composition, input preservation and prescribed-buffer gradients. The archive
comparison checks 132 tensors, including forced stages and energy gradients,
against the pinned reference implementation.

These are CPU float64 regression and representative gradient checks. Equilibrated
Held–Suarez climate statistics, accelerator execution and forced IMEX are not
qualified.

## Reproduce the archive comparison

From an installed-wheel check directory containing tools, run:

```bash
python -I tools/check_held_suarez_archive.py --archive-repo /path/to/vfs-a --report held-suarez.json
```

The tool reads only commit `849ccb829c7555ab8ecd1c11f0ae45993538f8ed` from local
Git history and executes the four required definitions in memory. It never
installs an old package or imports the old registry. A shallow checkout must
first fetch that commit explicitly (`git fetch origin 849ccb829c7555ab8ecd1c11f0ae45993538f8ed`).
It reports the source hash, method-AST equality, 132 exact tensor comparisons and
Torch/backend details; mismatch exits nonzero. Runtime wheels contain no archive.
