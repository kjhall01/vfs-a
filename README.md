# VFS — spherical atmospheric dynamics

VFS contains a conservative dry core and an energy-closing, transport-bounded
moist core on a spherical Voronoi C-grid. The default model steppers use SSPRK3.
Existing dry and centered moist ARK4 IMEX paths remain available through explicit
stepper instances. Centered moist ARK4 does **not** provide bounded species
transport, and is never selected as a fallback for bounded explicit stepping.

## Start here

- [Current configurations and follow one timestep](docs/CURRENT_PATHS.md): four
  retained paths, equation owners, units, conservation pairings and solver limits.
- [State/model ownership and short dry/moist examples](docs/STATE_OWNERSHIP.md).
- [Moist physical-field initialization](docs/PHYSICAL_FIELDS_API.md) and
  [bounded Euler → SSPRK3 reading guide](docs/BOUNDED_MOIST_READING_GUIDE.md).
- [Regression evidence and deliberate API removals](docs/READABILITY_RESULTS.md).
  [Historical reports](docs/HISTORY.md) describe earlier trees, not current APIs.

## Install and run

```bash
python -m pip install .
python examples/moist_atmosphere.py
python examples/bounded_moist.py
python examples/held_suarez.py
```

Use Python 3.10+; CI uses 3.12. For development, install `-e '.[test]'`.
The small test extra needs no plotting/cloud stack. See
[CONTRIBUTING.md](CONTRIBUTING.md) for installed-wheel checks outside the checkout.
Optional data/plotting extras serve external workflows. The local version
`1.0.0+extraction4` does not imply package publication; no license is selected here.

Models own fixed geometry and dynamics; ordinary states hold tensors and borrow
geometry. Move the model before initialization. Derived physical fields and
budgets are computed from the supplied state, without stale caches. Use explicit
`state_to_model` for existing states on the same geometry; no hidden cast,
detach, clipping or regridding is performed. Vapor must be strictly positive.

The bounded moist Euler update installs species amounts from its ledger. A generic
`state + dt * rhs` replacement would lose its cancellation-safe update. Gibbs and
limiter work are allocated with `release_as_heat=True`; the non-heat alternative
is removed. Energy closure is semidiscrete, not exact finite-step RK conservation.

[Held–Suarez thermal relaxation and Rayleigh drag](docs/HELD_SUAREZ.md) remain
available as a standalone source. Sum it explicitly with the conservative dry
RHS at each SSPRK3 stage and report thermal/drag budget contributions separately;
the model's default step remains unforced.

## Scope and validation

Retained: spherical geometry, mesh generation/transfers, conservative dry and
bounded moist dynamics, explicit and ARK4 stepping, differentiable diagnostics,
atmospheric initialization, regridding and I/O. ARK4 uses retained analytic wave
operators, residual-checked solvers and supported preconditioners. Evidence for
specific configurations is documented; arbitrary solver flags are not qualified.

Removed: other domains/ML, old dry recipes and general process/forcing composition,
canonical dry alternatives, ARK2/3 and pressure prototypes, compiled/coupled-JVP/
fixed-work IMEX alternatives and centered explicit moist stepping. Necessary
centered candidates, donor transport, limiter corrections and shared dry
operations remain inside the retained algorithms. Git history is the archive.

```bash
python -m pytest tests --fixture-mode=portable -q
python tools/check_baseline.py --mode portable
python tools/check_dry_helpers.py --mode portable
```

Portable checks retain independent positivity, conservation, limiter and exact
ledger checks. Frozen fixtures and tolerances are unchanged; exact mode remains
separate and can fail across CPU/backend environments. Same-environment pre/post
comparison covers states, RHS, energies, representative gradients, ledgers, ARK4
stages and solver diagnostics. One-step bounded derivative checks do not qualify
IMEX backward, arbitrary limiter switches, accelerators or full rollouts.
Read [the handoff](handoff/index.md) before further changes.
