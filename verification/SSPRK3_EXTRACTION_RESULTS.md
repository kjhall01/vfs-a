# Timed bounded SSPRK3 extraction — PR #16

Numerical baseline: `8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648`.
Approved interface commit: `ae132773e621671f70bfa6a698eebb1a82cffa94`.
Only runtime file changed: `vfs/atmosphere/bounded_moist.py`. The three Euler
calls and two original `MoistState.combine` expressions moved into
`bounded_ssprk3_with_info(euler_with_info, state, time, dt)`. The atmospheric
wrapper supplies its existing Euler method, ignores evaluation time, and keeps
its final physical validation. Reconstruction, limiter, heat, Euler ledgers,
state conventions, operators, fixtures and tolerances did not change.

The narrowly scoped Euler callback receives state, RHS evaluation time and
full positive budget dt. RHS times are t, t+dt, t+dt/2: the second convex
combination is the half-time RK state. These nonmonotonic evaluation times do
not reverse physical integration or negate the Euler budget.

A verification caller must replace wind before every Euler evaluation and
again at t+dt on return. Combined intermediate winds are scratch values, never
analytical velocities. Pressure, entropy and water advance through the actual
production Euler maps and combinations; water is installed from each Euler
amount ledger. Momentum is evaluated but its output wind is discarded. Entropy
and heat remain active. This verifies transport, not coupled energy closure.

## Local evidence

macOS Apple M3, Python 3.12.13, PyTorch 2.15.0, CPU float64, one thread.
Built and installed a wheel from a clean temporary staging tree, tested outside
the checkout. Original and extracted wheels used the same environment.

- `tools/capture_readability.py --save /tmp/vfs-1c-before.pt`, then `--compare`:
  **298 tensors exactly equal**, including states, tendencies, energy components,
  bounded Euler/SSPRK3 ledgers, diagnostic gradients and RHS VJPs. Non-tensor
  records also equal, including dry/moist ARK4 stages and solver diagnostics.
- Four full-step gradient tensors (wind, pressure, entropy, water) exactly equal
  using the existing smooth-case nonuniform objective at dt=10 s.
- Installed portable suite: **550 passed, one CUDA skip** before the additional
  pinned-source test; provenance/observer suite afterward: **13 passed**.
- Both standalone portable fixture tools passed (142 moist/dry tensors and
  93 dry-helper tensors). All six documented explicit/ARK4/forcing examples ran.
- Focused timed-stage test uses quadratic-in-time prescribed wind, poisons the
  unused Euler wind, checks stage times/budgets, exact ledger installation,
  unchanged input state, preserved geometry references and final-time wind.
- The provenance test accepts only the original or exactly pinned extracted
  source blob and rejects an additional edit to that same file.

The baseline fixture's exact mode remains unchanged and separate from portable
mode; this exact pre/post claim concerns two wheels in the same environment,
not cross-backend bitwise fixture reproducibility. No GPU or long-rollout
qualification is claimed. Existing active-limiter derivative probes still emit
the documented scalar-conversion warning for the discrete donor subcycle count.

GitHub Actions push run 41135325544 and PR run 41135332667 both passed.
Downloaded push artifacts: Linux Intel Xeon Platinum 8573C, Python 3.12.15,
PyTorch 2.15.0+cpu; portable pytest 551 passed/one CUDA skip, both portable tools,
examples and verification steps completed. Exact fixture pytest had two
informational failures, retained separately. These are distinct from the exact
same-environment pre/post extraction result.
C5 reference construction and return experiments belong in the dependent PR.
