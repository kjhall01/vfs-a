# Working on this extraction

Read `handoff/index.md` and `docs/PASS4_RESULTS.md` (and pass-1 history) first.

- Preserve checkpoint-54 numerical expressions and defaults during packaging.
- Current moist baseline is BoundedGibbs(base) with intrinsic heat release, not the
  centered base alone. Its Euler species-amount ledger must remain intact.
- Keep the package teachable, inspectable, object-oriented and concise. Avoid
  deep inheritance, plugin registries and hidden state conversions.
- State holds tensors. Xarray is an explicit initialization/output boundary;
  no xarray members, hidden clipping, regridding or cloud loads inside State.
- No forward tendencies generated using autograd. Tensor diagnostics should
  retain gradients; differentiate tests independently when appropriate.
- Scientific verification reports metrics, references and convergence without
  subjective acceptance thresholds. Numerical assertions in migration tests
  serve to detect accidental implementation changes.
- Keep the original baseline archive unchanged. Do not prune inherited helpers
  until dependencies are extracted and equivalence has been demonstrated.
- Run focused tests and the fixture comparison after relevant changes. Test a
  wheel outside the checkout to exclude accidental source/research imports.
- Update this handoff and results documentation; clearly distinguish tested
  CPU/float64 forward behavior from untested GPU/long-run/backward behavior.
- Do not publish a repository/package or choose a license without user direction.

- The new MoistAtmosphere facade requires positive vapor, matching the frozen
  moist kernel. Do not imply exact-zero-water dry equivalence or add a floor.
- Preserve the documented interface/edge staggering and tensor-only boundary.
