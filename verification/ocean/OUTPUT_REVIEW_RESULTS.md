# Output review follow-up validation

CPU float64, Python 3.12, Torch 2.15.0+cpu. Wheels built and installed in an
isolated environment; tests/tools copied outside the checkout.

- Full portable suite: 579 passed, one CUDA skip (four new output tests).
- Final focused output suite: five passed, including the subsequently added
  metadata-comparator mutation test. The full suite was not rerun for that
  test-only addition.
- 298 dynamics tensors and solver records exactly match pre-fix #24.
- 74 output tensors exactly match pre-fix #24. One intentional metadata key,
  density_convention=legacy_state_linear_eos, is explicitly allowed at the
  ocean output attrs path only. Other metadata changes still fail comparison.
- Both atmospheric portable comparison tools pass (142 and 93 tensors).
- Structural AST audit: 457 preserved executable definitions.

No historical fixtures, tolerance values or numerical kernels changed. The
existing donor-subcycle scalar-conversion warning remains. These checks do not
qualify GPU execution, full-rollout gradients, ocean positivity, active-EOS
energy closure or conservative regridding. No atmospheric campaign was run.

Structural reconciliation is draft #25. The Celsius label fix is independently
on #23. #24 retains its original numerical evidence and incorporates the output
fix separately. Gibbs reference policy (#19) and Ocean nn.Module ownership are
separate pending decisions.
