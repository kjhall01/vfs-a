# Verification

VFS separates numerical regression checks from physical accuracy measurements.
A passing test suite does not establish long-run stability, balanced flow or
accuracy at an arbitrary mesh and timestep.

## Package checks

Run tests against an installed wheel outside the source checkout. The complete
setup is in [Contributing](../CONTRIBUTING.md).

```bash
python -I -m pytest tests --fixture-mode=portable -q
python -I tools/check_baseline.py --reference-version corrected-gibbs-v1 --mode portable
python -I tools/check_dry_helpers.py --mode portable
```

The portable fixture policy defines quantity-specific roundoff allowances and
independent invariants. Exact comparisons remain separate; a portable pass is
not bitwise agreement. Historical references must not be regenerated to hide a
failure. See [comparison policy](../docs/PORTABLE_REGRESSION.md).

## Scientific evidence

| Area | Evidence |
| --- | --- |
| Atmospheric Gibbs identities and constant composition | [Derivation](CONSTANT_COMPOSITION_DERIVATION.md), [measurements](GIBBS_FIX_RESULTS.md) |
| Atmospheric operators and reference comparisons | [Results](RESULTS.md), [reference provenance](ATMOSPHERIC_INTEGRATION.md) |
| Atmospheric benchmark coverage | [Verification inventory](../docs/VERIFICATION_HANDOFF.md) |
| Ocean transport, balance, waves and ALE | [Ocean verification](ocean/README.md) |
| Bounded moist derivatives | [Derivative measurements and limits](../docs/BOUNDED_GRADIENT_LIMITATION.md) |

Reports record their source version, environment, geometry and configuration.
Frozen experiment drivers can require a pinned wheel; follow the corresponding
report rather than running every historical driver against the latest package.
Research scripts in `tools` are not the public model API.

Interpret energy measurements by separating instantaneous work, timestep drift,
remap changes and forcing. Small perturbation energies require care when taking
differences of large thermal backgrounds. Transport inventories, bounds, phase
errors and balance errors are distinct measurements.

Scientific comparisons report norms, units and refinement behavior without
subjective accuracy gates. CPU float64, short-step derivatives and small examples
do not establish GPU, long-backward or climate-run qualification.
