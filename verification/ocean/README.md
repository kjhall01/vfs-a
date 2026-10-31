# Ocean reference extraction

This is the first implementation slice of the 31 October ocean audit. It
restores the archived layered solver as an explicit, experimental reference.
It does not correct its energy functional, rotation, pressure gradient,
thickness positivity, vertical remap or nonlinear thermodynamics.

## Source and coordination

- Donor: `VFS-main.zip`, SHA-256
  `ced96143a44bf6eaac6ae86bff9ae9d80d06d45e5d4acf321befd1d0c20f1d3c`.
  No Git commit is established for this ZIP. Original archive unchanged.
- Base: PR #22, `941b794eb221c3e31d0244b936c1c5c79ec14e1b`, dependent on
  #21/#20 and merged main `3f806a947e2ff8a5947953137a96af02861d083f`.
- Branch: `ocean/extract-reference`, separate worktree. Do not merge the
  structural stack or rewrite the other agent's branches as part of this work.
- Runtime changes are confined to `vfs/ocean`. Shared mesh/integration/I/O,
  atmospheric code, atmospheric fixtures and all existing tolerances are intact.
- The only existing test contract changed is `test_package_scope.py`: four
  ocean numerical modules are now installed. Removed root model/diagnostic
  shells and the registry remain prohibited. No shared numerical API change.
- New documentation is ocean-owned, with one handoff entry linking here.
  The parent ownership document remains the record of the structural slice.

## Ownership and deliberate interface changes

`from vfs.ocean import Ocean, OceanState, OceanColumn, OceanDomain` is the small
routine API. The model owns geometry/dynamics/source instances and a shared
explicit integrator. State borrows those exact geometry objects, holds plain
tensors, and performs no casting or I/O. Compatible `Field` input unwraps by
identity, preserving the autograd graph. Tensor arithmetic retains the donor's
order. Removed state mesh forwarding is replaced by one direct mesh call in
the archived KE diagnostic. No equivalent kernel is maintained in parallel.

Equation and process constructors remain in `ocean.dynamics` and `ocean.physics`.
The base/subclass structure is retained for this first extraction because the
optional-process comparisons depend on it. Schema/registry methods, recipe
registrations, root model hierarchy and subjective health PASS rules are not
restored. `ocean.diagnostics` retains the actual measured quantities; its old
energy/work names have the limitations recorded below.

Snapshots are explicit `ocean.io.save_snapshot` and `load_snapshot` calls;
the old data format, metadata and size checks remain. Supply the original mesh
and column. The file does not establish topology/bathymetry identity. Output
remapping uses the parent stack's `ocean.regrid.state_to_latlon` adapter.

The facade supports unbatched tensors, explicit positive finite timesteps,
exact geometry identity and matching dtype/device. It rejects invalid output
without repairing it. Coasts must be projected explicitly using
`domain.project_state(state)` when desired; no hidden endpoint projection or
land-temperature reset is introduced. Advanced kernels retain their historical
array paths; GPU/batching/compile/long backward are not qualified.

## What is preserved and what is not a claim

Conservative h/hT/hS/hq transport, linear and JMD95 density, both pressure modes,
closed coast masks, the optional vertical and horizontal process formulas,
constant/CFL-scaled/biharmonic viscosity, restoring, heat/virtual-salt/stress
sources, bulk mixed-layer and convective formulas are retained. Relative-wind
attrs remain a historical advanced-source boundary, not a new coupling API.
ETOPO setup, old coupled models, registry recipes and old driver copies are not
restored. The default facade uses the base multilayer equations and SSPRK3.

The preserved `total_energy` is reconstructed KE plus free-surface PE; it omits
stratification and is not conserved by the preserved RHS. The facade calls it
`reference_mechanical_energy`. Integral hT is temperature content (m3 degC),
not joules; freshwater source is virtual salinity, not volume. Scalar upwinding
does not guarantee positive centered layer transport. Remap leaves velocity
unchanged and lacks a momentum/energy budget. Existing bottom-w diagnostics,
sloping stratified rest and pressure-dependent mixing remain unqualified.

These limitations are preserved intentionally for comparison. A later ocean
Hamiltonian scheme must be a separate numerical change, with explicit energy,
derivatives, metric-skew rotation and pressure/transport pairing. Algebraic
closure will not by itself qualify balance, positivity or convergence.

## Reproduction

CPU float64, one thread, Python 3.12.14 / PyTorch 2.15.0+cpu. Use the same
dependency environment in separate donor/extracted processes. Do not import
two different `vfs` trees into one interpreter.

`tools/capture_ocean.py --inputs tests/fixtures/ocean/inputs.npz --compare
tests/fixtures/ocean/archive_capture.npz --report comparison.json` compares the
installed package with the donor capture. Inputs contain exact mesh buffers,
topology, metrics and initial contents; they are not regenerated by the check.
The capture includes 1,416 arrays across global/coast configurations, both EOS
and pressure modes, zero/two tracers, all three explicit RK methods and actual
RHS input stages, source/mixing/remap paths and nonzero-flow gradient probes.

To reproduce the donor capture, point PYTHONPATH to the unchanged extracted
archive and run this same tool with those inputs and `--save new_archive.npz`.
The tool refuses to overwrite inputs or captures. `--create-inputs` is only for
creating a separately identified new experiment, never refreshing a fixture.

Build a wheel, install it in a fresh environment, copy `tests`, `tools` and
`examples` outside the checkout, then run:

```bash
python -I -m pytest tests --fixture-mode=portable -q
python -I tools/check_baseline.py --mode portable
python -I tools/check_dry_helpers.py --mode portable
python -I examples/ocean_reference.py
```

The existing atmospheric capture tool is run on the parent and changed wheels
in the same environment. It compares 298 tensors plus complete solver records.
Ocean same-environment comparisons are exact. Portable ocean tests permit only
a documented 512-epsilon per-array migration allowance, not a scientific
acceptance threshold. No existing fixture or tolerance is replaced.

See `extraction_comparison.json`, `source_preservation.json` and `RESULTS.md`.
