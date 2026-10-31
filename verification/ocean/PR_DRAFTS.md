# Prepared publication: two dependent draft PRs

Status: implementation and local validation complete; not published. Automatic
approval review rejected the synthetic solver-fixture upload to GitHub twice,
including after checking that every one of its 1,416 entries is float64 data
generated from seeded fields (no text/object payloads or observational data).
Explicit approval is needed for publishing these branches and their synthetic
verification fixtures/results to `https://github.com/kjhall01/vfs-a`.

No PR or remote branch has been created, no merge attempted, and no other
agent's branch changed. One small synthetic-input Git blob was accepted before
the larger fixture was rejected; it is unattached to any branch.

## Draft 1

Title: Restore the ocean reference with tensor ownership and exact archive comparisons

Head: `ocean/extract-reference` (local `d981099`).
Base: `structure/io-ownership` (pinned #22 head `941b794`).

The ocean was absent from the installed package. Restore its archived numerical
kernels under `vfs.ocean`, with a four-name routine API, tensor state ownership,
explicit snapshots and a thin model facade. Retire registry/schema/recipe shells;
keep the reference equations, defaults and operation order. Shared and atmospheric
runtime remains unchanged. This is an experimental numerical reference, not an
energy-closing ocean or a restoration of old coupled models.

Validation: 1,416 archive arrays match exactly on frozen geometry, including
RHS/terms, actual RK stages, gradients, masks and process/forcing outputs.
Installed-wheel CPU float64 suite: 566 passed, one CUDA skip. Atmospheric
298-tensor/solver and 74-output captures are exact; 235 portable fixture
comparisons pass with existing fixture bytes and tolerances unchanged.

Read `verification/ocean/README.md`, `RESULTS.md`, the capture/source manifests,
and `examples/ocean_reference.py`. The only existing scope test change permits
four restored ocean modules; removed root shells remain prohibited. Leave this
PR and the structural dependency stack unmerged for review.

## Draft 2

Title: Add an explicit constant-density ocean energy pairing and verification

Head: `ocean/constant-density-energy`.
Base: `ocean/extract-reference`.

The archived ocean's pressure-work diagnostic is not the derivative of its
reported energy. Add an explicit constant-density alternative with one discrete
mechanical energy, analytic derivatives, paired Bernoulli/continuity work and
rotation antisymmetrized in the actual layer-dependent kinetic metric. Keep the
archive as default and as an unchanged comparison. T/S are passive here; active
stratification, general carrier positivity and momentum/energy-consistent remap
remain separate future numerical changes. No shared numerical operator changes.

Installed-wheel CPU float64 suite: 575 passed, one CUDA skip. Existing reference
and atmospheric captures remain exact; 457 structural definitions unchanged.
Measured normalized spatial energy residuals are 0–9.2721e-17 in the stated
nonzero-flow global/coast/aligned cases. Wave projection slopes are 1.9434 and
1.8986; time-refinement orders 2.9942–3.0002. SSPRK3 still has temporal energy
error. Raw results, source hashes, mesh buffers, independent AD/FD checks and
limitations are in `verification/ocean/CONSTANT_DENSITY.md` and `energy_results`.

Numerical commit: `25c57ce`. A separate metadata commit, `b5b752a`, corrects the
ocean lat/lon temperature label from K to degC without changing values; its
focused output test passes. See `OUTPUT_METADATA.md`. The final documentation
commit only prepares this publication record. Hosted CI has not run.

Leave this PR unmerged for numerical review; do not promote the default or
merge it into the extraction. No active-EOS energy, long-run, GPU, compiled,
full backward rollout, coastal accuracy or complete Poisson-bracket claim.

## Local delivery

`VFS_ocean_implementation.bundle` contains both local branches and their commits
relative to the #22 base. It is a Git bundle, not an installed environment;
dependencies and the full old source archive are excluded. With the base commit
available in a clone, `git bundle verify /path/to/VFS_ocean_implementation.bundle`
checks its prerequisites before importing either branch.
