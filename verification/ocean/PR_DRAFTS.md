# Published ocean drafts

Published with explicit user approval: [reference extraction #23](https://github.com/kjhall01/vfs-a/pull/23)
and dependent [constant-density energy #24](https://github.com/kjhall01/vfs-a/pull/24).
Both are drafts and remain unmerged. Synthetic fixtures, verification meshes,
and results are included; no other agent's branch was changed.

At publication, #20–#22 were marked merged, but main (`aee9638`) contained only
the first ownership stage. #23 therefore targets the unchanged
`structure/io-ownership` head `941b794`; #24 targets `ocean/extract-reference`.
Retarget #23 after the complete ownership stack reaches main, with the atmosphere
owner coordinating that integration. No shared runtime change is part of either
ocean draft.

The GitHub publication API supplies new commit metadata, so published commit IDs
differ from the local bundle. Every uploaded implementation tree was checked for
exact Git tree identity, including all binary fixtures and verification meshes:

| Change | Local commit | Published commit | Identical tree |
| --- | --- | --- | --- |
| Reference extraction | `d981099` | `812c1a5493e9409ca0d166a6a2ea907f69bb4cdb` | `9b8082d517789be395073c4283ce595394f0de11` |
| Numerical energy pairing | `25c57ce` | `2d1ca581fd701468ba0a1cb7f5550c967c09d4ea` | `11bef7aa25b58951b3416fe15e63ed6f81251ed2` |
| Output metadata | `b5b752a` | `4124907f3f9b197a55c8ad11316377ef27473a55` | `d26b740b1d261bc088e4f6a945aaf74e1858fad2` |
| Original publication record | `68e1202` | `3ecff974ed1e8ad7dc1dd6fcb003b0e548762c8b` | `ad3a9c72ceb59631570b10c3555bdadb611a0159` |

This follow-up changes documentation only. Hosted checks are recorded on the PRs;
they are distinct from the local installed-wheel results below.

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
ocean PR unmerged for review.

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
focused output test passes. See `OUTPUT_METADATA.md` and the published commit
mapping above. The publication records change documentation only.

Leave this PR unmerged for numerical review; do not promote the default or
merge it into the extraction. No active-EOS energy, long-run, GPU, compiled,
full backward rollout, coastal accuracy or complete Poisson-bracket claim.

## Local delivery

`VFS_ocean_implementation.bundle` contains both local branches and their commits
relative to the #22 base. It is a Git bundle, not an installed environment;
dependencies and the full old source archive are excluded. With the base commit
available in a clone, `git bundle verify /path/to/VFS_ocean_implementation.bundle`
checks its prerequisites before importing either branch.
