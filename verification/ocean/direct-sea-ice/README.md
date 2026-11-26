# Thermal sea-ice ownership

Reference revision: `7cf3be1` on `experiment/direct-roquet-ocean`.

The campaign and the standalone example construct `vfs.ice.OceanIce(ocean)`.
`state.ocean` and `state.ice` are the inputs to the same joint RK stages.
`SeaIceState` stores category area, total ice-equivalent volume, edge velocity,
snow mass and sensible cold content. Snow labels part of the total frozen mass.
The optional `state.land` is a sibling and passes through these stages.
The shared atmospheric exchange still runs in the workflow. For a complete
initialization and one energy-budgeted step, run `python examples/sea_ice.py`.

## Forward comparisons

Use the same interpreter for both checkouts. Set up the pinned PyAM21 dependency
and the VFS test dependencies. Run from the experimental checkout.

```bash
git worktree add --detach ../ice-baseline 7cf3be1
driver=verification/ocean/direct-sea-ice/compare_stages.py
python "$driver" --checkout ../ice-baseline --save ice-stages.pt
python "$driver" --checkout . --compare ice-stages.pt
```

The CPU float64 comparison starts with snow-covered moving ice over liquid
ocean with a different velocity. It checks both split drag reports, positive
TKE gain, every RK input and tendency, the split-step endpoints, external-work
reports, frozen mass and total energy. SSPRK3 matched all **202 tensors** and
SSPRK(4,3) matched all **231 tensors** byte for byte.

The existing `verification/coupled/cleanup/compare.py` driver matched **1,249
tensors** including the aquaplanet, coastal runoff, active partial columns,
ALE and restart. Two separate on-disk CLI runs resumed a 60 s NetCDF checkpoint
from the ocean experiment and from main at `59b11db`. With
`--resume-compatible-manifest`, both second-120 endpoints matched their
uninterrupted control for all 36 tensors and all scalar entries after
excluding the campaign source fingerprint.

On PyTorch 2.15.0+cpu the installed-wheel portable suite passed **1,665 tests**
and skipped ten. Both separately enabled CPU Inductor tests passed. The
portable fixture checks passed for 142 moist and 93 dry quantities.

## Mature shelf replay

A 4,096-cell, 40-ocean-level coastal NetCDF checkpoint at second 1,088,400
provided the ocean, ice, snow and land state. The paired runs used its matching
mesh and geography, a 600 s SSPRK3 joint step, five mixing substeps, float64
and the shelf-benchmark ocean settings. Eager and CPU Inductor each matched
the parent byte for byte on **22 tensors**, including the step report and
combined energy. This diagnostic reads physical fields directly. The CLI
comparisons above exercised restart authentication separately.

```bash
driver=verification/ocean/direct-sea-ice/replay.py
inputs=(--mesh "$MESH" --source "$PYAM21_SOURCE" --restart "$RESTART")
python "$driver" --checkout ../ice-baseline "${inputs[@]}" --save ice-shelf.pt
python "$driver" --checkout . "${inputs[@]}" --compare ice-shelf.pt --repeats 6
```

Add `--compile` to both commands with a fresh reference file for the
Inductor comparison. Timing starts after a warmup and excludes file I/O,
atmosphere, radiation, the shared exchange, GM/Redi and ALE. The A100 coupled
benchmark uses the procedure in the [ocean verification report](../direct-roquet/README.md)
with `7cf3be1` as the reference checkout.

### CPU throughput

The timing comparison used the same four pinned CPU cores, eager float64,
one warmup per worker and one advancing worker at a time. Each sample
restarts from the same shelf state. The execution order alternated within pairs.

| Median seconds per 600 s joint step | Ocean branch | Ice branch |
| --- | ---: | ---: |
| Separate workers, six pairs with other tests running | 3.315 | 3.523 |
| Separate workers, nine pairs, other tests finished, worker launch order reversed | 3.193 | 3.368 |
| Same process and tensor allocations, ten pairs | 3.303 | 3.273 |

Across the nine-pair run, total phase time per step was 0.243 versus 0.268 s
and total ocean–ice RHS time was 1.781 versus 1.917 s. A CPU operator trace
on the small moving-ice case recorded identical counts for all 70 ATen
operations (23,022 calls per complete step).

The third comparison loaded the parent thermal and snow implementations into
the experimental interpreter and gave both steppers the same mechanical
operator, ocean, mesh and initial tensor objects. It used two warmups per
implementation and timed the complete step without per-stage instrumentation.
The separate-process slowdown of 5–6% did not appear in that comparison.
Its cause is unresolved. An end-to-end A100 measurement remains outstanding.

An initial timing attempt on a generated, unrelaxed island grid hit the
Euler outflow guard on the parent during warmup. The timing and equivalence
results here use the saved mesh and mature shelf state.
