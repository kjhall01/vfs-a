# Direct Roquet ocean comparison

The reference revision is `59b11db1`. The standalone forced ocean and the
coupled ocean–ice integrator both construct `vfs.ocean.Ocean`. The former
`vfs.ocean.Ocean` layered reference is available in `vfs.ocean.reference`
for reproducing historical experiments.

## Compare the numerical paths

From the experimental checkout, with the pinned PyAM21 dependency installed,
use the same Python interpreter for both runs. The comparison driver uses
small CPU float64 configurations and synthetic column radiation.

```bash
git worktree add --detach ../vfs-ocean-baseline 59b11db1
python verification/coupled/cleanup/compare.py \
    --checkout ../vfs-ocean-baseline --save baseline.pt
python verification/coupled/cleanup/compare.py \
    --checkout . --compare baseline.pt
```

| Comparison | Bitwise-identical tensors |
| --- | ---: |
| Dry and moist standalone atmosphere | 167 |
| Prescribed-SST atmosphere | 85 |
| Forced ocean, including process reports | 62 |
| Coupled aquaplanet | 192 |
| Coastal Earth and restart | 404 |
| Active partial columns, ALE and restart | 339 |

The coupled captures include stage tendencies and bounded transport ledgers.
The comparator checks dtype, shape, and tensor bytes. A separate radiation-off
CLI test started a NetCDF restart on the reference and resumed it on the
experimental branch using `--resume-compatible-manifest`. All 36 tensors and
33 scalar payload entries matched the uninterrupted reference at the next
interval, after excluding the campaign source fingerprint.

## Mature shelf timing

A separate CPU float64 comparison replayed one 1,200 s interval from a
4,096-cell day-12 shelf restart with 32 atmosphere levels, 40 ocean levels,
SSPRK43 ocean dynamics, four CPU threads and compilation disabled. Both
executables started each repetition from the same state and used the loaded
radiation cache. Fresh optical transfer and file I/O are outside this timing.
Reconstructing the restart from a different host required rebinding the
adapter identity for this diagnostic. The CLI restart test above exercised
the supported fingerprint checks separately.

Four processes ran in reference, candidate, candidate, reference order on
the same four CPU cores. Each process warmed up once, then timed three
repetitions. The pooled medians were **11.17 s** on the reference and
**10.96 s** on the candidate. The ranges overlapped (10.46–12.14 s and
9.49–11.43 s). The packed endpoint and report matched across all 67 tensors
and 34 scalar entries.

The same replay with CPU Inductor enabled gave pooled medians of **9.32 s**
on the reference and **9.89 s** on the candidate. The two compiled runs
matched for all 67 tensors and 34 scalar entries. These CPU measurements
exclude first-call compilation and do not establish A100 throughput.

A follow-up kept both compiled workers resident on the same four cores after
the test suite finished. Each advanced the same loaded interval six times.
Only one worker advanced at a time, and the first worker alternated within
each pair. Excluding one warmup per worker, medians were **9.57 s** on
the reference and **9.54 s** on the candidate. The initial compiled
slowdown did not repeat under paired timing.

The portable suite against a clean installed wheel completed with 1,661
passes and 14 skips using PyTorch 2.14.1+cpu. Two separately enabled native
Inductor checks passed, along with 142 moist and 93 dry portable fixture
comparisons.

## Compare on a GPU

Use two checkouts on the same GPU with the same absolute input paths.
Set `MESH`, `GEOGRAPHY`, `PYAM21_SOURCE`, and `PYAM21_RADIATION_DATA` to
the mesh, matching geography, upstream checkout, and radiation table directory.
Run this from the experimental checkout after creating the baseline worktree above.

```bash
for checkout in ../vfs-ocean-baseline .; do
    (
        cd "$checkout" || exit 1
        python -u -I tools/profile_coupled.py \
            --config workflows/coupled/earth_shelf_4096_benchmark.yaml \
            --mesh "$MESH" --geography "$GEOGRAPHY" \
            --source "$PYAM21_SOURCE" --radiation-data "$PYAM21_RADIATION_DATA" \
            --device cuda:0 --dtype float64 --threads 4 --no-compile \
            --warmup-intervals 4 --measure-intervals 24 \
            --output runs/ocean-refactor-eager
    ) || exit 1
done
```

Each checkout gets a separate output directory. Compare elapsed wall time,
SYPD, and the ice phase, mixing, GM–Redi and ocean–ice RHS rows. To compare
compiled performance, use `--compile` and a fresh output directory in both
checkouts. This preset uses ocean SSPRK3 and a 2,400 s physics interval.
