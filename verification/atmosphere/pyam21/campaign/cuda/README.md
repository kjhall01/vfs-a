# Single-device CUDA preparation

Base: merged PR #90, `969400ef0ae1f900fc148907d46a2541d0f149ff`.
The user reported a successful two-day, 16384-cell seasonal Mac run on that
branch. This pass adds explicit single-device CPU/CUDA execution while retaining
all physical configuration and float64. It does not implement multi-GPU coupling.

## Executed evidence

Environment: Python 3.12, PyTorch `2.15.0+cpu`; no CUDA hardware/runtime available.
VFS wheel installed into the environment; tests/workflows copied outside the
checkout into a directory with no `vfs` source package.

- **110 passed, 2 skipped** in the focused installed-wheel suite. The skips
  are the eager and compiled CUDA cases, not successful GPU results. The native
  **CPU Inductor** test was enabled and passed.
- **142 corrected-Gibbs / 93 dry-helper** frozen portable tensor comparisons pass.
- CPU before/after comparison against the merged base: **240 tensors and 127
  scalar/metadata leaves agree bitwise**, including initial state and four full
  coupled intervals. Configuration: 32 cells, 24 atmospheric levels, 12 ocean
  levels, 1200-second ocean/physics, 240-second atmosphere, hourly radiation,
  ten mixing and GM–Redi substeps, all ocean/ice options enabled. Analytic optics
  exercise the real radiation scheduler fresh/cache/cache/fresh. This does not
  claim real-table radiation or GPU qualification.

See [cpu-equivalence.json](cpu-equivalence.json), [focused-tests.log](focused-tests.log),
[baseline.json](baseline.json), [dry-helpers.json](dry-helpers.json).

## What changed

- Campaign YAML/CLI `device`, `--device`, host `--threads`, explicit model/state
  placement and startup hardware/runtime logging. Unavailable/unsupported
  devices fail without falling back or disabling physical processes.
- Move registered atmosphere/mesh geometry before constructing momentum maps,
  PyAM21 latitude/longitude and state. Ocean column and initializer tensors use
  the same selected device. Upstream PyAM21 source and packaged VFS kernels are
  unchanged; existing table caches follow the input device.
- CPU-stored checkpoints restored explicitly to the owner device. Preflight now
  exercises a real disk round trip. CUDA algorithm/workspace policy is explicit,
  requires deterministic kernels, and retains exact same-device restart checks.
  Device/runtime/algorithm policy is bound to restart identity.
- Synchronized device timings in a real-table CPU/CUDA comparison driver and
  the existing eager/compiled comparison driver. CUDA execution synchronizes
  before the campaign commits a completed interval.
- Eager/compiled CUDA tests are present for full physics with analytic optics,
  field placement, fresh/cache/fresh radiation, independent energy/water/salt,
  disk restart and NetCDF output. They require actual GPU execution.

## Reproduce

Use the same installed-wheel procedure as `.github/workflows/ci.yml`, copying
`tests`, `tools`, `workflows`, and `validation` outside the checkout. Then:

```bash
VFS_TEST_INDUCTOR=1 python -I -m pytest \
  tests/test_coupled_campaign.py tests/test_coupled_compilation.py \
  tests/test_coupled_device.py tests/test_radiation_mapping.py \
  tests/test_pyam21_workflow.py tests/test_coupling_momentum.py \
  tests/ocean/test_temperature_ale_roundoff.py -q
python -I tools/check_baseline.py --reference-version corrected-gibbs-v1 --mode portable
python -I tools/check_dry_helpers.py --mode portable
```

`capture_cpu.py --root CHECKOUT_OR_COPIED_TREE --output CAPTURE.pt` records the
CPU trajectory for either the base or this pass. Its records include full
state/cache tensors, independent inventories and stage reports. Compare them
recursively with exact tensor equality; no changed tolerance is involved.

## Remaining GPU checks

Follow [GPU.md](../../../../../workflows/coupled/GPU.md): run the CUDA-specific
tests, real-table preflight and CPU/CUDA comparison on the user's exact mesh
before the two-day run. Eager and compiled CUDA, actual GPU memory demand,
throughput, cross-device numerical agreement and long-run stability are not
established here. Existing eager scalar checks and iterative solvers remain;
GPU performance tuning is a later measured step. Start fresh when changing
source, device/runtime or compilation policy; old Mac restart migration is not
part of this pass.
