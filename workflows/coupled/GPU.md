# Full coupled campaign: device and precision

`--device cpu|cuda|cuda:N|mps` and `--dtype float32|float64` select execution
explicitly (YAML defaults: `device: cpu`, `dtype: float64`). The mesh, levels,
timestep schedule, forcing and enabled processes are retained.
`cuda:N` indexes the GPUs visible to the allocated process. The
entire atmosphere/column and ocean/ice calculation uses the same GPU and mesh.
No component is distributed across multiple GPUs in this pass.

| Device | Floating-point types | Example |
| --- | --- | --- |
| CPU | float64 (default), float32 | `--device cpu --dtype float64` |
| CUDA GPU | float64 (default), float32 | `--device cuda:0 --dtype float64` |
| Apple GPU / MPS | float32 | `--device mps --dtype float32` |

MPS cannot represent float64. Requesting that pair fails with a clear error;
there is no automatic downcast. Float16/bfloat16 are not supported by this
solver configuration. The Python `example` initializer also accepts an explicit
`device` and `dtype`; construction no longer changes the caller's default dtype.

The workflow retains seasonal/diurnal SeaEsfRad, RAS convection, clouds,
atmospheric mixing, Roquet ocean dynamics and TKE mixing, GM–Redi, eddy decay,
backscatter/biharmonic dissipation, adaptive thermal ALE, moving category ice,
snow and prognostic ice temperature as configured. CUDA never implicitly turns
off radiation or substitutes a simpler physics package. `--compile` remains an
independent option for the same nine tensor regions.

**Qualification:** this implementation was developed on CPU-only PyTorch.
CPU regression checks and GPU test definitions are distinct from executed GPU
evidence. CUDA/MPS correctness, real-table execution and GPU throughput require
the tests below on actual hardware. The reported
successful two-day Mac run belongs to the merged CPU branch, not this CUDA pass.

**Float32 is experimental.** Full-physics CPU float32 checks pass, but a 32-cell,
four-interval comparison at the default 1200/240/3600-second schedule has a
maximum scaled eddy-content difference of about 0.167 from float64 (maximum
specific eddy-energy difference 0.0753 J/kg). Completing
a run is not evidence that its precision is adequate for a climate campaign.
Use `--reference-dtype float64` in the comparison below. See the recorded
[precision measurements](../../verification/atmosphere/pyam21/campaign/precision/README.md).

The lagged mixing budget uses `max(1e-9, 32*eps(dtype))`; reciprocal ice-drag
Newton convergence uses `max(1e-12, 32*eps(dtype))`. The FP32 value is
3.814697265625e-6. These checks otherwise demand sub-ulp convergence in FP32.
Float64 thresholds remain unchanged; explicit drag tolerances remain literal.
Neither adjustment changes fluxes, imposes a state floor or disables a guard.
The existing energy/entropy checks already scale with native dtype epsilon.

## Environment and inputs

Use a GPU notebook, terminal or batch job. An A100 is a suitable FP64 target.
Request one GPU; a four-GPU node type does not require using all four.
Use a CUDA-enabled PyTorch environment, and install this checkout
and its pinned optional dependency into that same environment:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python -m pip install .
python -m pip install -r workflows/pyam21/requirements.txt
```

Keep the upstream namelist checkout at revision
`e1bdfcc3b23c05adea80b35af880702bb85718eb`, as described in [CAMPAIGN.md](CAMPAIGN.md).
Transfer your existing `mesh_16384_exact.npy`, local YAML and extracted
`input_data/sea_esf` directory to storage available to the job. Do not use the
`+cpu` wheel from this repository's Linux CPU CI recipe in the GPU environment.

Set these paths to the transferred files, from the VFS repository root:

```bash
CONFIG=/path/to/aquaplanet-local.yaml
MESH=/path/to/mesh_16384_exact.npy
PYAM_SOURCE=/path/to/PyAM21
RADIATION_DATA=/path/to/input_data/sea_esf
DEVICE=cuda:0
DTYPE=float64
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

For MPS, set `DEVICE=mps` and `DTYPE=float32` instead. Use a PyTorch installation
with MPS support and check `torch.backends.mps.is_available()` rather than the
CUDA availability command above. Leave `PYTORCH_ENABLE_MPS_FALLBACK` unset;
unsupported MPS operators must be reported rather than silently moved to CPU.
The cuBLAS variable is only relevant to CUDA.

GPU execution requires deterministic PyTorch algorithms for exact same-device
restart replay. The CLI configures the cuBLAS workspace if unset, rejects an
incompatible setting, and restores the caller's deterministic-algorithm policy
on exit. Set the environment variable before starting Python, particularly in
Jupyter where CUDA may already have been used. Unsupported deterministic
operations fail visibly; there is no silent CPU or eager fallback.

## Preflight, then the two-day segment

First use eager GPU execution to isolate device placement from compiler behavior:

```bash
python -I workflows/coupled/campaign.py \
  --config "$CONFIG" --mesh "$MESH" --source "$PYAM_SOURCE" \
  --radiation-data "$RADIATION_DATA" \
  --device "$DEVICE" --dtype "$DTYPE" --threads 4 --no-compile \
  --output gpu-preflight --preflight --verbose
```

The preflight executes all components, including fresh real-table radiation,
writes a CPU-stored restart, reloads it onto the GPU and requires exact replay.
It writes `run.log`, `manifest.json`, `status.json` and `preflight_restart.pt`.
The manifest identifies device, GPU model/memory/capability, CUDA/PyTorch runtime,
determinism policy, source hashes, radiation data hashes and physical settings.
Unavailable hardware is an explicit logged failure before model construction.

After qualification, start a fresh segment:

```bash
python -I workflows/coupled/campaign.py \
  --config "$CONFIG" --mesh "$MESH" --source "$PYAM_SOURCE" \
  --radiation-data "$RADIATION_DATA" \
  --device "$DEVICE" --dtype "$DTYPE" --threads 4 --no-compile \
  --output gpu-2day --stop-after-days 2 --verbose
```

Use the same flags and a new output directory on resume, adding
`--resume gpu-2day/restarts/restart_00172800.pt --stop-after-days 14` to advance
to absolute model day 14. Scheduler wall limits still apply. Choose a restart
cadence in the YAML appropriate to those limits before starting a long run;
an abrupt scheduler kill cannot run Python's last-good exception handler.

## CPU/GPU comparison with all physical settings retained

```bash
python -I validation/coupled_device/run.py \
  --config "$CONFIG" --mesh "$MESH" --source "$PYAM_SOURCE" \
  --radiation-data "$RADIATION_DATA" --device "$DEVICE" --dtype "$DTYPE" \
  --reference-dtype float64 --threads 4 \
  --steps 4 --warmup 1 --output gpu-comparison
```

This runs the same saved mesh and configuration first on CPU, then the selected
device, with explicit precision. Without `--reference-dtype`, the CPU reference
uses the candidate dtype. The command above instead measures the combined
device/precision difference from CPU float64. Real optical tables are mandatory. Four intervals at the
default 1200/3600-second physics/radiation cadence include fresh/cache/cache/fresh
radiation. A different cadence requires enough intervals to reach another fresh
call. Progress is printed after each interval; partial failures are retained in
`comparison.json`. Each mode saves a final NetCDF snapshot and checkpoint.

The report includes every state/cache tensor's absolute and scaled difference,
actual GM–Redi/decay/ALE counters, independent total-energy/water/salt inventories,
accumulated boundary forcing, local residuals, exact disk restart replay, GPU
allocated-memory peak and synchronized timing. `comparison_completed` means the
measurement/replay procedure completed; it is not a scientific pass threshold
for CPU/CUDA field differences. Inspect the reported differences and budgets.
Inventory reductions use each mode's native precision; FP32 budget residuals
include diagnostic rounding. Tensor difference reporting promotes copies to
CPU float64 without changing the simulation. CPU and GPU arithmetic need not be bitwise equal. Native solver checks remain
active and any rejection fails the comparison.

Setup and warmup are reported separately. Advance timings wait for CUDA work to
finish and exclude diagnostic/NetCDF/checkpoint output. Fresh-radiation intervals
can cost substantially more than cache hits, so inspect the individual timings
and use more intervals for a representative throughput estimate.

Repeat with `--compile --output gpu-compiled-comparison` to test native
Inductor, including its warmup cost. The installed PyTorch/compiler stack must
support the selected backend; compilation errors remain visible. Start with
`--no-compile` on MPS. For an eager/compiled comparison on the same GPU, the existing
`validation/coupled_compile/run.py` also accepts `--device` and `--dtype` and now
synchronizes its timing boundaries. It holds both model copies in GPU memory;
the CPU/CUDA comparison above runs one mode at a time.

## Focused GPU regression tests

Run against an installed wheel from a copied test/workflow tree outside the
checkout, following the installed-wheel CI convention. For a quick development
check at the checkout root:

```bash
python -m pip install pytest
VFS_TEST_CUDA_COMPILE=1 python -m pytest tests/test_coupled_device.py -q
```

Use `VFS_TEST_MPS_COMPILE=1` for the optional MPS compilation case instead.
The device cases exercise eager and compiled full coupling, device ownership,
analytic radiation fresh/cache/fresh, independent budgets, disk restart and
NetCDF. They skip when the required hardware is absent; a skipped case is not GPU evidence.
The real-table comparison above is a separate required GPU check.

## Ownership, restart and performance limits

Geometry/data preprocessing and NetCDF/file access stay on CPU. Geometry moves
before physics maps and prognostic states are constructed; hot tensors remain
on the chosen device. Upstream table caches follow their input tensors' device.
The upstream climatology reader still interpolates prescribed records on the
host and transfers profiles, so this is not a claim of zero host work/transfers.

Checkpoints store tensors on CPU and `unpack` explicitly restores device
placement. Storage portability does not bypass configuration/source/runtime
fingerprints: begin a **fresh run** when switching from the older CPU branch,
device, dtype, hardware/runtime identity or compilation policy. This pass does not
migrate the existing Mac checkpoint into a CUDA continuation.

Eager checks, iterative solves and scalar diagnostics still synchronize with
the host. They remain in place for this correctness pass; GPU speedup is not
guaranteed. Subsequent optimization should use the measured profile while
retaining the same budgets, checks and active physical processes.

## Diagnosing an endpoint/tendency rejection

For `PyAM21 endpoint and tendencies disagree beyond roundoff`, keep the failed
segment's `last_good_restart.pt`, `manifest.json` and `configuration.json`.
The message alone does not distinguish a physics inconsistency from accumulated
roundoff. Do not disable the check or change precision to continue this state.

On the original GPU/runtime and unchanged model checkout, run:

```bash
python -u -I tools/diagnose_pyam21_endpoint.py \
  --repo . --failed-run /path/to/failed-segment \
  --source "$PYAM_SOURCE" --radiation-data "$RADIATION_DATA" \
  --mesh "$MESH" --output /path/to/new-endpoint-diagnostic
```

This reuses the failed segment's complete configuration, compilation choice,
real radiation and normal restart identity check. It advances one interval via
the existing preflight. If the error recurs, `endpoint_failure.json` reports each
field's discrepancy, unchanged tolerance and worst cell/level. The accompanying
`endpoint_failure.pt` saves selected full columns and their actual PyAM21 moist
inputs for independent investigation. The original exception is re-raised;
`replay/status.json` remains failed. The tool exits successfully only to indicate
a completed capture, or a completed preflight if the failure did not reproduce.

The tool is under `tools/`, outside the campaign's hashed source directories.
It can be fetched separately into an unchanged checkout: no model-source update
or restart fingerprint bypass is needed. If the same-device run does not match
the original source/runtime identity, the ordinary restart guards still reject.
