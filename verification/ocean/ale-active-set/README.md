# Retain the ALE entropy constraint after a stored-state violation

The second uploaded seasonal run stops at second 38400, reporting column 16266,
entropy change/tolerance -1.00005422 and 24 iterations. It already includes the
stored-temperature acceptance fix and conservative radiation heating map.

## Diagnosis and correction

The iteration's acceptance test checks the actual stored temperature. However,
the minimum-change step decides whether to activate entropy using a *linearized
prediction*. These can disagree at the tolerance boundary. In the reproduced
nearby column, stored entropy remains outside tolerance, but the prediction
looks acceptable. The resulting 5.88e-17 K step is smaller than half a temperature
ulp and changes no stored temperature. The same rejected state then repeats
through all 24 iterations. This is a numerical active-constraint stall in ALE.

After a column's stored candidate violates entropy, the solver now remembers
that event and uses zero as its subsequent entropy activation threshold. A
negative predicted entropy change engages the existing constrained correction,
which targets zero change. It no longer repeatedly accepts an ineffective
unconstrained proposal just inside the tolerance. The nonlinear final checks,
256/512-epsilon energy/entropy tolerances, minimum-change equations, domain
checks, timestep settings and 24-iteration cap remain unchanged. No accepted
state is clipped or repaired after validation. The change is confined to the
research temperature remapper; packaged VFS and upstream PyAM21 are unchanged.

## Reproductions

| Case | Before | After |
| --- | --- | --- |
| Deterministic 16-column cold case, seed 26 | Column 0 rejects after 24 iterations; entropy/tolerance -1.00023884 | Accepts in 4; column 0 ratio +0.000520028 |
| Nearby state of uploaded column 16266 | Rejects after 24; ratio -1.00012837 | Accepts in 3; column 16266 ratio 0 |

The deterministic case uses salinity scale 4.257426495909136. Its independent
returned-state checks cover energy, entropy, depth, salt, passive tracer, TKE,
EKE and momentum, on one/four threads. A deliberately stalled solver still
rejects at the iteration limit. See [synthetic iteration trace](synthetic.json).

The nearby case starts from the captured pre-ALE ocean in the original Linux
checkpoint replay. Only column 16266's stored temperature content is changed
by `temp_h + h * (-8.9e-7 * torch.linspace(1, 0, 12))`. Every other field and
column is retained. This small controlled perturbation reproduces the same
stalling mechanism on Linux; **it is not a bitwise reproduction of the Mac
failure**. Maximum corrected column energy residual/tolerance is 0.00685802.
See [iteration trace](nearby-checkpoint-column.json) and [input provenance](provenance.json).

The unmodified uploaded checkpoint advances **38400 → 39600 seconds** on Linux
with the previous solver, as well as with the correction in eager and native
compiled execution. Its original radiation cache has 1201 s left on both
alarms, sufficient for this 1200 s interval. Fresh optics are guarded to fail;
none are called. All configured stages execute, including ten GM–Redi substeps,
eddy decay, mixing, sea ice and ALE in all 16384 columns. The native compiled
replay's maximum ALE column energy residual is **1.19795916e-5 J/m²**. See
[previous solver](original-checkpoint-before.json),
[corrected eager replay](original-checkpoint-after.json) and
[corrected compiled replay](original-checkpoint-compiled.json).

These are diagnostic reconstructions across source versions, not production
restart migrations. No fresh optical-table integration, long seasonal run,
Mac replay, GPU execution or full-coupling backward behavior is claimed.

## Validation and reproduction

Installed-wheel tests outside the checkout: **107 passed, one optional GSW
skip**, including native Inductor radiating intervals and bitwise compiled
restart. Frozen portable comparisons: **142 corrected-Gibbs / 93 dry-helper
tensors pass**. See [tests](focused.txt), [baseline](baseline.json) and
[dry helpers](dry.json). Environment: Linux CPU float64, Python 3.12,
PyTorch 2.15.0+cpu and pinned PyAM21 e1bdfcc3b23c05adea80b35af880702bb85718eb.

Use an installed VFS wheel and a check tree containing workflows, validation,
tests, tools and verification, without the vfs source directory:

```bash
python -I verification/ocean/ale-cold-columns/compare.py \
  --seed 26 --salinity-scale 4.257426495909136 --output after.json
# To reproduce the previous rejection, add --research-root pointing at the
# workflow/research tree from commit 2b557a49c5787e903a9fe85600db8806b607dd3b.

python -I validation/coupled_debug/replay_ale.py \
  --source /path/to/PyAM21 --config /path/to/configuration.json \
  --mesh /path/to/mesh_16384_exact.npy \
  --restart /path/to/last_good_restart.pt --output ale-diagnostic
```

The diagnostic writes `pre_ale.pt` and `result.json`, allowing a future failure
to be isolated without repeating a full campaign. Add `--compile` to exercise
native compilation. A fresh optical call raises rather than extending the cache.
For production, update the PR branch and start in a fresh output directory;
normal campaign fingerprint checks intentionally reject previous-source restarts.
