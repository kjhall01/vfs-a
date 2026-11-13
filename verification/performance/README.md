# Performance campaign execution checks

Installed wheel, Python 3.12 / PyTorch 2.15.0+cpu, Linux x86_64, float64.

* `smoke.json`: all 12 configurations complete at 16 cells × 4 layers, dt1,
  one warm-up and one measured step, including all three closure choices.
* `quick-validation.json`: all six workloads complete at 128 × 16, dt60,
  one warm-up and two evolving steps per repeat, two repeats, one CPU thread.
* `threads4.json`: final worker exercised at four CPU threads, six workloads,
  16 × 4, dt1, one warm-up and one measured step.
* `timeout.json`: forced timeout yields no SYPD and a failed campaign exit.
* Three accounting/JSON tests pass outside the checkout; portable corrected-Gibbs
  and dry-helper comparisons pass. No full regression suite rerun locally for
  this harness-only change; hosted CI provides the full suite separately.

Compact execution summaries are retained; these short runs are not statistically
qualified machine comparisons or long-run model validation. Early summaries
precede metadata-only package-JSON hashing and missing-score formatting changes.
The final CLI additionally rejects ALE timing windows shorter than its remap
cadence. No numerical workload changed between these checks. An initial test
environment installation ran out of disk downloading GPU dependencies; the
CPU-only PyTorch installation succeeded before any measurements were collected.

No package source, physical defaults, tolerances, fixtures or atmospheric
worktrees were modified. No Mac, MPS, GPU, compile or backward run was performed.
