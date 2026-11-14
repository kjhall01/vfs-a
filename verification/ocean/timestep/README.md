# Timestep campaign implementation checks

Parent main: 74224c194650bf4d98d192d19719d54c529b39fd. No package code changed.
Tests/drivers executed outside the checkout using the existing installed wheel
(SHA256 77e7066f7a56699f13e164b27052b4095be2ef453308b706b8d8bb6431d1c6a1),
CPU float64, Torch 2.15.0+cpu, one thread.

- Four unit checks pass: mixing cadence, completion-versus-accuracy distinction,
  weighted endpoint errors, atomic JSON rejection of nonfinite values.
- 32 cells x 4 layers: 60 simulated seconds, four cases, dt30/60 with dt30/15
  references. All trials completed; 60 s meets the illustrative budgets.
- 128 x 16: 360 simulated seconds, wave/front/forced, dt60/120/240 with dt60/30
  references. All completed; 240 s meets the budgets over this short interval.
  These are harness checks, not physically meaningful stability qualifications.
- Saved 16384-cell mesh (3000 Lloyd iterations): 4000 m depth, 40 layers,
  two-hour wave screen at dt240/600/1200/1800. All workers received signal 9
  before any step. Instrumented repeat localizes termination to initial energy
  evaluation. Resource termination is not a numerical rejection or CFL result;
  an out-of-memory cause is plausible but not proven by an OS diagnostic here.
- Same mesh with eight layers completed one 240 s wave step. This only checks
  that mesh construction and basic stepping work. It cannot qualify 40 layers.

Mesh generator SHA256:
3437509cb1520bb4e27aea7f3e9b989c64c19673d0f49523f0043ab81a9b0d0b.
The 3000-iteration mesh did not meet every original convergence tolerance;
we do not call it an exact SCVT. Minimum dc is 154995 m. At 4000 m depth the
external-wave speed estimate is 198.06 m/s: dt240 gives indicative Courant 0.307,
dt600 about 0.767, dt1200 about 1.53. These are not eigenvalue stability limits.

`trials.json` retains configuration, code/mesh hashes, statuses, budget maxima,
errors and progress without large endpoint tensors. Two summary files retain
the exact comparison errors and illustrative criteria. The earlier screen and
smoke predate instrumentation/explicit setup-wide no-grad; their source hashes
are retained as recorded. The functional 128-cell campaign uses the instrumented
version. Production stepping was no-grad in all runs. No full-suite rerun was
needed for this standalone harness; hosted CI is a separate check.

**No maximum timestep at 16384 x 40 has been established.** Run the documented
screen and reference campaign on the target machine, then extend near-limit
candidates to 24–72 hours before interpreting them as an operating choice.
Large local tensor endpoints and mesh generators are omitted whole from git.

The final coordinator avoids importing PyTorch and runs endpoint comparisons
in separate processes. Four unit checks and the complete 32x4 forced reference
comparison pass again. A solo 16384x40 attempt with this lightweight coordinator
still received signal 9 during initial energy evaluation, before any step.
This confirms the campaign overhead reduction did not remove that blocker.
