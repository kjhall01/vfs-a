# Explicit coupled ocean processes — 17 November 2026

**Follow-up:** the atmospheric failures archived here were subsequently traced
to short-edge momentum amplification in the coupling map. See the
[correction and new verification](../momentum-map/README.md). The failed-run
records below are retained unchanged as evidence.

The original campaign constructed an eddy-capable Roquet model but did not
schedule GM–Redi or ALE and initialized its unused mesoscale reservoir to zero.
The follow-up explicitly enables the existing energy-dependent GM transport,
Redi diffusion, eddy decay to heat, stage-wise funded backscatter and budgeted
rotational biharmonic sink, and post-exchange adaptive thermal ALE. See the
[run guide](../../../../../workflows/coupled/CAMPAIGN.md) for parameters/order.

Ocean dynamics/physics remain at 1200 s, radiation at 3600 s, with ten TKE
mixing and ten GM–Redi substeps (120 s each). The experiments here have real
AM2 physics and explicit **radiation OFF**, 256 generated cells, 24 atmospheric
levels, 12 ocean layers, 1000 m depth, CPU float64 and one thread. They are
implementation checks, not seasonal climate validation or a 16,384-cell result.

## Redi correction, separate from enabling the operators

The original rotated Redi operator rejected the default campaign after 7200 s:
near-zero TKE became negative. At the failing interval, 10, 40 and 100 eddy
substeps all rejected. The 10-substep candidate had minimum specific TKE
-4.62e-17 J/kg, from +2.28e-16 J/kg at that location. This is a lack of a
positivity property, not a violation of the explicit diffusion step bound.

`positive_redi=True` now conservatively limits whole local row fluxes, separately
for SA, passive tracers, TKE and mesoscale energy. `|R|^T |transfer|` bounds
withdrawal at each node; each row takes its donor entries' minimum allowance.
The transpose still conserves each tracer inventory and retains nonpositive
diffusion work with a reduced nonnegative row coefficient. The exact same
limited flux enters the local finite-H energy ledger. There is no clipping,
positive state floor, global throttle or unreported energy injection. Signed
CT diffusion is unchanged. A small floating-point margin affects only limited
fluxes. This is not a full maximum principle or an entropy-stability result.

Standalone `GMRedi` defaults remain unchanged; the campaign explicitly selects
the positive path. Separate commits contain the initial scheduling change and
this numerical correction. Unlimited smooth steps match the old operator
bitwise; independent sharp-store, coastal, inventory, variance and local/full-H
tests exercise the correction. Existing rejection tests remain intact.

## Validation

The external copied workflow/test tree imports the installed VFS wheel rather
than the source package. Focused checks: **186 passed, 1 skipped**; shared
exchange checks: **16 passed, 1 skipped** (202 passed total; optional GSW and
external real-radiation data are unavailable). The corrected-Gibbs and dry-helper portable
comparisons pass all **142 and 93** tensors. See the saved logs and JSON reports.

Campaign tests exercise actual backscatter RHS calls at all RK stages,
GM/Redi/decay activity, nontrivial ALE with combined ice-energy conservation,
water/salt/reservoir preservation, unchanged ice/snow/cold content, individual
switches, restart replay including cached synthetic radiation, failure without
input mutation, and NetCDF mesoscale energy. Actual operator counts and local
budget diagnostics are carried into checkpoints and CSV output.

The 240 s atmospheric configuration completes six hours: 180 GM–Redi substeps,
18 eddy-decay calls and 18 ALE calls, with all 256 columns remapped per call.
Numerical combined-energy residual is -0.836594 J/m2; water residual is
-3.49e-10 kg/m2 and relative salt change -2.14e-16. The separate longer attempt
stops after 20 h in the atmospheric CP energy solve, with max atmospheric speed
1107.76 m/s in the last accepted state. Raising the solve cap from 12 to 24 or
64 iterations did not resolve that failing interval; the solver was not changed.
The reduced-ocean control (GM–Redi/decay/backscatter/ALE disabled) also rejects
there after 18 h 20 min. Thus that failure does not require the added operators.

Reducing only the atmospheric step to 60 s improves the 18-hour numerical energy
residual from -2.4492 to -0.1150 J/m2, but that run also rejects in the same
atmospheric solve after 20 h 40 min. No smaller-timestep cure or completed day is
claimed. The requested 240 s atmospheric step remains explicit in the example;
the guide identifies the long-run blocker before the run commands. The ocean
and physics timesteps remain 1200 s in all of these comparisons.

`summary.json` retains configurations, statuses, source hashes and restart
fingerprints for these attempts; CSV and failure logs retain their raw output.
The old unbounded-Redi attempt has its own pre-correction source hashes. No
failed attempt is relabelled as a completed day. Binary snapshots/checkpoints
remain local run products. Full-year radiative execution, the user's saved
16,384-cell mesh, GPU, compiled performance and long differentiation remain
unqualified by this follow-up.
