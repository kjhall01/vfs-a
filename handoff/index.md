# Current ocean handoff — potential-enthalpy remap experiment

This branch follows #39 and adds an explicitly selected research EnthalpyRemapper.
Read [ENTHALPY_ALE.md](../verification/ocean/ENTHALPY_ALE.md). It conservatively
projects potential enthalpy and salt, then reconstructs the existing entropy state.
The density EOS, Hamiltonian, sharp-front solver, momentum remap and grid policy
are unchanged. All installed runtime files and atmosphere remain unchanged.

In short wave trajectories, the dominant ~63 kJ/m² reference-thermal loss disappears:
total remap energy change is +88 J/m² at eight layers, −132 J/m² at 32, and −167 J/m²
at 64. These remaining dynamic-enthalpy changes are not negligible relative to the
tiny wave energy. Entropy is produced by the mixed thermodynamic state and explicitly
accounted. The wave accuracy tradeoff remains visible; no total-energy-conserving
or universally entropy-stable PLM claim is made.

Wave comparisons use column SALINITY content so both remap variants share a
conserved observable. Runs cover 0.05 period; they are not full-period qualification.
Validation: 783 installed-wheel portable tests passed, one CUDA skip; 164 legacy
ocean tensors exactly unchanged. Twenty focused tests include repeated remaps,
complete-step entropy accounting and one derivative check. Leave draft/unmerged.
The next step is a local full-Hamiltonian treatment of the remaining mechanical
energy changes, together with wave and mixing accuracy checks; no global heat fix.

---

# Current ocean handoff — entropy ALE integration

This branch combines draft #38's entropy/salt sharp-front candidate with the
unchanged vertical remap machinery from #32. Read
[ENTROPY_ALE.md](../verification/ocean/ENTROPY_ALE.md) before interpreting success.
The new research wrapper conserves native entropy/salt/passive contents and edge
momentum, and reports energy and temperature-content changes separately. It does
not turn remapping into energy-conserving physical mixing. Rotation remains frozen.

Original-remap full-period wave RMS improves with vertical refinement: approximately
0.442/0.268/0.191/0.155/0.137 at 4/8/16/32/64 layers, versus 0.0627 without remap.
Energy defects do not decrease monotonically. Integrated entropy/ALE trajectories
are explicitly short (0.05 period); they exercise actual remapping, not just a
non-triggering grid policy. Reference-thermal energy loss is an unresolved issue.
Smooth single-remap temperature RMS decreases from 0.0703°C at eight layers to
0.00319°C at 64. Layer count alone is not a qualification criterion.

Validation: 763 installed-wheel portable tests passed, one CUDA skip; all 164
legacy ocean tensors exactly unchanged; 142 moist/93 dry atmosphere portable
comparisons pass. Nine new integration tests and the original 18 ALE tests pass.
Next scientific work should address the remap's thermodynamic energy defect and
required-remap wave accuracy, then longer integrated runs. No climate, GPU,
long-backward or atmosphere-coupling qualification. Leave draft and unmerged.

---

# Current ocean handoff — sharp-front transport research

The sharp-front pass builds on draft #37 (`bf5ad5cb8887ec05d26e0566dde0c636daf0676a`).
Read [the report](../verification/ocean/SHARP_FRONT.md) for derivation and measured limitations.
Rotation, kinetic reconstruction, installed runtime, defaults, fixtures and tolerances remain frozen.
A local flux-correction target controls entropy, salt and passives; a matched Hamiltonian
work budget blends active corrections toward the parent when needed for zero-velocity continuity.
Passive local bounds survive this blend; active local bounds are conditional. Temperature
has no maximum principle. No post-step clipping or ocean import from atmosphere is added.

The isolated 64-cell sharp front stays within SP 34–36 and passive 0–1 to roundoff,
compared with parent SP 33.235–36.773 and passive maximum 1.279. Smooth-profile
observed orders fall to about 1.6–1.7. A two-hour, 32-cell, three-layer active comparison
also preserves these SP/passive bounds; temperature still overshoots its initial range.
Energy drift improves from −43.05 to −5.52 J/m² on timestep halving. Numerical entropy
production is recorded explicitly and is not a calibrated physical mixing closure.

Validation: installed-wheel portable suite 736 passed, one CUDA skip; all 164 legacy
ocean tensors exactly unchanged; atmosphere portable comparisons pass (142 moist,
93 dry tensors). Fourteen focused tests cover independent energy derivatives/work,
Euler/SSP budgets, global admissibility, passive local bounds and single-edge zero
crossings with other faces limited. No GPU, long-run, full backward or ALE qualification.
Next work is broader front-candidate wave/jet qualification and the smooth-extrema
accuracy tradeoff; smaller rotation residuals are not a prerequisite. Draft and unmerged.

---

# Current ocean handoff — frozen-candidate qualification

This branch measures draft #36 at
`705720c79e38c2f07de39d45a33c9ede50707b87`; it does not change the candidate.
Read [the campaign report](../verification/ocean/QUALIFICATION_CAMPAIGN.md)
and its compact per-case JSON evidence before interpreting conservation as
accuracy. The campaign covers three-period nonlinear internal waves, seven-day
Williamson jets, one-day flat/sloping stratified rest, and isolated smooth/sharp
scalar transport. CPU float64 only; no GPU or backward qualification.

Smooth transport converges at approximately second order, but sharp fronts
oscillate beyond their initial extrema. Passive nonnegativity is not a maximum
principle. Rotating-jet drift persists under timestep refinement even though
energy loss improves; the fine initial tests show slower kinetic/rotation convergence than pressure.
Initial-balance tests extend to 65,536 cells, plus a mesh-relaxation comparison.
Flat/sloping stratified rest stays near roundoff. These findings remain exposed,
not corrected or assigned subjective acceptance thresholds in this branch.

Next numerical work should separately address sharp-front limiting with the
Gibbs/work/entropy pairing intact, and joint rotation/kinetic reconstruction
consistency. Then rerun these frozen cases before active runtime promotion,
baroclinic-growth qualification and ALE composition. ALE #32 remains separate;
no coupled atmosphere claim follows. All installed runtime files, defaults,
fixtures and tolerances are unchanged. No oversized artifacts are uploaded.

---

# Current ocean handoff — Gibbs/entropy transport candidate

Dependent on #35 at `c3e9d6cdf39f4d978ed30cfb9dc22d660f7c4e39`, branch
`ocean/entropy-compatible-transport`. New research state stores entropy and
salt contents, with explicit temperature recovery for unchanged JMD95 density.
The physical Hamiltonian is the same as #35. Surface calorics are derived from
GSW/TEOS-10; this is explicitly borrowed thermodynamic physics, not a new EOS.

Read [provenance, derivation, budgets and limits](../verification/ocean/ENTROPY_TRANSPORT.md).
Ocean-owned Gibbs face means cancel reference-caloric pressure; entropy-flux
changes are attached to actual salt-flux changes, adapting atmospheric ideas
without imports. Positive limiter work is mechanically paired; negative work
becomes nonnegative heat. The tilted-layer entropy failure in #35 is removed.

New per-incidence amount allowances recover the high trace as an individual
edge's velocity vanishes, even while other edges remain limited. Tests cover
that stronger limit as well as whole-flow rest. Positive salt, nonnegative
passives, energy work and every Euler entropy-source ledger are checked.

Scope: strictly **0 < SP < 42**, interior reference temperatures, nonnegative
passives; no salt floor. Exactly fresh active water is unsupported. Limiting
enforces admissibility, not local extrema; full stages leaving the nonlinear
caloric domain are rejected. No automatic dt retry or exact finite-step energy
claim. Equal heat allocation is numerical, not a calibrated mixing closure.

Installed-wheel portable: **717 passed, one CUDA skip**, including 20 new
tests. All 164 legacy-ocean tensors exactly equal; atmospheric portable checks
pass. Independent pressure refinement is about second order; 0.05-period
internal-wave errors decrease at 64/256 cells. Temporal energy refinement is
about third order in the tested short case. Long-wave/jet, GPU, performance and
full backward qualification remain open. All 100 runtime Python files, defaults,
fixtures and tolerances are unchanged. Keep drafts unmerged; ALE #32 separate.

Next: transport accuracy/overshoots, sustained wave phase/damping and rotating
balance; assess the per-face budgets' diffusion, then decide opt-in runtime
promotion and ALE composition. The new stepper remains a research candidate.

---

# Previous ocean handoff — seawater calorics and enthalpy coordinates

Dependent on #34 at `81481b0b81dcd193666dde86db265b82b7c70d61`, branch
`ocean/seawater-caloric-reference`. All 100 installed runtime Python files
remain unchanged. New research code fits one surface Gibbs potential to GSW
3.6.23 evaluations and derives enthalpy, entropy and heat capacity analytically.
Temperature scale and reference-composition salinity conversion are explicit.
JMD95 still supplies density; this is not a switch to full TEOS-10 dynamics.

Read [derivation, independent reference checks and limits](../verification/ocean/SEAWATER_CALORICS.md).
A small research state stores potential enthalpy content and recovers theta
for JMD95. Analytic transformed covectors and a matched centered force close
spatial work; signed thermal conversion tracks the nonconservative thermal
inventory contribution. Salt, volume and passive inventories remain conserved.
The same energy/entropy definitions also drive a separate thermodynamic
exchange reference. No forward autograd or atmospheric imports are added.

The tilted-layer counterexample still decreases entropy with calibrated
calorics. Do not promote this as a complete bounded active runtime stepper.
Positive salt alone also fails to guarantee joint thermodynamic admissibility:
the extreme exchange test is rejected outside the declared reference domain.

Installed-wheel portable suite: **697 passed, one CUDA skip**, including 17
new tests; 164 ocean comparison tensors exactly equal; atmospheric portable
comparisons pass. Independent GSW measurement runs outside the checkout and
is required in hosted CI. Reference data and compact results are committed;
all uploaded files stay below 128 KiB. No default, fixture, tolerance or ALE
change. Keep drafts unmerged. GPU, long runs and performance are unqualified.

Next: a joint enthalpy/salinity/geometry face construction with both the
Hamiltonian work identity and entropy inequality, plus full Euler caloric
admissibility, before runtime adoption and ALE composition.

---

# Previous ocean handoff — thermal work and entropy audit

Dependent on #33 at `ae8d1a1348fd154bc5a9c0fcda8b296eeac95664`, branch
`ocean/thermodynamic-work-closure`. All runtime files remain unchanged; new
research code/tests use explicit constant-cp and idealized composition entropy.
JMD95 density is retained. These calorics are not empirical seawater validation.

Read [the derivation, counterexamples and next boundary](../verification/ocean/THERMAL_WORK.md).
Signed thermal conversion closes instantaneous energy work without a force jump
at zero velocity, while accounting explicitly for changed temperature content.
However, that equal-endpoint thermal allocation decreases entropy on tilted
layers. Do not promote it as a bounded active ocean stepper.

A separate fixed-geometry thermodynamic exchange reference conserves spatial
energy/salt and produces nonnegative entropy, including a limited salt outflow
budget. It is not an advective-limiter repair. Large finite Euler stress updates
can still decrease entropy; do not infer a finite-step law from the spatial proof.

Installed-wheel portable suite: 680 passed, one CUDA skip; 21 new tests. The
164-tensor ocean capture is exact, and atmospheric portable comparisons pass.
No new runtime default, fixture, tolerance or ALE change. Keep drafts unmerged.
Next: validated reference calorics and matched thermodynamic face/exchange
operators, with stage positivity and entropy accounting, before runtime adoption.

---

# Previous ocean handoff — bounded horizontal transport first pass

Base main `5841cda8b4585418b18f0ba0486315870b1a59b2` includes JMD95 #31.
The independent `ocean/bounded-transport-first-pass` branch adds ocean-owned
donor/FCT volume/content ledgers, actual-carrier Euler budgets, all-face bounds,
and a research composition with the existing centered JMD95 momentum tendency.
It borrows atmospheric transport ideas without importing atmospheric code.
T and S are independent concentrations; no sum-to-one species constraint.

Read [derivation, measurements and limits](../verification/ocean/BOUNDED_TRANSPORT.md).
Installed-wheel suite: 659 passed, one CUDA skip; 17 new tests. The 164-tensor
legacy-ocean comparison is exact and atmospheric portable comparisons pass.
The active research composition preserves scalar bounds and inventories but
has a measured energy residual. The diagnostic all-mechanical pairing closes
work and reintroduces a zero-velocity force jump; it is never installed in a step.
No new active runtime stepper, default change, thermal reservoir or fixture change.

Keep this draft unmerged. Next: derive a physical thermodynamic treatment of
limiter work, preserving T/S inventories and continuous forces, before promoting
a bounded active stepper. ALE #32 remains separate; its composition and energy
accounting are later work. Long active runs, GPU and performance are unqualified.

---

# Previous ocean handoff — active JMD95 Hamiltonian candidate

User-directed numerical extension of #30, on `ocean/jmd95-hamiltonian`.
New opt-in JMD95Dynamics/JMD95Energy use active potential T and practical S,
a fixed-datum buoyancy integral, analytic fixed-content energy derivatives,
and centered energy-paired scalar fluxes/pressure. Use checked_ssprk3 explicitly
for full-Euler stage rejection. No atmosphere imports in the new ocean modules;
all pre-existing runtime files, defaults, fixtures and tolerances are unchanged.

Read [derivation, evidence, reproduction and limits](../verification/ocean/JMD95_HAMILTONIAN.md).
Installed-wheel suite: 642 passed, one CUDA skip; nine new tests.
Spatial work closes to roundoff; independent pressure refinement is about second
order; short internal-wave errors decrease with refinement; temporal order is
about three. This is adiabatic, centered, unforced and experimental: no bounded
mixing, heat budget, Poisson/Jacobi proof, long-run or GPU qualification.
The rotation/kinetic reconstruction issue from #28 remains open.
Keep the dependent draft unmerged. Next: rotating/many-period stratified tests
and a separately derived bounded/mixing energy budget.

---

# Current ocean handoff — three bounded milestones, no runtime adoption

Main baseline: `e959b7005832a5041e082b6e300838e15bfbaa9e` (merged #26/#27).
Draft #28 diagnoses rotating reconstruction; draft #29 establishes actual-carrier
admissibility limits. This dependent design derives linear-Boussinesq gravitational
energy and matching active-content work **outside the numerical package**.
Read [the derivation, independent probes and review gates](../verification/ocean/BOUSSINESQ_ENERGY_DESIGN.md).
Local installed-wheel suite: 633 passed, one CUDA skip. No active timestep,
new selectable model, default change, atmospheric numerical/policy change,
fixture replacement or tolerance change. Original evidence is retained.

Priorities: review joint rotation/kinetic consistency; introduce an explicit
per-stage acceptance policy without clipping; validate proposed active pressure
accuracy and rest before implementing a separately selected stratified candidate.
No long-run, GPU/compile or long-backward qualification. Keep drafts unmerged.

---

# Current ocean handoff — transport admissibility after balance diagnosis

Baseline remains `e959b7005832a5041e082b6e300838e15bfbaa9e`. Draft #28
records spatial reconstruction limitations without changing production. This
dependent milestone demonstrates why positive h and conservation are insufficient
for passive bounds, and derives the actual carrier-outflow restriction.
Read [the stage results and next implementation boundary](../verification/ocean/TRANSPORT_ADMISSIBILITY.md).
Local installed-wheel suite: 627 passed, one CUDA skip. Archived defaults,
atmospheric runtime/policy, fixtures and tolerances are unchanged. Do not merge
or promote a new ocean default. Next: explicit per-stage acceptance design,
and a separate active Boussinesq energy/transport derivation before implementation.

---

# Current handoff — ocean rotating-flow investigation

Starting main: `e959b7005832a5041e082b6e300838e15bfbaa9e`; GitHub and ancestry
confirm #26/#27 merged. Original atmospheric worktrees remain untouched.
The reference ocean remains default. Runtime and fixtures are unchanged.
Installed-wheel local validation: 619 passed, one CUDA skip; both atmospheric
portable tools and ocean examples pass. The 64-cell five-day replay is exact;
larger regenerated meshes reproduce reported errors without claiming bitwise
geometry equivalence. Reconstruction moment defects persist; no correction is
promoted. Actual-carrier transport admissibility is the next independent milestone.
Read [rotating-flow diagnosis](../verification/ocean/ROTATING_BALANCE.md).
Older draft/unmerged entries below are historical snapshots.

---

# Current handoff — Williamson rotating-balance measurements

Recovered publication work on 31 October 2026: 615 installed-wheel tests pass,
one CUDA skip; original artifact hashes and a new frozen-input replay are exact.
See `verification/ocean/recovery_results.json` and the recovery section of the
report below. The publication subset omits four artifacts over 128 KiB; read
`verification/ocean/publication_omissions.json`. The full environment manifest and detailed replay report are also omitted;
original artifact hashes, scientific JSON summaries and 64-cell frozen inputs
are retained. Original local evidence remains unchanged.
#26 and its dependent verification publication remain drafts for review.

The separate verification branch leaves every runtime file unchanged. Read
[the measured errors, balance terms and limitations](../verification/ocean/WILLIAMSON.md).
Five-day thickness/normal-velocity RMS errors decrease over 64/256/1024 cells,
but initial rotation and kinetic-gradient consistency is weak/nonmonotonic.
Energy cancellation does not establish balanced-flow accuracy. Investigate this
separately before promoting the candidate or extending its scientific claims.
Integration remains draft #26, based on corrected atmospheric main. Keep both
integration and verification unmerged for review.

---

# Current handoff — ocean integration onto corrected atmospheric main

This branch integrates the reviewed ocean #23/#24 history onto main
`4ec610ed25c818e60863a8f8b772107b33f9251c` (including #18/#19 and #25).
Ocean runtime is byte-identical to #24 head `8936334`; all non-ocean runtime
is byte-identical to starting main. The archived ocean remains the default;
constant-density energy pairing is explicit and T/S remain passive there.

Read [integration evidence](../verification/ocean/INTEGRATION.md). Atmospheric
regression remains `corrected-gibbs-v1`; historical fixtures and tolerances are
unchanged. Scientific drivers now require an explicit pinned `--ocean-revision`
in addition to the existing structural/numerical pins for this combined package.
Unknown or modified files are still rejected. No shared numerical code changes.

Williamson rotating-balance verification belongs to a separate dependent change.
Keep these drafts unmerged for review. Earlier handoffs below are historical
snapshots; their merge-status statements describe their original dates.

---

# Current handoff — atmospheric integration and versioned references

Main is 6c5bf938503b357ba01c50ab99ae7e2cc770e90a (#25). Ocean #23/#24 are not ancestors of main.
This isolated stack preserves the original #18/#19 commits and artifacts; no ocean changes or expanded campaigns.
Investigation and numerical correction remain separate. See verification/ATMOSPHERIC_INTEGRATION.md.

Current regression uses **corrected-gibbs-v1**, pinned to integrated runtime
`d972f55683bf40b4edf35d1f5193949ff420cb0d`. Historical scientific baseline and
fixtures remain unchanged. Local installed-wheel suite: 585 passed, one CUDA
skip; both original-to-integrated captures match exactly (419 tensors each).
See the integration report for commands, quantities, provenance and CI status.

## Historical numerical-fix handoff (superseded reference policy)

# Current handoff — proposed paired Gibbs numerical fix

The separately authorized fix is implemented at
`a7facc1bcdb88f7dc9c10f4fb6059f29b0a9bac7`, dependent on open #18.
Read [results, pairing audit and reproduction commands](../verification/GIBBS_FIX_RESULTS.md).
Only `gibbs_means.py` changes at runtime: total-density logarithmic mean times
average condensate fraction, with entropy/composition/specific-volume traces
recomputed together. No limiter, ledger, defaults, fixture or tolerance edits.

All 22 old/new C5 cases completed; constant-composition drift falls from up to
5.897e-6 to 5.465e-16. Deformational nonmonotonic convergence remains unresolved.
C2 sweeps and unaffected dry/centered/ARK4 captures are exact; C3/C4, zero-cloud
one-sided derivatives and existing smooth/active-limiter gradients are recorded.
The candidate is not an internal partial-density mean; the limiter and energy
pairing audit and changed limiter activity are explicit in the results.

Installed-wheel macOS and hosted Linux each report 570 passed, two historical
moist-reference failures and one CUDA skip. Required CI remains red; no failures
are reinterpreted as passes. Six examples and dry portable comparisons complete.
Keep the fix unmerged for review and decide reference policy separately.
Baseline remains `8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648`; manifests require an
explicit pinned numerical-revision opt-in. No GPU, rollout or IMEX backward
qualification. Historical evidence below remains intact.

---

# Historical handoff — constant-composition face investigation

Separate investigation branch follows #17; no production or fixture changes.
Read the [derivation and bounded probes](../verification/CONSTANT_COMPOSITION_DERIVATION.md)
and the separate, **unimplemented** [numerical-fix proposal](../verification/CONSTANT_COMPOSITION_FIX_PROPOSAL.md).

Candidate condensate weights are total-density logarithmic mean times arithmetic
endpoint condensate fraction. Recompute the entire existing entropy/fraction/
specific-volume tuple. Algebra proves constant-q preservation and the discrete
Gibbs identity; 114 small CPU float64 faces and production toy CP ledgers support
it numerically. Exact zeros and one-sided clouds need no floor or composition
switch. Five focused tests include finite one-sided composition derivatives. Local
installed-wheel suite: 568 passed, one CUDA skip; both portable tools passed.
The coupled candidate is not an internal mean of condensate partial densities;
that limitation and the density-only logarithmic-mean incompatibility are explicit.

Do not activate this candidate, update the baseline, or change frozen fixtures.
Full model work/energy/ledger/gradient checks and small C5 reruns are required in
a separately authorized numerical-fix PR. This does not explain all Case 4 errors.
Numerical baseline remains `8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648`, with only the
already approved #16 interface extraction recorded separately. No large runs,
GPU qualification or broad transport redesign were attempted.

---

# Historical handoff — C5 horizontal transport evidence, dependent on #16

PRs #13/#14 merged; #16 extraction is ready for review and remains unmerged.
Its required Linux CI passed; downloaded artifacts show 551 portable tests passed,
one CUDA skip, and two separately recorded informational exact-fixture failures.
This dependent branch adds no runtime changes. Numerical baseline remains
`8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648`; approved interface commit remains
`ae132773e621671f70bfa6a698eebb1a82cffa94`.

Read [C5 methods](../verification/INCREMENT_1C.md) and
[results](../verification/INCREMENT_1C_RESULTS.md). The installed production
transport is exercised with prescribed edge wind at t, t+dt, t+dt/2; wind is
replaced before each Euler and at the final time, carrier/entropy/species evolve.
Independent cell-average references and every-stage incidence audits accompany
16/64/256-cell runs with 120/240 positive steps. Local suite: 563 passed, one
CUDA skip. All 22 development runs completed; smooth/empty-cloud limiting,
inventories, bounds, quadrature uncertainty and dt sensitivity are recorded.

**Negative result:** constant composition drifts in full-period probes, with
maximum stage error 5.897e-6 kg/kg. Frozen Gibbs gas logarithmic means versus
condensate arithmetic means already provide a minimal counterexample. Both CP
interfaces must enter error norms. Do not hide the bottom-interface error or
change the core; a separate derivation/fix proposal is required. Deformational
return errors are nonmonotonic across these coarse grids. C5 is not qualified.

Keep both PRs unmerged. Next queued work remains terrain-rest construction;
vertical/combined transport and larger campaigns remain pending. Do not claim
coupled energy closure, IMEX backward, GPU or atmospheric rollout qualification.

---

# Historical handoff — approved SSPRK3 extraction (#16)

PRs #13 and #14 are merged. Numerical baseline remains
`8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648`. The sole runtime change is the
approved stage-sequence extraction at
`ae132773e621671f70bfa6a698eebb1a82cffa94` in `bounded_moist.py`.
Read [extraction evidence](../verification/SSPRK3_EXTRACTION_RESULTS.md).
The atmospheric wrapper still validates and returns the same state and ledgers.
The manifest verifier accepts only the original blob or this exact approved
blob, rejecting all other runtime modifications. Keep #16 unmerged for review.

Next: dependent C5 prescribed horizontal transport harness using actual timed
production Euler maps, not copied RK formulas or invented candidate fluxes.
Replace prescribed wind at all RHS times and final output; evolve carrier,
entropy and species. Record discrete divergence and reference quadrature error.
Keep terrain-rest construction queued afterward. No coupled energy, GPU,
IMEX backward or long-rollout qualification follows from this work.

---

# Historical handoff — verification increments 1a and 1b

Baseline remains `8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648` (merged PR #12).
PR #13 is increment 1a: geometry/provenance and the full C1–C12 tracker.
Its updated head e657729 passed both required hosted runs and remains unmerged
for review. No numerical package source, fixtures or tolerances changed.

Increment 1b is developed separately, dependent on #13. Read the
[tracker](../verification/README.md), [1b methods](../verification/INCREMENT_1B.md)
and [1b results](../verification/INCREMENT_1B_RESULTS.md).
It adds scalar-energy AD/analytic/FD sweeps (C2), component work matrices including
fixed-lid reservoir and bounded heat (C3), and observation of the three actual
production Euler stages with independent incidence/inventory audits (C4).
The shared manifest approach archives drivers, geometry, prognostic inputs,
directions and actual stage diagnostics. No RK formulas are duplicated.

Small 16/3 and 32/4 cell/layer CPU float64 cases run first. Local installed-wheel
suite: 548 passed, one CUDA skip. Broad campaign qualification remains pending.
No IMEX backward, rollout or GPU support is inferred from these diagnostics.

Next queued work is prescribed horizontal/vertical/combined transport (C5), then
physically consistent terrain-rest construction (C6). Do not treat the current
nonuniform-terrain diagnostic states as balanced rest. C10 integrated budgets,
C11 Held–Suarez statistics and C12 repeat/restart equivalence remain explicitly
pending. Numerical defects require separate reproductions/proposed fixes;
keep the runtime frozen, and do not silently expand benchmark physics.

---

# Previous handoff — implementation freeze for verification

Final bounded cleanup follows merged PRs #9–#11. Read the concise
[README](../README.md), [cleanup results](../docs/FINAL_CLEANUP_RESULTS.md), and
[verification handoff](../docs/VERIFICATION_HANDOFF.md). Conda packaging is deferred.
Recommend freezing the implementation after review, not another stylistic refactor.

IMEX has only the analytic ARK4 operator: obsolete operator/compile fields are
removed, default configs work, and public steppers preserve their effective
solver settings. Bounded moist heat release is intrinsic; release_as_heat is
removed from constructors, with its inactive branches deleted. Tensor-only
numerical paths no longer use v(x); standalone Field boundaries remain intact.
The pinned Held–Suarez archive comparison is now a committed developer tool.
No equations, stage formulas, species ledgers, boundary accounting or fixtures
were changed. Class-instance ownership and gradient connections remain.

Installed-wheel local CPU float64: 536 passed, 1 CUDA skip; portable tools pass
235 retained tensors. Existing 298 pre/post captures and full solver records,
15 additional default-step/bounded-gradient tensors, and 132 Held–Suarez archive
comparisons are exact. All explicit/ARK4/forcing example commands run; the moist
ARK4 example uses smooth positive composition, not the bounded sharp-cloud case.
Frozen exact/hosted outcomes are reported separately in results and the PR.

The next phase needs reproducible benchmark drivers, references and convergence
metrics. JW initialization and HS forcing exist; topographic rest requires a
consistent pressure setup, standalone dry tracer transport is absent, and a
hydrostatic mountain-wave case needs specification/initialization and attention
to the fixed lid and absent absorbing/open boundaries. No new benchmark physics
was added. IMEX backward, long-rollout/GPU and general limiter-switch derivatives
remain unqualified. Report missing capability work separately from verification.

---

# Historical handoff — readable retained atmospheric paths

The three dependent readability PRs start from merged PR #8: #9 removes obsolete
public paths; #10 consolidates retained equations and ownership; the final slice
reconciles current guidance. Read [current paths and timestep traces](../docs/CURRENT_PATHS.md),
[state/model examples](../docs/STATE_OWNERSHIP.md), and
[changes and evidence](../docs/READABILITY_RESULTS.md). Earlier reports are indexed
under [history](../docs/HISTORY.md), not current API instructions.

DryAtmosphere owns conservative dynamics with DrySSPRK3 or DryIMEX instances.
Moist default is bounded SSPRK3 with heat; centered ARK4 is an explicit choice
without species bounds. Required centered/donor/limiter ingredients and wave,
solver, preconditioner and transfer infrastructure remain. External dry method
binding/recipes are gone. ScalarTransportState names shared entropy/theta contents
without dry thermodynamic disguise, conversion or hidden state ownership.

Local installed wheel: 519 passed, 1 CUDA skip; both portable tools (235 retained
quantities) and examples pass. All 298 pre/post tensors and complete ARK4 solver
diagnostics exactly equal merged PR #8. Fixture files and retained tolerances
are unchanged; 100 removed canonical-core paths are explicitly retired. Frozen
exact mode fails separately on this macOS CPU. GitHub outcomes are recorded in
the results and PR descriptions, not inferred from local checks. No IMEX backward,
full-rollout or accelerator qualification follows from this refactor.

Held–Suarez thermal relaxation and Rayleigh drag remain as a standalone
[source](../docs/HELD_SUAREZ.md), with a runnable forced-step example and separate
thermal/drag budget contributions. Explicitly sum it with conservative dynamics
at each SSPRK3 stage; model(state, dt) alone remains unforced. No registry,
process framework, legacy dycore or forced IMEX support is restored. Local updated
wheel: 527 passed, 1 skip; both portable tools and all three examples pass.
Original forcing method ASTs and 132 tensor comparisons are exact, including
forced stages, energy and representative gradients. Hosted results are in PRs.

Next specific readability work, if requested: inspect the remaining dry State
uncached diagnostic/I/O conveniences and clarify their model/external boundaries;
review long analytic pressure-derivative blocks against their adjoint expressions.
Do not combine that work with numerical changes, performance optimization or a
scientific verification campaign. Development remains on GitHub.

---

# Historical handoff — scientific readability, supported-path scope slice

Start from merged PR #8 (849ccb8). Read [the pre-edit map](../docs/SIMPLIFICATION_MAP.md)
and [current paths](../docs/CURRENT_PATHS.md). First branch simplify/supported-paths
removes pressure-prototype solvers, ARK2/3 selectors and CenteredSSPRK3 model
selection; shared ARK4 machinery moves to dry_ark.py/tableau.py without changes
to the retained formulas. Bounded explicit remains default, release_as_heat=True;
centered ARK4 is explicit opt-in and has no bounded species guarantee.

Capture tools/capture_readability.py records retained outputs/stages/ledgers and
gradients outside fixtures. Same-environment pre/post: 298 tensors and complete
solver diagnostics exactly equal. Local installed wheel: 601 passed, 1 CUDA skip;
335 portable fixture tensors and both examples pass. IMEX backward is untested.

Next dependent implementation work: simplify dry equation ownership and remove
superseded dry recipes after extracting required helpers; replace entropy in
mtheta_i with a named shared scalar-transport input. Preserve the baseline capture
and constants/arithmetic/gradients. Final PR reconciles current and historical
guidance. No numerical or performance campaign.

---

# Historical handoff — tensor-only State ownership

PR #7 is merged. The focused branch `cleanup/state-tensor-ownership` makes dry
State/PrognosticFields expose plain tensors, matching the existing MoistState
boundary. Compatible Field inputs unwrap by identity; standalone Field remains
for placement-aware mesh/regridding. Dry IMEX accesses were adjusted mechanically.
No operators, arithmetic order, fixtures, tolerances, bounded ledger, moist
steppers or private entropy-adapter method bodies were changed.

Read [ownership rules, dry/moist examples and validation](../docs/STATE_OWNERSHIP.md).
Models own geometry (including column topography), dynamics and stepping setup;
states borrow geometry and share prognostic tensors. Dry model binding now rejects
unrelated geometry. Explicit state_to_model preserves gradients; mismatched
stepping dtype/device and implicit tracer casts are rejected. Derived dry fields
are available through model.physical_fields(state), with legacy State diagnostic
methods retained as uncached compatibility helpers. Xarray stays external.

Local installed wheel after PR review: 603 passed, 1 CUDA skip; both portable tools and examples
pass. All 335 capture tensors plus 47 additional forward/gradient tensors are
exactly equal pre/post on the same CPU environment. Frozen-fixture exact mode
still fails separately on this macOS host. No new accelerator/full-rollout claim.
GitHub check status is recorded separately in the PR. Development remains on GitHub.

PR #8 review follow-up fixes coarse Schur tracer restriction to use axis -3 for
[..., cell, full, tracer]. Three regression cases exercise actual transfers
with zero, one and two batch axes and check exact weighted values/shape without
input mutation. This covers coarse construction, not a complete batched solver.
Retained consumers were audited; other Field metadata accesses serve explicit
Field operations or schema metadata, not state tensors. Operators are unchanged.

Remaining legacy boundaries: private entropy in mtheta_i, dry State convenience
methods/attrs/restart shell, and the older dry recipe configuration. Keep future
cleanup of those interfaces separate from numerical changes and verification.

---

# Historical handoff — bounded tensor tolerance and one-step gradients

PR #6 was merged after confirming both required CPU checks at its final head.
The separate branch `fix/bounded-tensor-tolerance`,
[PR #7](https://github.com/kjhall01/vfs-a/pull/7), changes only bounded_transfer's
flux-sum validation to the equivalent tensor-valued inequality, preserving the
128*eps scale, finite-error rejection, operators, defaults and species ledger.
There is no detach, disabled-autograd workaround or limiter redesign.

Read [the fixed gradient failure, measurements and remaining limits](../docs/BOUNDED_GRADIENT_LIMITATION.md).
Backward now completes through one bounded SSPRK3 step for wind, ps, entropy
and water. Smooth CPU float64 directional checks use three finite-difference
sizes with inactive limiter margins. A sharp-cloud case also agrees along the
tested support-preserving directions with stable active masks and subcycle counts.
A controlled switch has different left/right derivatives: autograd follows the
selected branch, not a globally smooth limiter. The donor-count scalar-conversion
warning remains. No further backward exception appeared in these probes.

Installed-wheel results and any hosted CI evidence are recorded in the linked
document, separately from frozen-fixture exact failures. Same-environment
pre/post forward captures are exact; fixtures and portable tolerances are unchanged.
No full-rollout, arbitrary-switch, higher-derivative, IMEX-backward or accelerator
backward qualification is claimed. Positive vapor remains required; no floor,
new physics, integration scheme or scientific verification campaign was added.

Next work must distinguish fixed-branch derivatives from switch behavior; do not
turn this narrow fix into a limiter redesign or broad backward-support claim.
Development remains on GitHub.

---

# Historical handoff — moist module ownership and explicit steppers

PR #5 is merged. The ownership branch is `api/moist-module-ownership`, published as
[PR #6](https://github.com/kjhall01/vfs-a/pull/6).
Read the [module API](../docs/MOIST_MODULE_API.md) and
[validation results](../docs/MOIST_MODULE_RESULTS.md).
MoistAtmosphere registers mesh, column and dynamics once; dynamics owns kernel.
BoundedGibbs and the explicit stepper instances are plain non-owning collaborators.
Components cannot be replaced/deleted; borrowed references are checked before
advancement. IMEX binds its solver once and retains per-step wave construction.

Default bounded SSPRK3 still delegates to the exact existing Euler amount ledger
with release_as_heat=True. CenteredSSPRK3 and CenteredIMEX (existing ARK2/3/4
objects) are explicit opt-ins. There is no bounded IMEX option, generic RHS
substitution or duplicated RK formula. Centered IMEX rejects inadmissible stages;
it does not provide bounded species transport. Positive vapor remains required,
with no new floor. Private entropy adaptation and dry APIs remain unchanged.

Move the module before initialization. state_to_model explicitly converts only
prognostic tensors on identical shared geometry, retaining gradients. state_dict
is not a simulation restart: geometry layout and numerical/stepper configuration
must be reconstructed separately. See the API for supported ownership operations.

The [pre-existing gradient limitation](../docs/BOUNDED_GRADIENT_LIMITATION.md)
blocks gradient-enabled bounded timesteps even though diagnostics and conversion
retain gradients. Its allclose assertion remains unchanged; any fix must be a
separate narrow change. Local CPU float64 forward regression, short diagnostic
and centered one-step gradient probes, and one MPS float32 smoke are distinct
results, not full-rollout/accelerator numerical qualification. CUDA is unavailable.

Next: review this ownership PR and, separately, address the allclose scalar
validation failure without changing tolerance semantics or hiding gradients.
No new physics, integration scheme or scientific verification campaign belongs
in this change. Development remains on GitHub.

---

# Historical handoff — centered moist readability

`moist_conservative.py` now expands packed statements, uses consistent formatting
and names energy/transport intermediates explicitly. Follow the
[updated reading guide](../docs/BOUNDED_MOIST_READING_GUIDE.md#inside-the-centered-moist-base)
from tendency through the private adapter, mixture energy gradients, matched
momentum assembly and total-energy diagnostics. Function boundaries specify
units and staggering; familiar mathematical symbols remain defined locally.

The private State.mtheta_i buffer still aliases MoistState.entropy_content.
Inherited theta accessors mean specific entropy only on this adapter; it must
not escape to dry thermodynamic diagnostics or I/O. No tensor copies/detaches,
state redesign, new helper layer or cache were introduced. Arithmetic, evaluation
order, defaults, gradient connections and fixtures are unchanged. The bounded
wrapper's species amount ledger, release_as_heat=True and strictly positive
vapor requirement remain intact; no new floor or verification campaign is added.

The guide records local and hosted validation separately. Next: document the dry
RHS call order and wider State/Field ownership. Any adapter redesign or reuse of
intermediates needs a separate numerical/gradient review. CPU float64 checks do
not qualify GPU, long-run or full backward behavior.

Earlier handoffs below are historical snapshots.

---

# Historical handoff — bounded moist readability

Read [the bounded-step guide](../docs/BOUNDED_MOIST_READING_GUIDE.md) for the
execution order from MoistAtmosphere.step through SSPRK3, Euler transport,
Gibbs traces, species ledger limiting and heat allocation. Source comments now
state intermediate shapes, units, staggering and conservation roles; descriptive
locals distinguish Gibbs correction, limiter correction and RK stage states.

This is comments/docstrings/local renaming only. No operator, expression,
operation order, default, state convention, fixture or exact/portable policy
changed. Euler water still comes directly from the exact species amount ledger.
MoistAtmosphere retains release_as_heat=True and strictly positive input vapor;
no state floor is added. Existing transfer lower bounds remain unchanged.

The guide records validation and the boundary for any later structural work.
No scientific verification campaign, GPU/long-run/full-backward qualification,
new abstraction layer or caching/reorganization is included. Next readability
work remains the dry RHS call order and State/Field ownership guide; any
structural simplification needs its own numerical/gradient review.

Earlier handoffs below are historical snapshots.

---

# Historical handoff — atmospheric package scope

Development remains on GitHub. This scope-reduction PR removes ocean, land, ice,
surface-process, coupling, coupled graph orchestration and ML-specific code from
the distributed package. See [scope/migration/results](../docs/SCOPE_REDUCTION.md)
and [dependency trace](../docs/SCOPE_DEPENDENCIES.json). Removed code remains in
Git history; there is no second legacy package.

Retained: spherical geometry/mesh generation, dry/moist dycores, explicit and
existing experimental implicit integrators, conservation/differentiable
diagnostics, atmospheric initialization and I/O. Shared State/Field/configuration,
model/process support, ETOPO terrain ingestion and IMEX mesh hierarchy remain
because retained paths use them. The obsolete surface dependency extra is gone.

Numerical expressions/defaults, fixtures and integrators are unchanged. The
bounded amount ledger, release_as_heat=True MoistAtmosphere configuration and
strictly positive vapor requirement remain; no floor is added. Portable is the
required installed-wheel gate; exact is separate and retains failure status.
CPU float64 results do not qualify GPU, long runs or full backward rollouts.

Next: explain the dry RHS method index/call order, document State/Field ownership,
and annotate the bounded Euler/Gibbs/heat ledger sequence. Keep those focused
readability changes separate from kernel changes or scientific verification.
Earlier handoffs below are historical snapshots.

---

# Historical handoff — GitHub development and CI

Development is now on https://github.com/kjhall01/vfs-a. See
[CONTRIBUTING.md](../CONTRIBUTING.md) for the small test extra, local setup,
installed-wheel checks and focused PR preparation. The new GitHub Actions
workflow targets Ubuntu CPU/Python 3.12/PyTorch 2.15.0+cpu and runs the suite,
both baseline tools and examples/moist_atmosphere.py outside the checkout.
Portable mode is required in the full pytest suite and both tools. Exact mode
remains the default locally and runs separately in CI with failures reported
explicitly as informational outcomes. The public Linux CPU wheel was successfully downloaded.

See [CI setup results](../docs/CI_SETUP_RESULTS.md) for the initial local checks.
GitHub commit 911aada subsequently had one passing and one failing run with
identical recorded versions/source trees. The
[CI investigation](../docs/CI_911AADA_INVESTIGATION.md) includes all 335 tensor
differences from four Linux CPUs and macOS, controlled dispatch experiments,
and the reviewed quantity-specific portable tolerances. The
[portable implementation](../docs/PORTABLE_REGRESSION.md) now uses an exhaustive
path manifest, separate species scales and independent physical/discrete/ledger
checks. See [validation results](../docs/PORTABLE_REGRESSION_RESULTS.md) for
local and hosted outcomes separately. Exact comparison expressions remain
unchanged. Original CPU models were not logged.
The separate audit/ci-911aada branch collects diagnostic evidence only.
No numerical operators, defaults, state conventions or integrators changed.
MoistAtmosphere retains release_as_heat=True and its bounded species amount
ledger. Vapor remains strictly positive with no floor; exact-zero-water dry
equivalence is not established. GPU, long-run and full backward behavior remain
untested. Package publication and license selection still require direction.

Earlier sections below are historical snapshots.

---

# Historical handoff — extraction pass 4

Read docs/PASS4_RESULTS.md and docs/PHYSICAL_FIELDS_API.md first.
Earlier sections below are history.

Added MoistAtmosphere (composition, no new state type) and a physical-field
initialization/diagnostic/step workflow. Geometry and pressure ownership,
units, staggering, broadcasting and tensor boundaries are explicit.
Numerical kernels are unchanged. Run examples/moist_atmosphere.py.

Important finding: the inherited moist adapter rejects zero vapor. The facade
now requires positive vapor up front, with no floor. Zero condensates work.
This is NOT a demonstrated exact-zero-water reduction to the dry core.

Next: move this snapshot to GitHub and iterate on API/docs there. Before any
public distribution, select a license and review legacy package metadata.
Xarray initialization/regridding and a friendly dry facade remain later work.
Keep the separate scientific verification campaign and kernel changes explicit.

---

# Current handoff — extraction pass 3

Read [pass-3 results](../docs/PASS3_RESULTS.md), then [README](../README.md).
Earlier sections below are historical snapshots, not current next-step instructions.

Removed the six dry mixins. DryHydrostaticDynamics directly inherits nn.Module;
99 ordinary equation functions are explicitly bound in its class body.
Shared dry/conservative/moist numerical expressions and defaults are unchanged.
335 fixture tensors match exactly; 66 tests pass, including small dry/moist IMEX
checks. These checks do not establish production implicit performance or stability.

Next: define the small public model/state boundary and recognizable-variable
tensor initialization, preserving the current kernel and bounded amount ledger.
Then add explicit xarray adapters, including independent surface pressure and
topography DataArrays. Do not silently regrid or detach tensors. Legacy optional
modules and Field/state/configuration infrastructure still ship.
Checkpoint 54, pass 1, and pass 2 remain unchanged.

---

# Current handoff — extraction pass 2

Read [pass-2 results](../docs/PASS2_RESULTS.md), then [README](../README.md).
Pass 1 history follows below. Original checkpoint 54 and pass 1 are unchanged.

Current: default atmospheric imports reduced from 63 to 29 modules; optional
named exports resolve lazily; wildcard exports intentionally narrowed; learned
PGF imports only on selection. BoundedGibbs is now one explicit class, retaining
all update expressions. 142 tensors match exactly; 33 focused tests pass.

Do not describe this as a completed package downsizing: optional code still
ships and six dry mixins/state configuration remain. Next dedicated pass should
extract those dependencies with focused fixtures before public initialization
API work. No numerical defaults or implicit algorithms changed.

---

# Current handoff — extraction pass 1

Read [results](../docs/PASS1_RESULTS.md), then [README](../README.md).

Implemented: bounded moist extraction into four package modules; preserved
existing package numerical files byte-for-byte except adding an explicit export
in atmosphere/__init__.py. A single package tree is installed; no research tree
is shipped or imported. The old broad package contents remain for later cleanup.

Public entry: `from vfs.atmosphere import BoundedGibbs`; instantiate with
`MoistConservativeDynamics()` and `release_as_heat=True` for the campaign baseline.
Use its own `step(state, dt)`. Generic explicit integration of its dt-aware RHS
is not a substitute for the cancellation-safe species Euler update.

Next pass: reduce inherited/root import scope and simplify the dry/kernel
structure incrementally. Preserve old baseline fixtures and add coverage when
changing another path. The friendly MoistAtmosphere/physical-field constructor,
xarray/DataArray adapters (including pressure/terrain), and regridding remain
planned work. TrajectoryWriter/averaged output is deferred. Existing implicit
methods remain experimental and unchanged. User will set up GitHub after a few
passes; no GitHub operation or public release occurred here.

User priorities: teachable, hackable, inspectable, object-oriented, research QOL;
geometry/meshes, dry and moist atmosphere, explicit/implicit methods, tensor
conservation diagnostics and explicit I/O. No column-physics integration now.

---

# Historical handoff — structural ownership stack, unmerged

---

# Review follow-up: selected-density output

Use Ocean.to_latlon(state, remapper, fields="all") for model output. It supplies
the selected dynamics density, while legacy state-only output explicitly labels
its linear-EOS convention. See verification/ocean/OUTPUT_METADATA.md. #23 also
receives the Celsius label correction independently. #25 proposes integration
of the structural stack into main. Numerical kernels and fixtures are unchanged;
Gibbs reference policy and nn.Module migration remain separate tasks.

# Published ocean drafts

[Reference extraction #23](https://github.com/kjhall01/vfs-a/pull/23) targets
`structure/io-ownership`; [energy candidate #24](https://github.com/kjhall01/vfs-a/pull/24)
targets the extraction branch. Both remain drafts and unmerged. At publication,
#20–#22 were marked merged, but main contained only the first ownership stage;
coordinate completion of that stack before retargeting #23. No other agent's
branch or shared runtime was modified. See the [publication and exact tree
provenance](../verification/ocean/PR_DRAFTS.md). This status supersedes the older
structural-stack status in the handoff snapshots below.

# Current ocean handoff — explicit constant-density numerical candidate

The dependent `ocean/constant-density-energy` branch adds an ocean-owned H,
analytic derivatives, paired Bernoulli/continuity and metric-skew rotation.
Read the [derivation, usage and measured limitations](../verification/ocean/CONSTANT_DENSITY.md).
The archived reference remains the default and its 1,416-array comparison is
unchanged. Temperature/salinity are passive in the new candidate; no active
stratification or remapping closure is claimed.

Final installed-wheel CPU float64: 575 passed, one CUDA skip. Maximum measured
normalized semidiscrete H rate is 9.2721e-17; wave slopes 1.9434/1.8986;
SSPRK3 temporal orders 2.9942–3.0002. Shared and atmospheric runtime is unchanged.
Do not merge this numerical change with the extraction or promote its default
without reviewing the distinct evidence. Full structural integration into main
remained pending at publication.

---

# Current ocean handoff — extracted reference, dependent on structural stack

The ocean branch restores the archived layered solver on PR #22's pinned head,
without changes to shared runtime or atmospheric equations. Read the
[ocean boundary and coordination record](../verification/ocean/README.md) and
[results](../verification/ocean/RESULTS.md). New ordinary imports are
`from vfs.ocean import Ocean, OceanState, OceanColumn, OceanDomain`.

All 1,416 donor capture arrays match exactly. Installed-wheel portable suite:
566 passed, one CUDA skip. Atmospheric 298-tensor/solver and 74-output captures
are exact; existing fixtures/tolerances remain unchanged. This is extraction,
not Hamiltonian closure. Numerical corrections belong to a separate dependent
change. Leave the structural stack and ocean changes unmerged for review.
Atmospheric ownership and #18/#19 baseline policy below remain in force.

---

# Current handoff — structural ownership stack, unmerged

Atmospheric campaigns are paused. Structural baseline is merged main
`3f806a947e2ff8a5947953137a96af02861d083f`; #18/#19 remain separate and unmerged.
PR #20 owns dry state/model and pressure column in atmosphere. Dependent #21
separates atmospheric ARK4 systems from shared RK/GMRES. The final dependent
slice separates initialization/output interpretation from shared remapping/I/O.
Read [package map and migration notes](../docs/PACKAGE_OWNERSHIP.md) and
[validation](../docs/STRUCTURAL_RESULTS.md).

Final local wheel: 559 passes, one CUDA skip; portable fixtures and six examples
pass. 298 dynamics tensors and all solver records, plus 74 output/initialization
tensors, match the unchanged baseline exactly. The source audit preserves 457
relocated executable definitions. Historical moist/dry exact fixture failures
remain separately reported. #20/#21 hosted CI passed; final-slice status is
reported in its PR. No new GPU, IMEX-backward, rollout or scientific claims.

Ocean archive was inventoried but no ocean physics/state restored. Only its
existing output adapter is separated. Shared implementation modules have no
component state imports. Old imports are thin compatibility surfaces; new code
uses owner paths. Gibbs means, bounded ledgers, fixtures, tolerances and supported
ARK4 settings are unchanged. Follow the explicit #19 rebase notes in the package
map; scientific source guards require reviewed structural mapping, not bypasses.
Leave the entire structural stack unmerged for review.

---

# Historical handoff — approved SSPRK3 extraction (#16)

PRs #13 and #14 are merged. Numerical baseline remains
`8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648`. The sole runtime change is the
approved stage-sequence extraction at
`ae132773e621671f70bfa6a698eebb1a82cffa94` in `bounded_moist.py`.
Read [extraction evidence](../verification/SSPRK3_EXTRACTION_RESULTS.md).
The atmospheric wrapper still validates and returns the same state and ledgers.
The manifest verifier accepts only the original blob or this exact approved
blob, rejecting all other runtime modifications. Keep #16 unmerged for review.

Next: dependent C5 prescribed horizontal transport harness using actual timed
production Euler maps, not copied RK formulas or invented candidate fluxes.
Replace prescribed wind at all RHS times and final output; evolve carrier,
entropy and species. Record discrete divergence and reference quadrature error.
Keep terrain-rest construction queued afterward. No coupled energy, GPU,
IMEX backward or long-rollout qualification follows from this work.

---

# Historical handoff — verification increments 1a and 1b

Baseline remains `8f5edc749bdd6fe3e0c54b74122d5ec8ab2a8648` (merged PR #12).
PR #13 is increment 1a: geometry/provenance and the full C1–C12 tracker.
Its updated head e657729 passed both required hosted runs and remains unmerged
for review. No numerical package source, fixtures or tolerances changed.

Increment 1b is developed separately, dependent on #13. Read the
[tracker](../verification/README.md), [1b methods](../verification/INCREMENT_1B.md)
and [1b results](../verification/INCREMENT_1B_RESULTS.md).
It adds scalar-energy AD/analytic/FD sweeps (C2), component work matrices including
fixed-lid reservoir and bounded heat (C3), and observation of the three actual
production Euler stages with independent incidence/inventory audits (C4).
The shared manifest approach archives drivers, geometry, prognostic inputs,
directions and actual stage diagnostics. No RK formulas are duplicated.

Small 16/3 and 32/4 cell/layer CPU float64 cases run first. Local installed-wheel
suite: 548 passed, one CUDA skip. Broad campaign qualification remains pending.
No IMEX backward, rollout or GPU support is inferred from these diagnostics.

Next queued work is prescribed horizontal/vertical/combined transport (C5), then
physically consistent terrain-rest construction (C6). Do not treat the current
nonuniform-terrain diagnostic states as balanced rest. C10 integrated budgets,
C11 Held–Suarez statistics and C12 repeat/restart equivalence remain explicitly
pending. Numerical defects require separate reproductions/proposed fixes;
keep the runtime frozen, and do not silently expand benchmark physics.

---

# Previous handoff — implementation freeze for verification

Final bounded cleanup follows merged PRs #9–#11. Read the concise
[README](../README.md), [cleanup results](../docs/FINAL_CLEANUP_RESULTS.md), and
[verification handoff](../docs/VERIFICATION_HANDOFF.md). Conda packaging is deferred.
Recommend freezing the implementation after review, not another stylistic refactor.

IMEX has only the analytic ARK4 operator: obsolete operator/compile fields are
removed, default configs work, and public steppers preserve their effective
solver settings. Bounded moist heat release is intrinsic; release_as_heat is
removed from constructors, with its inactive branches deleted. Tensor-only
numerical paths no longer use v(x); standalone Field boundaries remain intact.
The pinned Held–Suarez archive comparison is now a committed developer tool.
No equations, stage formulas, species ledgers, boundary accounting or fixtures
were changed. Class-instance ownership and gradient connections remain.

Installed-wheel local CPU float64: 536 passed, 1 CUDA skip; portable tools pass
235 retained tensors. Existing 298 pre/post captures and full solver records,
15 additional default-step/bounded-gradient tensors, and 132 Held–Suarez archive
comparisons are exact. All explicit/ARK4/forcing example commands run; the moist
ARK4 example uses smooth positive composition, not the bounded sharp-cloud case.
Frozen exact/hosted outcomes are reported separately in results and the PR.

The next phase needs reproducible benchmark drivers, references and convergence
metrics. JW initialization and HS forcing exist; topographic rest requires a
consistent pressure setup, standalone dry tracer transport is absent, and a
hydrostatic mountain-wave case needs specification/initialization and attention
to the fixed lid and absent absorbing/open boundaries. No new benchmark physics
was added. IMEX backward, long-rollout/GPU and general limiter-switch derivatives
remain unqualified. Report missing capability work separately from verification.

---

# Historical handoff — readable retained atmospheric paths

The three dependent readability PRs start from merged PR #8: #9 removes obsolete
public paths; #10 consolidates retained equations and ownership; the final slice
reconciles current guidance. Read [current paths and timestep traces](../docs/CURRENT_PATHS.md),
[state/model examples](../docs/STATE_OWNERSHIP.md), and
[changes and evidence](../docs/READABILITY_RESULTS.md). Earlier reports are indexed
under [history](../docs/HISTORY.md), not current API instructions.

DryAtmosphere owns conservative dynamics with DrySSPRK3 or DryIMEX instances.
Moist default is bounded SSPRK3 with heat; centered ARK4 is an explicit choice
without species bounds. Required centered/donor/limiter ingredients and wave,
solver, preconditioner and transfer infrastructure remain. External dry method
binding/recipes are gone. ScalarTransportState names shared entropy/theta contents
without dry thermodynamic disguise, conversion or hidden state ownership.

Local installed wheel: 519 passed, 1 CUDA skip; both portable tools (235 retained
quantities) and examples pass. All 298 pre/post tensors and complete ARK4 solver
diagnostics exactly equal merged PR #8. Fixture files and retained tolerances
are unchanged; 100 removed canonical-core paths are explicitly retired. Frozen
exact mode fails separately on this macOS CPU. GitHub outcomes are recorded in
the results and PR descriptions, not inferred from local checks. No IMEX backward,
full-rollout or accelerator qualification follows from this refactor.

Held–Suarez thermal relaxation and Rayleigh drag remain as a standalone
[source](../docs/HELD_SUAREZ.md), with a runnable forced-step example and separate
thermal/drag budget contributions. Explicitly sum it with conservative dynamics
at each SSPRK3 stage; model(state, dt) alone remains unforced. No registry,
process framework, legacy dycore or forced IMEX support is restored. Local updated
wheel: 527 passed, 1 skip; both portable tools and all three examples pass.
Original forcing method ASTs and 132 tensor comparisons are exact, including
forced stages, energy and representative gradients. Hosted results are in PRs.

Next specific readability work, if requested: inspect the remaining dry State
uncached diagnostic/I/O conveniences and clarify their model/external boundaries;
review long analytic pressure-derivative blocks against their adjoint expressions.
Do not combine that work with numerical changes, performance optimization or a
scientific verification campaign. Development remains on GitHub.

---

# Historical handoff — scientific readability, supported-path scope slice

Start from merged PR #8 (849ccb8). Read [the pre-edit map](../docs/SIMPLIFICATION_MAP.md)
and [current paths](../docs/CURRENT_PATHS.md). First branch simplify/supported-paths
removes pressure-prototype solvers, ARK2/3 selectors and CenteredSSPRK3 model
selection; shared ARK4 machinery moves to dry_ark.py/tableau.py without changes
to the retained formulas. Bounded explicit remains default, release_as_heat=True;
centered ARK4 is explicit opt-in and has no bounded species guarantee.

Capture tools/capture_readability.py records retained outputs/stages/ledgers and
gradients outside fixtures. Same-environment pre/post: 298 tensors and complete
solver diagnostics exactly equal. Local installed wheel: 601 passed, 1 CUDA skip;
335 portable fixture tensors and both examples pass. IMEX backward is untested.

Next dependent implementation work: simplify dry equation ownership and remove
superseded dry recipes after extracting required helpers; replace entropy in
mtheta_i with a named shared scalar-transport input. Preserve the baseline capture
and constants/arithmetic/gradients. Final PR reconciles current and historical
guidance. No numerical or performance campaign.

---

# Historical handoff — tensor-only State ownership

PR #7 is merged. The focused branch `cleanup/state-tensor-ownership` makes dry
State/PrognosticFields expose plain tensors, matching the existing MoistState
boundary. Compatible Field inputs unwrap by identity; standalone Field remains
for placement-aware mesh/regridding. Dry IMEX accesses were adjusted mechanically.
No operators, arithmetic order, fixtures, tolerances, bounded ledger, moist
steppers or private entropy-adapter method bodies were changed.

Read [ownership rules, dry/moist examples and validation](../docs/STATE_OWNERSHIP.md).
Models own geometry (including column topography), dynamics and stepping setup;
states borrow geometry and share prognostic tensors. Dry model binding now rejects
unrelated geometry. Explicit state_to_model preserves gradients; mismatched
stepping dtype/device and implicit tracer casts are rejected. Derived dry fields
are available through model.physical_fields(state), with legacy State diagnostic
methods retained as uncached compatibility helpers. Xarray stays external.

Local installed wheel after PR review: 603 passed, 1 CUDA skip; both portable tools and examples
pass. All 335 capture tensors plus 47 additional forward/gradient tensors are
exactly equal pre/post on the same CPU environment. Frozen-fixture exact mode
still fails separately on this macOS host. No new accelerator/full-rollout claim.
GitHub check status is recorded separately in the PR. Development remains on GitHub.

PR #8 review follow-up fixes coarse Schur tracer restriction to use axis -3 for
[..., cell, full, tracer]. Three regression cases exercise actual transfers
with zero, one and two batch axes and check exact weighted values/shape without
input mutation. This covers coarse construction, not a complete batched solver.
Retained consumers were audited; other Field metadata accesses serve explicit
Field operations or schema metadata, not state tensors. Operators are unchanged.

Remaining legacy boundaries: private entropy in mtheta_i, dry State convenience
methods/attrs/restart shell, and the older dry recipe configuration. Keep future
cleanup of those interfaces separate from numerical changes and verification.

---

# Historical handoff — bounded tensor tolerance and one-step gradients

PR #6 was merged after confirming both required CPU checks at its final head.
The separate branch `fix/bounded-tensor-tolerance`,
[PR #7](https://github.com/kjhall01/vfs-a/pull/7), changes only bounded_transfer's
flux-sum validation to the equivalent tensor-valued inequality, preserving the
128*eps scale, finite-error rejection, operators, defaults and species ledger.
There is no detach, disabled-autograd workaround or limiter redesign.

Read [the fixed gradient failure, measurements and remaining limits](../docs/BOUNDED_GRADIENT_LIMITATION.md).
Backward now completes through one bounded SSPRK3 step for wind, ps, entropy
and water. Smooth CPU float64 directional checks use three finite-difference
sizes with inactive limiter margins. A sharp-cloud case also agrees along the
tested support-preserving directions with stable active masks and subcycle counts.
A controlled switch has different left/right derivatives: autograd follows the
selected branch, not a globally smooth limiter. The donor-count scalar-conversion
warning remains. No further backward exception appeared in these probes.

Installed-wheel results and any hosted CI evidence are recorded in the linked
document, separately from frozen-fixture exact failures. Same-environment
pre/post forward captures are exact; fixtures and portable tolerances are unchanged.
No full-rollout, arbitrary-switch, higher-derivative, IMEX-backward or accelerator
backward qualification is claimed. Positive vapor remains required; no floor,
new physics, integration scheme or scientific verification campaign was added.

Next work must distinguish fixed-branch derivatives from switch behavior; do not
turn this narrow fix into a limiter redesign or broad backward-support claim.
Development remains on GitHub.

---

# Historical handoff — moist module ownership and explicit steppers

PR #5 is merged. The ownership branch is `api/moist-module-ownership`, published as
[PR #6](https://github.com/kjhall01/vfs-a/pull/6).
Read the [module API](../docs/MOIST_MODULE_API.md) and
[validation results](../docs/MOIST_MODULE_RESULTS.md).
MoistAtmosphere registers mesh, column and dynamics once; dynamics owns kernel.
BoundedGibbs and the explicit stepper instances are plain non-owning collaborators.
Components cannot be replaced/deleted; borrowed references are checked before
advancement. IMEX binds its solver once and retains per-step wave construction.

Default bounded SSPRK3 still delegates to the exact existing Euler amount ledger
with release_as_heat=True. CenteredSSPRK3 and CenteredIMEX (existing ARK2/3/4
objects) are explicit opt-ins. There is no bounded IMEX option, generic RHS
substitution or duplicated RK formula. Centered IMEX rejects inadmissible stages;
it does not provide bounded species transport. Positive vapor remains required,
with no new floor. Private entropy adaptation and dry APIs remain unchanged.

Move the module before initialization. state_to_model explicitly converts only
prognostic tensors on identical shared geometry, retaining gradients. state_dict
is not a simulation restart: geometry layout and numerical/stepper configuration
must be reconstructed separately. See the API for supported ownership operations.

The [pre-existing gradient limitation](../docs/BOUNDED_GRADIENT_LIMITATION.md)
blocks gradient-enabled bounded timesteps even though diagnostics and conversion
retain gradients. Its allclose assertion remains unchanged; any fix must be a
separate narrow change. Local CPU float64 forward regression, short diagnostic
and centered one-step gradient probes, and one MPS float32 smoke are distinct
results, not full-rollout/accelerator numerical qualification. CUDA is unavailable.

Next: review this ownership PR and, separately, address the allclose scalar
validation failure without changing tolerance semantics or hiding gradients.
No new physics, integration scheme or scientific verification campaign belongs
in this change. Development remains on GitHub.

---

# Historical handoff — centered moist readability

`moist_conservative.py` now expands packed statements, uses consistent formatting
and names energy/transport intermediates explicitly. Follow the
[updated reading guide](../docs/BOUNDED_MOIST_READING_GUIDE.md#inside-the-centered-moist-base)
from tendency through the private adapter, mixture energy gradients, matched
momentum assembly and total-energy diagnostics. Function boundaries specify
units and staggering; familiar mathematical symbols remain defined locally.

The private State.mtheta_i buffer still aliases MoistState.entropy_content.
Inherited theta accessors mean specific entropy only on this adapter; it must
not escape to dry thermodynamic diagnostics or I/O. No tensor copies/detaches,
state redesign, new helper layer or cache were introduced. Arithmetic, evaluation
order, defaults, gradient connections and fixtures are unchanged. The bounded
wrapper's species amount ledger, release_as_heat=True and strictly positive
vapor requirement remain intact; no new floor or verification campaign is added.

The guide records local and hosted validation separately. Next: document the dry
RHS call order and wider State/Field ownership. Any adapter redesign or reuse of
intermediates needs a separate numerical/gradient review. CPU float64 checks do
not qualify GPU, long-run or full backward behavior.

Earlier handoffs below are historical snapshots.

---

# Historical handoff — bounded moist readability

Read [the bounded-step guide](../docs/BOUNDED_MOIST_READING_GUIDE.md) for the
execution order from MoistAtmosphere.step through SSPRK3, Euler transport,
Gibbs traces, species ledger limiting and heat allocation. Source comments now
state intermediate shapes, units, staggering and conservation roles; descriptive
locals distinguish Gibbs correction, limiter correction and RK stage states.

This is comments/docstrings/local renaming only. No operator, expression,
operation order, default, state convention, fixture or exact/portable policy
changed. Euler water still comes directly from the exact species amount ledger.
MoistAtmosphere retains release_as_heat=True and strictly positive input vapor;
no state floor is added. Existing transfer lower bounds remain unchanged.

The guide records validation and the boundary for any later structural work.
No scientific verification campaign, GPU/long-run/full-backward qualification,
new abstraction layer or caching/reorganization is included. Next readability
work remains the dry RHS call order and State/Field ownership guide; any
structural simplification needs its own numerical/gradient review.

Earlier handoffs below are historical snapshots.

---

# Historical handoff — atmospheric package scope

Development remains on GitHub. This scope-reduction PR removes ocean, land, ice,
surface-process, coupling, coupled graph orchestration and ML-specific code from
the distributed package. See [scope/migration/results](../docs/SCOPE_REDUCTION.md)
and [dependency trace](../docs/SCOPE_DEPENDENCIES.json). Removed code remains in
Git history; there is no second legacy package.

Retained: spherical geometry/mesh generation, dry/moist dycores, explicit and
existing experimental implicit integrators, conservation/differentiable
diagnostics, atmospheric initialization and I/O. Shared State/Field/configuration,
model/process support, ETOPO terrain ingestion and IMEX mesh hierarchy remain
because retained paths use them. The obsolete surface dependency extra is gone.

Numerical expressions/defaults, fixtures and integrators are unchanged. The
bounded amount ledger, release_as_heat=True MoistAtmosphere configuration and
strictly positive vapor requirement remain; no floor is added. Portable is the
required installed-wheel gate; exact is separate and retains failure status.
CPU float64 results do not qualify GPU, long runs or full backward rollouts.

Next: explain the dry RHS method index/call order, document State/Field ownership,
and annotate the bounded Euler/Gibbs/heat ledger sequence. Keep those focused
readability changes separate from kernel changes or scientific verification.
Earlier handoffs below are historical snapshots.

---

# Historical handoff — GitHub development and CI

Development is now on https://github.com/kjhall01/vfs-a. See
[CONTRIBUTING.md](../CONTRIBUTING.md) for the small test extra, local setup,
installed-wheel checks and focused PR preparation. The new GitHub Actions
workflow targets Ubuntu CPU/Python 3.12/PyTorch 2.15.0+cpu and runs the suite,
both baseline tools and examples/moist_atmosphere.py outside the checkout.
Portable mode is required in the full pytest suite and both tools. Exact mode
remains the default locally and runs separately in CI with failures reported
explicitly as informational outcomes. The public Linux CPU wheel was successfully downloaded.

See [CI setup results](../docs/CI_SETUP_RESULTS.md) for the initial local checks.
GitHub commit 911aada subsequently had one passing and one failing run with
identical recorded versions/source trees. The
[CI investigation](../docs/CI_911AADA_INVESTIGATION.md) includes all 335 tensor
differences from four Linux CPUs and macOS, controlled dispatch experiments,
and the reviewed quantity-specific portable tolerances. The
[portable implementation](../docs/PORTABLE_REGRESSION.md) now uses an exhaustive
path manifest, separate species scales and independent physical/discrete/ledger
checks. See [validation results](../docs/PORTABLE_REGRESSION_RESULTS.md) for
local and hosted outcomes separately. Exact comparison expressions remain
unchanged. Original CPU models were not logged.
The separate audit/ci-911aada branch collects diagnostic evidence only.
No numerical operators, defaults, state conventions or integrators changed.
MoistAtmosphere retains release_as_heat=True and its bounded species amount
ledger. Vapor remains strictly positive with no floor; exact-zero-water dry
equivalence is not established. GPU, long-run and full backward behavior remain
untested. Package publication and license selection still require direction.

Earlier sections below are historical snapshots.

---

# Historical handoff — extraction pass 4

Read docs/PASS4_RESULTS.md and docs/PHYSICAL_FIELDS_API.md first.
Earlier sections below are history.

Added MoistAtmosphere (composition, no new state type) and a physical-field
initialization/diagnostic/step workflow. Geometry and pressure ownership,
units, staggering, broadcasting and tensor boundaries are explicit.
Numerical kernels are unchanged. Run examples/moist_atmosphere.py.

Important finding: the inherited moist adapter rejects zero vapor. The facade
now requires positive vapor up front, with no floor. Zero condensates work.
This is NOT a demonstrated exact-zero-water reduction to the dry core.

Next: move this snapshot to GitHub and iterate on API/docs there. Before any
public distribution, select a license and review legacy package metadata.
Xarray initialization/regridding and a friendly dry facade remain later work.
Keep the separate scientific verification campaign and kernel changes explicit.

---

# Current handoff — extraction pass 3

Read [pass-3 results](../docs/PASS3_RESULTS.md), then [README](../README.md).
Earlier sections below are historical snapshots, not current next-step instructions.

Removed the six dry mixins. DryHydrostaticDynamics directly inherits nn.Module;
99 ordinary equation functions are explicitly bound in its class body.
Shared dry/conservative/moist numerical expressions and defaults are unchanged.
335 fixture tensors match exactly; 66 tests pass, including small dry/moist IMEX
checks. These checks do not establish production implicit performance or stability.

Next: define the small public model/state boundary and recognizable-variable
tensor initialization, preserving the current kernel and bounded amount ledger.
Then add explicit xarray adapters, including independent surface pressure and
topography DataArrays. Do not silently regrid or detach tensors. Legacy optional
modules and Field/state/configuration infrastructure still ship.
Checkpoint 54, pass 1, and pass 2 remain unchanged.

---

# Current handoff — extraction pass 2

Read [pass-2 results](../docs/PASS2_RESULTS.md), then [README](../README.md).
Pass 1 history follows below. Original checkpoint 54 and pass 1 are unchanged.

Current: default atmospheric imports reduced from 63 to 29 modules; optional
named exports resolve lazily; wildcard exports intentionally narrowed; learned
PGF imports only on selection. BoundedGibbs is now one explicit class, retaining
all update expressions. 142 tensors match exactly; 33 focused tests pass.

Do not describe this as a completed package downsizing: optional code still
ships and six dry mixins/state configuration remain. Next dedicated pass should
extract those dependencies with focused fixtures before public initialization
API work. No numerical defaults or implicit algorithms changed.

---

# Current handoff — extraction pass 1

Read [results](../docs/PASS1_RESULTS.md), then [README](../README.md).

Implemented: bounded moist extraction into four package modules; preserved
existing package numerical files byte-for-byte except adding an explicit export
in atmosphere/__init__.py. A single package tree is installed; no research tree
is shipped or imported. The old broad package contents remain for later cleanup.

Public entry: `from vfs.atmosphere import BoundedGibbs`; instantiate with
`MoistConservativeDynamics()` and `release_as_heat=True` for the campaign baseline.
Use its own `step(state, dt)`. Generic explicit integration of its dt-aware RHS
is not a substitute for the cancellation-safe species Euler update.

Next pass: reduce inherited/root import scope and simplify the dry/kernel
structure incrementally. Preserve old baseline fixtures and add coverage when
changing another path. The friendly MoistAtmosphere/physical-field constructor,
xarray/DataArray adapters (including pressure/terrain), and regridding remain
planned work. TrajectoryWriter/averaged output is deferred. Existing implicit
methods remain experimental and unchanged. User will set up GitHub after a few
passes; no GitHub operation or public release occurred here.

User priorities: teachable, hackable, inspectable, object-oriented, research QOL;
geometry/meshes, dry and moist atmosphere, explicit/implicit methods, tensor
conservation diagnostics and explicit I/O. No column-physics integration now.
