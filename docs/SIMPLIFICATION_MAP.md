# Scientific-readability simplification map (before structural edits)

Baseline: merged PR #8, `849ccb829c7555ab8ecd1c11f0ae45993538f8ed`.
This map precedes structural changes. The delivery is three dependent PRs:
(1) supported-path scope/public selection, (2) equation/transport consolidation,
(3) current guides and removal of stale guidance. No performance/new-method work.

| Path | Current execution | Retain and clarify |
|---|---|---|
| Conservative dry explicit | `core.integrators.ssprk3(d, state, dt)` → `DryHydrostaticConservativeDynamics.tendency` → `_assemble` → hybrid mass flux, matched diagonal kinetic-energy/pressure/rotation/transport terms → three existing RK updates | Conservative unbatched dry State, fixed positive lid, direct_u_gl2_adjoint column; class-instance model/stepper entry, unchanged formula/order |
| Bounded moist explicit | `MoistAtmosphere.forward` → `BoundedSSPRK3.advance` → `BoundedGibbs.step_with_info` → three `euler_with_info` calls → `evaluate` → graph/carrier flux, centered candidates + bounded donor correction, Gibbs work and heat → authoritative species amounts → convex combinations | Default release_as_heat=True, positive vapor (no floor), nonnegative condensates, direct Euler amount ledger and full dt at each Euler stage |
| Conservative dry ARK4 | `DryPEIMEXARK(..., method='ark4')` → inherited `_step` in ark2.py → `DryPELinearOperator` → `ConservativePressureWave` → full/Schur GMRES, optional geometric multilevel preconditioner → six ARK stages and common weighted sum | ARK4 tableau and shared solver/wave/transfer machinery; distinguish observed solver configurations from available flags |
| Moist ARK4 | `MoistAtmosphere` + `CenteredIMEX(ARK4_TABLEAU)` → `MoistPEIMEXARK.step_with_info` → `MoistPressureWave` → full/Schur GMRES → six stage validations and final validation | Existing eager analytic centered mixture split only; no bounded species ledger, SSP guarantee, or implicit heat/limiter scheme |

Both ARK4 paths freeze an analytic wave operator L at the outer-step anchor y0.
The split is F(y)-L(y-y0) explicitly and L(y-y0) implicitly. Stage residual
checks/failures and physical validation remain; an accepted residual is not a
positivity proof. No IMEX backward qualification follows from forward agreement.

## Keep/remove/refactor

| Area | Action and dependency boundary |
|---|---|
| Geometry/column/mesh generation/regridding/I/O | Keep spherical C-grid, hybrid column, prescribed terrain, transfers and explicit external data boundaries. Multilevel transfer is necessary ARK infrastructure. |
| Conservative dry + moist kernels | Keep matched formulas and diagnostic gradients. Replace external method binding with cohesive equation ownership only after captures. |
| Bounded transport/Gibbs/mixture thermodynamics | Keep centered candidate fluxes, donors, limiter, Gibbs correction, heat allocation and ledger. They are ingredients, not alternative model selections. |
| Dry canonical/strategy recipes | Remove superseded public recipes. Initially trace inherited dependencies; move required continuity, energy and transport routines into retained implementation before deleting their old owners. No legacy package. |
| Entropy adapter | Replace private State(mtheta_i=entropy) with a named transport input whose scalar content/specific scalar are neutral and explicitly labeled at dry/moist boundaries. Preserve formulas, sharing and gradients. |
| ARK2/ARK3 + pressure prototypes | Remove separate options and `legacy_pressure.py`. Move shared ark2.py machinery to an accurately named owner and retain ARK4 coefficients unchanged. No generic tableau registry. |
| Public composition | Fixed model ownership, tensor states, explicit conversion, class-instance steppers. Remove historical recipe/registry/physics composition entry points outside retained scope; no compatibility dispatch. |
| Diagnostics/initialization | Preserve physical budgets, tensor derivatives and relevant atmospheric initialization. Remove diagnostics tied solely to removed algorithms; record intentionally retired fixture paths rather than regenerate fixtures. |
| Documentation/tests | Preserve independent conservation/positivity/derivative checks and frozen fixture values/tolerances. Replace tests of intentionally removed APIs with explicit absence tests; keep current entry guides distinct from historical reports. |

## Evidence and capture plan

Existing conservative dry tests cover energy/continuity, independent derivatives,
analytic wave JVPs, full/Schur/multilevel agreement and time refinement **using
ARK2**. They do not alone qualify dry ARK4. Before edits, run ARK4 on those same
representative states/configurations and record actual stages, residuals,
iterations and outputs. No assumption that coupled-JVP, compiled, fixed-iteration
or arbitrary option combinations are qualified.

Existing moist ARK4 evidence: no-runtime-autograd 1800 s conservation case,
full-vs-Schur 600 s agreement, DOP853 time refinement at 600/300/150 s, stage
rejection and input-preservation checks. Only eager analytic waves with `none`
or `hydrostatic_schur` are accepted; multilevel and compiled moist paths reject.

Before restructuring, capture states, tendencies, energy terms and representative
diagnostic gradients for retained dry/moist cases; bounded Euler/SSPRK3 ledgers;
and each ARK4 nonlinear stage and complete solver diagnostics. Keep captures
outside baseline fixtures and compare pre/post on the same CPU environment.
Then run installed-wheel portable tests/tools, independent invariant checks and
examples. Exact frozen-fixture results remain separate from portable/pre-post.

## First PR implementation and validation

The first scope slice removes pressure prototypes (2205 lines), ARK2/3 choices,
and the centered explicit model selector. Shared solver code moves from ark2.py
to dry_ark.py, ARK4 coefficients to tableau.py; no compatibility modules remain.
The broader dry dependency consolidation follows in the dependent implementation
PR, so this slice does not yet claim to remove all dry recipe internals.

Pre/post capture: 298 tensors exactly equal, including stage states, tendencies,
energy terms, representative gradients, bounded ledgers and all five ARK4 solver
configurations above; complete convergence diagnostics also equal. Installed
wheel outside checkout: 601 passed, 1 CUDA skip; both portable tools (335 fixture
tensors) and both examples pass on macOS CPU float64/PyTorch 2.15.0 Accelerate.
Two fewer tests than PR #8 reflects removed ARK2/3/centered-explicit parameter
cases and new absence checks; retained numerical assertions are unchanged, except
the dry refinement check is now explicitly fourth-order against DOP853.
