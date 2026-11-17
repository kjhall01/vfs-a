# Coupling momentum debugging — 17 November 2026

The previous 20-hour failure originates in the external VFS/PyAM21 wind map.
The CP energy-transfer solver detects the excessive native kinetic energy on
the next call; increasing its iteration count cannot fix the momentum defect.
The package's dynamics and upstream PyAM21 sources are unchanged by this fix.

## Reproduction and diagnosis

All integrations here use actual pinned PyAM21 physics with **radiation OFF**,
CPU float64, 256 generated cells, 24 atmospheric and 12 ocean layers. Ocean and
physics intervals are 1200 s, atmospheric substeps 240 s, mixing/eddy substeps 10.
The full-ocean cases enable GM–Redi, eddy decay, backscatter/biharmonic and ALE.
The radiation interval remains configured as 3600 s but no radiation calls occur.

The unchanged atmospheric-dynamics-only control completes 24 hours. In the old
coupled map, the interval starting at 70,800 s has native maximum wind 46.015 m/s
after dynamics. PyAM21 requests at most 1.327 m/s zonal and 3.581 m/s meridional
changes. Mapping those changes produces a native maximum of **1107.762 m/s**.
The following call rejects in CP energy transfer. See
[original stage trace](original-failure-stages.json) and the preserved
[earlier ocean-process controls](../ocean-processes/README.md).

The unweighted diagnostic reconstruction observes a short edge as strongly as
a long one. Its transpose therefore puts an order-one cell impulse onto an
edge whose native mass is proportional to its very small dual length. This is
a finite impulse instability despite the transpose's exact work identity.

The campaign mesh has dual-edge lengths from 1064.39 to 1,351,250.78 m.
For the cell reconstruction R, cell mass matrix W and native edge mass M, the
largest eigenvalue of `M^(-1/2) R^T W R M^(-1/2)` is **144.270574** under the old
map and **1.017534** under the correction at uniform layer mass. A complete
cell-wind removal becomes `u1=(I-M^-1 R^T W R)u`; eigenvalues above two amplify
native kinetic energy. The regression also checks horizontally varying mass.

## Correction and retained contracts

`workflows/momentum.py` fits the two tangent components using weights
`w_ce = (dc_e*dv_e) * I_ec / A_c`, where `I_ec` is the native cell-to-edge
interpolation's ownership and `A_c` is cell area. The local normal matrix is
`G_c = sum_e w_ce n_ce n_ce^T`; the fit is `G_c^-1 sum_e w_ce n_ce u_e`.
Its coefficients vanish with a short edge's kinetic-energy weight, so the
subsequent division by edge mass no longer introduces a `1/dv` singularity.
The two-dimensional fit reproduces a locally constant tangent vector.

Atmosphere, ocean and ice surface exchange use this same readout and its exact
transpose. Native generalized impulses remain reciprocal; the finite energy
ledger and CP solve are unchanged. No new velocity cap, timestep reduction,
regularizer, thermal correction or prescribed gust floor is introduced. Legacy
atmospheric mode retains its historical map. Packaged diagnostic reconstruction
and ocean–ice internal mechanics remain unchanged. The new map is not a proof
of unconditional stability for arbitrary meshes, physics or timesteps.

Standalone restart schema is now 4; the campaign also hashes source. Start
fresh after updating, rather than relabelling a checkpoint from the old map.

## Verification

- External installed-wheel focused suite: **173 passed, 1 skipped** (real
  radiation tables unavailable). Includes local tangent reproduction, an
  independent differentiated work identity, short-edge finite-drag regression,
  reciprocal exchange, energy/water budgets, restart replay and ocean operators.
- Frozen portable comparisons: **142 corrected-Gibbs and 93 dry-helper tensors**.
- Default full-stack preflight: **bitwise restart replay**.
- A fresh one-day full-ocean campaign completes at the requested timesteps:
  endpoint maximum native air wind **16.613 m/s**, numerical combined-energy
  residual **-3.24740 J/m2**, water **-1.16e-9 kg/m2**, relative salt **-4.28e-16**.
  This residual includes finite-step dynamics; the upstream imbalance remains
  separately reported, rather than balanced away.
- The final-source fresh **48-hour** debugging run also completes: 1440 GM–Redi
  substeps and 144 ALE calls; endpoint native air wind **23.179 m/s** (maximum
  across recorded stages **31.818 m/s**), numerical energy **-6.39188 J/m2**,
  water **-2.56e-9 kg/m2**, relative salt **-6.42e-16**. Its day-one result
  reproduces the normal campaign driver. See [summary and source identity](summary.json),
  [all stage extrema](full-48h-stages.csv) and [budget history](full-48h-diagnostics.csv).

The focused suite and baseline/preflight outputs are archived alongside this
file. The earlier 202-pass count covered a different selection; neither count
is a final full-repository regression result.

## Remaining limits and controls

An independent physics-only control with exactly resting air develops zero
gustiness and rejects its second interval, after 1200 accepted seconds. The
pinned upstream surface routine divides by zero wind/gust magnitude and returns
nonfinite `q_star`, stress derivatives and `q_surf`. The adapter now identifies
this as a **PyAM21 surface-layer** failure before it contaminates column physics
or the shared exchange. This upstream degeneracy is diagnosed, not repaired;
the coupled run advances dynamics before physics and passes this point.

Actual SeaEsfRad tables are unavailable here. No seasonal-radiation integration,
user 16,384-cell mesh, full-year stability, GPU or full-coupling backward result
is claimed. A successful short radiation-off run does not establish climate
fidelity or safety of the same timestep on a different mesh.

For independent debugging from the repository root:

```bash
python -I validation/coupled_debug/run.py --source /path/to/PyAM21 \
  --output debug-coupled --hours 48
python -I validation/coupled_debug/run.py --source /path/to/PyAM21 \
  --output debug-dynamics --mode dynamics --hours 24
python -I validation/coupled_debug/run.py --source /path/to/PyAM21 \
  --output debug-physics --mode physics --hours 24
```

These are explicitly radiation-off controls; `--mesh /path/to/mesh.npy` and
`--config FILE.yaml` are supported. `stages.jsonl` records before-dynamics,
before-physics, actual AM2 increments and accepted native endpoint extrema.
`manifest.json` pins code/configuration; `diagnostics.csv` and `status.json`
record budgets and outcome. An exception returns nonzero and saves only the
last accepted checkpoint. Use `workflows/coupled/campaign.py` for the actual
seasonal experiment, snapshots and supported continuation workflow.
