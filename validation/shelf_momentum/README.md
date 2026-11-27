# Day-50 shelf and ice momentum experiments

These CPU, float64 replays compare two candidate numerical changes on the 4,096-cell Earth-shelf checkpoints. The atmospheric integrator and the campaign defaults stay as supplied. The experimental operators are installed in a fresh model for each comparison. The shelf experiment uses the campaign's existing native velocity slots.

## Reproduce

Use the VFS environment with PyTorch, SciPy and netCDF4, and make PyAM21 importable for the water scheduler and ALE scripts. Run from the repository root. Set the three shell variables to your local files.

```bash
python -m validation.shelf_momentum.checks "$restart50" "$restart43"
python -m validation.shelf_momentum.smooth_drag "$restart50"
python -m validation.shelf_momentum.ale_check "$restart43" "$manifest"
python -m validation.shelf_momentum.ocean_replay "$restart50" --hours 24
python -m validation.shelf_momentum.replay "$restart50" --hours 6
python -m validation.shelf_momentum.water_replay "$restart50" "$manifest" --hours 6
python -m validation.shelf_momentum.water_replay "$restart50" "$manifest" --hours 1 --variants edge both --trace-first-interval
```

Each command writes JSON records to standard output. The mask for pack-ice statistics requires at least 15% ice cover in both neighboring cells. Outputs track the same initially fastest edge through time separately from spatial maxima. Time starts at the checkpoint.

| Experiment | What advances |
|---|---|
| Drag and smooth-flow checks | Reciprocal drag alone, including the actual TKE increment |
| Shelf and ALE checks | Instantaneous shelf RHS, homogeneous rest, stratified rest, imposed sea-level slope, or one scheduled rezone |
| Ocean trajectory | Unforced shelf transport, pressure and rotation with the four-stage third-order integrator and 1,200 s step |
| Mechanical ocean–ice trajectory | Ocean dynamics, ice transport/rotation/ridging, and two 600 s drag half-steps |
| Campaign water trajectory | The water scheduler with thermal ice, snow, mixing, backscatter, GM/Redi, eddy dissipation and calendar-scheduled ALE. Atmosphere, land and their new surface fluxes are omitted |

The edge drag uses the prognostic relative normal velocity and estimates tangential relative speed to set the magnitude of quadratic stress. Equal and opposite edge impulses close momentum. Their midpoint work is credited to surface TKE. The normal-only law is included as a directional control.

The shelf candidate computes kinetic energy from physical opening, differentiates it at fixed edge velocity, and uses opening-weighted transport in skew rotation. It retains the split tracer fluxes. Matching ALE uses open thickness for remapped momentum and partitions actual edge kinetic energy into columns. The old column identity involving the thickness derivative requires a homogeneous degree-one mass metric, an assumption that fails with a fixed bed and clipped openings. The isolated ALE comparison includes the intentionally mismatched case so the effect can be reproduced.

The replays compare energy changes against each formulation's own initial Hamiltonian. The one-day ocean-only ledger reports mechanical and dynamic enthalpy changes. CT is conserved and the linear reference-enthalpy total is omitted from that ledger. Six-hour water runs include thermal reference energy, TKE, EKE and ice/snow energy. None of the replays project velocities when switching metrics in a checkpoint. A general campaign restart across distinct Hamiltonians would need an explicit policy for that transition.
