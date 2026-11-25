# Partial-cell shelf experiment

From the repository root in a CPU PyTorch environment, run

```bash
PYTHONPATH=. python validation/temperature_coordinate/shelf_campaign.py \
  --mesh input-data/mesh_4096_exact_3000.npy \
  --geography input-data/earth_4096_geography.npz
```

The geography file must use the same cell order as the mesh and contain
`land` and `depth_m`. Both paths are supplied by the caller.

The experiment uses float64, four CPU threads and 40 reference ocean levels.
It exercises split wet faces, four 120 s unforced continuity and donor-momentum
updates, and a separate GM/Redi and ALE experiment. The stdout report includes
per-edge momentum and energy residuals, minimum TKE, and the ALE column energy
residual. Positive TKE is required when a local momentum update consumes energy.
The coupled campaign still uses its existing vertical layout. Rotation, pressure
integration, coupled restart and A100 throughput remain unqualified.
