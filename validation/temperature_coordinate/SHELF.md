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
Four composed 120 s steps exercise continuity, donor momentum, GM/Redi and
ALE on the evolving partial columns. All material inventories, including
optional passives and the turbulent and eddy reservoirs, follow the current
wet geometry. The runner also audits each isolated budget. Positive TKE is
required when a local unforced momentum update consumes energy.

The coupled Earth campaign shares the wet-opening transport principles and
keeps its existing terrain-following storage. See the [coupled shelf
configuration](../../workflows/coupled/EARTH_SHELF.md).
