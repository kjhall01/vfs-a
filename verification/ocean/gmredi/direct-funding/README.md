# Direct local GM funding benchmark

The experimental solver preserves the donor-parcel GM proposal wherever its local energy settlement succeeds. A rejection identifies shared streamfunction interfaces incident to depleted cell-layers. Interfaces affecting a common cell-layer form one patch. On each patch the solver seeks the largest common amplitude affordable at every affected cell-layer. Holding the exterior transport fixed preserves the scope of each change.

Each affected carrier varies linearly with the patch amplitude. The solver splits the interval where a face changes upwind donor. On each piece, at fixed layer thickness the Roquet Hamiltonian gives a quadratic remaining energy as a function of amplitude. The solver intersects the feasible intervals and retains a numerical margin of up to 1e−7 J/kg, capped at 2% of the failing layer’s specific eddy energy at substep start. This margin slightly reduces the accepted transport. No energy is added. The native GM and Redi operators verify the assembled candidate, including tracer positivity and local work. When the solver cannot find or verify a candidate it replays the existing halving strategy. Geometry from the substep input is reused on retries.

The solver is an opt-in forward prototype for the fixed-thickness donor-parcel Roquet operator. It uses host-side float64 patch calculations and treats the selected scales as constants in autograd. Behavior with widespread depletion, accelerator costs and gradients of the allocation require separate evaluation.

## Reproduce the timings

Use the environment and mesh from the original donor-parcel experiment. From this checkout, the following command times the entire coupled update using the original Sea ESF radiation, compile policy, and runtime fingerprint:

```sh
python -u -I tools/benchmark_coupled_gm_funding.py \
  --intervals 1 --pairs 4 --report "$HOME/Desktop/gm-funding-coupled.json"
```

Defaults point to the strict donor restart at `~/Desktop/vfs-a/gm-parcel-day30/last_good_restart.pt`, its adjacent manifest, `~/Desktop/PyAM21`, and `~/Desktop/sea_esf`. The mesh path comes from the manifest. Use `--help` for overrides. The benchmark advances private in-memory states. It writes a JSON timing report and has no campaign restart output. Run timings on an otherwise idle machine.

For the ocean/ice replay and the isolated operator cases, run:

```sh
run="$HOME/Desktop/vfs-a/gm-parcel-day30"
python -u -I tools/benchmark_gm_funding.py \
  --manifest "$run/manifest.json" --resume "$run/last_good_restart.pt" \
  --mesh "$HOME/Desktop/mesh_4096_exact_3000.npy" --source "$HOME/Desktop/PyAM21" \
  --output "$HOME/Desktop/gm-funding-ocean-benchmark"
```

The microbenchmark discovers the first native GM failure from the saved strict checkpoint, then reuses the full state just before it. Each paired trial starts both methods from identical input and alternates their run order. Both methods run the same per-substep transport and energy audit. Window and sustained cases evolve each trajectory independently after their first intervention. The sustained case isolates GM and Redi. The two ocean/ice legs include dynamics and eddy decay up to the next atmosphere–surface exchange. The coupled benchmark additionally executes the atmosphere, surface exchange, phase change and ALE remap. The reported kernel timings exclude initialization, compilation warmup, checkpoint and snapshot I/O.

## CPU timing results

The [raw paired results](results.json) cover an AMD EPYC 9V74 Linux host, four PyTorch threads, PyTorch 2.14.1+cpu, eager CPU float64, 4,096 horizontal cells, 60 ocean layers and 32 atmospheric levels. The checkpoint time was 936000 s. All measurements below use four alternating pairs after an excluded warmup pair. Speedup is the median of the individual halving/direct time ratios. The two wall times are independently calculated medians in seconds.

| Timed operation | Halving (s) | Direct (s) | Paired speedup | Paired range |
|---|---:|---:|---:|---:|
| Four calls from a funded input | 1.66 | 1.60 | 1.062× | 0.929–1.147× |
| Four calls from the first failed input | 2.33 | 2.21 | 1.044× | 0.989–1.110× |
| 20-minute ocean and sea-ice window | 22.69 | 21.78 | 1.042× | 1.013–1.065× |
| 30 consecutive binding GM/Redi steps | 23.11 | 17.58 | 1.358× | 1.215–1.434× |
| 20-minute coupled advance, radiation off | 29.49 | 29.53 | 0.999× | 0.980–1.019× |

Halving made 27 GM settlement attempts and direct funding made 24 over the two ocean/ice legs. Both intervened on four of the 20 substeps. On the first failure, direct funding retained 76.20% of the affected edge-length-weighted streamfunction magnitude and halving retained 50%. Over the entire window the direct solver reduced this magnitude by 0.003183% domain-wide, compared with 0.003300% for halving. In the sustained isolated case, attempts fell from 102 to 60 across 30 accepted substeps. The new trajectories remained locally energy-limited and had nonnegative specific eddy energy. No fallback occurred on these checkpoint cases. The largest absolute GM/Redi cell-layer residual measured on either method was approximately 4.01e−12 J/m².

The coupled timing used the real checkpoint’s physical state and column memory, with radiation disabled and eager kernels. It includes atmosphere dynamics, moist column physics, surface exchange, ocean/ice/snow, GM/Redi, eddy decay and ALE. Its rate was 0.1115 simulated years per wall day for halving and 0.1113 for direct funding. The Sea ESF coefficient tables and the original Mac were unavailable for timing. Use the real-table command above for a production-environment comparison.

Six focused 32-cell cases cover a bitwise funded control, overlapping and disjoint patches, a donor reversal, forced fallback and a Redi funding rejection. The last case shows why the native check of the assembled GM and Redi update matters. The independent two-interval 32-cell coupled check matched the native funded control bit for bit across 36 tensors and 68 scalar leaves.

## Isolating the effect of the funding solve

With retry geometry cached in both arms, 12 consecutive depleted GM/Redi steps took 9.36 s with halving and 6.94 s with direct funding. The median paired speedup was 1.348× across three alternating pairs (1.279–1.417×) after one warmup pair. GM settlement attempts fell from 39 to 24. Cached halving reproduced the original halving endpoint in all seven ocean state tensors and all 20 substep transport and budget records byte for byte in the ocean/ice window.

To reproduce this comparison, use `tools/benchmark_gm_cache_ablation.py` in the ocean replay command above and give it a fresh output directory. The archived timings used the same driver and operator code via a workspace-specific import path. The published helper resolves its imports from the checkout and also checks the per-substep records.

## Limits under severe depletion

On a separate 32-cell, eight-layer cold-front test initialized everywhere with 1e−9 J/kg of specific eddy energy, both methods completed ten 60-second GM/Redi steps and rejected the eleventh. Direct funding used the halving fallback on all ten accepted steps. Both methods reached the same minimum reservoir of 9.51e−15 J/kg before rejection. This test bounds the interpretation of the timing gain on the day-10 checkpoint.

In an installed-wheel environment outside the checkout, all six focused tests and 49 existing GM/Redi, feedback and reservoir tests passed. The portable dry-helper and corrected-Gibbs checks passed 93 and 142 tensor comparisons respectively.
