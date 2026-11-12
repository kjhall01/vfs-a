# Equiangular diagnostics on WeatherBench2 ERA5

An optional validation campaign for global wind diagnostics on real data. Nothing
in `vfs` imports WeatherBench2, opens a cloud dataset, or requires this directory.
Normal tests use small generated fields and a temporary local NetCDF file. Cloud
and plotting dependencies are needed only when running the corresponding options.

## Run

Install VFS with the coordinate-aware EquiangularGrid, then install the optional
reader/plotting tools in your validation environment:

```bash
python -m pip install 'zarr>=3.1' 'fsspec[http]' matplotlib
python -I validation/equiangular_wb2/run.py --weatherbench2 --output /path/to/results
```

The default selects four 2020 snapshots (January/April/July/October 15 at 12 UTC),
at 250 and 500 hPa, from the public 0.25-degree WB2 ERA5 store. Downloads are
explicit. The reader uses anonymous HTTPS, respects proxy settings, and verifies
TLS certificates. For `gs://` input, install `gcsfs` separately. Select just one
snapshot for a shorter first run:

```bash
python -I validation/equiangular_wb2/run.py --weatherbench2 \
  --times 2020-01-15T12:00:00 --levels 250 500 --output /path/to/results
```

A local file works without cloud dependencies:

```bash
python -I validation/equiangular_wb2/run.py --input snapshot.nc \
  --u u --v v --geopotential z --output /path/to/results
```

Coordinates default to `latitude`, `longitude`; override with `--lat`/`--lon`.
A pressure axis, if present, must be named `level` and be in hPa. `--levels`
selects it exactly. A time axis, if present, must be named `time`: local input
uses its first timestamp unless `--times` is supplied. Two-dimensional local
snapshots without time/level axes also work. Select extra axes upstream.

Wind components must be geographic eastward/northward, in m/s. Geopotential,
when present, must be in m²/s². The optional archived reference fields default
to `divergence` and `vorticity`, both in s⁻¹, and must both be present for that
comparison. The script assumes these units; it does not infer or convert them.
Latitude/longitude must describe a complete global regular grid. No masking,
land filling, missing-wind filling, or regional boundary conditions are applied.

The spherical radius defaults to 6,367,500 m to match WB2's reference convention;
`--radius` sets it explicitly for another dataset. This does not change VFS's
package default. Archive comparisons need matching radii to be interpretable.

Only selected times/variables are loaded, one timestamp at a time. WB2's pressure
variables are chunked with all 13 levels together: selecting two levels does not
mean downloading only two levels. The script writes results, not copies of the
input dataset. `--no-plots` removes the matplotlib dependency; `--no-coarse`
skips the common-scale comparisons. A local NetCDF run with both flags needs
only the ordinary VFS dependencies. Re-running into a nonempty output directory
can leave obsolete files; use a fresh directory for each campaign.

## What is tested

- Global and regional divergence integrals against boundary-only flux sums.
- Native dual-cell curl integrals against boundary-only circulation sums.
- Regions covering the date line, the stored longitude seam, and both polar caps.
- Invariance under latitude reversal, longitude roll, and longitude convention.
- Geopotential curl(grad) and gradient/Laplacian integration by parts.
- Independent spherical finite-difference divergence and vorticity comparisons.
- Archived WB2 comparisons and an independent replay of its published formulas.
- Common-scale comparisons on centered 32×64 and pole-inclusive 121×240 layouts,
  when the source is finer. These use exact spherical scalar area overlaps.
- The difference between differentiating coarse winds and averaging fine-grid
  diagnostics. Winds are averaged in Cartesian coordinates and projected onto
  target tangent directions. This avoids mixing local bases, but is not an
  energy/momentum-conserving vector remapping claim or a convergence oracle.

Discrete budgets and invariance have roundoff-scaled assertions. Agreement with
another differentiation method is measured, without an arbitrary percentage
pass threshold. The analytic tests in `tools/qualify_equiangular.py` remain the
accuracy reference. Global area weighting must not conceal polar error.

## Reference conventions matter

The inspected [WB2 implementation](https://github.com/google-research/weatherbench2/blob/1f1ff39b954fdb79c96e11ad9750212eb7dd14eb/weatherbench2/derived_variables.py)
computes divergence as `d_dx(u) + d_dy(v)` and curl as `d_dx(v) - d_dy(u)`,
where the zonal derivative divides by R cos(latitude), but the meridional
derivative acts on the component itself. Longitude differentiation is not
periodic at its first/last sample. It zeros the zonal derivative at the poles.
These formulas omit the curvature contributions −v tan(latitude)/R to
divergence and +u tan(latitude)/R to vorticity. Thus the archived fields are
comparison products, not exact spherical truth. Their historical production
commit is not assumed; the replay comparison measures consistency empirically.

Our independent NumPy comparison instead evaluates the complete spherical forms:

\[
D=\frac{\partial_\lambda u+\partial_\phi(v\cos\phi)}{R\cos\phi},
\qquad
\zeta=\frac{\partial_\lambda v-\partial_\phi(u\cos\phi)}{R\cos\phi}.
\]

It uses periodic centered longitude differences and second-order latitude
finite differences. Its pole point values are undefined and marked NaN. The VFS
polar results are cap averages, so we test their budgets separately. Finite
reference masks and sample counts are explicit in the metrics. No NaN reference
is replaced with zero for comparison. This FD implementation is independent of
VFS, but is not itself an exact solution or a conservative operator pair.

## Outputs

`metrics.json` records source, selected cases, coordinate hashes, installed
operator/script hashes and environment versions. Each `case-NN.json` contains
full-resolution statistics, input hashes, budgets, invariance, and coarse-scale
comparisons. Statistics are reported globally, below 85°, in the ≥75° polar
bands, at sampled poles, and at the stored seam columns.

With plotting enabled, each case has a six-panel `.webp` comparison map, a
`-latitude.webp` error profile, and a `.csv` of per-latitude RMS differences.
Color scales are clipped at the 99th percentile and maps are subsampled for
rendering only; metrics use every source grid point. Reference pole values are
excluded from profiles because no pointwise FD answer is supplied there.

Recorded results are linked from [RESULTS.md](RESULTS.md). No input dataset is
committed. Files above the project's 128 KiB upload limit are omitted and listed
in the result manifest rather than resized or silently included.
