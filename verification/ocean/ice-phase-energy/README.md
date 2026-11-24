# Reduced thermal ice phase Hamiltonian

The coupled float64 forward path under `torch.no_grad()` computes the energy difference from the fully melted column at its freezing point. Its right-hand side is the original target minus the energy of that same all-liquid state. The initial ocean may have any supported temperature. Float32 and gradient-enabled execution use the original Hamiltonian solve.

## Energy per unit horizontal area

Let $m$ be trial solid mass (kg/m²), $H$ the all-liquid thickness (m), $d=-m/\rho_0$ the liquid thickness change and $h=H+d$. The layer bottom $b$ is fixed. Salt content $K=hS$ and liquidus temperature content $U=h\,CT=-\gamma K$ are constant. The liquid sensible term $\rho_0 c_p U$ cancels. The free-surface and ice loading terms combine with the category variance energy to give

$$
\tfrac12\rho_0g(\eta+m/\rho_0)^2+C\sum_k V_k^2/a_k,
\qquad C=\tfrac12\rho_i g(1-\rho_i/\rho_0).
$$

The square is constant. $V_k$ is the category volume per unit cell area and $a_k$ its area fraction. Empty categories contribute zero. Along the prescribed phase path each nonempty $V_k$ is proportional to $m$. Its energy is linear in $m$ when area scales at fixed thickness and quadratic when area stays fixed. Snow-first melting can change regimes at different masses for the two categories. Frozen cold content scales with retained solid volume during melt and stays constant during growth.

The remaining liquid energy comes from integrating the Roquet parcel potential. Substituting $CT=U/h$ and $S=K/h$ into the polynomial and integrating over $[b,b+h]$ gives the following factored difference and derivative. The per-column coefficients $A,B,D,F$ are assembled from the model EOS parameters and the initial column geometry.

$$
\Delta E_{\rm Roquet}=d\left[A+d(B+dD)+F/h\right],\qquad
\partial_m\Delta E_{\rm Roquet}=-\left[A+2Bd+3Dd^2+FH/h^2\right]/\rho_0 .
$$

The energy curve adds the category geometry, retained cold content and $-L_f m$ to this expression. Factoring out $d$ preserves small exchanges near complete melt. The solve still uses 56 bisections and three Newton updates. The third uses an independently evaluated residual from the reconstructed state and the original Hamiltonian. A finite Newton proposal outside the water/EOS bracket retains the prior bracketed mass.

## CPU float64 verification

The paired solver checks used the saved 4,096-cell, 40-level ocean at second 1,846,800 (day 21.375). Trial masses covered melt, new ice, growth and full coverage. Maximum differences against the direct Hamiltonian were 1.79e-7 J/m² in energy differences and 1.17e-10 J/kg in slopes. Near-boundary trials used the next representable energy inside each physical bracket.

Three independent 600 s phase calls used zero heat flux, 150 sin(latitude) W/m² and 60,000 sin(latitude) W/m². Maximum endpoint differences were 2.28e-13 kg/m² of ice and 4.45e-16 K in ocean temperature. Including prognostic ice cold content, the largest absolute global energy balance error per area was 5.59e-8 J/m². The largest local water residual was 1.03e-12 kg/m².

A separate fresh coupled comparison advanced two 2,400 s intervals using the 4,096-cell multirate preset, 32 atmosphere levels, 40 ocean levels and the saved mesh. Both runs used eager CPU float64 with radiation off. Maximum differences after 4,800 s were 2.10e-13 K in ocean temperature, 1.21e-13 g/kg in salinity and 7.67e-14 in ice category area fraction. The tests also exercise distinct snow-retreat and coverage regimes, a one-ulp margin at both phase bracket endpoints, and the heat-flux gradient against a forward finite difference.

## Paired phase benchmark

Each row times the complete 600 s phase call on four CPU threads, in float64 with 16 measurements per method after two warmup pairs. The day-21 comparison reused the saved ocean and ice fields. The fresh runs used the 4,096-cell campaign mesh and initial conditions.

| State and mode | Original Hamiltonian | Reduced Hamiltonian | Speedup |
| --- | ---: | ---: | ---: |
| Fresh, eager | 190.85 ms | 129.70 ms | 1.47× |
| Fresh, regional Inductor | 163.08 ms | 110.75 ms | 1.47× |
| Day 21.375, eager | 148.79 ms | 98.47 ms | 1.51× |

CUDA and fully coupled throughput were unmeasured.

Run from the checkout with VFS and the pinned PyAM21 installed. This example uses the fresh campaign surface, four host threads and the same regional compilation setting as the compiled A100 profile. Each method gets two warmup calls. Timed calls alternate order, synchronize the accelerator, start from the same state and apply the same 600 s, 150 sin(latitude) W/m² forcing. Optical data are unnecessary for this isolated operator.

```bash
python -u -I tools/benchmark_ice_phase.py --source "$PYAM21_SOURCE" \
    --mesh "$VFS_MESH" --device cuda:0 --compile --repeats 16
```
