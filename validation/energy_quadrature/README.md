# Discrete energy cost and accuracy

Run outside the checkout with the candidate wheel installed:

```sh
python -I validation/energy_quadrature/accuracy.py --output accuracy.json
python -I validation/energy_quadrature/kernels.py --compile --output kernels.json
python -I validation/performance/field_reuse.py --package /tmp/unpacked-wheel \
  --case full --cells 1024 --layers 64 --steps 1 --warm \
  --energy-quadrature 2 --output full-q2.json
```

Compare to order 4 with the same physical input and timestep. `--compile-thermo`
and `--compile-gibbs` together select the per-model compiled thermodynamics;
individually they select isolated kernel experiments. Run all timing processes
sequentially; do not overlap tests, compilations or other timing jobs. Compare
endpoint tensors using `validation/performance/compare_field_reuse.py` with
`--allow-roundoff`, which reports differences but does not certify accuracy.

Accuracy references use the same layerwise physical states and higher quadrature,
not an analytical wave or continuum solution. Energy derivatives are independently
checked by `tests/ocean/test_energy_quadrature.py`. The full timing profiles are
100 m deep, so deep-ocean accuracy comes from the separate operator campaign.
Tensor captures are local intermediates and must not be uploaded if over 128 KiB.

For a local editable checkout, `--package .` selects that checkout explicitly;
you do not need a wheel to try the benchmark. Installed-wheel execution is used
for the recorded regression evidence. The compiler experiments require a working
PyTorch CPU compilation toolchain; Mac/MPS behavior is not qualified here.
