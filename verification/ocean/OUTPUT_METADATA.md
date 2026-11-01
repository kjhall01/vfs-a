# Ocean output follows the selected equations

Ocean temperature is stored in degrees Celsius and labelled `degC`. The same
metadata correction is also applied to parent #23 so it stands alone correctly.
No temperature values, numerical kernels or fixtures change.

Use the model-owned output path for selected dynamics:

```python
output = model.to_latlon(state, remapper, fields="all")
```

This validates model/state ownership and passes `model.dynamics.density` to the
ocean adapter. ConstantDensityDynamics therefore outputs rho0 even when passive
temperature and salinity vary. The archived linear and JMD95 options use their
selected EOS and coefficients. Output tensors retain their autograd connections.
The interpolation remains nonconservative.

The lower-level `state_to_latlon` accepts an explicit `density_provider(state)`.
For compatibility, omission retains `state.density()` and labels its metadata
`density_convention=legacy_state_linear_eos`. A state alone does not identify
which dynamics created it. Model output uses `selected_dynamics` metadata and
records the provider class and EOS name where available. Density is evaluated
only when requested.

The regression suite covers constant, linear (nondefault alpha), and JMD95
models with spatially varying temperature and salinity, matching native model
physical fields after remapping. It checks gradients for active EOS choices,
Celsius and density units, rejected unrelated columns, and lazy density calls.

Historical energy experiment source hashes retain the files that actually
produced those measurements; no campaign is rerun or requalified by this output
fix. Ocean remains a plain facade: an nn.Module ownership migration is separate.
