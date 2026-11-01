# Rotating-flow diagnosis, with production frozen

Starting main: `e959b7005832a5041e082b6e300838e15bfbaa9e`. #26 merge
`735785c7cf5edae2f363f8c61ca7d0c8409496e3` and #27 merge `e959b700...`
are ancestors. This isolated ocean investigation changes no runtime code,
fixtures, tolerances or defaults. Diagnosis precedes any numerical proposal.

The five-day axis-aligned Williamson case is replayed before operator probes.
The 64-cell mesh is frozen published input; larger meshes are regenerated,
explicitly not a byte-for-byte claim about omitted original mesh archives.
The baseline's exact ocean/atmosphere source checks remain enabled.

The diagnostic driver separates point/cell/dual sampling, gradient truncation,
kinetic reconstruction, vorticity reconstruction, raw/shared-skew/layer-skew
rotation, and local vector moments. Results and limitations follow in a separate
evidence commit. No ablated expression is used to advance a benchmark state.
