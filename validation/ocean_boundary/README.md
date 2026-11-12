# Boundary column audit

Run `python validation/ocean_boundary/run.py` against an installed wheel.
NumPy and SciPy are validation dependencies; no external data are downloaded.

The transient uses VFS wind-kick and closed vertical-diffusion kernels with exact
f-plane rotation. Its independent reference is a dense augmented matrix
exponential for the same constant-viscosity column. Timestep halving checks the
split first-order scheme; a work/loss ledger checks finite-step energy.

The steady comparison solves the independent finite-volume linear system and
compares against **cell averages**, not point samples, of the finite-depth
analytic Ekman spiral, with prescribed top stress and zero bottom stress.
For z in [-D,0], complex U=u+iv and lambda=(1+i)*sqrt(f/(2*nu)),

    U(z) = tau*cosh(lambda*(z+D))/(rho*nu*lambda*sinh(lambda*D)).

Its integrated transport is -i*tau/(rho*f). The top-to-bottom layer ordering
matches VFS. A closed, undamped barotropic inertial mode means transient
spin-up need not settle to this steady solution; the two tests are distinct.

Quadratic drag is checked against the exact fixed-direction spin-down law.
This test harness is not a new grid or a replacement for the spherical rotation
operator, and does not validate evolving-TKE entrainment or bottom Ekman layers.
The separate `examples/ocean_boundary_stress.py` exercises actual spherical
resolved dynamics, surface stress, bottom drag and per-layer TKE together.

Run `python validation/ocean_boundary/bathymetry.py` for varying-depth drag on
32/128/512/2048-cell meshes. The reference uses analytic smooth vector speed
with the same discrete interpolated thickness, isolating wind reconstruction
error. It reports unchanged rest-pressure residuals separately.

`radiation.py` compares analytic layer-integrated two-band absorption on
1/4/16/64 layers for both bottom policies, and measures independent Hamiltonian
energy increments. `examples/ocean_radiation.py` checks short combined evolution.

`compare_heating.py --repository /path/to/vfs-a` loads the pinned pre-extraction
TKE module from local git history and compares its outputs exactly with the
installed candidate. It does not modify the checkout or install another wheel.

`exchange.py` checks closed mass/salinity targets and constant-flux subdivision
consistency for the prescribed freshwater/energy operator. The combined example
`examples/ocean_surface_fluxes.py` also exercises radiation, stress and TKE.
