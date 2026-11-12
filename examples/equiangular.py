"""Global integral and spherical-harmonic operator errors (CPU float64)."""
import torch
from vfs.geometry import EquiangularGrid

for nlat in (16, 32, 64, 128):
    grid = EquiangularGrid(nlat, 2*nlat)
    latitude, longitude = map(torch.deg2rad, grid.meshgrid())
    # Degree-one spherical harmonic: Laplacian(q) = -2 q / radius².
    q = latitude.cos()*longitude.cos()
    lap = grid.laplacian(q)
    scaled_error = grid.radius**2*lap + 2*q
    rms = grid.mean(scaled_error.square()).sqrt()
    polar_max = scaled_error[[0, -1]].abs().max()
    east, north = grid.grad(q)
    closure = grid.integral(q*lap) + grid.face_inner(east,north,east,north)
    print(f'{nlat}x{2*nlat}: scaled RMS={rms.item():.6g}, '
          f'polar max={polar_max.item():.6g}, adjoint residual={closure.item():.3g}')

# Column mass in kg/m² gives total mass in kg. Leading axes are retained.
column_mass = torch.full(grid.shape, 10000., dtype=grid.cell_area.dtype)
print('Uniform column mass (kg):', grid.integral(column_mass).item())

# Illustrative eastward stress (N/m²), supplied at cell centers.
tau_east = 0.1*latitude.cos()
tau_north = torch.zeros_like(tau_east)
stress_curl = grid.curl_from_collocated(tau_east, tau_north)  # N/m³, cell averages
native_curl = grid.curl_from_collocated(tau_east, tau_north, location='corner')
print('Global wind-stress curl integral (N/m):', grid.corner_integral(native_curl).item())
print('Cell wind-stress curl range (N/m³):', stress_curl.min().item(), stress_curl.max().item())
