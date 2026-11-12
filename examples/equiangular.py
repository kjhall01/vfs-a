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
