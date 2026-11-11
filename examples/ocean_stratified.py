"""Installed-package example: active entropy/salt transport and explicit ALE.

CPU float64, three layers, no forcing. This is an execution example, not a
wave/balance accuracy benchmark. Run with python -I ocean_stratified.py.
"""
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean, AdaptiveThermalRemapper


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    mesh = VoronoiMesh(n_cells=32, lloyd_iters=2, verbose=False)
    column = OceanColumn(3, total_depth=3000.)
    ocean = StratifiedOcean(mesh, column, remapper=AdaptiveThermalRemapper())
    h = torch.tensor([500., 1000., 1500.]).expand(mesh.nCells, 3).clone()
    temperature = torch.tensor([17., 10., 4.])[None, :] + .4*mesh.cell_xyz[:, :1]
    salinity = torch.tensor([34., 35., 35.5])[None, :] + .05*mesh.cell_xyz[:, 1:2]
    state = ocean.state(u_e=torch.zeros(mesh.nEdges, 3), h=h,
        temperature=temperature, salinity=salinity, tracer=torch.ones(mesh.nCells, 3, 1))
    initial = ocean.budgets(state)
    with torch.no_grad():
        for _ in range(3):
            state, step_report = ocean.step_with_report(state, 10.)
        # Keep the new column depth; explicitly choose the target fractions.
        target_h = state.h.sum(-1, keepdim=True)*torch.tensor([.2, .3, .5])
        state, remap_report = ocean.remap_with_report(state, target_h)
    final = ocean.budgets(state)
    area = mesh.cell_area.sum()
    print('Energy change including timestep drift (J/m²):', float((final['total']-initial['total'])/area))
    print('Remap energy change only (J/m²):', float(remap_report.energy_change['total']/area))
    print('Salt inventory relative change:', float((final['salt']-initial['salt'])/initial['salt']))
    print('Temperature range (IPTS-68 C):', float(ocean.physical_fields(state)['temperature'].min()),
          float(ocean.physical_fields(state)['temperature'].max()))


if __name__ == '__main__':
    main()
