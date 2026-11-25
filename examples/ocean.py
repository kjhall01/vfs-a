"""Force the standalone campaign ocean with a prescribed surface heat flux."""
import torch

from vfs import VoronoiMesh
from vfs.ocean import Ocean, OceanColumn
from vfs.ocean.temperature.roquet import RoquetHeating


def main():
    torch.set_default_dtype(torch.float64)
    mesh = VoronoiMesh(n_cells=32, lloyd_iters=1, verbose=False)
    column = OceanColumn(3, total_depth=300.)
    ocean = Ocean(mesh, column, heating=RoquetHeating(0.6, (0.5, 20.)))
    h = torch.full((mesh.nCells, 3), 100.)
    state = ocean.state(u_e=h.new_zeros(mesh.nEdges, 3), h=h,
                        conservative_temperature=torch.full_like(h, 10.),
                        absolute_salinity=torch.full_like(h, 35.))
    state, report = ocean.step_with_report(state, 60., surface_heat=h.new_full((mesh.nCells,), 50.))
    print("Surface CT (°C):", state.temperature()[:, 0].mean().item())
    print("Applied heat (J):", report.heating.external_heat.item())


if __name__ == "__main__":
    main()
