"""Advance a Roquet ocean with two sea-ice categories and inspect the energy budget."""
import torch

from vfs import VoronoiMesh
from vfs.ice import OceanIce
from vfs.ocean import Ocean, OceanColumn


def main():
    torch.set_default_dtype(torch.float64)
    mesh = VoronoiMesh(n_cells=32, lloyd_iters=1, verbose=False)
    ocean = Ocean(mesh, OceanColumn(3, total_depth=300.))
    h = torch.full((mesh.nCells, 3), 100.)
    water = ocean.state(u_e=h.new_zeros(mesh.nEdges, 3), h=h,
                        conservative_temperature=torch.full_like(h, -1.836),
                        absolute_salinity=torch.full_like(h, 34.), tke=1e-4)
    model = OceanIce(ocean)
    area = h.new_tensor([0.4, 0.1]).expand(mesh.nCells, -1)
    state = model.state(water, area=area, volume=area * h.new_tensor([0.3, 1.0]),
                        velocity=h.new_zeros(mesh.nEdges, 1), temperature=265.)
    energy = model.total_energy(state)
    end, report = model.step(state, 60.)
    residual = (model.total_energy(end) - energy - report.external_energy) / mesh.cell_area.sum()
    print("Cold content in cell 0 (J/m²)", end.ice.cold_content[0].tolist())
    print("Combined numerical energy residual (J/m²)", residual.item())


if __name__ == "__main__":
    with torch.no_grad():
        main()
