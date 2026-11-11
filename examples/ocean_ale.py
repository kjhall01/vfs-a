"""Active JMD95 ocean with bounded vertical ALE remapping after each full step.

CPU float64 demonstration. Horizontal transport remains centered and checked,
not bounded. Report remap energy changes explicitly; no exact energy claim.
"""
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import Ocean, OceanColumn
from vfs.ocean.jmd95 import JMD95Dynamics
from vfs.ocean.ale import ALEStepper


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    mesh = VoronoiMesh(n_cells=32, lloyd_iters=2, verbose=False)
    column = OceanColumn(4, total_depth=3000., rest_thickness=torch.tensor([300.,600.,900.,1200.]))
    stepper = ALEStepper()
    model = Ocean(mesh, column, dynamics=JMD95Dynamics(), stepper=stepper)
    h = column.rest_thickness.expand(mesh.nCells,4).clone()
    temperature = torch.tensor([18.,12.,7.,3.])[None,:] + .2*mesh.cell_xyz[:,:1]
    salinity = torch.tensor([34.,34.5,35.,35.5])[None,:].expand_as(h)
    state = model.state(u_e=torch.zeros(mesh.nEdges,4),h=h,temperature=temperature,salinity=salinity)
    initial = model.budgets(state)
    remap_energy = 0.
    with torch.no_grad():
        for _ in range(20):
            state, report = stepper.step_with_report(model,state,30.)
            remap_energy += float(report.energy_change['total'])
    final = model.budgets(state)
    for name in ('volume','temperature_content','salinity_content'):
        print(name, 'relative change:',float((final[name]-initial[name])/initial[name]))
    energy_change = float(final['jmd95_hamiltonian']-initial['jmd95_hamiltonian'])
    print('Hamiltonian change (J):',energy_change)
    print('sum of remap changes (J):',remap_energy)
    print('remaining dynamics/time-integration change (J):',energy_change-remap_energy)
    print('minimum layer thickness (m):',float(state.h.min()))


if __name__ == '__main__':
    main()
